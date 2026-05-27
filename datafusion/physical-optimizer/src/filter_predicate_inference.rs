// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Infer additional physical filter predicates from existing predicates.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::PhysicalOptimizerRule;

use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_expr::Operator;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion_physical_expr::intervals::cp_solver::{
    ExprIntervalGraph, PropagationResult,
};
use datafusion_physical_expr::utils::{collect_columns, conjunction, split_conjunction};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter::{FilterExec, FilterExecBuilder};

const MAX_PROPAGATION_PASSES: usize = 8;

#[derive(Debug, Default)]
pub struct FilterPredicateInference {}

impl FilterPredicateInference {
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for FilterPredicateInference {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_filter_predicate_inference {
            return Ok(plan);
        }

        plan.transform_down(|node| match try_transform_filter(&node)? {
            None => Ok(Transformed::no(node)),
            Some(InferenceOutcome::Augmented(plan)) => Ok(Transformed::yes(plan)),
            Some(InferenceOutcome::Infeasible(schema)) => {
                Ok(Transformed::yes(Arc::new(EmptyExec::new(schema))))
            }
        })
        .data()
    }

    fn name(&self) -> &str {
        "FilterPredicateInference"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

enum InferenceOutcome {
    Augmented(Arc<dyn ExecutionPlan>),
    Infeasible(arrow::datatypes::SchemaRef),
}

fn try_transform_filter(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<InferenceOutcome>> {
    let Some(filter) = plan.downcast_ref::<FilterExec>() else {
        return Ok(None);
    };

    let predicate = filter.predicate();
    let input_schema = filter.input().schema();
    let output_schema = filter.schema();

    let mut output_terms: Vec<Arc<dyn PhysicalExpr>> =
        split_conjunction(predicate).into_iter().cloned().collect();
    let mut working_terms = output_terms.clone();
    let mut derived_seen = Vec::new();
    let original_len = output_terms.len();

    // Rebuild the whole predicate graph until no new predicate is found.
    // A single cp_solver pass may only tighten one occurrence of a repeated
    // column, so equality chains such as `a = 5 AND a = b` need another pass.
    for _ in 0..MAX_PROPAGATION_PASSES {
        let working_predicate = conjunction(working_terms.clone());
        let Some(derived) = infer_predicates(&working_predicate, input_schema.as_ref())?
        else {
            return Ok(Some(InferenceOutcome::Infeasible(output_schema)));
        };

        let mut changed = false;
        for predicate in derived {
            if !derived_seen
                .iter()
                .any(|existing| is_duplicate_predicate(existing, &predicate))
            {
                derived_seen.push(Arc::clone(&predicate));
                working_terms.push(Arc::clone(&predicate));
                changed = true;
            }

            if output_terms
                .iter()
                .any(|existing| is_duplicate_predicate(existing, &predicate))
            {
                continue;
            }
            output_terms.push(predicate);
            changed = true;
        }

        if !changed {
            break;
        }
    }

    if output_terms.len() == original_len {
        return Ok(None);
    }

    let new_filter = FilterExecBuilder::from(filter)
        .with_predicate(conjunction(output_terms))
        .build()?;

    Ok(Some(InferenceOutcome::Augmented(Arc::new(new_filter))))
}

fn infer_predicates(
    predicate: &Arc<dyn PhysicalExpr>,
    schema: &arrow::datatypes::Schema,
) -> Result<Option<Vec<Arc<dyn PhysicalExpr>>>> {
    let mut graph = match ExprIntervalGraph::try_new(Arc::clone(predicate), schema) {
        Ok(graph) => graph,
        Err(_) => return Ok(Some(vec![])),
    };

    let mut columns: Vec<_> = collect_columns(predicate).into_iter().collect();
    columns.sort_by_key(|column| column.index());
    if columns.is_empty() {
        return Ok(Some(vec![]));
    }

    let col_exprs: Vec<Arc<dyn PhysicalExpr>> = columns
        .iter()
        .map(|column| Arc::new(column.clone()) as Arc<dyn PhysicalExpr>)
        .collect();
    let node_indices = graph.gather_node_indices(&col_exprs);

    let mut leaves: Vec<(usize, Interval)> = columns
        .iter()
        .zip(node_indices.iter())
        .filter_map(|(column, (_, node_idx))| {
            if *node_idx == usize::MAX {
                return None;
            }
            let dt = column.data_type(schema).ok()?;
            let unbounded = Interval::make_unbounded(&dt).ok()?;
            Some((*node_idx, unbounded))
        })
        .collect();

    if leaves.is_empty() {
        return Ok(Some(vec![]));
    }

    match graph.update_ranges(&mut leaves, Interval::TRUE) {
        Ok(PropagationResult::Infeasible) => return Ok(None),
        Ok(PropagationResult::CannotPropagate) => return Ok(Some(vec![])),
        Ok(PropagationResult::Success) => {}
        Err(_) => return Ok(Some(vec![])),
    }

    let propagated: BTreeMap<usize, Interval> = columns
        .iter()
        .zip(leaves.iter())
        .map(|(column, (_, interval))| (column.index(), interval.clone()))
        .collect();

    intervals_to_predicates(&propagated, schema).map(Some)
}

fn is_duplicate_predicate(
    existing: &Arc<dyn PhysicalExpr>,
    candidate: &Arc<dyn PhysicalExpr>,
) -> bool {
    if existing.eq(candidate) {
        return true;
    }

    let Some(existing) = existing.downcast_ref::<BinaryExpr>() else {
        return false;
    };
    let Some(candidate) = candidate.downcast_ref::<BinaryExpr>() else {
        return false;
    };

    reversed_operator(existing.op()) == Some(*candidate.op())
        && existing.left().eq(candidate.right())
        && existing.right().eq(candidate.left())
}

fn reversed_operator(operator: &Operator) -> Option<Operator> {
    match operator {
        Operator::Eq => Some(Operator::Eq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        _ => None,
    }
}

fn intervals_to_predicates(
    new: &BTreeMap<usize, Interval>,
    schema: &arrow::datatypes::Schema,
) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
    let mut out = Vec::new();
    for (&idx, interval) in new {
        let field = schema.field(idx);
        let baseline = Interval::make_unbounded(field.data_type())?;
        if interval == &baseline {
            continue;
        };

        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new(field.name(), idx));
        let lower = interval.lower();
        let upper = interval.upper();

        if !lower.is_null() && lower == upper {
            out.push(Arc::new(BinaryExpr::new(
                Arc::clone(&column),
                Operator::Eq,
                Arc::new(Literal::new(lower.clone())),
            )) as _);
            continue;
        }

        if !lower.is_null() && lower != baseline.lower() {
            out.push(Arc::new(BinaryExpr::new(
                Arc::clone(&column),
                Operator::GtEq,
                Arc::new(Literal::new(lower.clone())),
            )) as _);
        }
        if !upper.is_null() && upper != baseline.upper() {
            out.push(Arc::new(BinaryExpr::new(
                Arc::clone(&column),
                Operator::LtEq,
                Arc::new(Literal::new(upper.clone())),
            )) as _);
        }
    }

    Ok(out)
}
