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

//! [`PropagateEqualities`] propagates constant equalities across equivalent columns.

use std::collections::HashMap;

use datafusion_common::tree_node::Transformed;
use datafusion_common::{Result, internal_datafusion_err};
use datafusion_expr::expr::BinaryExpr;
use datafusion_expr::logical_plan::{Filter, LogicalPlan};
use datafusion_expr::utils::{conjunction, split_conjunction_owned};
use datafusion_expr::{Expr, Operator};

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};

/// Optimization rule that propagates equality predicates through equivalent columns.
///
/// For example, `a = b AND a = 5` is rewritten to
/// `a = b AND a = 5 AND b = 5`.
#[derive(Default, Debug)]
pub struct PropagateEqualities;

impl PropagateEqualities {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for PropagateEqualities {
    fn name(&self) -> &str {
        "propagate_equalities"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Filter(Filter {
            predicate, input, ..
        }) = plan
        else {
            return Ok(Transformed::no(plan));
        };

        let predicates = split_conjunction_owned(predicate);
        if predicates.len() < 2 {
            return filter_from_predicates(predicates, input).map(Transformed::no);
        }

        let mut equivalence_classes = EquivalenceClasses::new();

        // Phase 1: classify predicates.
        for pred in &predicates {
            match classify_predicate(pred) {
                PredicateKind::ColumnEquality { left, right } => {
                    equivalence_classes.union_columns(left, right);
                }
                PredicateKind::ColumnConstant { column, .. } => {
                    equivalence_classes.ensure_registered(column);
                }
                PredicateKind::Other => {}
            }
        }

        // Phase 2: bind constants to equivalence classes.
        for pred in &predicates {
            if let PredicateKind::ColumnConstant { column, constant } =
                classify_predicate(pred)
            {
                equivalence_classes.bind_constant(column, constant.clone());
            }
        }

        // Phase 3: generate new predicates from propagated constants.
        let propagated = equivalence_classes.propagated_constants();
        let mut existing_col_const = vec![];
        for pred in &predicates {
            if let PredicateKind::ColumnConstant { column, constant } =
                classify_predicate(pred)
            {
                existing_col_const.push((column, constant));
            }
        }

        let mut new_predicates = vec![];
        for (column, constant) in propagated {
            if !existing_col_const
                .iter()
                .any(|(existing_col, existing_const)| {
                    *existing_col == &column && *existing_const == &constant
                })
            {
                new_predicates.push(Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(column),
                    Operator::Eq,
                    Box::new(constant),
                )));
            }
        }

        if new_predicates.is_empty() {
            return filter_from_predicates(predicates, input).map(Transformed::no);
        };

        let predicates = predicates
            .into_iter()
            .chain(new_predicates)
            .collect::<Vec<_>>();
        filter_from_predicates(predicates, input).map(Transformed::yes)
    }
}

fn filter_from_predicates(
    predicates: Vec<Expr>,
    input: std::sync::Arc<LogicalPlan>,
) -> Result<LogicalPlan> {
    let predicate = conjunction(predicates)
        .ok_or_else(|| internal_datafusion_err!("Filter predicate unexpectedly empty"))?;
    Ok(LogicalPlan::Filter(Filter::try_new(predicate, input)?))
}

#[derive(Default, Debug)]
struct EquivalenceClasses {
    expr_to_index: HashMap<Expr, usize>,
    parent: Vec<usize>,
    constants: HashMap<usize, Expr>,
}

impl EquivalenceClasses {
    fn new() -> Self {
        Self::default()
    }

    fn propagated_constants(&mut self) -> Vec<(Expr, Expr)> {
        let mut expressions = self
            .expr_to_index
            .iter()
            .map(|(expr, index)| (expr.clone(), *index))
            .collect::<Vec<_>>();
        expressions.sort_by_key(|(_, index)| *index);

        let mut propagated = vec![];

        for (expr, index) in expressions {
            let root = self.find(index);
            let Some(constant) = self.constants.get(&root).cloned() else {
                continue;
            };

            propagated.push((expr, constant));
        }

        propagated
    }

    fn union_columns(&mut self, left: &Expr, right: &Expr) {
        let left = self.ensure_registered(left);
        let right = self.ensure_registered(right);
        let left_root = self.find(left);
        let right_root = self.find(right);

        if left_root == right_root {
            return;
        }

        self.parent[right_root] = left_root;
        if let Some(constant) = self.constants.remove(&right_root) {
            self.constants.entry(left_root).or_insert(constant);
        }
    }

    fn bind_constant(&mut self, column: &Expr, literal: Expr) {
        let index = self.ensure_registered(column);
        let root = self.find(index);
        self.constants.entry(root).or_insert(literal);
    }

    fn ensure_registered(&mut self, expr: &Expr) -> usize {
        if let Some(id) = self.expr_to_index.get(expr) {
            *id
        } else {
            let id = self.parent.len();
            self.expr_to_index.insert(expr.clone(), id);
            self.parent.push(id);
            id
        }
    }

    fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }
}

fn equality_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
    match expr {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) => Some((left.as_ref(), right.as_ref())),
        _ => None,
    }
}

enum PredicateKind<'a> {
    ColumnEquality {
        left: &'a Expr,
        right: &'a Expr,
    },
    ColumnConstant {
        column: &'a Expr,
        constant: &'a Expr,
    },
    Other,
}

fn classify_predicate(expr: &Expr) -> PredicateKind<'_> {
    let Some((left, right)) = equality_operands(expr) else {
        return PredicateKind::Other;
    };

    match (left, right) {
        (Expr::Column(_), Expr::Column(_)) => {
            PredicateKind::ColumnEquality { left, right }
        }
        (Expr::Column(_), Expr::Literal(value, _)) if !value.is_null() => {
            PredicateKind::ColumnConstant {
                column: left,
                constant: right,
            }
        }
        (Expr::Literal(value, _), Expr::Column(_)) if !value.is_null() => {
            PredicateKind::ColumnConstant {
                column: right,
                constant: left,
            }
        }
        _ => PredicateKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion_common::{Result, ScalarValue};
    use datafusion_expr::logical_plan::builder::LogicalPlanBuilder;
    use datafusion_expr::{Expr, col, lit};

    use crate::OptimizerContext;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::propagate_equalities::PropagateEqualities;
    use crate::test::test_table_scan;

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn crate::OptimizerRule + Send + Sync>> =
                vec![Arc::new(PropagateEqualities::new())];
            assert_optimized_plan_eq_snapshot!(
                optimizer_ctx,
                rules,
                $plan,
                @ $expected,
            )
        }};
    }

    #[test]
    fn propagates_constant_through_column_equality() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .filter(col("a").eq(col("b")).and(col("a").eq(lit(5u32))))?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Filter: test.a = test.b AND test.a = UInt32(5) AND test.b = UInt32(5)
          TableScan: test
        ")
    }

    #[test]
    fn propagates_constant_through_transitive_equalities() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .filter(
                col("a")
                    .eq(col("b"))
                    .and(col("b").eq(col("c")))
                    .and(col("c").eq(lit(5u32))),
            )?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Filter: test.a = test.b AND test.b = test.c AND test.c = UInt32(5) AND test.a = UInt32(5) AND test.b = UInt32(5)
          TableScan: test
        ")
    }

    #[test]
    fn does_not_duplicate_existing_constant_predicates() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .filter(
                col("a")
                    .eq(col("b"))
                    .and(col("a").eq(lit(5u32)))
                    .and(col("b").eq(lit(5u32))),
            )?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Filter: test.a = test.b AND test.a = UInt32(5) AND test.b = UInt32(5)
          TableScan: test
        ")
    }

    #[test]
    fn ignores_non_conjunctive_equalities() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .filter(col("a").eq(col("b")).or(col("a").eq(lit(5u32))))?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Filter: test.a = test.b OR test.a = UInt32(5)
          TableScan: test
        ")
    }

    #[test]
    fn does_not_propagate_null_literals() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .filter(
                col("a")
                    .eq(col("b"))
                    .and(col("a").eq(Expr::Literal(ScalarValue::Int32(None), None))),
            )?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Filter: test.a = test.b AND test.a = Int32(NULL)
          TableScan: test
        ")
    }
}
