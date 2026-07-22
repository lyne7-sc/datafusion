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

use crate::PhysicalOptimizerRule;
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_expr::{LimitEffect, WindowFrameBound, WindowFrameUnits};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::window::{
    PlainAggregateWindowExpr, SlidingAggregateWindowExpr, StandardWindowExpr,
    StandardWindowFunctionExpr, WindowExpr,
};
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowUDFExpr};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use std::cmp;
use std::collections::HashSet;
use std::sync::Arc;

/// This rule inspects [`ExecutionPlan`]'s attempting to find fetch limits that were not pushed
/// down by `LimitPushdown` because [BoundedWindowAggExec]s were "in the way". If the window is
/// bounded by [WindowFrameUnits::Rows] then we calculate the adjustment needed to grow the limit
/// and continue pushdown.
#[derive(Default, Clone, Debug)]
pub struct LimitPushPastWindows;

impl LimitPushPastWindows {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Eq, PartialEq)]
enum Phase {
    FindOrGrow,
    Apply,
}

struct TraverseState {
    pub limit: Option<usize>,
    pub lookahead: usize,
    /// Columns required by windows above the current node. `None` means their
    /// lineage could not be determined safely.
    required_columns: Option<HashSet<usize>>,
}

impl Default for TraverseState {
    fn default() -> Self {
        Self {
            limit: None,
            lookahead: 0,
            required_columns: Some(HashSet::new()),
        }
    }
}

impl TraverseState {
    pub fn reset_limit(&mut self, limit: Option<usize>) {
        self.limit = limit;
        self.lookahead = 0;
        self.required_columns = Some(HashSet::new());
    }
}

impl PhysicalOptimizerRule for LimitPushPastWindows {
    fn optimize(
        &self,
        original: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_window_limits {
            return Ok(original);
        }
        let mut ctx = TraverseState::default();
        let mut phase = Phase::FindOrGrow;
        let result = original.transform_down(|node| {
            // helper closure to DRY out most the early return cases
            let reset = |node,
                         ctx: &mut TraverseState|
             -> datafusion_common::Result<
                Transformed<Arc<dyn ExecutionPlan>>,
            > {
                ctx.reset_limit(None);
                Ok(Transformed::no(node))
            };

            // traversing sides of joins will require more thought
            if node.children().len() > 1 {
                return reset(node, &mut ctx);
            }

            // grab the latest limit we see
            if phase == Phase::FindOrGrow && get_limit(&node, &mut ctx) {
                return Ok(Transformed::no(node));
            }

            // grow the limit if we hit a window function
            if let Some(window) = node.downcast_ref::<BoundedWindowAggExec>() {
                phase = Phase::Apply;
                if !grow_limit(window, &mut ctx) {
                    return reset(node, &mut ctx);
                }
                return Ok(Transformed::no(node));
            }

            // Apply the limit if we hit a sortpreservingmerge node
            if phase == Phase::Apply
                && let Some(out) = apply_limit(&node, &mut ctx)
            {
                return Ok(out);
            }

            // nodes along the way
            if !node.supports_limit_pushdown() {
                return reset(node, &mut ctx);
            }
            if let Some(part) = node.downcast_ref::<RepartitionExec>() {
                let output = part.partitioning().partition_count();
                let input = part.input().output_partitioning().partition_count();
                if output < input {
                    return reset(node, &mut ctx);
                }
            }
            match node.cardinality_effect() {
                CardinalityEffect::Unknown => return reset(node, &mut ctx),
                CardinalityEffect::LowerEqual => return reset(node, &mut ctx),
                CardinalityEffect::Equal => {}
                CardinalityEffect::GreaterEqual => {}
            }

            ctx.required_columns = ctx
                .required_columns
                .take()
                .and_then(|columns| map_required_columns_to_input(&node, columns));
            Ok(Transformed::no(node))
        })?;
        Ok(result.data)
    }

    fn name(&self) -> &str {
        "LimitPushPastWindows"
    }

    fn schema_check(&self) -> bool {
        false // we don't change the schema
    }
}

fn grow_limit(window: &BoundedWindowAggExec, ctx: &mut TraverseState) -> bool {
    let mut max_rel = 0;
    let mut max_frame = 0;
    for expr in window.window_expr().iter() {
        // grow based on function requirements
        match get_limit_effect(expr) {
            LimitEffect::None => {}
            LimitEffect::Unknown => return false,
            LimitEffect::Relative(rel) => max_rel = max_rel.max(rel),
            LimitEffect::Absolute(val) => {
                let cur = ctx.limit.unwrap_or(0);
                ctx.limit = Some(cur.max(val))
            }
        }

        // grow based on frames
        let frame = expr.get_window_frame();
        if frame.units != WindowFrameUnits::Rows {
            return false; // expression-based limits not statically evaluatable
        }
        let Some(end_bound) = bound_to_usize(&frame.end_bound) else {
            return false; // can't optimize unbounded window expressions
        };
        max_frame = max_frame.max(end_bound);
    }

    // Each window operator needs enough input for both its largest frame and
    // its largest relative function requirement.
    let Some(window_lookahead) = max_frame.checked_add(max_rel) else {
        return false;
    };

    let input_len = window.input().schema().fields().len();
    let accumulate = match (
        &mut ctx.required_columns,
        collect_window_required_columns(window, input_len),
    ) {
        (Some(previous), Some(current)) => {
            let accumulate = previous.iter().any(|index| *index >= input_len);
            previous.retain(|index| *index < input_len);
            previous.extend(current);
            accumulate
        }
        (required_columns, _) => {
            // Unknown lineage requires conservative accumulation.
            *required_columns = None;
            true
        }
    };
    ctx.lookahead = if accumulate {
        let Some(lookahead) = ctx.lookahead.checked_add(window_lookahead) else {
            return false;
        };
        lookahead
    } else {
        ctx.lookahead.max(window_lookahead)
    };

    true
}

fn apply_limit(
    node: &Arc<dyn ExecutionPlan>,
    ctx: &mut TraverseState,
) -> Option<Transformed<Arc<dyn ExecutionPlan>>> {
    if !node.is::<SortExec>() && !node.is::<SortPreservingMergeExec>() {
        return None;
    }
    let latest = ctx.limit.take();
    let Some(fetch) = latest else {
        ctx.reset_limit(None);
        return Some(Transformed::no(Arc::clone(node)));
    };
    let Some(fetch) = fetch.checked_add(ctx.lookahead) else {
        ctx.reset_limit(None);
        return Some(Transformed::no(Arc::clone(node)));
    };
    let fetch = match node.fetch() {
        None => fetch,
        Some(existing) => cmp::min(existing, fetch),
    };
    Some(Transformed::complete(node.with_fetch(Some(fetch)).unwrap()))
}

fn get_limit(node: &Arc<dyn ExecutionPlan>, ctx: &mut TraverseState) -> bool {
    if let Some(limit) = node.downcast_ref::<GlobalLimitExec>() {
        ctx.reset_limit(
            limit
                .fetch()
                .and_then(|fetch| fetch.checked_add(limit.skip())),
        );
        return true;
    }
    // In distributed execution, GlobalLimitExec becomes LocalLimitExec
    // per partition. Handle it the same way (LocalLimitExec has no skip).
    if let Some(limit) = node.downcast_ref::<LocalLimitExec>() {
        ctx.reset_limit(Some(limit.fetch()));
        return true;
    }
    if let Some(limit) = node.downcast_ref::<SortPreservingMergeExec>() {
        ctx.reset_limit(limit.fetch());
        return true;
    }
    false
}

fn collect_window_required_columns(
    window: &BoundedWindowAggExec,
    input_len: usize,
) -> Option<HashSet<usize>> {
    let required_columns = window
        .window_expr()
        .iter()
        .flat_map(|expr| {
            expr.expressions()
                .into_iter()
                .chain(expr.partition_by().iter().cloned())
                .chain(expr.order_by().iter().map(|sort| Arc::clone(&sort.expr)))
        })
        .flat_map(|expr| collect_columns(&expr))
        .map(|column| column.index())
        .collect::<HashSet<_>>();

    (!required_columns.iter().any(|index| *index >= input_len))
        .then_some(required_columns)
}

fn map_required_columns_to_input(
    node: &Arc<dyn ExecutionPlan>,
    columns: HashSet<usize>,
) -> Option<HashSet<usize>> {
    if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
        let mut input_columns = HashSet::new();
        for index in columns {
            let projection_expr = projection.expr().get(index)?;
            input_columns.extend(
                collect_columns(&projection_expr.expr)
                    .into_iter()
                    .map(|column| column.index()),
            );
        }
        return Some(input_columns);
    }

    let children = node.children();
    let Some(input) = children.first() else {
        return Some(columns);
    };

    (node.schema() == input.schema()
        && columns
            .iter()
            .all(|index| *index < input.schema().fields().len()))
    .then_some(columns)
}

/// Examines the `WindowExpr` and decides:
/// 1. The expression does not change the window size
/// 2. The expression grows it by X amount
/// 3. We don't know
///
/// # Arguments
///
/// * `expr` the expression to examine
///
/// # Returns
///
/// The effect on the limit
fn get_limit_effect(expr: &Arc<dyn WindowExpr>) -> LimitEffect {
    // White list aggregates
    if expr.as_any().is::<PlainAggregateWindowExpr>()
        || expr.as_any().is::<SlidingAggregateWindowExpr>()
    {
        return LimitEffect::None;
    }

    // Grab the window function
    let Some(swe) = expr.as_any().downcast_ref::<StandardWindowExpr>() else {
        return LimitEffect::Unknown; // should be only remaining type
    };
    let swfe = swe.get_standard_func_expr();
    let Some(udf) = swfe.as_any().downcast_ref::<WindowUDFExpr>() else {
        return LimitEffect::Unknown; // should be only remaining type
    };
    udf.limit_effect()
}

fn bound_to_usize(bound: &WindowFrameBound) -> Option<usize> {
    match bound {
        WindowFrameBound::Preceding(_) => Some(0),
        WindowFrameBound::CurrentRow => Some(0),
        WindowFrameBound::Following(ScalarValue::UInt64(Some(scalar))) => {
            usize::try_from(*scalar).ok()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_expr::WindowFrame;
    use datafusion_functions_window::lead_lag::lead_udwf;
    use datafusion_functions_window::row_number::row_number_udwf;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::InputOrderMode;
    use datafusion_physical_plan::displayable;
    use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
    use datafusion_physical_plan::windows::{
        BoundedWindowAggExec, create_udwf_window_expr,
    };
    use insta::assert_snapshot;

    fn plan_str(plan: &dyn ExecutionPlan) -> String {
        displayable(plan).indent(true).to_string()
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]))
    }

    /// Build: LocalLimitExec or GlobalLimitExec → BoundedWindowAggExec(row_number) → SortExec
    fn build_window_plan(
        use_local_limit: bool,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        let s = schema();
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(PlaceholderRowExec::new(Arc::clone(&s)));

        let ordering =
            LexOrdering::new(vec![PhysicalSortExpr::new_default(col("a", &s)?).asc()])
                .unwrap();

        let sort: Arc<dyn ExecutionPlan> = Arc::new(
            SortExec::new(ordering.clone(), input).with_preserve_partitioning(true),
        );

        let window_expr = Arc::new(StandardWindowExpr::new(
            create_udwf_window_expr(
                &row_number_udwf(),
                &[],
                &s,
                "row_number".to_string(),
                false,
            )?,
            &[],
            ordering.as_ref(),
            Arc::new(WindowFrame::new_bounds(
                WindowFrameUnits::Rows,
                WindowFrameBound::Preceding(ScalarValue::UInt64(None)),
                WindowFrameBound::CurrentRow,
            )),
        ));

        let window: Arc<dyn ExecutionPlan> = Arc::new(BoundedWindowAggExec::try_new(
            vec![window_expr],
            sort,
            InputOrderMode::Sorted,
            true,
        )?);

        let limit: Arc<dyn ExecutionPlan> = if use_local_limit {
            Arc::new(LocalLimitExec::new(window, 100))
        } else {
            Arc::new(GlobalLimitExec::new(window, 0, Some(100)))
        };

        Ok(limit)
    }

    fn lead_window(
        input: Arc<dyn ExecutionPlan>,
        argument: &str,
        name: &str,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let ordering = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("a", &schema)?).asc(),
        ])
        .unwrap();
        let window_expr = Arc::new(StandardWindowExpr::new(
            create_udwf_window_expr(
                &lead_udwf(),
                &[col(argument, &schema)?],
                &schema,
                name.to_string(),
                false,
            )?,
            &[],
            ordering.as_ref(),
            Arc::new(WindowFrame::new_bounds(
                WindowFrameUnits::Rows,
                WindowFrameBound::Preceding(ScalarValue::UInt64(None)),
                WindowFrameBound::CurrentRow,
            )),
        ));

        Ok(Arc::new(BoundedWindowAggExec::try_new(
            vec![window_expr],
            input,
            InputOrderMode::Sorted,
            true,
        )?))
    }

    fn project_all(
        input: Arc<dyn ExecutionPlan>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let expressions = schema
            .fields()
            .iter()
            .map(|field| Ok((col(field.name(), &schema)?, field.name().to_string())))
            .collect::<datafusion_common::Result<Vec<_>>>()?;
        Ok(Arc::new(ProjectionExec::try_new(expressions, input)?))
    }

    fn build_dependent_window_plan() -> datafusion_common::Result<Arc<dyn ExecutionPlan>>
    {
        let schema = schema();
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(PlaceholderRowExec::new(Arc::clone(&schema)));
        let ordering = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("a", &schema)?).asc(),
        ])
        .unwrap();
        let sort: Arc<dyn ExecutionPlan> =
            Arc::new(SortExec::new(ordering, input).with_preserve_partitioning(true));

        let bottom = lead_window(sort, "a", "bottom_value")?;
        let middle = lead_window(project_all(bottom)?, "a", "middle_value")?;
        let top = lead_window(project_all(middle)?, "bottom_value", "top_value")?;

        Ok(Arc::new(LocalLimitExec::new(top, 3)))
    }

    fn optimize(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let mut config = ConfigOptions::new();
        config.optimizer.enable_window_limits = true;
        LimitPushPastWindows::new().optimize(plan, &config).unwrap()
    }

    /// GlobalLimitExec above a windowed sort should push fetch into the SortExec.
    #[test]
    fn global_limit_pushes_past_window() {
        let plan = build_window_plan(false).unwrap();
        let optimized = optimize(plan);
        assert_snapshot!(plan_str(optimized.as_ref()), @r#"
        GlobalLimitExec: skip=0, fetch=100
          BoundedWindowAggExec: wdw=[row_number: Field { "row_number": UInt64 }, frame: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW], mode=[Sorted]
            SortExec: TopK(fetch=100), expr=[a@0 ASC], preserve_partitioning=[true]
              PlaceholderRowExec
        "#);
    }

    /// LocalLimitExec above a windowed sort should also push fetch into the SortExec.
    /// This is the case in distributed execution where GlobalLimitExec becomes LocalLimitExec.
    #[test]
    fn local_limit_pushes_past_window() {
        let plan = build_window_plan(true).unwrap();
        let optimized = optimize(plan);
        assert_snapshot!(plan_str(optimized.as_ref()), @r#"
        LocalLimitExec: fetch=100
          BoundedWindowAggExec: wdw=[row_number: Field { "row_number": UInt64 }, frame: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW], mode=[Sorted]
            SortExec: TopK(fetch=100), expr=[a@0 ASC], preserve_partitioning=[true]
              PlaceholderRowExec
        "#);
    }

    #[test]
    fn local_limit_accumulates_dependent_window_lookahead_through_projection() {
        let plan = build_dependent_window_plan().unwrap();
        let optimized = optimize(plan);
        assert_snapshot!(plan_str(optimized.as_ref()), @r#"
        LocalLimitExec: fetch=3
          BoundedWindowAggExec: wdw=[top_value: Field { "top_value": nullable Int64 }, frame: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW], mode=[Sorted]
            ProjectionExec: expr=[a@0 as a, bottom_value@1 as bottom_value, middle_value@2 as middle_value]
              BoundedWindowAggExec: wdw=[middle_value: Field { "middle_value": nullable Int64 }, frame: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW], mode=[Sorted]
                ProjectionExec: expr=[a@0 as a, bottom_value@1 as bottom_value]
                  BoundedWindowAggExec: wdw=[bottom_value: Field { "bottom_value": nullable Int64 }, frame: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW], mode=[Sorted]
                    SortExec: TopK(fetch=5), expr=[a@0 ASC], preserve_partitioning=[true]
                      PlaceholderRowExec
        "#);
    }
}
