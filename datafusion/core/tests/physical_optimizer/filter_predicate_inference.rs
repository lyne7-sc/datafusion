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

use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion_physical_expr::{PhysicalExpr, conjunction};
use datafusion_physical_optimizer::{
    PhysicalOptimizerRule, filter_predicate_inference::FilterPredicateInference,
};
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::{ExecutionPlan, displayable};

use super::test_utils::schema;

fn optimize(plan: Arc<dyn ExecutionPlan>, enable: bool) -> Vec<String> {
    let mut config = ConfigOptions::default();
    config.optimizer.enable_filter_predicate_inference = enable;

    let optimized = FilterPredicateInference::new()
        .optimize(plan, &config)
        .unwrap();

    displayable(optimized.as_ref())
        .indent(false)
        .to_string()
        .trim()
        .lines()
        .map(ToString::to_string)
        .collect()
}

fn filter(predicate: Arc<dyn PhysicalExpr>) -> Arc<dyn ExecutionPlan> {
    Arc::new(
        FilterExec::try_new(
            predicate,
            Arc::new(EmptyExec::new(schema())) as Arc<dyn ExecutionPlan>,
        )
        .unwrap(),
    )
}

#[test]
fn disabled_by_default() {
    let plan = filter(conjunction([
        Arc::new(BinaryExpr::new(
            col("a", &schema()).unwrap(),
            Operator::Eq,
            col("b", &schema()).unwrap(),
        )) as Arc<dyn PhysicalExpr>,
        Arc::new(BinaryExpr::new(
            col("b", &schema()).unwrap(),
            Operator::Gt,
            lit(10i64),
        )),
    ]));

    insta::assert_snapshot!(
        optimize(plan, false).join("\n"),
        @r"
    FilterExec: a@0 = b@1 AND b@1 > 10
      EmptyExec
    "
    );
}

#[test]
fn infers_predicate_from_equality() {
    let plan = filter(conjunction([
        Arc::new(BinaryExpr::new(
            col("a", &schema()).unwrap(),
            Operator::Eq,
            col("b", &schema()).unwrap(),
        )) as Arc<dyn PhysicalExpr>,
        Arc::new(BinaryExpr::new(
            col("b", &schema()).unwrap(),
            Operator::Gt,
            lit(10i64),
        )),
    ]));

    insta::assert_snapshot!(
        optimize(plan, true).join("\n"),
        @r"
    FilterExec: a@0 = b@1 AND b@1 > 10 AND a@0 >= 11 AND b@1 >= 11
      EmptyExec
    "
    );
}

#[test]
fn infers_literal_through_equality_chain() {
    let plan = filter(conjunction([
        Arc::new(BinaryExpr::new(
            lit(5i64),
            Operator::Eq,
            col("a", &schema()).unwrap(),
        )) as Arc<dyn PhysicalExpr>,
        Arc::new(BinaryExpr::new(
            col("a", &schema()).unwrap(),
            Operator::Eq,
            col("b", &schema()).unwrap(),
        )),
    ]));

    insta::assert_snapshot!(
        optimize(plan, true).join("\n"),
        @r"
    FilterExec: 5 = a@0 AND a@0 = b@1 AND b@1 = 5
      EmptyExec
    "
    );
}

#[test]
fn infers_two_sided_bounds_through_equality() {
    let plan = filter(conjunction([
        Arc::new(BinaryExpr::new(
            col("a", &schema()).unwrap(),
            Operator::Eq,
            col("b", &schema()).unwrap(),
        )) as Arc<dyn PhysicalExpr>,
        Arc::new(BinaryExpr::new(
            col("b", &schema()).unwrap(),
            Operator::GtEq,
            lit(5i64),
        )),
        Arc::new(BinaryExpr::new(
            col("b", &schema()).unwrap(),
            Operator::LtEq,
            lit(10i64),
        )),
    ]));

    insta::assert_snapshot!(
        optimize(plan, true).join("\n"),
        @r"
    FilterExec: a@0 = b@1 AND b@1 >= 5 AND b@1 <= 10 AND a@0 >= 5 AND a@0 <= 10
      EmptyExec
    "
    );
}

#[test]
fn replaces_infeasible_filter_with_empty_exec() {
    let plan = filter(conjunction([
        Arc::new(BinaryExpr::new(
            col("a", &schema()).unwrap(),
            Operator::Gt,
            lit(10i64),
        )) as Arc<dyn PhysicalExpr>,
        Arc::new(BinaryExpr::new(
            col("a", &schema()).unwrap(),
            Operator::Lt,
            lit(5i64),
        )),
    ]));

    insta::assert_snapshot!(
        optimize(plan, true).join("\n"),
        @r"
    EmptyExec
    "
    );
}

#[test]
fn preserves_filter_projection() {
    let predicate = conjunction([
        Arc::new(BinaryExpr::new(
            col("a", &schema()).unwrap(),
            Operator::Eq,
            col("b", &schema()).unwrap(),
        )) as Arc<dyn PhysicalExpr>,
        Arc::new(BinaryExpr::new(
            col("b", &schema()).unwrap(),
            Operator::Gt,
            lit(10i64),
        )),
    ]);
    let input = Arc::new(
        ProjectionExec::try_new(
            vec![
                (col("a", &schema()).unwrap(), "a".to_string()),
                (col("b", &schema()).unwrap(), "b".to_string()),
            ],
            Arc::new(EmptyExec::new(schema())),
        )
        .unwrap(),
    );
    let plan = Arc::new(
        FilterExecBuilder::new(predicate, input)
            .apply_projection(Some(vec![1]))
            .unwrap()
            .build()
            .unwrap(),
    );

    insta::assert_snapshot!(
        optimize(plan, true).join("\n"),
        @r"
    FilterExec: a@0 = b@1 AND b@1 > 10 AND a@0 >= 11 AND b@1 >= 11, projection=[b@1]
      ProjectionExec: expr=[a@0 as a, b@1 as b]
        EmptyExec
    "
    );
}
