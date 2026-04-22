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

//! Criterion benchmarks for HashJoinExec and SymmetricHashJoinExec.
//!
//! These benchmarks feed in-memory RecordBatches directly into the join
//! operators to focus on execution behavior rather than scan or planner cost.

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_plan::joins::{
    HashJoinExec, JoinOn, PartitionMode, StreamJoinPartitionMode, SymmetricHashJoinExec,
};
use datafusion_physical_plan::test::TestMemoryExec;
use datafusion_physical_plan::{ExecutionPlan, collect};
use tokio::runtime::Runtime;

const INPUT_BATCH_SIZE: usize = 4096;
const UNIQUE_ROWS: usize = 50_000;
const MULTI_KEY_ROWS: usize = 50_000;
const NULL_HEAVY_ROWS: usize = 50_000;
const DUPLICATE_ROWS: usize = 16_384;
const DUPLICATE_KEY_MOD: usize = 512;
const SYMMETRIC_DUPLICATE_ROWS: usize = 8_192;
const SYMMETRIC_DUPLICATE_KEY_MOD: usize = 256;

#[derive(Clone)]
struct JoinInput {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl JoinInput {
    fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

#[derive(Clone)]
struct BenchCase {
    name: &'static str,
    left: JoinInput,
    right: JoinInput,
    on_columns: &'static [&'static str],
    null_equality: NullEquality,
}

fn single_key_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("payload", DataType::Int64, false),
    ]))
}

fn two_key_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key1", DataType::Int32, true),
        Field::new("key2", DataType::Int32, true),
        Field::new("payload", DataType::Int64, false),
    ]))
}

fn split_batch(batch: RecordBatch) -> Vec<RecordBatch> {
    let mut batches = Vec::new();
    let mut offset = 0;
    while offset < batch.num_rows() {
        let len = (batch.num_rows() - offset).min(INPUT_BATCH_SIZE);
        batches.push(batch.slice(offset, len));
        offset += len;
    }
    batches
}

fn build_single_key_input<F>(num_rows: usize, key_fn: F, payload_offset: i64) -> JoinInput
where
    F: Fn(usize) -> Option<i32>,
{
    let schema = single_key_schema();
    let keys = Int32Array::from_iter((0..num_rows).map(key_fn));
    let payloads =
        Int64Array::from_iter_values((0..num_rows).map(|i| payload_offset + i as i64));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(keys), Arc::new(payloads)],
    )
    .unwrap();
    JoinInput {
        schema,
        batches: split_batch(batch),
    }
}

fn build_two_key_input<F1, F2>(
    num_rows: usize,
    key1_fn: F1,
    key2_fn: F2,
    payload_offset: i64,
) -> JoinInput
where
    F1: Fn(usize) -> Option<i32>,
    F2: Fn(usize) -> Option<i32>,
{
    let schema = two_key_schema();
    let key1 = Int32Array::from_iter((0..num_rows).map(key1_fn));
    let key2 = Int32Array::from_iter((0..num_rows).map(key2_fn));
    let payloads =
        Int64Array::from_iter_values((0..num_rows).map(|i| payload_offset + i as i64));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(key1), Arc::new(key2), Arc::new(payloads)],
    )
    .unwrap();
    JoinInput {
        schema,
        batches: split_batch(batch),
    }
}

fn make_exec(input: &JoinInput) -> Arc<dyn ExecutionPlan> {
    TestMemoryExec::try_new_exec(
        &[input.batches.clone()],
        Arc::clone(&input.schema),
        None,
    )
    .unwrap()
}

fn build_join_on(
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    on_columns: &[&str],
) -> JoinOn {
    on_columns
        .iter()
        .map(|name| {
            (
                col(name, left_schema).unwrap(),
                col(name, right_schema).unwrap(),
            )
        })
        .collect()
}

fn execute_hash_join(case: &BenchCase, rt: &Runtime) -> usize {
    let left = make_exec(&case.left);
    let right = make_exec(&case.right);
    let join_on = build_join_on(&case.left.schema, &case.right.schema, case.on_columns);
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            join_on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            case.null_equality,
            false,
        )
        .unwrap(),
    );

    let task_ctx = Arc::new(TaskContext::default());
    rt.block_on(async {
        let batches = collect(join, task_ctx).await.unwrap();
        batches.iter().map(RecordBatch::num_rows).sum()
    })
}

fn execute_symmetric_hash_join(case: &BenchCase, rt: &Runtime) -> usize {
    let left = make_exec(&case.left);
    let right = make_exec(&case.right);
    let join_on = build_join_on(&case.left.schema, &case.right.schema, case.on_columns);
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        SymmetricHashJoinExec::try_new(
            left,
            right,
            join_on,
            None,
            &JoinType::Inner,
            case.null_equality,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )
        .unwrap(),
    );

    let task_ctx = Arc::new(TaskContext::default());
    rt.block_on(async {
        let batches = collect(join, task_ctx).await.unwrap();
        batches.iter().map(RecordBatch::num_rows).sum()
    })
}

fn hash_join_cases() -> Vec<BenchCase> {
    vec![
        BenchCase {
            name: "inner_unique",
            left: build_single_key_input(UNIQUE_ROWS, |i| Some(i as i32), 0),
            right: build_single_key_input(UNIQUE_ROWS, |i| Some(i as i32), 1_000_000),
            on_columns: &["key"],
            null_equality: NullEquality::NullEqualsNothing,
        },
        BenchCase {
            name: "inner_dense_duplicates",
            left: build_single_key_input(
                DUPLICATE_ROWS,
                |i| Some((i % DUPLICATE_KEY_MOD) as i32),
                0,
            ),
            right: build_single_key_input(
                DUPLICATE_ROWS,
                |i| Some((i % DUPLICATE_KEY_MOD) as i32),
                1_000_000,
            ),
            on_columns: &["key"],
            null_equality: NullEquality::NullEqualsNothing,
        },
        BenchCase {
            name: "inner_multi_key",
            left: build_two_key_input(
                MULTI_KEY_ROWS,
                |i| Some((i % 1024) as i32),
                |i| Some((i / 1024) as i32),
                0,
            ),
            right: build_two_key_input(
                MULTI_KEY_ROWS,
                |i| Some((i % 1024) as i32),
                |i| Some((i / 1024) as i32),
                1_000_000,
            ),
            on_columns: &["key1", "key2"],
            null_equality: NullEquality::NullEqualsNothing,
        },
        BenchCase {
            name: "inner_null_heavy",
            left: build_single_key_input(
                NULL_HEAVY_ROWS,
                |i| {
                    if i % 4 == 0 { None } else { Some(i as i32) }
                },
                0,
            ),
            right: build_single_key_input(
                NULL_HEAVY_ROWS,
                |i| {
                    if i % 4 == 0 { None } else { Some(i as i32) }
                },
                1_000_000,
            ),
            on_columns: &["key"],
            null_equality: NullEquality::NullEqualsNothing,
        },
    ]
}

fn symmetric_hash_join_cases() -> Vec<BenchCase> {
    vec![BenchCase {
        name: "inner_dense_duplicates",
        left: build_single_key_input(
            SYMMETRIC_DUPLICATE_ROWS,
            |i| Some((i % SYMMETRIC_DUPLICATE_KEY_MOD) as i32),
            0,
        ),
        right: build_single_key_input(
            SYMMETRIC_DUPLICATE_ROWS,
            |i| Some((i % SYMMETRIC_DUPLICATE_KEY_MOD) as i32),
            1_000_000,
        ),
        on_columns: &["key"],
        null_equality: NullEquality::NullEqualsNothing,
    }]
}

fn bench_hash_join(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hash_join");

    for case in hash_join_cases() {
        group.bench_function(BenchmarkId::new(case.name, case.left.num_rows()), |b| {
            b.iter(|| execute_hash_join(&case, &rt))
        });
    }

    group.finish();
}

fn bench_symmetric_hash_join(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("symmetric_hash_join");

    for case in symmetric_hash_join_cases() {
        group.bench_function(BenchmarkId::new(case.name, case.left.num_rows()), |b| {
            b.iter(|| execute_symmetric_hash_join(&case, &rt))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hash_join, bench_symmetric_hash_join);
criterion_main!(benches);
