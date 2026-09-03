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

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col};
use datafusion_physical_plan::sorts::partial_sort::PartialSortExec;
use datafusion_physical_plan::test::TestMemoryExec;
use datafusion_physical_plan::{ExecutionPlan, collect};

const NUM_BATCHES: usize = 32;
const BATCH_SIZE: usize = 8192;
const NUM_ROWS: usize = NUM_BATCHES * BATCH_SIZE;

#[derive(Clone, Copy)]
enum PrefixLayout {
    BatchesPerPrefix(usize),
    PrefixesPerBatch(usize),
    RowsPerPrefix(usize),
}

impl PrefixLayout {
    fn prefix_and_suffix(self, batch_idx: usize, row_idx: usize) -> (u64, u64) {
        match self {
            Self::BatchesPerPrefix(batches_per_prefix) => {
                let rows_per_prefix = batches_per_prefix * BATCH_SIZE;
                let offset = (batch_idx % batches_per_prefix) * BATCH_SIZE + row_idx;
                (
                    (batch_idx / batches_per_prefix) as u64,
                    (rows_per_prefix - offset - 1) as u64,
                )
            }
            Self::PrefixesPerBatch(prefixes_per_batch) => {
                let rows_per_prefix = BATCH_SIZE / prefixes_per_batch;
                let prefix_in_batch = row_idx / rows_per_prefix;
                (
                    (batch_idx * prefixes_per_batch + prefix_in_batch) as u64,
                    (rows_per_prefix - row_idx % rows_per_prefix - 1) as u64,
                )
            }
            Self::RowsPerPrefix(rows_per_prefix) => {
                let global_idx = batch_idx * BATCH_SIZE + row_idx;
                (
                    (global_idx / rows_per_prefix) as u64,
                    (rows_per_prefix - global_idx % rows_per_prefix - 1) as u64,
                )
            }
        }
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(
        [
            "prefix",
            "suffix",
            "payload_0",
            "payload_1",
            "payload_2",
            "payload_3",
        ]
        .into_iter()
        .map(|name| Field::new(name, DataType::UInt64, false))
        .collect::<Vec<_>>(),
    ))
}

fn make_batches(layout: PrefixLayout) -> Vec<RecordBatch> {
    let schema = schema();
    (0..NUM_BATCHES)
        .map(|batch_idx| {
            let rows = 0..BATCH_SIZE;
            let prefix = UInt64Array::from_iter_values(
                rows.clone()
                    .map(|row_idx| layout.prefix_and_suffix(batch_idx, row_idx).0),
            );
            let suffix = UInt64Array::from_iter_values(
                rows.clone()
                    .map(|row_idx| layout.prefix_and_suffix(batch_idx, row_idx).1),
            );
            let payload = |row_idx: usize| (batch_idx * BATCH_SIZE + row_idx) as u64;
            let payload_0 = UInt64Array::from_iter_values(rows.clone().map(payload));
            let payload_1 = UInt64Array::from_iter_values(
                rows.clone()
                    .map(|row_idx| payload(row_idx).wrapping_mul(31)),
            );
            let payload_2 = UInt64Array::from_iter_values(
                rows.clone().map(|row_idx| payload(row_idx).rotate_left(13)),
            );
            let payload_3 = UInt64Array::from_iter_values(
                rows.map(|row_idx| payload(row_idx).wrapping_neg()),
            );

            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(prefix) as ArrayRef,
                    Arc::new(suffix) as ArrayRef,
                    Arc::new(payload_0) as ArrayRef,
                    Arc::new(payload_1) as ArrayRef,
                    Arc::new(payload_2) as ArrayRef,
                    Arc::new(payload_3) as ArrayRef,
                ],
            )
            .unwrap()
        })
        .collect()
}

fn make_plan(batches: &[RecordBatch]) -> Arc<dyn ExecutionPlan> {
    let schema = batches[0].schema();
    let input =
        TestMemoryExec::try_new_exec(&[batches.to_vec()], Arc::clone(&schema), None)
            .unwrap();
    let ordering = LexOrdering::new([
        PhysicalSortExpr::new(col("prefix", &schema).unwrap(), SortOptions::default()),
        PhysicalSortExpr::new(col("suffix", &schema).unwrap(), SortOptions::default()),
    ])
    .unwrap();
    Arc::new(PartialSortExec::new(ordering, input, 1))
}

fn partial_sort_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let task_ctx = Arc::new(TaskContext::default());
    let mut group = c.benchmark_group("partial_sort");
    group.sample_size(10);

    let cases = [
        (
            BenchmarkId::new("prefix_across_batches", "batches_per_prefix=1"),
            PrefixLayout::BatchesPerPrefix(1),
        ),
        (
            BenchmarkId::new("prefix_across_batches", "batches_per_prefix=2"),
            PrefixLayout::BatchesPerPrefix(2),
        ),
        (
            BenchmarkId::new("prefix_across_batches", "batches_per_prefix=4"),
            PrefixLayout::BatchesPerPrefix(4),
        ),
        (
            BenchmarkId::new("prefixes_within_batch", "prefixes_per_batch=2"),
            PrefixLayout::PrefixesPerBatch(2),
        ),
        (
            BenchmarkId::new("prefixes_within_batch", "prefixes_per_batch=8"),
            PrefixLayout::PrefixesPerBatch(8),
        ),
        (
            BenchmarkId::new("prefixes_within_batch", "prefixes_per_batch=32"),
            PrefixLayout::PrefixesPerBatch(32),
        ),
        (
            BenchmarkId::new("prefixes_within_batch", "prefixes_per_batch=128"),
            PrefixLayout::PrefixesPerBatch(128),
        ),
        (
            BenchmarkId::new("unaligned_mixed", "rows_per_prefix=1000"),
            PrefixLayout::RowsPerPrefix(1000),
        ),
    ];

    for (id, layout) in cases {
        let batches = make_batches(layout);
        let validation = runtime
            .block_on(collect(make_plan(&batches), Arc::clone(&task_ctx)))
            .unwrap();
        assert_eq!(
            validation.iter().map(RecordBatch::num_rows).sum::<usize>(),
            NUM_ROWS
        );
        drop(validation);

        group.bench_function(id, |b| {
            b.iter_batched(
                || make_plan(&batches),
                |plan| {
                    let output = runtime
                        .block_on(collect(plan, Arc::clone(&task_ctx)))
                        .unwrap();
                    black_box(output);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, partial_sort_benchmark);
criterion_main!(benches);
