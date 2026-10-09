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

//! Benchmarks for `BoundedWindowAggExec` with many partitions.
//!
//! The streaming window operator keeps per-partition state keyed by
//! `PartitionKey` (`Vec<ScalarValue>`) and, in `Linear` mode (input sorted
//! by the ORDER BY column but not by the partition columns), visits every
//! live partition on every batch while never retiring partitions until the
//! input is exhausted. The cases here stress that path in different ways.
//!
//! Case names spell out the input order mode, the key
//! layout (`dense` / `sparse`), the window functions, an optional frame
//! variant, and the partition count:
//!
//! - `linear dense count N partitions`: dense round-robin keys -- every
//!   partition receives rows in every batch, so per-visit fixed costs
//!   dominate.
//! - `linear sparse count N partitions`: keys are clustered in time, so
//!   each batch touches only a small, fresh subset of keys while the set of
//!   live partitions keeps growing -- per-batch work on quiet partitions
//!   dominates.
//! - `linear dense count rows-frame N partitions`: the dense layout with a
//!   ROWS frame, whose results can only be finalized as more rows of the
//!   same partition arrive.
//! - `linear dense count+sum N partitions`: two window expressions over the
//!   dense layout, doubling the per-partition evaluation sweeps.
//! - `linear dense row_number N partitions` / `linear sparse row_number N
//!   partitions`: the dense / sparse layouts evaluated through
//!   `StandardWindowExpr` and a `PartitionEvaluator` rather than an
//!   aggregate accumulator.
//! - `linear sparse lead N partitions`: the sparse layout with a non-causal
//!   function, whose result for the last buffered row of a partition stays
//!   pending until that partition receives another row.
//! - `linear dense row_number TYPE N partitions`: fixed key-type matrix with
//!   16-byte strings and four-element lists, isolating partition-key lookup costs.
//! - `partially_sorted dense row_number TYPE N partitions per prefix`: keys
//!   cycle within each sorted prefix, which spans four batches before changing.
//! - `linear dense rank N partitions`: the dense layout with an evaluator
//!   that compares ORDER BY values row by row.
//! - `sorted count N partitions`: control; input sorted by partition key,
//!   as `Sorted` mode requires, so finished partitions are pruned eagerly
//!   and the state maps stay small.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, ListArray, StringArray,
    StringViewArray, UInt64Array,
};
use arrow::datatypes::Int32Type;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::ScalarValue;
use datafusion_execution::TaskContext;
use datafusion_expr::{
    WindowFrame, WindowFrameBound, WindowFrameUnits, WindowFunctionDefinition,
};
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_functions_window::lead_lag::lead_udwf;
use datafusion_functions_window::rank::rank_udwf;
use datafusion_functions_window::row_number::row_number_udwf;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::test::TestMemoryExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, create_window_expr};
use datafusion_physical_plan::{ExecutionPlan, InputOrderMode, collect};

const BATCH_SIZE: usize = 8192;
const N_BATCHES: usize = 16;
/// Distinct partition keys per batch in the sparse layout. Each batch
/// introduces this many previously-unseen keys, so the total partition count
/// is `N_BATCHES * SPARSE_KEYS_PER_BATCH`.
const SPARSE_KEYS_PER_BATCH: usize = 2048;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("pk", DataType::UInt64, false),
        Field::new("ts", DataType::UInt64, false),
    ]))
}

/// Batches with `ts` ascending across the whole input and partition keys
/// chosen by `pk_of_row`.
fn make_batches(pk_of_row: impl Fn(usize) -> u64) -> Vec<RecordBatch> {
    (0..N_BATCHES)
        .map(|b| {
            let start = b * BATCH_SIZE;
            let pk: UInt64Array = (start..start + BATCH_SIZE)
                .map(|i| Some(pk_of_row(i)))
                .collect();
            let ts: UInt64Array = (start..start + BATCH_SIZE)
                .map(|i| Some(i as u64))
                .collect();
            RecordBatch::try_new(schema(), vec![Arc::new(pk), Arc::new(ts)]).unwrap()
        })
        .collect()
}

/// Round-robin over `n_partitions`: every partition receives rows in every
/// batch (when `n_partitions <= BATCH_SIZE`).
fn dense_batches(n_partitions: usize) -> Vec<RecordBatch> {
    make_batches(move |i| (i % n_partitions) as u64)
}

/// Fixed key-type matrix with the same dense layout and timing boundary.
fn typed_dense_batches(data_type: &DataType, n_partitions: usize) -> Vec<RecordBatch> {
    (0..N_BATCHES)
        .map(|b| {
            let start = b * BATCH_SIZE;
            let groups = (start..start + BATCH_SIZE).map(|i| i % n_partitions);
            let pk: ArrayRef = match data_type {
                DataType::Utf8 => Arc::new(StringArray::from_iter_values(
                    groups.map(|g| format!("{g:016}")),
                )),
                DataType::Utf8View => Arc::new(StringViewArray::from_iter_values(
                    groups.map(|g| format!("{g:016}")),
                )),
                DataType::UInt64 => {
                    Arc::new(UInt64Array::from_iter_values(groups.map(|g| g as u64)))
                }
                DataType::Float64 => {
                    Arc::new(Float64Array::from_iter_values(groups.map(|g| g as f64)))
                }
                DataType::Date32 => {
                    Arc::new(Date32Array::from_iter_values(groups.map(|g| g as i32)))
                }
                DataType::Boolean => {
                    Arc::new(BooleanArray::from_iter(groups.map(|g| Some(g != 0))))
                }
                DataType::List(_) => {
                    Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
                        groups.map(|g| Some(vec![Some(g as i32); 4])),
                    ))
                }
                _ => unreachable!("unsupported benchmark key"),
            };
            let schema = Arc::new(Schema::new(vec![
                Field::new("pk", pk.data_type().clone(), false),
                Field::new("ts", DataType::UInt64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    pk,
                    Arc::new(UInt64Array::from_iter_values(
                        (start..start + BATCH_SIZE).map(|i| i as u64),
                    )),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// Four sorted prefixes, each spanning four batches of round-robin keys.
fn partially_sorted_batches(
    data_type: &DataType,
    partitions_per_prefix: usize,
) -> Vec<RecordBatch> {
    typed_dense_batches(data_type, partitions_per_prefix)
        .into_iter()
        .enumerate()
        .map(|(b, batch)| {
            RecordBatch::try_from_iter([
                ("pk", Arc::clone(batch.column(0))),
                ("ts", Arc::clone(batch.column(1))),
                (
                    "prefix",
                    Arc::new(UInt64Array::from(vec![(b / 4) as u64; BATCH_SIZE])),
                ),
            ])
            .unwrap()
        })
        .collect()
}

/// Keys clustered in time: batch `b` only contains keys in
/// `[b * SPARSE_KEYS_PER_BATCH, (b + 1) * SPARSE_KEYS_PER_BATCH)`, cycled so
/// that consecutive rows belong to different partitions. Previously-seen
/// keys never recur, but `Linear` mode cannot know that, so the live
/// partition set grows for the whole run.
fn sparse_batches() -> Vec<RecordBatch> {
    make_batches(|i| {
        ((i / BATCH_SIZE) * SPARSE_KEYS_PER_BATCH + (i % SPARSE_KEYS_PER_BATCH)) as u64
    })
}

/// Input laid out partition-by-partition (the `Sorted` layout).
fn sorted_batches(n_partitions: usize) -> Vec<RecordBatch> {
    let rows_per_partition = BATCH_SIZE * N_BATCHES / n_partitions;
    make_batches(move |i| (i / rows_per_partition) as u64)
}

fn sort_expr(name: &str) -> PhysicalSortExpr {
    PhysicalSortExpr {
        expr: col(name, &schema()).unwrap(),
        options: Default::default(),
    }
}

/// `RANGE BETWEEN CURRENT ROW AND 10 FOLLOWING`
fn range_frame() -> WindowFrame {
    WindowFrame::new_bounds(
        WindowFrameUnits::Range,
        WindowFrameBound::CurrentRow,
        WindowFrameBound::Following(ScalarValue::UInt64(Some(10))),
    )
}

/// `ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING`
fn rows_frame() -> WindowFrame {
    WindowFrame::new_bounds(
        WindowFrameUnits::Rows,
        WindowFrameBound::CurrentRow,
        WindowFrameBound::Following(ScalarValue::UInt64(Some(2))),
    )
}

/// `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, the default frame
/// of a window that has an ORDER BY clause.
fn default_frame() -> WindowFrame {
    WindowFrame::new(Some(false))
}

/// A window function to benchmark: definition, display name, and arguments.
type BenchWindowFn = (
    WindowFunctionDefinition,
    &'static str,
    Vec<Arc<dyn PhysicalExpr>>,
);

/// `<fn>(<args>) OVER (PARTITION BY [prefix,] pk ORDER BY ts <window_frame>)` for each
/// window function in `functions`.
fn window_exec(
    batches: Vec<RecordBatch>,
    mode: InputOrderMode,
    input_ordering: Vec<PhysicalSortExpr>,
    window_frame: &WindowFrame,
    functions: &[BenchWindowFn],
) -> Arc<dyn ExecutionPlan> {
    let schema = batches[0].schema();
    let source = TestMemoryExec::try_new(&[batches], Arc::clone(&schema), None)
        .expect("memory exec")
        .try_with_sort_information(LexOrdering::new(input_ordering).into_iter().collect())
        .expect("sort information");
    let input = Arc::new(TestMemoryExec::update_cache(&Arc::new(source)));
    let mut partitionby_exprs = vec![col("pk", &schema).unwrap()];
    if matches!(mode, InputOrderMode::PartiallySorted(_)) {
        partitionby_exprs.insert(0, col("prefix", &schema).unwrap());
    }
    let orderby_exprs = vec![PhysicalSortExpr {
        expr: col("ts", &schema).unwrap(),
        options: Default::default(),
    }];
    let window_expr = functions
        .iter()
        .map(|(fun, name, args)| {
            create_window_expr(
                fun,
                name.to_string(),
                args,
                &partitionby_exprs,
                &orderby_exprs,
                Arc::new(window_frame.clone()),
                input.schema(),
                false,
                false,
                None,
            )
            .expect("window expr")
        })
        .collect::<Vec<_>>();
    Arc::new(
        BoundedWindowAggExec::try_new(window_expr, input, mode, true)
            .expect("bounded window exec"),
    )
}

fn ts_arg() -> Vec<Arc<dyn PhysicalExpr>> {
    vec![col("ts", &schema()).unwrap()]
}

fn count() -> BenchWindowFn {
    (
        WindowFunctionDefinition::AggregateUDF(count_udaf()),
        "count",
        ts_arg(),
    )
}

fn sum() -> BenchWindowFn {
    (
        WindowFunctionDefinition::AggregateUDF(sum_udaf()),
        "sum",
        ts_arg(),
    )
}

fn row_number() -> BenchWindowFn {
    (
        WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        "row_number",
        vec![],
    )
}

fn lead() -> BenchWindowFn {
    (
        WindowFunctionDefinition::WindowUDF(lead_udwf()),
        "lead",
        ts_arg(),
    )
}

fn rank() -> BenchWindowFn {
    (
        WindowFunctionDefinition::WindowUDF(rank_udwf()),
        "rank",
        vec![],
    )
}

fn bounded_window_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("bounded_window_partitions");
    group.sample_size(10);

    let mut run_case = |name: String, plan: Arc<dyn ExecutionPlan>| {
        group.bench_function(name, |b| {
            b.iter(|| {
                let task_ctx = Arc::new(TaskContext::default());
                let batches = rt
                    .block_on(collect(Arc::clone(&plan), task_ctx))
                    .expect("execution");
                assert_eq!(
                    batches.iter().map(|b| b.num_rows()).sum::<usize>(),
                    BATCH_SIZE * N_BATCHES
                );
            })
        });
    };

    for n_partitions in [100, 10_000] {
        run_case(
            format!("linear dense count {n_partitions} partitions"),
            window_exec(
                dense_batches(n_partitions),
                InputOrderMode::Linear,
                vec![sort_expr("ts")],
                &range_frame(),
                &[count()],
            ),
        );
    }

    run_case(
        format!(
            "linear sparse count {} partitions",
            N_BATCHES * SPARSE_KEYS_PER_BATCH
        ),
        window_exec(
            sparse_batches(),
            InputOrderMode::Linear,
            vec![sort_expr("ts")],
            &range_frame(),
            &[count()],
        ),
    );

    run_case(
        "linear dense count rows-frame 10000 partitions".to_string(),
        window_exec(
            dense_batches(10_000),
            InputOrderMode::Linear,
            vec![sort_expr("ts")],
            &rows_frame(),
            &[count()],
        ),
    );

    run_case(
        "linear dense count+sum 10000 partitions".to_string(),
        window_exec(
            dense_batches(10_000),
            InputOrderMode::Linear,
            vec![sort_expr("ts")],
            &range_frame(),
            &[count(), sum()],
        ),
    );

    run_case(
        "linear dense row_number 10000 partitions".to_string(),
        window_exec(
            dense_batches(10_000),
            InputOrderMode::Linear,
            vec![sort_expr("ts")],
            &default_frame(),
            &[row_number()],
        ),
    );

    run_case(
        format!(
            "linear sparse row_number {} partitions",
            N_BATCHES * SPARSE_KEYS_PER_BATCH
        ),
        window_exec(
            sparse_batches(),
            InputOrderMode::Linear,
            vec![sort_expr("ts")],
            &default_frame(),
            &[row_number()],
        ),
    );

    run_case(
        format!(
            "linear sparse lead {} partitions",
            N_BATCHES * SPARSE_KEYS_PER_BATCH
        ),
        window_exec(
            sparse_batches(),
            InputOrderMode::Linear,
            vec![sort_expr("ts")],
            &default_frame(),
            &[lead()],
        ),
    );

    run_case(
        "linear dense rank 10000 partitions".to_string(),
        window_exec(
            dense_batches(10_000),
            InputOrderMode::Linear,
            vec![sort_expr("ts")],
            &default_frame(),
            &[rank()],
        ),
    );

    for (key_name, data_type, n_partitions) in [
        ("Utf8", DataType::Utf8, 100),
        ("Utf8View", DataType::Utf8View, 100),
        ("UInt64", DataType::UInt64, 100),
        ("Float64", DataType::Float64, 100),
        ("Date32", DataType::Date32, 100),
        (
            "List<Int32>",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            100,
        ),
        ("Boolean", DataType::Boolean, 2),
        ("Utf8", DataType::Utf8, 10_000),
    ] {
        run_case(
            format!("linear dense row_number {key_name} {n_partitions} partitions"),
            window_exec(
                typed_dense_batches(&data_type, n_partitions),
                InputOrderMode::Linear,
                vec![sort_expr("ts")],
                &default_frame(),
                &[row_number()],
            ),
        );
    }

    for (key_name, data_type, partitions_per_prefix) in [
        ("Utf8", DataType::Utf8, 100),
        ("Utf8View", DataType::Utf8View, 100),
        ("UInt64", DataType::UInt64, 100),
        ("Utf8", DataType::Utf8, 10_000),
    ] {
        let batches = partially_sorted_batches(&data_type, partitions_per_prefix);
        let ordering = vec![
            PhysicalSortExpr {
                expr: col("prefix", &batches[0].schema()).unwrap(),
                options: Default::default(),
            },
            sort_expr("ts"),
        ];
        run_case(
            format!(
                "partially_sorted dense row_number {key_name} {partitions_per_prefix} partitions per prefix"
            ),
            window_exec(
                batches,
                InputOrderMode::PartiallySorted(vec![0]),
                ordering,
                &default_frame(),
                &[row_number()],
            ),
        );
    }

    // Control: the same query over partition-sorted input, where finished
    // partitions are pruned eagerly and the state maps stay small.
    run_case(
        "sorted count 10000 partitions".to_string(),
        window_exec(
            sorted_batches(10_000),
            InputOrderMode::Sorted,
            vec![sort_expr("pk"), sort_expr("ts")],
            &range_frame(),
            &[count()],
        ),
    );

    group.finish();
}

criterion_group!(benches, bounded_window_benchmark);
criterion_main!(benches);
