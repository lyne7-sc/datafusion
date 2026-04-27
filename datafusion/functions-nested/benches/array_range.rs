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

//! Benchmarks for range and generate_series functions.

use arrow::array::{
    ArrayRef, Date32Array, Int64Array, IntervalMonthDayNanoArray,
    TimestampNanosecondArray,
};
use arrow::datatypes::{
    DataType, Field, IntervalMonthDayNano, IntervalUnit::MonthDayNano, TimeUnit,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion_common::{config::ConfigOptions, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_functions_nested::range::{gen_series_udf, Range};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::sync::Arc;

const NUM_ROWS: &[usize] = &[100, 1000, 10000];
const INT_STEPS: &[i64] = &[1, 5, 50];
/// Each row produces at most RANGE_SIZE elements in its Int64 list
const RANGE_SIZE: i64 = 200;
const SEED: u64 = 42;

// Date32: days since Unix epoch. 2020-01-01 = 18262
const DATE_2020_01_01: i32 = 18262;

// TimestampNanosecond: nanoseconds since Unix epoch. 2020-01-01T00:00:00Z
const TS_2020_01_01: i64 = 1_577_836_800_000_000_000;

fn criterion_benchmark(c: &mut Criterion) {
    // Integer benchmarks – range()
    bench_range_int64_single_arg(c);
    bench_range_int64_two_args(c);
    bench_range_int64_three_args(c);

    // Integer benchmarks – generate_series()
    bench_generate_series_int64(c);

    // Date32 benchmarks
    bench_range_date32(c);
    bench_generate_series_date32(c);

    // Timestamp benchmarks
    bench_range_timestamp(c);
    bench_generate_series_timestamp(c);
}

// ---------------------------------------------------------------------------
// Integer – range(start)          equivalent to range(0, start, 1)
// ---------------------------------------------------------------------------
fn bench_range_int64_single_arg(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_int64_single_arg");

    for &num_rows in NUM_ROWS {
        let stop_array = make_int64_positive_array(num_rows, RANGE_SIZE);

        let args = vec![ColumnarValue::Array(stop_array.clone())];

        group.bench_with_input(
            BenchmarkId::new("stop_only", num_rows),
            &num_rows,
            |b, _| {
                let udf = Range::new();
                b.iter(|| {
                    black_box(
                        udf.invoke_with_args(ScalarFunctionArgs {
                            args: args.clone(),
                            arg_fields: vec![Arc::new(Field::new(
                                "stop",
                                DataType::Int64,
                                true,
                            ))],
                            number_rows: num_rows,
                            return_field: Arc::new(Field::new(
                                "result",
                                DataType::List(Arc::new(Field::new_list_field(
                                    DataType::Int64,
                                    true,
                                ))),
                                true,
                            )),
                            config_options: Arc::new(ConfigOptions::default()),
                        })
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Integer – range(start, stop)
// ---------------------------------------------------------------------------
fn bench_range_int64_two_args(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_int64_two_args");

    for &num_rows in NUM_ROWS {
        let (start_array, stop_array) =
            make_int64_start_stop_arrays(num_rows, RANGE_SIZE);

        let args = vec![
            ColumnarValue::Array(start_array.clone()),
            ColumnarValue::Array(stop_array.clone()),
        ];

        group.bench_with_input(
            BenchmarkId::new("start_stop", num_rows),
            &num_rows,
            |b, _| {
                let udf = Range::new();
                b.iter(|| {
                    black_box(
                        udf.invoke_with_args(ScalarFunctionArgs {
                            args: args.clone(),
                            arg_fields: vec![
                                Arc::new(Field::new("start", DataType::Int64, true)),
                                Arc::new(Field::new("stop", DataType::Int64, true)),
                            ],
                            number_rows: num_rows,
                            return_field: Arc::new(Field::new(
                                "result",
                                DataType::List(Arc::new(Field::new_list_field(
                                    DataType::Int64,
                                    true,
                                ))),
                                true,
                            )),
                            config_options: Arc::new(ConfigOptions::default()),
                        })
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Integer – range(start, stop, step)
// ---------------------------------------------------------------------------
fn bench_range_int64_three_args(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_int64_three_args");

    for &num_rows in NUM_ROWS {
        let (start_array, stop_array) =
            make_int64_start_stop_arrays(num_rows, RANGE_SIZE);

        for &step in INT_STEPS {
            let args = vec![
                ColumnarValue::Array(start_array.clone()),
                ColumnarValue::Array(stop_array.clone()),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(step))),
            ];

            group.bench_with_input(
                BenchmarkId::new(format!("step_{step}"), num_rows),
                &num_rows,
                |b, _| {
                    let udf = Range::new();
                    b.iter(|| {
                        black_box(
                            udf.invoke_with_args(ScalarFunctionArgs {
                                args: args.clone(),
                                arg_fields: vec![
                                    Arc::new(Field::new("start", DataType::Int64, true)),
                                    Arc::new(Field::new("stop", DataType::Int64, true)),
                                    Arc::new(Field::new("step", DataType::Int64, true)),
                                ],
                                number_rows: num_rows,
                                return_field: Arc::new(Field::new(
                                    "result",
                                    DataType::List(Arc::new(Field::new_list_field(
                                        DataType::Int64,
                                        true,
                                    ))),
                                    true,
                                )),
                                config_options: Arc::new(ConfigOptions::default()),
                            })
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Integer – generate_series(start, stop, step)
// ---------------------------------------------------------------------------
fn bench_generate_series_int64(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_series_int64");

    for &num_rows in NUM_ROWS {
        let (start_array, stop_array) =
            make_int64_start_stop_arrays(num_rows, RANGE_SIZE);

        for &step in INT_STEPS {
            let args = vec![
                ColumnarValue::Array(start_array.clone()),
                ColumnarValue::Array(stop_array.clone()),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(step))),
            ];

            group.bench_with_input(
                BenchmarkId::new(format!("step_{step}"), num_rows),
                &num_rows,
                |b, _| {
                    let udf = gen_series_udf();
                    b.iter(|| {
                        black_box(
                            udf.invoke_with_args(ScalarFunctionArgs {
                                args: args.clone(),
                                arg_fields: vec![
                                    Arc::new(Field::new("start", DataType::Int64, true)),
                                    Arc::new(Field::new("stop", DataType::Int64, true)),
                                    Arc::new(Field::new("step", DataType::Int64, true)),
                                ],
                                number_rows: num_rows,
                                return_field: Arc::new(Field::new(
                                    "result",
                                    DataType::List(Arc::new(Field::new_list_field(
                                        DataType::Int64,
                                        true,
                                    ))),
                                    true,
                                )),
                                config_options: Arc::new(ConfigOptions::default()),
                            })
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Date32 – range(start, stop, step)
// ---------------------------------------------------------------------------
fn bench_range_date32(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_date32");

    // Interval steps: 1 month, 3 months, 1 year
    let intervals: &[(i32, i32, i32)] = &[
        (1, 0, 0),   // 1 month
        (3, 0, 0),   // 3 months
        (12, 0, 0),  // 1 year
    ];

    for &num_rows in NUM_ROWS {
        for &(months, days, ns) in intervals {
            let (start, stop, step) = make_date32_arrays(num_rows, months, days, ns);

            let args = vec![
                ColumnarValue::Array(start),
                ColumnarValue::Array(stop),
                ColumnarValue::Array(step),
            ];

            let label = if days == 0 && ns == 0 {
                format!("interval_{months}m")
            } else {
                format!("interval_{months}m_{days}d")
            };

            group.bench_with_input(
                BenchmarkId::new(label, num_rows),
                &num_rows,
                |b, _| {
                    let udf = Range::new();
                    b.iter(|| {
                        black_box(
                            udf.invoke_with_args(ScalarFunctionArgs {
                                args: args.clone(),
                                arg_fields: vec![
                                    Arc::new(Field::new("start", DataType::Date32, true)),
                                    Arc::new(Field::new("stop", DataType::Date32, true)),
                                    Arc::new(Field::new(
                                        "step",
                                        DataType::Interval(MonthDayNano),
                                        true,
                                    )),
                                ],
                                number_rows: num_rows,
                                return_field: Arc::new(Field::new(
                                    "result",
                                    DataType::List(Arc::new(Field::new_list_field(
                                        DataType::Date32,
                                        true,
                                    ))),
                                    true,
                                )),
                                config_options: Arc::new(ConfigOptions::default()),
                            })
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Date32 – generate_series(start, stop, step)
// ---------------------------------------------------------------------------
fn bench_generate_series_date32(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_series_date32");

    let intervals: &[(i32, i32, i32)] = &[
        (1, 0, 0),   // 1 month
        (3, 0, 0),   // 3 months
        (12, 0, 0),  // 1 year
    ];

    for &num_rows in NUM_ROWS {
        for &(months, days, ns) in intervals {
            let (start, stop, step) = make_date32_arrays(num_rows, months, days, ns);

            let args = vec![
                ColumnarValue::Array(start),
                ColumnarValue::Array(stop),
                ColumnarValue::Array(step),
            ];

            let label = if days == 0 && ns == 0 {
                format!("interval_{months}m")
            } else {
                format!("interval_{months}m_{days}d")
            };

            group.bench_with_input(
                BenchmarkId::new(label, num_rows),
                &num_rows,
                |b, _| {
                    let udf = gen_series_udf();
                    b.iter(|| {
                        black_box(
                            udf.invoke_with_args(ScalarFunctionArgs {
                                args: args.clone(),
                                arg_fields: vec![
                                    Arc::new(Field::new("start", DataType::Date32, true)),
                                    Arc::new(Field::new("stop", DataType::Date32, true)),
                                    Arc::new(Field::new(
                                        "step",
                                        DataType::Interval(MonthDayNano),
                                        true,
                                    )),
                                ],
                                number_rows: num_rows,
                                return_field: Arc::new(Field::new(
                                    "result",
                                    DataType::List(Arc::new(Field::new_list_field(
                                        DataType::Date32,
                                        true,
                                    ))),
                                    true,
                                )),
                                config_options: Arc::new(ConfigOptions::default()),
                            })
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Timestamp – range(start, stop, step)
// ---------------------------------------------------------------------------
fn bench_range_timestamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_timestamp");

    let intervals: &[(i32, i32, i64)] = &[
        (0, 1, 0),                   // 1 day
        (0, 7, 0),                   // 7 days
        (1, 0, 0),                   // 1 month
        (0, 0, 3_600_000_000_000),   // 1 hour in ns
    ];

    for &num_rows in NUM_ROWS {
        for &(months, days, ns) in intervals {
            let (start, stop, step) =
                make_timestamp_arrays(num_rows, months, days, ns);

            let args = vec![
                ColumnarValue::Array(start),
                ColumnarValue::Array(stop),
                ColumnarValue::Array(step),
            ];

            let label = make_interval_label(months, days, ns);

            group.bench_with_input(
                BenchmarkId::new(label, num_rows),
                &num_rows,
                |b, _| {
                    let udf = Range::new();
                    b.iter(|| {
                        black_box(
                            udf.invoke_with_args(ScalarFunctionArgs {
                                args: args.clone(),
                                arg_fields: vec![
                                    Arc::new(Field::new(
                                        "start",
                                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                                        true,
                                    )),
                                    Arc::new(Field::new(
                                        "stop",
                                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                                        true,
                                    )),
                                    Arc::new(Field::new(
                                        "step",
                                        DataType::Interval(MonthDayNano),
                                        true,
                                    )),
                                ],
                                number_rows: num_rows,
                                return_field: Arc::new(Field::new(
                                    "result",
                                    DataType::List(Arc::new(Field::new_list_field(
                                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                                        true,
                                    ))),
                                    true,
                                )),
                                config_options: Arc::new(ConfigOptions::default()),
                            })
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Timestamp – generate_series(start, stop, step)
// ---------------------------------------------------------------------------
fn bench_generate_series_timestamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_series_timestamp");

    let intervals: &[(i32, i32, i64)] = &[
        (0, 1, 0),                   // 1 day
        (0, 7, 0),                   // 7 days
        (1, 0, 0),                   // 1 month
        (0, 0, 3_600_000_000_000),   // 1 hour in ns
    ];

    for &num_rows in NUM_ROWS {
        for &(months, days, ns) in intervals {
            let (start, stop, step) =
                make_timestamp_arrays(num_rows, months, days, ns);

            let args = vec![
                ColumnarValue::Array(start),
                ColumnarValue::Array(stop),
                ColumnarValue::Array(step),
            ];

            let label = make_interval_label(months, days, ns);

            group.bench_with_input(
                BenchmarkId::new(label, num_rows),
                &num_rows,
                |b, _| {
                    let udf = gen_series_udf();
                    b.iter(|| {
                        black_box(
                            udf.invoke_with_args(ScalarFunctionArgs {
                                args: args.clone(),
                                arg_fields: vec![
                                    Arc::new(Field::new(
                                        "start",
                                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                                        true,
                                    )),
                                    Arc::new(Field::new(
                                        "stop",
                                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                                        true,
                                    )),
                                    Arc::new(Field::new(
                                        "step",
                                        DataType::Interval(MonthDayNano),
                                        true,
                                    )),
                                ],
                                number_rows: num_rows,
                                return_field: Arc::new(Field::new(
                                    "result",
                                    DataType::List(Arc::new(Field::new_list_field(
                                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                                        true,
                                    ))),
                                    true,
                                )),
                                config_options: Arc::new(ConfigOptions::default()),
                            })
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build Date32 start, stop and IntervalMonthDayNano step arrays.
///
/// Each row gets a start date around 2020-01-01 and a stop date that is
/// approximately `num_intervals` steps later.
fn make_date32_arrays(
    num_rows: usize,
    months: i32,
    days: i32,
    ns: i32,
) -> (ArrayRef, ArrayRef, ArrayRef) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let num_intervals: i32 = 18;

    let start_values: Vec<i32> = (0..num_rows)
        .map(|_| DATE_2020_01_01 + rng.random_range(-365..365))
        .collect();
    let stop_values: Vec<i32> = start_values
        .iter()
        .map(|s| s + num_intervals * (months * 30 + days))
        .collect();
    let step_values: Vec<IntervalMonthDayNano> = (0..num_rows)
        .map(|_| IntervalMonthDayNano::new(months, days, ns as i64))
        .collect();

    let start = Arc::new(Date32Array::from(start_values));
    let stop = Arc::new(Date32Array::from(stop_values));
    let step = Arc::new(IntervalMonthDayNanoArray::from(step_values));

    (start, stop, step)
}

/// Build TimestampNanosecond start, stop and IntervalMonthDayNano step arrays.
fn make_timestamp_arrays(
    num_rows: usize,
    months: i32,
    days: i32,
    ns: i64,
) -> (ArrayRef, ArrayRef, ArrayRef) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let thirty_days_ns: i64 = 30i64 * 24 * 3600 * 1_000_000_000;
    let num_intervals: i64 = 12;

    let start_values: Vec<i64> = (0..num_rows)
        .map(|_| TS_2020_01_01 + rng.random_range(-thirty_days_ns..thirty_days_ns))
        .collect();
    let stop_base_add: i64 = num_intervals
        * (months as i64 * thirty_days_ns
            + days as i64 * 24 * 3600 * 1_000_000_000
            + ns);
    let stop_values: Vec<i64> = start_values.iter().map(|s| s + stop_base_add).collect();
    let step_values: Vec<IntervalMonthDayNano> = (0..num_rows)
        .map(|_| IntervalMonthDayNano::new(months, days, ns))
        .collect();

    let start = Arc::new(TimestampNanosecondArray::from(start_values));
    let stop = Arc::new(TimestampNanosecondArray::from(stop_values));
    let step = Arc::new(IntervalMonthDayNanoArray::from(step_values));

    (start, stop, step)
}

/// Build a human-readable label for an (months, days, ns) interval tuple.
fn make_interval_label(months: i32, days: i32, ns: i64) -> String {
    if months != 0 && days == 0 && ns == 0 {
        format!("interval_{months}m")
    } else if months == 0 && days != 0 && ns == 0 {
        format!("interval_{days}d")
    } else if months == 0 && days == 0 && ns != 0 {
        let hours = ns / 3_600_000_000_000;
        format!("interval_{hours}h")
    } else {
        format!("interval_{months}m_{days}d_{ns}ns")
    }
}

/// Build an Int64Array of `num_rows` small positive values in [1, max_val].
/// Used as the `stop` argument for single-arg `range(stop)` benchmarks.
fn make_int64_positive_array(num_rows: usize, max_val: i64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(SEED);
    let values: Vec<i64> = (0..num_rows)
        .map(|_| rng.random_range(1..=max_val))
        .collect();
    Arc::new(Int64Array::from(values))
}

/// Build (start, stop) Int64Arrays where each stop = start + offset,
/// with offset in [1, max_range]. This ensures every row produces a
/// bounded-size list, avoiding OOM from unbounded i64 ranges.
fn make_int64_start_stop_arrays(
    num_rows: usize,
    max_range: i64,
) -> (ArrayRef, ArrayRef) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut starts: Vec<i64> = Vec::with_capacity(num_rows);
    let mut stops: Vec<i64> = Vec::with_capacity(num_rows);
    for _ in 0..num_rows {
        let s = rng.random_range(0..max_range);
        let offset = rng.random_range(1..=max_range);
        starts.push(s);
        stops.push(s + offset);
    }
    (Arc::new(Int64Array::from(starts)), Arc::new(Int64Array::from(stops)))
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
