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

use arrow::{
    array::{Int32Array, StringArray},
    datatypes::{DataType, Field},
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

fn prepare_sha2_args(
    size: usize,
    bit_len: i32,
    config_options: Arc<ConfigOptions>,
) -> ScalarFunctionArgs {
    let mut rng = StdRng::seed_from_u64(42);
    let null_density = 0.2;

    let input: StringArray = (0..size)
        .map(|_| {
            if rng.random::<f32>() < null_density {
                None
            } else {
                let len = rng.random_range::<usize, _>(1..100);
                let s: String = (0..len)
                    .map(|_| (rng.random_range::<u8, _>(97..123)) as char)
                    .collect();
                Some(s)
            }
        })
        .collect();

    // 根据传入的 bit_len 生成参数列
    let bit_length_array = Arc::new(Int32Array::from(vec![bit_len; size]));
    let args = vec![
        ColumnarValue::Array(Arc::new(input)),
        ColumnarValue::Array(bit_length_array),
    ];

    let arg_fields = args
        .iter()
        .enumerate()
        .map(|(idx, arg)| Field::new(format!("arg_{idx}"), arg.data_type(), true).into())
        .collect::<Vec<_>>();

    ScalarFunctionArgs {
        args,
        arg_fields,
        number_rows: size,
        return_field: Arc::new(Field::new("f", DataType::Utf8, true)),
        config_options,
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let sha2_fn = datafusion_spark::function::hash::sha2();
    let config_options = Arc::new(ConfigOptions::default());

    let sizes = [1024, 2048, 4096];
    let bit_lengths = [224, 256, 384, 512];

    let mut group = c.benchmark_group("sha2_variants_comparison");

    for bit_len in bit_lengths {
        for size in sizes {
            // 构造参数
            let bench_args = prepare_sha2_args(size, bit_len, config_options.clone());

            // 构造 Benchmark ID，例如 "SHA2-256/1024"
            let id = BenchmarkId::new(format!("SHA2-{}", bit_len), size);

            group.bench_with_input(id, &bench_args, |b, args| {
                b.iter(|| sha2_fn.invoke_with_args(args.clone()).unwrap())
            });
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
