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

use arrow::array::{
    ArrayRef, BinaryArray, BinaryViewArray, Int32Array, ListArray, StringArray,
    StringViewArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::Field;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
use datafusion_functions_nested::map::map_udf;
use datafusion_functions_nested::map_extract::map_extract_udf;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cmp::max;
use std::hint::black_box;
use std::sync::Arc;

const SEED: u64 = 42;
const MAP_ROWS: usize = 1000;
const MAX_MAP_LENGTHS: [usize; 2] = [8, 16];
const HIT_MODES: [HitMode; 3] =
    [HitMode::FoundEarly, HitMode::FoundRandom, HitMode::NotFound];

#[derive(Clone, Copy)]
enum HitMode {
    FoundEarly,
    FoundRandom,
    NotFound,
}

impl HitMode {
    fn name(self) -> &'static str {
        match self {
            Self::FoundEarly => "found_early",
            Self::FoundRandom => "found_random",
            Self::NotFound => "not_found",
        }
    }
}

fn list_array(values: ArrayRef, lengths: &[usize]) -> ArrayRef {
    Arc::new(ListArray::new(
        Arc::new(Field::new_list_field(values.data_type().clone(), true)),
        OffsetBuffer::from_lengths(lengths.iter().copied()),
        values,
        None,
    ))
}

fn build_map_array(keys: ArrayRef, values: ArrayRef) -> ArrayRef {
    let number_rows = keys.len();
    let keys_arg = ColumnarValue::Array(keys);
    let values_arg = ColumnarValue::Array(values);
    let return_type = map_udf()
        .return_type(&[keys_arg.data_type(), values_arg.data_type()])
        .expect("should get return type");
    let arg_fields = vec![
        Field::new("keys", keys_arg.data_type(), true).into(),
        Field::new("values", values_arg.data_type(), true).into(),
    ];
    let return_field = Field::new("map", return_type, true).into();
    let config_options = Arc::new(ConfigOptions::default());
    let map_args = ScalarFunctionArgs {
        args: vec![keys_arg, values_arg],
        arg_fields,
        number_rows,
        return_field,
        config_options,
    };

    match map_udf()
        .invoke_with_args(map_args)
        .expect("map should work on valid values")
    {
        ColumnarValue::Array(array) => array,
        other => panic!("expected array result, got {other:?}"),
    }
}

fn bench_map_extract_case(
    c: &mut Criterion,
    name: &str,
    map_array: ArrayRef,
    key_arg: &ColumnarValue,
) {
    let number_rows = map_array.len();
    let map_arg = ColumnarValue::Array(map_array);
    let return_type = map_extract_udf()
        .return_type(&[map_arg.data_type(), key_arg.data_type()])
        .expect("should get return type");
    let arg_fields = vec![
        Field::new("map", map_arg.data_type(), true).into(),
        Field::new("key", key_arg.data_type(), true).into(),
    ];
    let return_field = Field::new("result", return_type, true).into();
    let config_options = Arc::new(ConfigOptions::default());

    c.bench_function(name, |b| {
        b.iter(|| {
            black_box(
                map_extract_udf()
                    .invoke_with_args(ScalarFunctionArgs {
                        args: vec![map_arg.clone(), key_arg.clone()],
                        arg_fields: arg_fields.clone(),
                        number_rows,
                        return_field: Arc::clone(&return_field),
                        config_options: Arc::clone(&config_options),
                    })
                    .expect("map_extract should work on valid values"),
            );
        });
    });
}

fn query_index(rng: &mut StdRng, len: usize, mode: HitMode) -> Option<usize> {
    match mode {
        HitMode::FoundEarly => Some(rng.random_range(0..max(1, len / 5))),
        HitMode::FoundRandom => Some(rng.random_range(0..len)),
        HitMode::NotFound => None,
    }
}

fn build_string_data(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (Vec<String>, Vec<String>) {
    let mut keys = Vec::with_capacity(lengths.iter().sum());
    let mut queries = Vec::with_capacity(lengths.len());

    for (row, len) in lengths.iter().copied().enumerate() {
        let row_keys = (0..len)
            .map(|index| format!("k_{row}_{index}"))
            .collect::<Vec<_>>();
        let query = match query_index(rng, len, mode) {
            Some(index) => row_keys[index].clone(),
            None => format!("missing_{row}"),
        };

        keys.extend(row_keys);
        queries.push(query);
    }
    (keys, queries)
}

fn build_i32_data(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (Vec<i32>, Vec<i32>) {
    let mut keys = Vec::with_capacity(lengths.iter().sum());
    let mut queries = Vec::with_capacity(lengths.len());

    for (row, len) in lengths.iter().copied().enumerate() {
        let base = (row as i32) * 1_000;
        let row_keys = (0..len)
            .map(|index| base + index as i32)
            .collect::<Vec<_>>();
        let query = match query_index(rng, len, mode) {
            Some(index) => row_keys[index],
            None => base + 10_000,
        };
        keys.extend(row_keys);
        queries.push(query);
    }
    (keys, queries)
}

fn build_binary_data(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut keys = Vec::with_capacity(lengths.iter().sum());
    let mut queries = Vec::with_capacity(lengths.len());

    for (row, len) in lengths.iter().copied().enumerate() {
        let row_keys = (0..len)
            .map(|index| format!("k_{row}_{index}").into_bytes())
            .collect::<Vec<_>>();
        let query = match query_index(rng, len, mode) {
            Some(index) => row_keys[index].clone(),
            None => format!("missing_{row}").into_bytes(),
        };

        keys.extend(row_keys);
        queries.push(query);
    }

    (keys, queries)
}

fn build_utf8_case(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (ArrayRef, ArrayRef) {
    let (keys, queries) = build_string_data(rng, lengths, mode);
    (
        list_array(Arc::new(StringArray::from(keys)) as ArrayRef, lengths),
        Arc::new(StringArray::from(queries)) as ArrayRef,
    )
}

fn build_utf8_view_case(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (ArrayRef, ArrayRef) {
    let (keys, queries) = build_string_data(rng, lengths, mode);
    (
        list_array(Arc::new(StringViewArray::from(keys)) as ArrayRef, lengths),
        Arc::new(StringViewArray::from(queries)) as ArrayRef,
    )
}

fn build_int32_case(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (ArrayRef, ArrayRef) {
    let (keys, queries) = build_i32_data(rng, lengths, mode);
    (
        list_array(Arc::new(Int32Array::from(keys)) as ArrayRef, lengths),
        Arc::new(Int32Array::from(queries)) as ArrayRef,
    )
}

fn build_binary_case(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (ArrayRef, ArrayRef) {
    let (keys, queries) = build_binary_data(rng, lengths, mode);
    (
        list_array(
            Arc::new(BinaryArray::from_iter_values(keys)) as ArrayRef,
            lengths,
        ),
        Arc::new(BinaryArray::from_iter_values(queries)) as ArrayRef,
    )
}

fn build_binary_view_case(
    rng: &mut StdRng,
    lengths: &[usize],
    mode: HitMode,
) -> (ArrayRef, ArrayRef) {
    let (keys, queries) = build_binary_data(rng, lengths, mode);
    (
        list_array(
            Arc::new(BinaryViewArray::from_iter_values(keys)) as ArrayRef,
            lengths,
        ),
        Arc::new(BinaryViewArray::from_iter_values(queries)) as ArrayRef,
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut map_extract_cases = Vec::new();

    macro_rules! add_cases {
        ($type_name:literal, $case_builder:expr) => {
            for max_len in MAX_MAP_LENGTHS {
                let lengths = (0..MAP_ROWS)
                    .map(|_| rng.random_range(1..=max_len))
                    .collect::<Vec<_>>();
                let values = {
                    let values = (0..lengths.iter().sum::<usize>())
                        .map(|value| value as i32)
                        .collect::<Vec<_>>();
                    list_array(Arc::new(Int32Array::from(values)) as ArrayRef, &lengths)
                };

                for mode in HIT_MODES {
                    let (keys, query_keys) = $case_builder(&mut rng, &lengths, mode);
                    map_extract_cases.push((
                        format!(
                            "map_extract_{}_max_len{max_len}_{}",
                            $type_name,
                            mode.name()
                        ),
                        build_map_array(keys, Arc::clone(&values)),
                        ColumnarValue::Array(query_keys),
                    ));
                }
            }
        };
    }

    add_cases!("utf8", build_utf8_case);
    add_cases!("int32", build_int32_case);
    add_cases!("string_view", build_utf8_view_case);
    add_cases!("binary", build_binary_case);
    add_cases!("binary_view", build_binary_view_case);

    for (name, map_array, key_arg) in map_extract_cases {
        bench_map_extract_case(c, &name, map_array, &key_arg);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
