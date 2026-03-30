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
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::Field;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
use datafusion_functions_nested::map::map_udf;
use datafusion_functions_nested::map_extract::map_extract_udf;
use rand::Rng;
use rand::prelude::ThreadRng;
use std::collections::HashSet;
use std::hash::Hash;
use std::hint::black_box;
use std::sync::Arc;

const MAP_ROWS: usize = 1000;
const MAP_KEYS_PER_ROW: usize = 1000;

fn gen_unique_values<T>(
    rng: &mut ThreadRng,
    mut make_value: impl FnMut(i32) -> T,
) -> Vec<T>
where
    T: Eq + Hash,
{
    let mut values = HashSet::with_capacity(MAP_KEYS_PER_ROW);

    while values.len() < MAP_KEYS_PER_ROW {
        values.insert(make_value(rng.random_range(0..10000)));
    }

    values.into_iter().collect()
}

fn gen_repeat_values<T: Clone>(values: &[T], repeats: usize) -> Vec<T> {
    let mut repeated = Vec::with_capacity(values.len() * repeats);

    for _ in 0..repeats {
        repeated.extend_from_slice(values);
    }

    repeated
}

fn gen_utf8_values(rng: &mut ThreadRng) -> Vec<String> {
    gen_unique_values(rng, |value| value.to_string())
}

fn gen_binary_values(rng: &mut ThreadRng) -> Vec<Vec<u8>> {
    gen_unique_values(rng, |value| value.to_le_bytes().to_vec())
}

fn gen_primitive_values(rng: &mut ThreadRng) -> Vec<i32> {
    gen_unique_values(rng, |value| value)
}

fn list_array(values: ArrayRef, row_count: usize, values_per_row: usize) -> ArrayRef {
    let offsets = (0..=row_count)
        .map(|index| (index * values_per_row) as i32)
        .collect::<Vec<_>>();
    Arc::new(ListArray::new(
        Arc::new(Field::new_list_field(values.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
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

    match map_udf()
        .invoke_with_args(ScalarFunctionArgs {
            args: vec![keys_arg, values_arg],
            arg_fields,
            number_rows,
            return_field,
            config_options,
        })
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
    query_keys: ArrayRef,
) {
    let number_rows = map_array.len();
    let map_arg = ColumnarValue::Array(map_array);
    let key_arg = ColumnarValue::Array(query_keys);
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

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();
    let primitive_values = gen_primitive_values(&mut rng);
    let utf8_values = gen_utf8_values(&mut rng);
    let binary_values = gen_binary_values(&mut rng);
    let values = Arc::new(Int32Array::from(gen_repeat_values(
        &primitive_values,
        MAP_ROWS,
    ))) as ArrayRef;
    let values = list_array(values, MAP_ROWS, MAP_KEYS_PER_ROW);

    let map_extract_cases = [
        (
            "map_extract_1000_utf8_found_middle",
            build_map_array(
                list_array(
                    Arc::new(StringArray::from(gen_repeat_values(&utf8_values, MAP_ROWS)))
                        as ArrayRef,
                    MAP_ROWS,
                    MAP_KEYS_PER_ROW,
                ),
                Arc::clone(&values),
            ),
            Arc::new(StringArray::from(vec![
                utf8_values[MAP_KEYS_PER_ROW / 2]
                    .clone();
                MAP_ROWS
            ])) as ArrayRef,
        ),
        (
            "map_extract_1000_utf8_found_last",
            build_map_array(
                list_array(
                    Arc::new(StringArray::from(gen_repeat_values(&utf8_values, MAP_ROWS)))
                        as ArrayRef,
                    MAP_ROWS,
                    MAP_KEYS_PER_ROW,
                ),
                Arc::clone(&values),
            ),
            Arc::new(StringArray::from(vec![
                utf8_values[MAP_KEYS_PER_ROW - 1]
                    .clone();
                MAP_ROWS
            ])) as ArrayRef,
        ),
        (
            "map_extract_1000_binary_found_last",
            build_map_array(
                list_array(
                    Arc::new(BinaryArray::from_iter_values(gen_repeat_values(
                        &binary_values,
                        MAP_ROWS,
                    ))) as ArrayRef,
                    MAP_ROWS,
                    MAP_KEYS_PER_ROW,
                ),
                Arc::clone(&values),
            ),
            Arc::new(BinaryArray::from_iter_values(vec![
                binary_values[MAP_KEYS_PER_ROW - 1].clone();
                MAP_ROWS
            ])) as ArrayRef,
        ),
        (
            "map_extract_1000_utf8_view_found_last",
            build_map_array(
                list_array(
                    Arc::new(StringViewArray::from(gen_repeat_values(
                        &utf8_values,
                        MAP_ROWS,
                    ))) as ArrayRef,
                    MAP_ROWS,
                    MAP_KEYS_PER_ROW,
                ),
                Arc::clone(&values),
            ),
            Arc::new(StringViewArray::from(vec![
                utf8_values[MAP_KEYS_PER_ROW - 1]
                    .clone();
                MAP_ROWS
            ])) as ArrayRef,
        ),
        (
            "map_extract_1000_binary_view_found_last",
            build_map_array(
                list_array(
                    Arc::new(BinaryViewArray::from_iter_values(gen_repeat_values(
                        &binary_values,
                        MAP_ROWS,
                    ))) as ArrayRef,
                    MAP_ROWS,
                    MAP_KEYS_PER_ROW,
                ),
                Arc::clone(&values),
            ),
            Arc::new(BinaryViewArray::from_iter_values(vec![
                binary_values[MAP_KEYS_PER_ROW - 1].clone();
                MAP_ROWS
            ])) as ArrayRef,
        ),
        (
            "map_extract_1000_int32_found_last",
            build_map_array(
                list_array(
                    Arc::new(Int32Array::from(gen_repeat_values(
                        &primitive_values,
                        MAP_ROWS,
                    ))) as ArrayRef,
                    MAP_ROWS,
                    MAP_KEYS_PER_ROW,
                ),
                Arc::clone(&values),
            ),
            Arc::new(Int32Array::from(vec![
                primitive_values[MAP_KEYS_PER_ROW - 1];
                MAP_ROWS
            ])) as ArrayRef,
        ),
    ];

    for (name, map_array, query_keys) in map_extract_cases {
        bench_map_extract_case(c, name, map_array, query_keys);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
