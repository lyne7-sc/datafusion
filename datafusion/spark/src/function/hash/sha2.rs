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

extern crate datafusion_functions;

use crate::function::error_utils::{
    invalid_arg_count_exec_err, unsupported_data_type_exec_err,
};
use arrow::array::{
    Array, ArrayRef, AsArray, StringBuilder,
};
use arrow::datatypes::{DataType, Int32Type};
use datafusion_common::{Result, ScalarValue, exec_err, internal_datafusion_err};
use datafusion_expr::Signature;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Volatility};
pub use datafusion_functions::crypto::basic::{sha224, sha256, sha384, sha512};
use sha2::{Digest, Sha224, Sha256, Sha384, Sha512};
use std::any::Any;
use std::sync::Arc;

/// <https://spark.apache.org/docs/latest/api/sql/index.html#sha2>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkSha2 {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for SparkSha2 {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkSha2 {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            aliases: vec![],
        }
    }
}

impl ScalarUDFImpl for SparkSha2 {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "sha2"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types[1].is_null() {
            return Ok(DataType::Null);
        }
        Ok(match arg_types[0] {
            DataType::Utf8View
            | DataType::LargeUtf8
            | DataType::Utf8
            | DataType::Binary
            | DataType::BinaryView
            | DataType::LargeBinary => DataType::Utf8,
            DataType::Null => DataType::Null,
            _ => {
                return exec_err!(
                    "{} function can only accept strings or binary arrays.",
                    self.name()
                );
            }
        })
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args: [ColumnarValue; 2] = args.args.try_into().map_err(|_| {
            internal_datafusion_err!("Expected 2 arguments for function sha2")
        })?;

        sha2(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 2 {
            return Err(invalid_arg_count_exec_err(
                self.name(),
                (2, 2),
                arg_types.len(),
            ));
        }
        let expr_type = match &arg_types[0] {
            DataType::Utf8View
            | DataType::LargeUtf8
            | DataType::Utf8
            | DataType::Binary
            | DataType::BinaryView
            | DataType::LargeBinary
            | DataType::Null => Ok(arg_types[0].clone()),
            _ => Err(unsupported_data_type_exec_err(
                self.name(),
                "String, Binary",
                &arg_types[0],
            )),
        }?;
        let bit_length_type = if arg_types[1].is_numeric() {
            Ok(DataType::Int32)
        } else if arg_types[1].is_null() {
            Ok(DataType::Null)
        } else {
            Err(unsupported_data_type_exec_err(
                self.name(),
                "Numeric Type",
                &arg_types[1],
            ))
        }?;

        Ok(vec![expr_type, bit_length_type])
    }
}

pub fn sha2(args: [ColumnarValue; 2]) -> Result<ColumnarValue> {
    let [input_values, bit_lengths] = args;

    match bit_lengths {
        ColumnarValue::Scalar(ScalarValue::Int32(bit_opt)) => {
            if let Some(bits) = bit_opt {
                compute_sha2_bulk(bits, &input_values)
            } else {
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)))
            }
        }

        ColumnarValue::Array(bit_array) => {
            let bit_array = bit_array.as_primitive::<Int32Type>();
            let len = bit_array.len();
            let input_array = input_values.into_array(len)?;

            let mut builder = StringBuilder::with_capacity(len, len * 64);

            for i in 0..len {
                if bit_array.is_null(i) || input_array.is_null(i) {
                    builder.append_null();
                    continue;
                }

                if let Some(algo) = DigestAlgorithm::from_bits(bit_array.value(i)) {
                    let bytes = get_bytes_at(&input_array, i);
                    builder.append_value(algo.compute_and_hex(bytes));
                } else {
                    builder.append_null();
                }
            }
            Ok(ColumnarValue::Array(Arc::new(builder.finish())))
        }
        _ => exec_err!("sha2 second argument must be Int32"),
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum DigestAlgorithm {
    Sha224,
    Sha256,
    Sha384,
    Sha512,
}

impl DigestAlgorithm {
    fn hex_length(&self) -> usize {
        match self {
            DigestAlgorithm::Sha224 => 56,
            DigestAlgorithm::Sha256 => 64,
            DigestAlgorithm::Sha384 => 96,
            DigestAlgorithm::Sha512 => 128,
        }
    }

    fn from_bits(bits: i32) -> Option<Self> {
        match bits {
            0 | 256 => Some(Self::Sha256),
            224 => Some(Self::Sha224),
            384 => Some(Self::Sha384),
            512 => Some(Self::Sha512),
            _ => None,
        }
    }

    fn compute_and_hex(&self, bytes: &[u8]) -> String {
        let digest = match self {
            Self::Sha224 => Sha224::digest(bytes).to_vec(),
            Self::Sha256 => Sha256::digest(bytes).to_vec(),
            Self::Sha384 => Sha384::digest(bytes).to_vec(),
            Self::Sha512 => Sha512::digest(bytes).to_vec(),
        };
        hex_encode_lower(digest)
    }
}

const HEX_CHARS_LOWER: &[u8; 16] = b"0123456789abcdef";

#[inline]
fn hex_encode_lower<T: AsRef<[u8]>>(data: T) -> String {
    let input = data.as_ref();
    let mut output = vec![0u8; input.len() * 2];
    for (i, &byte) in input.iter().enumerate() {
        output[i * 2] = HEX_CHARS_LOWER[(byte >> 4) as usize];
        output[i * 2 + 1] = HEX_CHARS_LOWER[(byte & 0x0F) as usize];
    }
    String::from_utf8(output).unwrap()
}

fn get_bytes_at(array: &ArrayRef, i: usize) -> &[u8] {
    match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value(i).as_bytes(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(i).as_bytes(),
        DataType::Utf8View => array.as_string_view().value(i).as_bytes(),
        DataType::Binary => array.as_binary::<i32>().value(i),
        DataType::LargeBinary => array.as_binary::<i64>().value(i),
        DataType::BinaryView => array.as_binary_view().value(i),
        _ => &[],
    }
}

fn compute_sha2_bulk(bits: i32, input: &ColumnarValue) -> Result<ColumnarValue> {
    let algo = match DigestAlgorithm::from_bits(bits) {
        Some(a) => a,
        None => return Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None))),
    };

    match input {
        ColumnarValue::Scalar(scalar) => {
            let bytes = match scalar {
                ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => {
                    Some(s.as_bytes())
                }
                ScalarValue::Utf8View(Some(s)) => Some(s.as_bytes()),
                ScalarValue::Binary(Some(b))
                | ScalarValue::LargeBinary(Some(b))
                | ScalarValue::BinaryView(Some(b)) => Some(b.as_slice()),
                _ => None,
            };
            let res = bytes.map(|b| algo.compute_and_hex(b));
            Ok(ColumnarValue::Scalar(ScalarValue::Utf8(res)))
        }
        ColumnarValue::Array(array) => {
            let len = array.len();
            let mut builder = StringBuilder::with_capacity(len, len * algo.hex_length());

            for i in 0..len {
                if array.is_null(i) {
                    builder.append_null();
                } else {
                    let bytes = get_bytes_at(array, i);
                    builder.append_value(algo.compute_and_hex(bytes));
                }
            }
            Ok(ColumnarValue::Array(Arc::new(builder.finish())))
        }
    }
}
