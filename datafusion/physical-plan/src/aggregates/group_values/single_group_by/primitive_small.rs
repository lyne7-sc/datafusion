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

use crate::aggregates::group_values::GroupValues;
use arrow::array::{
    ArrayRef, ArrowPrimitiveType, NullBufferBuilder, PrimitiveArray, cast::AsArray,
};
use arrow::datatypes::DataType;
use datafusion_common::Result;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use std::mem::size_of;
use std::sync::Arc;

const UNSET: u32 = u32::MAX;

pub trait SmallIndexValue: Copy + Default {
    const RANGE: usize;

    fn to_index(self) -> usize;
}

impl SmallIndexValue for i8 {
    const RANGE: usize = 1 << 8;

    fn to_index(self) -> usize {
        (self as i16 - i8::MIN as i16) as usize
    }
}

impl SmallIndexValue for u8 {
    const RANGE: usize = 1 << 8;

    fn to_index(self) -> usize {
        self as usize
    }
}

impl SmallIndexValue for i16 {
    const RANGE: usize = 1 << 16;

    fn to_index(self) -> usize {
        (self as i32 - i16::MIN as i32) as usize
    }
}

impl SmallIndexValue for u16 {
    const RANGE: usize = 1 << 16;

    fn to_index(self) -> usize {
        self as usize
    }
}

/// A [`GroupValues`] storing a single column of small-range integer values.
///
/// Instead of hashing, the value range is used as a direct index lookup table.
pub struct GroupValuesPrimitiveSmall<T: ArrowPrimitiveType>
where
    T::Native: SmallIndexValue,
{
    data_type: DataType,
    group_ids: Vec<u32>,
    null_group: Option<usize>,
    values: Vec<T::Native>,
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitiveSmall<T>
where
    T::Native: SmallIndexValue,
{
    pub fn new(data_type: DataType) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        Self {
            data_type,
            group_ids: vec![UNSET; T::Native::RANGE],
            null_group: None,
            values: Vec::with_capacity(128),
        }
    }

    fn rebuild_group_ids(&mut self) {
        self.group_ids.fill(UNSET);
        for (group_id, value) in self.values.iter().copied().enumerate() {
            if self.null_group == Some(group_id) {
                continue;
            }
            self.group_ids[value.to_index()] = group_id as u32;
        }
    }
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitiveSmall<T>
where
    T::Native: SmallIndexValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        groups.clear();

        for value in cols[0].as_primitive::<T>() {
            let group_id = match value {
                None => *self.null_group.get_or_insert_with(|| {
                    let group_id = self.values.len();
                    self.values.push(Default::default());
                    group_id
                }),
                Some(value) => {
                    let slot = &mut self.group_ids[value.to_index()];
                    if *slot == UNSET {
                        let group_id = self.values.len();
                        self.values.push(value);
                        *slot = group_id as u32;
                        group_id
                    } else {
                        *slot as usize
                    }
                }
            };
            groups.push(group_id);
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.group_ids.allocated_size() + self.values.allocated_size() + size_of::<Self>()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        fn build_primitive<T: ArrowPrimitiveType>(
            values: Vec<T::Native>,
            null_idx: Option<usize>,
        ) -> PrimitiveArray<T> {
            let nulls = null_idx.map(|null_idx| {
                let mut buffer = NullBufferBuilder::new(values.len());
                buffer.append_n_non_nulls(null_idx);
                buffer.append_null();
                buffer.append_n_non_nulls(values.len() - null_idx - 1);
                buffer.finish().unwrap()
            });
            PrimitiveArray::<T>::new(values.into(), nulls)
        }

        let array: PrimitiveArray<T> = match emit_to {
            EmitTo::All => {
                self.group_ids.fill(UNSET);
                build_primitive(std::mem::take(&mut self.values), self.null_group.take())
            }
            EmitTo::First(n) => {
                assert!(n <= self.len());

                let null_group = match self.null_group {
                    Some(group_id) if group_id < n => self.null_group.take(),
                    Some(group_id) => {
                        self.null_group = Some(group_id - n);
                        None
                    }
                    None => None,
                };

                let remaining = self.values.split_off(n);
                let emitted = std::mem::replace(&mut self.values, remaining);
                self.rebuild_group_ids();
                build_primitive(emitted, null_group)
            }
        };

        Ok(vec![Arc::new(array.with_data_type(self.data_type.clone()))])
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.group_ids.fill(UNSET);
        self.null_group = None;
        self.values.clear();
        self.values.shrink_to(num_rows);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int8Array, UInt16Array, cast::AsArray};
    use arrow::datatypes::DataType;
    use datafusion_expr::EmitTo;

    use crate::aggregates::group_values::GroupValues;

    use super::GroupValuesPrimitiveSmall;

    #[test]
    fn test_intern_int8_with_nulls() {
        let mut group_values =
            GroupValuesPrimitiveSmall::<arrow::array::types::Int8Type>::new(
                DataType::Int8,
            );
        let mut groups = vec![];
        let input: ArrayRef =
            Arc::new(Int8Array::from(vec![Some(-2), None, Some(-2), Some(3)]));

        group_values.intern(&[input], &mut groups).unwrap();

        assert_eq!(groups, vec![0, 1, 0, 2]);

        let emitted = group_values.emit(EmitTo::All).unwrap();
        let emitted = emitted[0].as_primitive::<arrow::array::types::Int8Type>();
        assert_eq!(emitted, &Int8Array::from(vec![Some(-2), None, Some(3)]));
    }

    #[test]
    fn test_emit_first_reindexes_remaining_values() {
        let mut group_values =
            GroupValuesPrimitiveSmall::<arrow::array::types::UInt16Type>::new(
                DataType::UInt16,
            );
        let mut groups = vec![];
        let input: ArrayRef = Arc::new(UInt16Array::from(vec![
            Some(9),
            Some(7),
            Some(9),
            None,
            Some(12),
        ]));

        group_values.intern(&[input], &mut groups).unwrap();
        assert_eq!(groups, vec![0, 1, 0, 2, 3]);

        let first = group_values.emit(EmitTo::First(2)).unwrap();
        let first = first[0].as_primitive::<arrow::array::types::UInt16Type>();
        assert_eq!(first, &UInt16Array::from(vec![Some(9), Some(7)]));

        let next_input: ArrayRef =
            Arc::new(UInt16Array::from(vec![Some(12), None, Some(7), Some(18)]));
        group_values.intern(&[next_input], &mut groups).unwrap();

        assert_eq!(groups, vec![1, 0, 2, 3]);

        let remaining = group_values.emit(EmitTo::All).unwrap();
        let remaining = remaining[0].as_primitive::<arrow::array::types::UInt16Type>();
        assert_eq!(
            remaining,
            &UInt16Array::from(vec![None, Some(12), Some(7), Some(18)])
        );
    }
}
