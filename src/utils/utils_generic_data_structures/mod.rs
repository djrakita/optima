use std::fmt::Debug;
use serde_with::{serde_as};
use serde::de::DeserializeOwned;
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SquareArray2D<T> where T: Clone + Debug + Serialize + DeserializeOwned + Default + Combinable {
    #[serde_as(as = "Vec<Vec<_>>")]
    array: Vec<Vec<T>>,
    side_length: usize,
    symmetric: bool
}
impl <T> SquareArray2D <T> where T: Clone + Debug + Serialize + DeserializeOwned + Default + Combinable {
    pub fn new(side_length: usize, symmetric: bool) -> Self {
        let mut array = vec![];

        for _ in 0..side_length {
            let mut tmp = vec![];
            for _ in 0..side_length {
                tmp.push(T::default());
            }
            array.push(tmp);
        }

        Self {
            array,
            side_length,
            symmetric
        }
    }
    pub fn insert_data(&mut self, data: T, row_idx: usize, col_idx: usize) -> Result<(), OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(row_idx, self.side_length, file!(), line!())?;
        OptimaError::new_check_for_out_of_bound_error(col_idx, self.side_length, file!(), line!())?;

        if self.symmetric {
            self.array[row_idx][col_idx] = data.clone();
            self.array[col_idx][row_idx] = data;
        } else {
            self.array[row_idx][col_idx] = data;
        }


        Ok(())
    }
    pub fn combine(&mut self, other: &Self) -> Result<(), OptimaError> {
        if self.side_length != other.side_length {
            return Err(OptimaError::new_generic_error_str("Cannot combine SquareArray2Ds of different sizes.", file!(), line!()));
        }

        let side_length = self.side_length;

        for col in 0..side_length {
            for row in 0..side_length {
                self.array[col][row] = self.array[col][row].combine(&other.array[col][row]);
            }
        }

        Ok(())
    }
    pub fn data_cell(&self, row_idx: usize, col_idx: usize) -> Result<&T, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(row_idx, self.side_length, file!(), line!())?;
        OptimaError::new_check_for_out_of_bound_error(col_idx, self.side_length, file!(), line!())?;

        Ok(&self.array[row_idx][col_idx])
    }
    pub fn data_cell_mut(&mut self, row_idx: usize, col_idx: usize) -> Result<&mut T, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(row_idx, self.side_length, file!(), line!())?;
        OptimaError::new_check_for_out_of_bound_error(col_idx, self.side_length, file!(), line!())?;

        Ok(&mut self.array[row_idx][col_idx])
    }
}

pub trait Combinable {
    fn combine(&self, other: &Self) -> Self;
}
impl Combinable for f64 {
    fn combine(&self, other: &Self) -> Self {
        (*self + *other) / 2.0
    }
}
impl Combinable for bool {
    fn combine(&self, other: &Self) -> Self {
        return *self || *other;
    }
}

