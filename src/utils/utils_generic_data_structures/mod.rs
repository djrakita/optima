use std::fmt::Debug;
use serde_with::{serde_as};
use serde::de::DeserializeOwned;
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Array1D<T> where T: Clone + Debug + Serialize + DeserializeOwned + Default + Mixable {
    #[serde_as(as = "Vec<_>")]
    array: Vec<T>
}
impl <T> Array1D <T> where T: Clone + Debug + Serialize + DeserializeOwned + Default + Mixable {
    pub fn new(length: usize, initialization_value: Option<T>) -> Self {
        return match initialization_value {
            None => { Self::new_without_initialization_value(length) }
            Some(initialization_value) => { Self::new_with_initialization_value(length, initialization_value) }
        }
    }
    fn new_without_initialization_value(length: usize) -> Self {
        return Self::new_with_initialization_value(length, T::default())
    }
    fn new_with_initialization_value(length: usize, initialization_value: T) -> Self {
        let mut array = vec![];
        for _ in 0..length { array.push(initialization_value.clone()) }
        return Self {
            array
        }
    }
    pub fn replace_data(&mut self, data: T, idx: usize) -> Result<(), OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(idx, self.array.len(), file!(), line!())?;

        self.array[idx] = data;

        Ok(())
    }
    pub fn replace_data_on_every_cell(&mut self, data: T) {
        for cell in &mut self.array {
            *cell = data.clone();
        }
    }
    pub fn append_new_cell(&mut self, data: T) {
        self.array.push(data);
    }
    pub fn mix(&mut self, other: &Self) -> Result<(), OptimaError> {
        if self.array.len() != other.array.len() {
            return Err(OptimaError::new_generic_error_str("Cannot combine Array1Ds of different sizes.", file!(), line!()));
        }

        let length = self.array.len();

        for i in 0..length {
            self.array[i] = self.array[i].mix(&other.array[i]);
        }

        Ok(())
    }
    pub fn data_cell(&self, idx: usize) -> Result<&T, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(idx, self.array.len(), file!(), line!())?;

        Ok(&self.array[idx])
    }
    pub fn data_cell_mut(&mut self, idx: usize) -> Result<&mut T, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(idx, self.array.len(), file!(), line!())?;

        Ok(&mut self.array[idx])
    }
    pub fn len(&self) -> usize { self.array.len() }
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SquareArray2D<T> where T: Clone + Debug + Serialize + DeserializeOwned + Default + Mixable {
    #[serde_as(as = "Vec<Vec<_>>")]
    array: Vec<Vec<T>>,
    side_length: usize,
    symmetric: bool
}
impl <T> SquareArray2D <T> where T: Clone + Debug + Serialize + DeserializeOwned + Default + Mixable {
    pub fn new(side_length: usize, symmetric: bool, initialization_value: Option<T>) -> Self {
        return match initialization_value {
            None => { Self::new_without_initialization_value(side_length, symmetric) }
            Some(initialization_value) => { Self::new_with_initialization_value(side_length, symmetric, initialization_value) }
        }
    }
    /// Concatenation places the second matrix in the lower right corner such that an m x m matrix
    /// concatenated with an n x n matrix will become an m + n x m + n matrix.  Off diagonal terms
    /// will be T::default
    pub fn new_concatenated(s1: &SquareArray2D<T>, s2: &SquareArray2D<T>, symmetric: bool) -> Self {
        let l1 = s1.side_length;
        let l2 = s2.side_length;

        let mut out_self = Self::new(l1 + l2, symmetric, None);

        for i in 0..l1 {
            for j in 0..l1 {
                out_self.array[i][j] = s1.array[i][j].clone();
            }
        }

        for i in 0..l2 {
            for j in 0..l2 {
                out_self.array[i + l1][j + l1] = s2.array[i][j].clone();
            }
        }

        return out_self
    }
    fn new_without_initialization_value(side_length: usize, symmetric: bool) -> Self {
        return Self::new_with_initialization_value(side_length, symmetric, T::default())
    }
    fn new_with_initialization_value(side_length: usize, symmetric: bool, initialization_value: T) -> Self {
        let mut array = vec![];

        for _ in 0..side_length {
            let mut tmp = vec![];
            for _ in 0..side_length {
                tmp.push(initialization_value.clone());
            }
            array.push(tmp);
        }

        Self {
            array,
            side_length,
            symmetric
        }
    }
    pub fn replace_data(&mut self, data: T, row_idx: usize, col_idx: usize) -> Result<(), OptimaError> {
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
    pub fn replace_data_on_every_cell(&mut self, data: T) {
        let l = self.side_length;
        for row in 0..l {
            for col in 0..l {
                self.array[col][row] = data.clone();
            }
        }
    }
    pub fn adjust_data<F: Fn(&mut T)>(&mut self, adjustment: F, row_idx: usize, col_idx: usize) -> Result<(), OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(row_idx, self.side_length, file!(), line!())?;
        OptimaError::new_check_for_out_of_bound_error(col_idx, self.side_length, file!(), line!())?;

        let data = &mut self.array[row_idx][col_idx];
        adjustment(data);

        if self.symmetric {
            let data = &mut self.array[col_idx][row_idx];
            adjustment(data);
        }

        Ok(())
    }
    pub fn adjust_data_on_every_cell<F: Fn(&mut T)>(&mut self, adjustment: F) {
        let l = self.side_length;
        for row in 0..l {
            for col in 0..l {
                let data = &mut self.array[row][col];
                adjustment(data);
            }
        }
    }
    pub fn append_new_row_and_column(&mut self, data: Option<T>) {
        for col in &mut self.array {
            match &data {
                None => { col.push( T::default() ) }
                Some(data) => { col.push(data.clone()) }
            }
        }

        self.side_length += 1;

        let mut new_row = vec![];
        for _ in 0..self.side_length {
            match &data {
                None => { new_row.push( T::default() ) }
                Some(data) => { new_row.push(data.clone()) }
            }
        }
        self.array.push(new_row);
    }
    pub fn mix(&mut self, other: &Self) -> Result<(), OptimaError> {
        if self.side_length != other.side_length {
            return Err(OptimaError::new_generic_error_str("Cannot combine SquareArray2Ds of different sizes.", file!(), line!()));
        }

        let side_length = self.side_length;

        for col in 0..side_length {
            for row in 0..side_length {
                self.array[col][row] = self.array[col][row].mix(&other.array[col][row]);
            }
        }

        Ok(())
    }
    /// Concatenation places the second matrix in the lower right corner such that an m x m matrix
    /// concatenated with an n x n matrix will become an m + n x m + n matrix.  Off diagonal terms
    /// will be T::default
    pub fn concatenate_in_place(&mut self, other: &Self) {
        let l1 = self.side_length;
        let l2 = other.side_length;
        for _ in 0..l2 { self.append_new_row_and_column(None); }
        for i in 0..l2 {
            for j in 0..l2 {
                self.array[i + l1][j + l1] = other.array[i][j].clone();
            }
        }
    }
    pub fn data_cell(&self, row_idx: usize, col_idx: usize) -> Result<&T, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(row_idx, self.side_length, file!(), line!())?;
        OptimaError::new_check_for_out_of_bound_error(col_idx, self.side_length, file!(), line!())?;

        Ok(&self.array[row_idx][col_idx])
    }
}

pub trait Mixable {
    fn mix(&self, other: &Self) -> Self;
}
impl Mixable for f64 {
    fn mix(&self, other: &Self) -> Self {
        (*self + *other) / 2.0
    }
}
impl Mixable for bool {
    fn mix(&self, other: &Self) -> Self {
        return *self || *other;
    }
}
impl Mixable for usize {
    fn mix(&self, other: &Self) -> Self {
        *self + *other
    }
}
impl Mixable for OptimaSE3Pose {
    fn mix(&self, other: &Self) -> Self {
        self.multiply(other, false).expect("error")
    }
}
impl Mixable for AveragingFloat {
    fn mix(&self, other: &Self) -> Self {
        let mut out_self = AveragingFloat::new();
        out_self.absorb(self.value);
        out_self.absorb(other.value);
        out_self.counter = self.counter + other.counter;
        return out_self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AveragingFloat {
    total_sum: f64,
    counter: f64,
    value: f64
}
impl AveragingFloat {
    pub fn new() -> Self {
        Self {
            total_sum: 0.0,
            counter: 0.0,
            value: 0.0
        }
    }
    pub fn absorb(&mut self, value: f64) {
        self.total_sum += value;
        self.counter += 1.0;
        self.value = self.total_sum / self.counter;
    }
    pub fn value(&self) -> f64 { self.value }
}
impl Default for AveragingFloat {
    fn default() -> Self {
        Self::new()
    }
}

