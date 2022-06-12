use nalgebra::{DMatrix, DVector};
use ndarray::{Array, ArrayD};
use serde::{Serialize, Deserialize};
use crate::utils::utils_generic_data_structures::EnumMapToSignature;

#[derive(Clone, Debug)]
pub enum OptimaTensor {
    Scalar(OptimaTensor0D),
    Vector(OptimaTensor1D),
    Matrix(OptimaTensor2D),
    TensorND(OptimaTensorND)
}
impl OptimaTensor {
    pub fn new_zeros(dimensions: Vec<usize>) -> Self {
        let l = dimensions.len();
        return if l == 0 {
            Self::Scalar(OptimaTensor0D::new_zero())
        } else if l == 1 {
            Self::Vector(OptimaTensor1D::new_zeros(dimensions[0]))
        } else if l == 2 {
            Self::Matrix(OptimaTensor2D::new_zeros(dimensions[0], dimensions[1]))
        } else {
            Self::TensorND(OptimaTensorND::new_zeros(dimensions))
        }
    }
    pub fn convert(&self, target: &OptimaTensorSignature) -> OptimaTensor {
        return match self {
            OptimaTensor::Scalar(t) => { t.convert(target) }
            OptimaTensor::Vector(t) => { t.convert(target) }
            OptimaTensor::Matrix(t) => { t.convert(target) }
            OptimaTensor::TensorND(t) => { t.convert(target) }
        }
    }
    pub fn dimensions(&self) -> Vec<usize> {
        match self {
            OptimaTensor::Scalar(t) => { t.dimensions() }
            OptimaTensor::Vector(t) => { t.dimensions() }
            OptimaTensor::Matrix(t) => { t.dimensions() }
            OptimaTensor::TensorND(t) => { t.dimensions() }
        }
    }
    pub fn set_value(&mut self, indices: Vec<usize>, value: f64) {
        match self {
            OptimaTensor::Scalar(t) => { t.set_value(indices, value); }
            OptimaTensor::Vector(t) => { t.set_value(indices, value); }
            OptimaTensor::Matrix(t) => { t.set_value(indices, value); }
            OptimaTensor::TensorND(t) => { t.set_value(indices, value); }
        }
    }
    pub fn get_value(&self, indices: Vec<usize>) -> f64 {
        return match self {
            OptimaTensor::Scalar(t) => { t.get_value(indices) }
            OptimaTensor::Vector(t) => { t.get_value(indices) }
            OptimaTensor::Matrix(t) => { t.get_value(indices) }
            OptimaTensor::TensorND(t) => { t.get_value(indices) }
        }
    }
    fn vectorized_data(&self) -> &[f64] {
        match self {
            OptimaTensor::Scalar(t) => { t.vectorized_data() }
            OptimaTensor::Vector(t) => { t.vectorized_data() }
            OptimaTensor::Matrix(t) => { t.vectorized_data() }
            OptimaTensor::TensorND(t) => { t.vectorized_data() }
        }
    }
    fn vectorized_data_mut(&mut self) -> &mut [f64] {
        match self {
            OptimaTensor::Scalar(t) => { t.vectorized_data_mut() }
            OptimaTensor::Vector(t) => { t.vectorized_data_mut() }
            OptimaTensor::Matrix(t) => { t.vectorized_data_mut() }
            OptimaTensor::TensorND(t) => { t.vectorized_data_mut() }
        }
    }
    fn vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        match self {
            OptimaTensor::Scalar(t) => { t.vectorized_idx_to_indices(vectorized_idx) }
            OptimaTensor::Vector(t) => { t.vectorized_idx_to_indices(vectorized_idx) }
            OptimaTensor::Matrix(t) => { t.vectorized_idx_to_indices(vectorized_idx) }
            OptimaTensor::TensorND(t) => { t.vectorized_idx_to_indices(vectorized_idx) }
        }
    }
    fn indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        return match self {
            OptimaTensor::Scalar(t) => { t.indices_to_vectorized_idx(indices) }
            OptimaTensor::Vector(t) => { t.indices_to_vectorized_idx(indices) }
            OptimaTensor::Matrix(t) => { t.indices_to_vectorized_idx(indices) }
            OptimaTensor::TensorND(t) => { t.indices_to_vectorized_idx(indices) }
        }
    }
}
impl EnumMapToSignature<OptimaTensorSignature> for OptimaTensor {
    fn map_to_signature(&self) -> OptimaTensorSignature {
        match self {
            OptimaTensor::Scalar(_) => { OptimaTensorSignature::Scalar }
            OptimaTensor::Vector(_) => { OptimaTensorSignature::Vector }
            OptimaTensor::Matrix(_) => { OptimaTensorSignature::Matrix }
            OptimaTensor::TensorND(_) => { OptimaTensorSignature::TensorND }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptimaTensorSignature {
    Scalar,
    Vector,
    Matrix,
    TensorND
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaTensor0D {
    value: [f64; 1]
}
impl OptimaTensor0D {
    pub fn new_zero() -> Self {
        Self::new(0.0)
    }
    pub fn new(value: f64) -> Self {
        Self {
            value: [value]
        }
    }
    pub fn convert(&self, target: &OptimaTensorSignature) -> OptimaTensor {
        return match target {
            OptimaTensorSignature::Scalar => {
                OptimaTensor::Scalar(self.clone())
            },
            OptimaTensorSignature::Vector => {
                let mut vector = DVector::zeros(1);
                vector[0] = self.value[0];
                OptimaTensor::Vector(OptimaTensor1D::new(vector))
            }
            OptimaTensorSignature::Matrix => {
                let mut matrix = DMatrix::zeros(1, 1);
                matrix[(0, 0)] = self.value[0];
                OptimaTensor::Matrix(OptimaTensor2D::new(matrix))
            }
            OptimaTensorSignature::TensorND => {
                let mut tensor = OptimaTensorND::new_zeros(vec![1]);
                tensor.vectorized_data_mut()[0] = self.value[0];
                OptimaTensor::TensorND(tensor)
            }
        }
    }
    pub fn dimensions(&self) -> Vec<usize> { vec![] }
    pub fn set_value(&mut self, indices: Vec<usize>, value: f64) {
        if indices.len() > 0 {
            panic!("incorrect indices.");
        }
        self.value = [value];
    }
    pub fn get_value(&self, indices: Vec<usize>) -> f64 {
        if indices.len() > 0 {
            panic!("incorrect indices.");
        }
        return self.value[0];
    }
    fn vectorized_data(&self) -> &[f64] { &self.value }
    fn vectorized_data_mut(&mut self) -> &mut [f64] { &mut self.value }
    fn vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        if vectorized_idx > 0 {
            panic!("invalid vectorized idx.");
        }
        return vec![];
    }
    fn indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        if indices.len() > 0 {
            panic!("invalid indices.");
        }
        return 0;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaTensor1D {
    vector: DVector<f64>
}
impl OptimaTensor1D {
    pub fn new_zeros(len: usize) -> Self {
        Self::new(DVector::zeros(len))
    }
    pub fn new(vector: DVector<f64>) -> Self {
        Self {
            vector
        }
    }
    pub fn convert(&self, target: &OptimaTensorSignature) -> OptimaTensor {
        return match target {
            OptimaTensorSignature::Scalar => {
                if self.vectorized_data().len() != 1 {
                    panic!("illegal conversion.");
                }
                OptimaTensor::Scalar(OptimaTensor0D::new(self.vectorized_data()[0]))
            }
            OptimaTensorSignature::Vector => {
                OptimaTensor::Vector(self.clone())
            }
            OptimaTensorSignature::Matrix => {
                let mut out = OptimaTensor::Matrix(OptimaTensor2D::new_zeros(self.vector.nrows(), self.vector.ncols()));
                copy_vectorized_data(self.vectorized_data(), out.vectorized_data_mut());
                out
            }
            OptimaTensorSignature::TensorND => {
                let mut out = OptimaTensor::TensorND(OptimaTensorND::new_zeros(self.dimensions()));
                copy_vectorized_data(self.vectorized_data(), out.vectorized_data_mut());
                out
            }
        }
    }
    pub fn dimensions(&self) -> Vec<usize> { vec![self.vector.len()] }
    pub fn set_value(&mut self, indices: Vec<usize>, value: f64) {
        if indices.len() != 1 {
            panic!("incorrect indices.");
        }
        self.vector[indices[0]] = value;
    }
    pub fn get_value(&self, indices: Vec<usize>) -> f64 {
        if indices.len() != 1 {
            panic!("incorrect indices.");
        }
        return self.vector[indices[0]];
    }
    fn vectorized_data(&self) -> &[f64] { self.vector.as_slice() }
    fn vectorized_data_mut(&mut self) -> &mut [f64] { self.vector.as_mut_slice() }
    fn vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        if vectorized_idx > self.vector.len() {
            panic!("invalid vectorized idx.");
        }
        return vec![vectorized_idx];
    }
    fn indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        if indices.len() != 1 {
            panic!("invalid indices.");
        }
        return indices[0];
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaTensor2D {
    matrix: DMatrix<f64>
}
impl OptimaTensor2D {
    pub fn new_zeros(nrows: usize, ncols: usize) -> Self {
        let mat = DMatrix::zeros(nrows, ncols);
        return Self::new(mat);
    }
    pub fn new(matrix: DMatrix<f64>) -> Self {
        Self {
            matrix
        }
    }
    pub fn convert(&self, target: &OptimaTensorSignature) -> OptimaTensor {
        return match target {
            OptimaTensorSignature::Scalar => {
                if self.vectorized_data().len() != 1 {
                    panic!("illegal conversion.");
                }
                OptimaTensor::Scalar(OptimaTensor0D::new(self.vectorized_data()[0]))
            }
            OptimaTensorSignature::Vector => {
                let dimensions = self.dimensions();
                if dimensions.len() > 1 {
                    panic!("illegal conversion.");
                }
                OptimaTensor::Vector(OptimaTensor1D::new(DVector::from_column_slice(self.vectorized_data())))
            }
            OptimaTensorSignature::Matrix => {
                OptimaTensor::Matrix(self.clone())
            }
            OptimaTensorSignature::TensorND => {
                let mut out = OptimaTensor::TensorND(OptimaTensorND::new_zeros(self.dimensions()));
                copy_vectorized_data(self.vectorized_data(), out.vectorized_data_mut());
                out
            }
        }
    }
    pub fn dimensions(&self) -> Vec<usize> { vec![ self.matrix.nrows(), self.matrix.ncols() ] }
    pub fn set_value(&mut self, indices: Vec<usize>, value: f64) {
        if indices.len() != 2 {
            panic!("incorrect indices.");
        }
        self.matrix[(indices[0], indices[1])] = value;
    }
    pub fn get_value(&self, indices: Vec<usize>) -> f64 {
        if indices.len() != 2 {
            panic!("incorrect indices.");
        }
        return self.matrix[(indices[0], indices[1])];
    }
    fn vectorized_data(&self) -> &[f64] { self.matrix.as_slice() }
    fn vectorized_data_mut(&mut self) -> &mut [f64] { self.matrix.as_mut_slice() }
    fn vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        if vectorized_idx >= self.vectorized_data().len() {
            panic!("invalid vectorized idx.");
        }

        let mut out_vec = vec![];
        let mut remainder = vectorized_idx;

        let strides_tuple = self.matrix.strides();
        let strides = vec![strides_tuple.0, strides_tuple.1];
        for dim in strides {
            if remainder == 0 {
                out_vec.push(0);
            } else if remainder >= dim as usize {
                let div = remainder / dim as usize;
                out_vec.push(div);
                remainder %= dim as usize;
            } else {
                out_vec.push(0);
            }
        }

        out_vec
    }
    fn indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        if indices.len() != 2 {
            panic!("invalid indices.");
        }

        let mut out = 0;

        let strides_tuple = self.matrix.strides();
        let strides = vec![strides_tuple.0, strides_tuple.1];

        for (i, idx) in indices.iter().enumerate() {
            out += strides[i] * *idx;
        }

        out
    }
}

#[derive(Clone, Debug)]
pub struct OptimaTensorND {
    ndim: usize,
    tensor: ArrayD<f64>
}
impl OptimaTensorND {
    pub fn new_zeros(dimensions: Vec<usize>) -> Self {
        if dimensions.len() > 6 { panic!("cannot support greater than 6 dimensions.") }
        Self {
            ndim: dimensions.len(),
            tensor: Array::<f64, _>::zeros(dimensions)
        }
    }
    pub fn convert(&self, target: &OptimaTensorSignature) -> OptimaTensor {
        return match target {
            OptimaTensorSignature::Scalar => {
                if self.vectorized_data().len() != 1 {
                    panic!("illegal conversion.");
                }
                OptimaTensor::Scalar(OptimaTensor0D::new(self.vectorized_data()[0]))
            }
            OptimaTensorSignature::Vector => {
                let dimensions = self.dimensions();
                if dimensions.len() > 1 {
                    panic!("illegal conversion.");
                }
                OptimaTensor::Vector(OptimaTensor1D::new(DVector::from_column_slice(self.vectorized_data())))
            }
            OptimaTensorSignature::Matrix => {
                let dimensions = self.dimensions();
                if dimensions.len() > 2 {
                    panic!("illegal conversion.");
                }
                let mut out = OptimaTensor2D::new_zeros(dimensions[0], dimensions[1]);
                copy_vectorized_data(self.vectorized_data(), out.vectorized_data_mut());
                OptimaTensor::Matrix(out)
            }
            OptimaTensorSignature::TensorND => {
                OptimaTensor::TensorND(self.clone())
            }
        }
    }
    pub fn dimensions(&self) -> Vec<usize> { self.tensor.shape().to_vec() }
    pub fn set_value(&mut self, indices: Vec<usize>, value: f64) {
        let dimensions = self.dimensions();
        let l = dimensions.len();
        if l != indices.len() {
            panic!("incorrect indices.")
        }

        let mut_val = if l == 1 {
            self.cell_mut_ref1(indices)
        } else if l == 2 {
            self.cell_mut_ref2(indices)
        } else if l == 3 {
            self.cell_mut_ref3(indices)
        } else if l == 4 {
            self.cell_mut_ref4(indices)
        } else if l == 5 {
            self.cell_mut_ref5(indices)
        } else if l == 6 {
            self.cell_mut_ref6(indices)
        } else {
            unreachable!()
        };

        *mut_val = value;
    }
    pub fn get_value(&self, indices: Vec<usize>) -> f64 {
        let dimensions = self.dimensions();
        let l = dimensions.len();
        if l != indices.len() {
            panic!("incorrect indices.")
        }

        let val = if l == 1 {
            self.cell_ref1(indices)
        } else if l == 2 {
            self.cell_ref2(indices)
        } else if l == 3 {
            self.cell_ref3(indices)
        } else if l == 4 {
            self.cell_ref4(indices)
        } else if l == 5 {
            self.cell_ref5(indices)
        } else if l == 6 {
            self.cell_ref6(indices)
        } else {
            unreachable!()
        };

        return *val;
    }

    fn vectorized_data(&self) -> &[f64] { self.tensor.as_slice().unwrap() }
    fn vectorized_data_mut(&mut self) -> &mut [f64] { self.tensor.as_slice_mut().unwrap() }
    fn vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        if vectorized_idx >= self.vectorized_data().len() {
            panic!("invalid vectorized idx.");
        }

        let mut out_vec = vec![];
        let mut remainder = vectorized_idx;

        let strides = self.tensor.strides();
        for dim in strides {
            if remainder == 0 {
                out_vec.push(0);
            } else if remainder >= *dim as usize {
                let div = remainder / *dim as usize;
                out_vec.push(div);
                remainder %= *dim as usize;
            } else {
                out_vec.push(0);
            }
        }

        out_vec
    }
    fn indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        let mut out = 0;

        let strides = self.tensor.strides();

        for (i, idx) in indices.iter().enumerate() {
            out += strides[i] as usize * *idx;
        }

        out
    }

    fn cell_mut_ref1(&mut self, indices: Vec<usize>) -> &mut f64 {
        return &mut self.tensor[[indices[0]]]
    }
    fn cell_mut_ref2(&mut self, indices: Vec<usize>) -> &mut f64 {
        return &mut self.tensor[[indices[0], indices[1]]]
    }
    fn cell_mut_ref3(&mut self, indices: Vec<usize>) -> &mut f64 {
        return &mut self.tensor[[indices[0], indices[1], indices[2]]]
    }
    fn cell_mut_ref4(&mut self, indices: Vec<usize>) -> &mut f64 {
        return &mut self.tensor[[indices[0], indices[1], indices[2], indices[3]]]
    }
    fn cell_mut_ref5(&mut self, indices: Vec<usize>) -> &mut f64 {
        return &mut self.tensor[[indices[0], indices[1], indices[2], indices[3], indices[4]]]
    }
    fn cell_mut_ref6(&mut self, indices: Vec<usize>) -> &mut f64 {
        return &mut self.tensor[[indices[0], indices[1], indices[2], indices[3], indices[4], indices[5]]]
    }

    fn cell_ref1(&self, indices: Vec<usize>) -> &f64 {
        return &self.tensor[[indices[0]]]
    }
    fn cell_ref2(&self, indices: Vec<usize>) -> &f64 {
        return &self.tensor[[indices[0], indices[1]]]
    }
    fn cell_ref3(&self, indices: Vec<usize>) -> &f64 {
        return &self.tensor[[indices[0], indices[1], indices[2]]]
    }
    fn cell_ref4(&self, indices: Vec<usize>) -> &f64 {
        return &self.tensor[[indices[0], indices[1], indices[2], indices[3]]]
    }
    fn cell_ref5(&self, indices: Vec<usize>) -> &f64 {
        return &self.tensor[[indices[0], indices[1], indices[2], indices[3], indices[4]]]
    }
    fn cell_ref6(&self, indices: Vec<usize>) -> &f64 {
        return &self.tensor[[indices[0], indices[1], indices[2], indices[3], indices[4], indices[5]]]
    }
}

fn copy_vectorized_data(from: &[f64], to: &mut [f64]) {
    let l1 = from.len();
    let l2 = to.len();
    if l1 != l2 {
        panic!("lengths must be the same");
    }

    for i in 0..l1 {
        to[i] = from[i];
    }
}

pub enum OptimaTensorSliceScope {
    Free,
    Fixed(usize)
}
