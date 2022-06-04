use std::fmt::Debug;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array2, ArrayD, Axis};
use ndarray_einsum_beta::tensordot;
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_generic_data_structures::{EnumBinarySearchSignatureContainer, EnumHashMapSignatureContainer, EnumMapToSignature, EnumSignatureContainer, EnumSignatureContainerType};

#[allow(unused_variables)]
pub trait OptimaTensorFunction {
    fn call(&self, input: &OptimaTensor, vars: &OTFVars) -> Result<OptimaTensor, OptimaError> {
        let mut precomputation_vars = OTFPrecomputationVars::new(EnumSignatureContainerType::default());
        return self.call_raw(input, vars, &mut precomputation_vars);
    }
    fn call_raw(&self, input: &OptimaTensor, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OptimaTensor, OptimaError>;
    fn derivative(&self, input: &OptimaTensor, vars: &OTFVars, mode: Option<OTFDerivativeMode>) -> Result<OptimaTensor, OptimaError> {
        let mut precomputation_vars = OTFPrecomputationVars::new(EnumSignatureContainerType::default());

        match mode {
            None => {
                {
                    let res = self.derivative_analytical(input, vars, &mut precomputation_vars)?;
                    match res {
                        OTFDerivativeResult::Unimplemented => { }
                        OTFDerivativeResult::Complete(t) => { return Ok(t); }
                    }
                }

                {
                    let res = self.derivative_finite_difference(input, vars, &mut precomputation_vars)?;
                    match res {
                        OTFDerivativeResult::Unimplemented => {}
                        OTFDerivativeResult::Complete(t) => { return Ok(t); }
                    }
                }


                panic!("Called an Unimplemented Derivative on OTF.")

            }
            Some(mode) => {
                let res = match mode {
                    OTFDerivativeMode::Analytical => { self.derivative_analytical(input, vars, &mut precomputation_vars) }
                    OTFDerivativeMode::FiniteDifference => { self.derivative_finite_difference(input,  vars, &mut precomputation_vars) }
                    OTFDerivativeMode::Test1 => { self.derivative_test1(input, vars, &mut precomputation_vars)  }
                    OTFDerivativeMode::Test2 => { self.derivative_test2(input, vars, &mut precomputation_vars) }
                    OTFDerivativeMode::Test3 => { self.derivative_test3(input, vars, &mut precomputation_vars) }
                }?;
                match res {
                    OTFDerivativeResult::Unimplemented => {
                        panic!("Called an Unimplemented Derivative on OTF.")
                    }
                    OTFDerivativeResult::Complete(t) => { return Ok(t); }
                }
            }
        }
    }
    fn derivative_analytical(&self, input: &OptimaTensor, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn derivative_finite_difference(&self, input: &OptimaTensor, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        /*
        let num_input_elements = input.total_number_of_elements();

        let x_0 = self.call(input, vars)?;
        let x_0_values = x_0.vectorized_data();
        let perturbation = 0.000001;

        for i in 0..num_input_elements {
            let mut input_copy = input.clone();
            input_copy.vectorized_data_mut()[i] += perturbation;
            let mut x_h = self.call(&input_copy, vars)?;

            for (j, x_h_value) in x_h.vectorized_data_mut().iter_mut().enumerate() {
                *x_h_value = (-x_0_values[j] + *x_h_value) / perturbation;
            }

            output.vectorized_data_mut()[i] = x_h;
        }
        */

        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn derivative_test1(&self, input: &OptimaTensor, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn derivative_test2(&self, input: &OptimaTensor, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn derivative_test3(&self, input: &OptimaTensor, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
}

#[derive(Clone, Debug)]
pub enum OTFDerivativeResult {
    Unimplemented, Complete(OptimaTensor)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OTFDerivativeMode {
    Analytical, FiniteDifference, Test1, Test2, Test3
}

/// Creates an optima tensor that is (inner_dimensions) x (outer_dimensions).
/// Can be thought of as an outer_dimensions matrix of inner_dimensions matrices. 
#[derive(Clone, Debug)]
pub struct OptimaTensor {
    ndim: usize,
    tensor_nd: Option<ArrayD<f64>>,
    tensor_2d: Option<Array2<f64>>,
    inner_dimensions: Vec<usize>,
    outer_dimensions: Vec<usize>,
    combined_dimensions: Vec<usize>,
    inner_dimensions_axes: Vec<Axis>,
    outer_dimensions_axes: Vec<Axis>,
    inner_dimensions_strides: Vec<isize>,
    outer_dimensions_strides: Vec<isize>,
    combined_strides: Vec<isize>
}
impl OptimaTensor {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let l = dimensions.len();

        if l < 2 {
            panic!("dimension length cannot be less than 2");
        }

        let mut inner_dimensions = vec![];
        let mut outer_dimensions = vec![];

        for i in 0..l-1 {
            inner_dimensions.push(dimensions[i]);
        }

        outer_dimensions.push(dimensions[l-1]);

        return Self::new_from_inner_and_outer_dimensions(inner_dimensions, outer_dimensions);
    }
    pub fn new_from_inner_and_outer_dimensions(inner_dimensions: Vec<usize>, outer_dimensions: Vec<usize>) -> Self {
        if inner_dimensions.is_empty() || outer_dimensions.is_empty() {
            panic!("innerdimensions and outerdimensions must be non-empty.");
        }

        let mut combined_dimensions = vec![];

        let mut inner_dimensions_axes = vec![];
        let mut outer_dimensions_axes = vec![];

        let inner_dimensions_len = inner_dimensions.len();
        let outer_dimensions_len = outer_dimensions.len();

        for i in 0..inner_dimensions_len {
            inner_dimensions_axes.push(Axis(i));
            combined_dimensions.push(inner_dimensions[i]);
        }

        for i in 0..outer_dimensions_len {
            outer_dimensions_axes.push(Axis(i + inner_dimensions_len));
            combined_dimensions.push(outer_dimensions[i]);
        }

        let mut combined_strides = vec![];
        let mut inner_dimensions_strides = vec![];
        let mut outer_dimensions_strides = vec![];

        let tensor_2d = if combined_dimensions.len() == 2 {
            let tensor = Array2::zeros((inner_dimensions[0], outer_dimensions[0]));
            combined_strides = tensor.strides().to_vec();
            Some(tensor)
        } else {
            None
        };
        let tensor_nd = if combined_dimensions.len() == 2 {
            None
        } else {
            let tensor = Array::<f64, _>::zeros(combined_dimensions.clone());
            combined_strides = tensor.strides().to_vec();
            Some(tensor)
        };

        for i in 0..inner_dimensions_len {
            inner_dimensions_strides.push(combined_strides[i]);
        }

        for i in 0..outer_dimensions_len {
            outer_dimensions_strides.push(combined_strides[i + inner_dimensions_len]);
        }

        Self {
            ndim: combined_dimensions.len(),
            tensor_nd,
            tensor_2d,
            inner_dimensions,
            outer_dimensions,
            combined_dimensions,
            inner_dimensions_axes,
            outer_dimensions_axes,
            inner_dimensions_strides,
            outer_dimensions_strides,
            combined_strides
        }
    }
    pub fn new_from_vector(v: Vec<f64>, t: VectorType) -> Self {
        let mut out = match &t {
            VectorType::Row => { Self::new_from_inner_and_outer_dimensions(vec![1], vec![v.len()]) }
            VectorType::Column => { Self::new_from_inner_and_outer_dimensions(vec![v.len()], vec![1]) }
        };

        let tensor = out.tensor_2d_mut();
        for (i, val) in v.iter().enumerate() {
            match &t {
                VectorType::Row => { tensor[(0,i)] = *val; }
                VectorType::Column => { tensor[(i,0)] = *val; }
            }
        }

        out
    }
    pub fn new_from_2d_vector(v: Vec<Vec<f64>>) -> Self {
        let mut out = Self::new_from_inner_and_outer_dimensions(vec![v.len()], vec![v[0].len()]);

        let tensor = out.tensor_2d_mut();
        for (i, vec) in v.iter().enumerate() {
            for (j, val) in vec.iter().enumerate() {
                tensor[(i,j)] = *val;
            }
        }

        out
    }
    pub fn tensor_nd_mut(&mut self) -> &mut ArrayD<f64> {
        return match &mut self.tensor_nd {
            None => { panic!("tensor_nd is None.  Try tensor_2d instead.") }
            Some(t) => { t }
        }
    }
    pub fn tensor_nd(&self) -> &ArrayD<f64> {
        return match &self.tensor_nd {
            None => { panic!("tensor_nd is None.  Try tensor_2d instead.") }
            Some(t) => { t }
        }
    }
    pub fn tensor_2d_mut(&mut self) -> &mut Array2<f64> {
        return match &mut self.tensor_2d {
            None => { panic!("tensor_2d is None.  Try tensor_nd instead.") }
            Some(t) => { t }
        }
    }
    pub fn tensor_2d(&self) -> &Array2<f64> {
        return match &self.tensor_2d {
            None => { panic!("tensor_2d is None.  Try tensor_nd instead.") }
            Some(t) => { t }
        }
    }
    pub fn convert_to_dmatrix(&self) -> DMatrix<f64> {
        if self.ndim > 2 {
            panic!("Cannot convert an optima tensor of dimension > 2 to a DMatrix.");
        }

        let tensor = self.tensor_2d();
        let shape = tensor.shape();

        let mut out = DMatrix::zeros(shape[0], shape[1]);

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                out[(i,j)] = tensor[(i,j)];
            }
        }

        out
    }
    pub fn convert_to_dvector(&self) -> DVector<f64> {
        if self.ndim > 2 {
            panic!("Cannot convert an optima tensor of dimension > 2 to a DMatrix.");
        }

        let tensor = self.tensor_2d();
        let shape = tensor.shape();

        if shape[0] == 1 {
            let mut out = DVector::zeros(shape[1]);

            for i in 0..shape[1] {
                out[i] = tensor[(0, i)];
            }

            return out;
        } else if shape[1] == 1 {
            let mut out = DVector::zeros(shape[0]);

            for i in 0..shape[0] {
                out[i] = tensor[(i, 0)];
            }

            return out;
        } else {
            panic!("Could not convert optima tensor to a dvector.");
        }
    }
    pub fn scalar_multiplication(&mut self, val: f64) -> OptimaTensor {
        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            *tensor = val * tensor_self;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            *tensor = val * tensor_self;
        }

        out
    }
    pub fn scalar_multiplication_inplace(&mut self, val: f64) {
        if let Some(tensor) = &mut self.tensor_nd {
            *tensor *= val;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            *tensor *= val;
        }
    }
    pub fn elementwise_multiplication(&mut self, other: &OptimaTensor) -> OptimaTensor {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor = tensor_self * tensor_other;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor = tensor_self * tensor_other;
        }

        out
    }
    pub fn elementwise_multiplication_inplace(&mut self, other: &OptimaTensor) {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        if let Some(tensor) = &mut self.tensor_nd {
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor *= tensor_other;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor *= tensor_other;
        }
    }
    pub fn dot(&self, other: &OptimaTensor) -> OptimaTensor {
        if self.outer_dimensions() != other.inner_dimensions() {
            let error_str = format!("OptimaTensor dimensions are incompatible for dot (self.outer_dimensions: {:?} x other.inner_dimensions: {:?})", self.outer_dimensions(), other.inner_dimensions());
            panic!("{}", error_str);
        }

        let mut out = OptimaTensor::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), other.outer_dimensions.clone());

        if let Some(self_tensor) = &self.tensor_2d {
            if let Some(other_tensor) = &other.tensor_2d {
                let res = self_tensor.dot(other_tensor);
                out.tensor_2d = Some(res);
            }
            else if let Some(other_tensor) = &other.tensor_nd {
                let res = tensordot(self_tensor, other_tensor, &self.outer_dimensions_axes, &other.inner_dimensions_axes);
                out.tensor_nd = Some(res);
            } else {
                unreachable!();
            }
        } else if let Some(self_tensor) = &self.tensor_nd {
            if let Some(other_tensor) = &other.tensor_2d {
                let res = tensordot(self_tensor, other_tensor, &self.outer_dimensions_axes, &other.inner_dimensions_axes);
                out.tensor_nd = Some(res);            }
            else if let Some(other_tensor) = &other.tensor_nd {
                let res = tensordot(self_tensor, other_tensor, &self.outer_dimensions_axes, &other.inner_dimensions_axes);
                out.tensor_nd = Some(res);
            } else {
                unreachable!();
            }
        }

        return out;
    }
    pub fn dot_inplace(&mut self, other: &OptimaTensor) {
        if self.outer_dimensions() != other.inner_dimensions() {
            let error_str = format!("OptimaTensor dimensions are incompatible for dot (self.outer_dimensions: {:?} x other.inner_dimensions: {:?})", self.outer_dimensions(), other.inner_dimensions());
            panic!("{}", error_str);
        }

        if let Some(self_tensor) = &self.tensor_2d {
            if let Some(other_tensor) = &other.tensor_2d {
                let res = self_tensor.dot(other_tensor);
                self.tensor_2d = Some(res);
            }
            else if let Some(other_tensor) = &other.tensor_nd {
                let res = tensordot(self_tensor, other_tensor, &self.outer_dimensions_axes, &other.inner_dimensions_axes);
                self.tensor_nd = Some(res);
            } else {
                unreachable!();
            }
        } else if let Some(self_tensor) = &self.tensor_nd {
            if let Some(other_tensor) = &other.tensor_2d {
                let res = tensordot(self_tensor, other_tensor, &self.outer_dimensions_axes, &other.inner_dimensions_axes);
                self.tensor_nd = Some(res);
            }
            else if let Some(other_tensor) = &other.tensor_nd {
                let res = tensordot(self_tensor, other_tensor, &self.outer_dimensions_axes, &other.inner_dimensions_axes);
                self.tensor_nd = Some(res);
            } else {
                unreachable!();
            }
        }

        self.outer_dimensions = other.outer_dimensions.clone();

        let inner_dimensions_len = self.inner_dimensions.len();
        let outer_dimensions_len = self.outer_dimensions.len();

        self.inner_dimensions_axes = vec![];
        self.outer_dimensions_axes = vec![];

        for i in 0..inner_dimensions_len {
            self.inner_dimensions_axes.push(Axis(i));
        }

        for i in 0..outer_dimensions_len {
            self.outer_dimensions_axes.push(Axis(i + inner_dimensions_len));
        }

    }
    pub fn inner_dimensions(&self) -> &Vec<usize> {
        &self.inner_dimensions
    }
    pub fn outer_dimensions(&self) -> &Vec<usize> {
        &self.outer_dimensions
    }
    pub fn ndim(&self) -> usize {
        self.ndim
    }
    pub fn combined_dimensions(&self) -> &Vec<usize> {
        &self.combined_dimensions
    }
    pub fn print_summary(&self) {
        if let Some(tensor) = &self.tensor_nd {
            println!("{}", tensor);
        }

        if let Some(tensor) = &self.tensor_2d {
            println!("{}", tensor);
        }
    }

    pub fn vectorized_idx_to_dimensions(&self, vectorized_idx: usize) -> Vec<usize> {
        let mut out_vec = vec![];
        let mut remainder = vectorized_idx;

        println!("{:?}", self.inner_dimensions_strides);
        for dim in &self.inner_dimensions_strides {
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
}

#[derive(Clone, Debug)]
pub enum VectorType {
    Row, Column
}

pub struct OTFVars {
    pub c: Box<dyn EnumSignatureContainer<OTFVarsObject, OTFVarsObjectSignature>>
}
impl OTFVars {
    pub fn new(enum_signature_container_type: EnumSignatureContainerType) -> Self {
        Self {
            c: match enum_signature_container_type {
                EnumSignatureContainerType::BinarySearch => { Box::new(EnumBinarySearchSignatureContainer::new()) }
                EnumSignatureContainerType::HashMap => { Box::new(EnumHashMapSignatureContainer::new()) }
            }
        }
    }
}

pub enum OTFVarsObject {
    Test
}
impl EnumMapToSignature<OTFVarsObjectSignature> for OTFVarsObject {
    fn map_to_signature(&self) -> OTFVarsObjectSignature {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OTFVarsObjectSignature {
    Test
}

pub struct OTFPrecomputationVars {
    c: Box<dyn EnumSignatureContainer<OTFPrecomputationVarsObject, OTFPrecomputationVarsObjectSignature>>
}
impl OTFPrecomputationVars {
    pub fn new(enum_signature_container_type: EnumSignatureContainerType) -> Self {
        Self {
            c: match enum_signature_container_type {
                EnumSignatureContainerType::BinarySearch => { Box::new(EnumBinarySearchSignatureContainer::new()) }
                EnumSignatureContainerType::HashMap => { Box::new(EnumHashMapSignatureContainer::new()) }
            }
        }
    }
    pub fn object_mut_ref(&mut self, vars: &OTFVars, signature: &OTFPrecomputationVarsObjectSignature) -> &mut OTFPrecomputationVarsObject {
        signature.precomputation(vars, self);
        let o = self.c.object_mut_ref(signature);
        return o.expect("Should be impossible that this is None after precomputation.  If it is, fix the precomputation code.")
    }
    pub fn object_ref(&mut self, vars: &OTFVars, signature: &OTFPrecomputationVarsObjectSignature) -> &OTFPrecomputationVarsObject {
        let o = self.object_mut_ref(vars, signature);
        return o;
    }
}

pub enum OTFPrecomputationVarsObject {
    Test
}
impl EnumMapToSignature<OTFPrecomputationVarsObjectSignature> for OTFPrecomputationVarsObject {
    fn map_to_signature(&self) -> OTFPrecomputationVarsObjectSignature {
        match self {
            OTFPrecomputationVarsObject::Test => { OTFPrecomputationVarsObjectSignature::Test }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OTFPrecomputationVarsObjectSignature {
    Test
}
impl OTFPrecomputationVarsObjectSignature {
    pub fn precomputation(&self, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) {
        if precomputation_vars.c.contains_object(self) { return; }
        let o = self.precompute_raw(vars, precomputation_vars);
        let signature = o.map_to_signature();
        if &signature != self {
            panic!("OTF Precomputation did not return expected type (Expected {:?} and got {:?}.)", self, o.map_to_signature());
        }
        precomputation_vars.c.insert_or_replace_object(o);
    }
    #[allow(unused_variables)]
    fn precompute_raw(&self, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> OTFPrecomputationVarsObject {
        todo!()
    }
}

//////////////// GARBAGE CODE

/*
pub enum TensorDimensionsInfo {
    Any,
    PerDimension(Vec<TensorSingleDimensionInfo>)
}
impl TensorDimensionsInfo {
    pub fn sample(&self) -> OptimaTensor1 {
        let dims = self.sample_dims();
        return OptimaTensor1::new_random(dims);
    }
    fn sample_dims(&self) -> Vec<usize> {
        return match self {
            TensorDimensionsInfo::Any => {
                let mut out_vec = vec![];

                let s = SimpleSamplers::uniform_samples_i32(&vec![(1,5)])[0] as usize;
                for _ in 0..s { out_vec.push(TensorSingleDimensionInfo::Any.sample_dims()) }

                out_vec
            }
            TensorDimensionsInfo::PerDimension(vec) => {
                let mut out_vec = vec![];
                for v in vec {
                    out_vec.push(v.sample_dims());
                }
                out_vec
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TensorSingleDimensionInfo {
    Any,
    Fixed(usize)
}
impl TensorSingleDimensionInfo {
    fn sample_dims(&self) -> usize {
        return match self {
            TensorSingleDimensionInfo::Any => {
                let res = SimpleSamplers::uniform_samples_i32(&vec![(0,10)]);
                res[0] as usize
            }
            TensorSingleDimensionInfo::Fixed(a) => { *a }
        }
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimaTensorGeneric<T> where T: Default + Debug + Clone + Serialize + DeserializeOwned {
    order: usize,
    total_number_of_elements: usize,
    dimensions: Vec<usize>,
    dimension_vectorization_helper: Vec<usize>,
    #[serde_as(as = "Vec<_>")]
    vectorized_data: Vec<T>
}
impl <T> OptimaTensorGeneric<T> where T: Default + Debug + Clone + Serialize + DeserializeOwned {
    pub fn new_default(dimensions: Vec<usize>) -> Self {
        let order = dimensions.len();

        let mut total_number_of_elements = 1;
        for d in &dimensions { total_number_of_elements *= *d; }
        let vectorized_data = vec![T::default(); total_number_of_elements];

        let mut dimension_vectorization_helper = vec![0; dimensions.len()];
        for (i, _) in dimensions.iter().enumerate() {
            let mut total_number_of_elements = 1;
            for (j, d) in dimensions.iter().enumerate() {
                if j > i {
                    total_number_of_elements *= *d;
                }
            }
            dimension_vectorization_helper[i] = total_number_of_elements;
        }

        Self {
            order,
            total_number_of_elements,
            dimensions,
            dimension_vectorization_helper,
            vectorized_data,
        }
    }
    pub fn get_element(&self, indices: Vec<usize>) -> Result<&T, OptimaError> {
        let idx = self.get_idx_of_vectorized_data(indices)?;
        return Ok(&self.vectorized_data[idx]);
    }
    pub fn set_element(&mut self, indices: Vec<usize>, value: T) -> Result<(), OptimaError> {
        let idx = self.get_idx_of_vectorized_data(indices)?;
        self.vectorized_data[idx] = value;
        Ok(())
    }
    pub fn set_all_elements(&mut self, value: T) {
        for e in &mut self.vectorized_data {
            *e = value.clone();
        }
    }
    pub fn vectorized_data(&self) -> &Vec<T> {
        &self.vectorized_data
    }
    pub fn get_idx_of_vectorized_data(&self, indices: Vec<usize>) -> Result<usize, OptimaError> {
        if indices.len() != self.order {
            return Err(OptimaError::new_generic_error_str(&format!("Given indices are wrong for given tensor vector.  Given indices: {:?} | Dimensions: {:?}", indices, self.dimensions), file!(), line!()));
        }

        for (i, idx) in indices.iter().enumerate() {
            if *idx >= self.dimensions[i] {
            return Err(OptimaError::new_generic_error_str(&format!("Given indices are wrong for given tensor vector.  Given indices: {:?} | Dimensions: {:?}", indices, self.dimensions), file!(), line!()));
            }
        }

        let mut out_idx = 0;
        for (i, idx) in indices.iter().enumerate() {
            out_idx += *idx * self.dimension_vectorization_helper[i];
        }

        if out_idx >= self.total_number_of_elements {
            return Err(OptimaError::new_generic_error_str(&format!("Given indices are wrong for given tensor vector.  Given indices: {:?} | Dimensions: {:?}", indices, self.dimensions), file!(), line!()));
        }

        return Ok(out_idx);
    }
    pub fn get_indices_from_vectorized_data_idx(&self, idx: usize) -> Result<Vec<usize>, OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(idx, self.total_number_of_elements, file!(), line!())?;

        let mut out_vec = vec![];
        let mut remainder = idx;

        for dim in &self.dimension_vectorization_helper {
            if remainder == 0 {
                out_vec.push(0);
            } else if remainder >= *dim {
                let div = remainder / *dim;
                out_vec.push(div);
                remainder %= *dim;
            } else {
                out_vec.push(0);
            }
        }

        Ok(out_vec)
    }
    pub fn order(&self) -> usize {
        self.order
    }
    pub fn total_number_of_elements(&self) -> usize {
        self.total_number_of_elements
    }
    pub fn dimensions(&self) -> &Vec<usize> {
        &self.dimensions
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaTensor1 {
    dimensions: Vec<usize>,
    o: OptimaTensorGeneric<f64>
}
impl OptimaTensor1 {
    pub fn new_default(dimensions: Vec<usize>) -> Self {
        let o = OptimaTensorGeneric::<f64>::new_default(dimensions.clone());
        Self {
            dimensions,
            o
        }
    }
    pub fn new_random(dimensions: Vec<usize>) -> Self {
        let mut out_self = Self::new_default(dimensions);
        let total_num_elements = out_self.total_number_of_elements();
        let s = SimpleSamplers::uniform_samples(&vec![(-1.0, 1.0); total_num_elements]);
        out_self.o.vectorized_data = s;
        out_self
    }
    pub fn new_from_vec(vec: Vec<f64>) -> Self {
        let dimensions = vec![vec.len()];
        let mut out = Self::new_default(dimensions);
        out.o.vectorized_data = vec;
        out
    }
    pub fn new_from_2d_vec(vec: Vec<Vec<f64>>) -> Result<Self, OptimaError> {
        let dimensions = vec![vec.len(), vec[0].len()];
        let mut out = Self::new_default(dimensions);
        for (i, v1) in vec.iter().enumerate() {
            for (j, v2) in v1.iter().enumerate() {
                out.o.set_element(vec![i, j], *v2)?;
            }
        }
        Ok(out)
    }
    pub fn get_element(&self, indices: Vec<usize>) -> Result<&f64, OptimaError> {
        return self.o.get_element(indices)
    }
    pub fn set_element(&mut self, indices: Vec<usize>, value: f64) -> Result<(), OptimaError> {
        self.o.set_element(indices, value)
    }
    pub fn set_all_elements(&mut self, value: f64) {
        self.o.set_all_elements(value);
    }
    pub fn order(&self) -> usize {
        self.o.order
    }
    pub fn total_number_of_elements(&self) -> usize {
        self.o.total_number_of_elements
    }
    pub fn dimensions(&self) -> &Vec<usize> {
        &self.o.dimensions
    }
    pub fn vectorized_data(&self) -> &Vec<f64> {
        &self.o.vectorized_data
    }
    pub fn vectorized_data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.o.vectorized_data
    }
}
impl Default for OptimaTensor1 {
    fn default() -> Self {
        OptimaTensor1::new_default(vec![1])
    }
}

/// inner_dimensions x outer_dimensions ``matrix''
/// An outer dimensions generic tensor of inner dimension tensor vectors
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaTensor2 {
    inner_dimensions: Vec<usize>,
    outer_dimensions: Vec<usize>,
    o: OptimaTensorGeneric<OptimaTensor1>
}
impl OptimaTensor2 {
    pub fn new_default(inner_dimensions: Vec<usize>, outer_dimensions: Vec<usize>) -> Self {
        let mut o = OptimaTensorGeneric::new_default(outer_dimensions.clone());

        for tv in &mut o.vectorized_data { *tv = OptimaTensor1::new_default(inner_dimensions.clone()) }

        Self {
            inner_dimensions,
            outer_dimensions,
            o
        }
    }
    pub fn new_random(inner_dimensions: Vec<usize>, outer_dimensions: Vec<usize>) -> Self {
        let mut out_self = Self::new_default(inner_dimensions.clone(), outer_dimensions);
        for m in &mut out_self.o.vectorized_data {
            *m = OptimaTensor1::new_random(inner_dimensions.clone());
        }
        out_self
    }
    pub fn multiply(&self, other: &OptimaTensor2) -> Result<OptimaTensor2, OptimaError> {
        if self.outer_dimensions != other.inner_dimensions {
            return Err(OptimaError::new_generic_error_str(&format!("OptimaTensorMatrix multiplication error.  Dimensions do not align.  ({:?} and {:?})", self.outer_dimensions, other.inner_dimensions), file!(), line!()));
        }

        let mut out = OptimaTensor2::new_default(self.inner_dimensions.clone(), other.outer_dimensions.clone());

        for other_vectorized_data_idx in 0..other.o.total_number_of_elements {
            let coeff_vectorized_data = &other.o.vectorized_data[other_vectorized_data_idx].o.vectorized_data;
            let mut sum_tensor = OptimaTensor1::new_default(self.inner_dimensions.clone());
            for self_vectorized_data_idx in 0..self.o.total_number_of_elements {
                let curr_tensor_vectorized_data = &self.o.vectorized_data[self_vectorized_data_idx].o.vectorized_data;
                for (i, sum) in sum_tensor.o.vectorized_data.iter_mut().enumerate() {
                    *sum += coeff_vectorized_data[self_vectorized_data_idx] * curr_tensor_vectorized_data[i];
                }
            }
            out.o.vectorized_data[other_vectorized_data_idx] = sum_tensor;
        }

        return Ok(out);
    }
    pub fn inner_dimensions(&self) -> &Vec<usize> {
        &self.inner_dimensions
    }
    pub fn outer_dimensions(&self) -> &Vec<usize> {
        &self.outer_dimensions
    }
    pub fn get_element(&self, indices: Vec<usize>) -> Result<&OptimaTensor1, OptimaError> {
        return self.o.get_element(indices);
    }
    pub fn set_element(&mut self, indices: Vec<usize>, value: OptimaTensor1) -> Result<(), OptimaError> {
        if value.dimensions != self.inner_dimensions {
            return Err(OptimaError::new_generic_error_str(&format!("Could not set element as the given optima tensor vector is the wrong size."), file!(), line!()));
        }

        self.o.set_element(indices, value)?;

        Ok(())
    }
    pub fn set_all_elements(&mut self, value: OptimaTensor1) -> Result<(), OptimaError> {
        if value.dimensions != self.inner_dimensions {
            return Err(OptimaError::new_generic_error_str(&format!("Could not set element as the given optima tensor vector is the wrong size."), file!(), line!()));
        }

        self.o.set_all_elements(value);

        Ok(())
    }
    pub fn vectorized_data(&self) -> &Vec<OptimaTensor1> {
        &self.o.vectorized_data
    }
    pub fn vectorized_data_mut(&mut self) -> &mut Vec<OptimaTensor1> {
        &mut self.o.vectorized_data
    }
}
*/
/*
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum OTFPrecomputationBlock {

}
impl OTFPrecomputationBlock {
    pub fn precomputation(&self, input: &OptimaTensorVector, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) {

    }
    fn prerequisite_precomputation_blocks(&self) -> Vec<OTFPrecomputationBlock> {
        todo!()
    }
    fn prerequisite_precomputation(&self) -> Result<(), OptimaError> {
        let blocks = self.prerequisite_precomputation_blocks();
        for block in blocks {

        }
        Ok(())
    }
}
*/
/*
pub struct OTFVars {
    pub c: EnumObjectContainer<OTFVarsObject, OTFVarsSignature>
}
impl OTFVars {
    pub fn new() -> Self {
        Self {
            c: EnumObjectContainer::new().expect("error")
        }
    }
}

#[derive(EnumCount, Clone, Debug)]
pub enum OTFVarsObject {

}
impl EnumIndexWrapper for OTFVarsObject {
    fn enum_index_wrapper(&self) -> usize {
        todo!()
    }
}

#[derive(EnumCount, Clone, Debug)]
pub enum OTFVarsSignature {

}
impl EnumIndexWrapper for OTFVarsSignature {
    fn enum_index_wrapper(&self) -> usize {
        todo!()
    }
}

pub struct OTFPrecomputationVars {
    pub c: EnumObjectContainer<OTFPrecomputationVarsObject, OTFPrecomputationVarsSignature>
}
impl OTFPrecomputationVars {
    pub fn new() -> Self {
        Self {
            c: EnumObjectContainer::new().expect("error")
        }
    }
}


#[derive(EnumCount, Clone, Debug)]
pub enum OTFPrecomputationVarsObject {

}
impl EnumIndexWrapper for OTFPrecomputationVarsObject {
    fn enum_index_wrapper(&self) -> usize {
        todo!()
    }
}

#[derive(EnumCount, Clone, Debug)]
pub enum OTFPrecomputationVarsSignature {

}
impl EnumIndexWrapper for OTFPrecomputationVarsSignature {
    fn enum_index_wrapper(&self) -> usize {
        todo!()
    }
}
*/

