use std::fmt::Debug;
use serde_with::{serde_as};
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_generic_data_structures::{EnumBinarySearchSignatureContainer, EnumHashMapSignatureContainer, EnumMapToSignature, EnumSignatureContainer, EnumSignatureContainerType};
use crate::utils::utils_sampling::SimpleSamplers;

#[allow(unused_variables)]
pub trait OptimaTensorFunction {
    fn call(&self, input: &OptimaTensorVector, vars: &OTFVars) -> Result<OptimaTensorVector, OptimaError> {
        let mut precomputation_vars = OTFPrecomputationVars::new(EnumSignatureContainerType::default());

        OptimaError::new_check_for_optima_tensor_function_input_error(input, &self.input_dimensions(vars, &mut precomputation_vars), file!(), line!())?;
        let mut output = OptimaTensorVector::new_default(self.output_dimensions(input, vars, &mut precomputation_vars));

        self.call_raw(input, &mut output, vars, &mut precomputation_vars);

        return Ok(output);
    }
    fn call_raw(&self, input: &OptimaTensorVector, output: &mut OptimaTensorVector, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars);
    fn derivative(&self, input: &OptimaTensorVector, vars: &OTFVars, mode: Option<OTFDerivativeMode>) -> Result<OptimaTensorMatrix, OptimaError> {
        let mut precomputation_vars = OTFPrecomputationVars::new(EnumSignatureContainerType::default());

        OptimaError::new_check_for_optima_tensor_function_input_error(input, &self.input_dimensions(vars, &mut precomputation_vars), file!(), line!())?;

        let input_dimensions = self.input_dimensions(vars, &mut precomputation_vars);
        let output_dimensions = self.output_dimensions(input, vars, &mut precomputation_vars);

        let mut output = OptimaTensorMatrix::new_default(output_dimensions, input_dimensions);

        match mode {
            None => {
                let mut computation_completed = false;

                if !computation_completed {
                    let res = self.derivative_analytical(input, &mut output, vars, &mut precomputation_vars)?;
                    match res {
                        OTFDerivativeResult::Unimplemented => {}
                        OTFDerivativeResult::Complete => { computation_completed = true; }
                    }
                }

                if !computation_completed {
                    let res = self.derivative_finite_difference(input, &mut output, vars, &mut precomputation_vars)?;
                    match res {
                        OTFDerivativeResult::Unimplemented => {}
                        OTFDerivativeResult::Complete => { computation_completed = true; }
                    }
                }

                if !computation_completed {
                    panic!("Called an Unimplemented Derivative on OTF.")
                }
            }
            Some(mode) => {
                let res = match mode {
                    OTFDerivativeMode::Analytical => { self.derivative_analytical(input, &mut output, vars, &mut precomputation_vars) }
                    OTFDerivativeMode::FiniteDifference => { self.derivative_finite_difference(input, &mut output, vars, &mut precomputation_vars) }
                    OTFDerivativeMode::Test1 => { self.derivative_test1(input, &mut output, vars, &mut precomputation_vars)  }
                    OTFDerivativeMode::Test2 => { self.derivative_test2(input, &mut output, vars, &mut precomputation_vars) }
                    OTFDerivativeMode::Test3 => { self.derivative_test3(input, &mut output, vars, &mut precomputation_vars) }
                }?;
                match res {
                    OTFDerivativeResult::Unimplemented => {
                        panic!("Called an Unimplemented Derivative on OTF.")
                    }
                    OTFDerivativeResult::Complete => { }
                }
            }
        }

        return Ok(output);
    }
    fn derivative_analytical(&self, input: &OptimaTensorVector, output: &mut OptimaTensorMatrix, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn derivative_finite_difference(&self, input: &OptimaTensorVector, output: &mut OptimaTensorMatrix, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
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

        Ok(OTFDerivativeResult::Complete)
    }
    fn derivative_test1(&self, input: &OptimaTensorVector, output: &mut OptimaTensorMatrix, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn derivative_test2(&self, input: &OptimaTensorVector, output: &mut OptimaTensorMatrix, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn derivative_test3(&self, input: &OptimaTensorVector, output: &mut OptimaTensorMatrix, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Result<OTFDerivativeResult, OptimaError> {
        Ok(OTFDerivativeResult::Unimplemented)
    }
    fn input_dimensions(&self, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Vec<usize>;
    fn output_dimensions(&self, input: &OptimaTensorVector, vars: &OTFVars, precomputation_vars: &mut OTFPrecomputationVars) -> Vec<usize>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OTFDerivativeResult {
    Unimplemented, Complete
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OTFDerivativeMode {
    Analytical, FiniteDifference, Test1, Test2, Test3
}

pub enum TensorDimensionsInfo {
    Any,
    PerDimension(Vec<TensorSingleDimensionInfo>)
}
impl TensorDimensionsInfo {
    pub fn sample(&self) -> OptimaTensorVector {
        let dims = self.sample_dims();
        return OptimaTensorVector::new_random(dims);
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
pub struct OptimaTensorVector {
    dimensions: Vec<usize>,
    o: OptimaTensorGeneric<f64>
}
impl OptimaTensorVector {
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
impl Default for OptimaTensorVector {
    fn default() -> Self {
        OptimaTensorVector::new_default(vec![1])
    }
}

/// inner_dimensions x outer_dimensions ``matrix''
/// An outer dimensions generic tensor of inner dimension tensor vectors
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaTensorMatrix {
    inner_dimensions: Vec<usize>,
    outer_dimensions: Vec<usize>,
    o: OptimaTensorGeneric<OptimaTensorVector>
}
impl OptimaTensorMatrix {
    pub fn new_default(inner_dimensions: Vec<usize>, outer_dimensions: Vec<usize>) -> Self {
        let mut o = OptimaTensorGeneric::new_default(outer_dimensions.clone());

        for tv in &mut o.vectorized_data { *tv = OptimaTensorVector::new_default(inner_dimensions.clone()) }

        Self {
            inner_dimensions,
            outer_dimensions,
            o
        }
    }
    pub fn new_random(inner_dimensions: Vec<usize>, outer_dimensions: Vec<usize>) -> Self {
        let mut out_self = Self::new_default(inner_dimensions.clone(), outer_dimensions);
        for m in &mut out_self.o.vectorized_data {
            *m = OptimaTensorVector::new_random(inner_dimensions.clone());
        }
        out_self
    }
    pub fn multiply(&self, other: &OptimaTensorMatrix) -> Result<OptimaTensorMatrix, OptimaError> {
        if self.outer_dimensions != other.inner_dimensions {
            return Err(OptimaError::new_generic_error_str(&format!("OptimaTensorMatrix multiplication error.  Dimensions do not align.  ({:?} and {:?})", self.outer_dimensions, other.inner_dimensions), file!(), line!()));
        }

        let mut out = OptimaTensorMatrix::new_default(self.inner_dimensions.clone(), other.outer_dimensions.clone());

        for other_vectorized_data_idx in 0..other.o.total_number_of_elements {
            let coeff_vectorized_data = &other.o.vectorized_data[other_vectorized_data_idx].o.vectorized_data;
            let mut sum_tensor = OptimaTensorVector::new_default(self.inner_dimensions.clone());
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
    pub fn get_element(&self, indices: Vec<usize>) -> Result<&OptimaTensorVector, OptimaError> {
        return self.o.get_element(indices);
    }
    pub fn set_element(&mut self, indices: Vec<usize>, value: OptimaTensorVector) -> Result<(), OptimaError> {
        if value.dimensions != self.inner_dimensions {
            return Err(OptimaError::new_generic_error_str(&format!("Could not set element as the given optima tensor vector is the wrong size."), file!(), line!()));
        }

        self.o.set_element(indices, value)?;

        Ok(())
    }
    pub fn set_all_elements(&mut self, value: OptimaTensorVector) -> Result<(), OptimaError> {
        if value.dimensions != self.inner_dimensions {
            return Err(OptimaError::new_generic_error_str(&format!("Could not set element as the given optima tensor vector is the wrong size."), file!(), line!()));
        }

        self.o.set_all_elements(value);

        Ok(())
    }
    pub fn vectorized_data(&self) -> &Vec<OptimaTensorVector> {
        &self.o.vectorized_data
    }
    pub fn vectorized_data_mut(&mut self) -> &mut Vec<OptimaTensorVector> {
        &mut self.o.vectorized_data
    }
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

