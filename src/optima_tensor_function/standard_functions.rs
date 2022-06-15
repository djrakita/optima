use crate::optima_tensor_function::{OptimaTensor, OptimaTensorFunction, OTFImmutVars, OTFMutVars, OTFMutVarsSessionKey, OTFResult};
use crate::utils::utils_errors::OptimaError;

#[derive(Clone)]
pub struct OTFComposition<F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction {
    f: F,
    g: G
}
impl<F, G> OTFComposition<F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction {
    pub fn new(outer_function: F, enveloped_function: G) -> Self {
        Self {
            f: outer_function,
            g: enveloped_function
        }
    }
}
impl<F, G> OptimaTensorFunction for OTFComposition<F, G>
    where F: OptimaTensorFunction + Clone + 'static,
          G: OptimaTensorFunction + Clone + 'static {
    fn output_dimensions(&self) -> Vec<usize> {
        self.f.output_dimensions()
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let g_res = self.g.call_raw(input, immut_vars, mut_vars, session_key)?;
        let g = g_res.unwrap_tensor();
        let f_res = self.f.call_raw(g, immut_vars, mut_vars, session_key)?;
        return Ok(f_res);
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let g_res = self.g.call_raw(input, immut_vars, mut_vars, session_key)?;
        let g = g_res.unwrap_tensor();
        let dg_dx_res = self.g.derivative(input, immut_vars, mut_vars, None)?;
        let dg_dx = dg_dx_res.unwrap_tensor();
        let df_dg_res = self.f.derivative(g, immut_vars, mut_vars, None)?;
        let df_dg = df_dg_res.unwrap_tensor();
        let df_dx = df_dg.dot(&dg_dx);
        return Ok(OTFResult::Complete(df_dx));
    }
}

#[derive(Clone)]
pub struct OTFWeightedSum {
    functions: Vec<Box<dyn OptimaTensorFunction>>,
    weights: Vec<f64>
}
impl OTFWeightedSum {
    pub fn new() -> Self {
        Self {
            functions: vec![],
            weights: vec![]
        }
    }
    pub fn add_function<F: OptimaTensorFunction + 'static>(&mut self, f: F, weight: Option<f64>) -> usize {
        if !self.functions.is_empty() {
            assert_eq!(f.output_dimensions(), self.output_dimensions())
        }

        let add_idx = self.functions.len();

        self.functions.push(Box::new(f));
        match weight {
            None => { self.weights.push(1.0); }
            Some(weight) => { self.weights.push(weight); }
        }

        add_idx
    }
    pub fn adjust_weight(&mut self, idx: usize, new_weight: f64) {
        assert!(idx <= self.weights.len());
        self.weights[idx] = new_weight;
    }
}
impl OptimaTensorFunction for OTFWeightedSum {
    fn output_dimensions(&self) -> Vec<usize> {
        if self.functions.is_empty() { return vec![] }
        else { return self.functions[0].output_dimensions() }
    }
    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.call_raw(input, immut_vars, mut_vars, session_key)?;
            let f = f_res.unwrap_tensor_mut();
            f.scalar_multiplication(self.weights[i]);
            if i == 0 {
                out = f.clone();
            } else {
                out = out.elementwise_addition(f);
            }
        }
        return Ok(OTFResult::Complete(out));
    }
    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative(input, immut_vars, mut_vars, None)?;
            let f = f_res.unwrap_tensor_mut();
            f.scalar_multiplication(self.weights[i]);
            if i == 0 {
                out = f.clone();
            } else {
                out = out.elementwise_addition(f);
            }
        }
        return Ok(OTFResult::Complete(out));
    }
    fn derivative2_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative2(input, immut_vars, mut_vars, None)?;
            let f = f_res.unwrap_tensor_mut();
            f.scalar_multiplication(self.weights[i]);
            if i == 0 {
                out = f.clone();
            } else {
                out = out.elementwise_addition(f);
            }
        }
        return Ok(OTFResult::Complete(out));
    }
    fn derivative3_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative3(input, immut_vars, mut_vars, None)?;
            let f = f_res.unwrap_tensor_mut();
            f.scalar_multiplication(self.weights[i]);
            if i == 0 {
                out = f.clone();
            } else {
                out = out.elementwise_addition(f);
            }
        }
        return Ok(OTFResult::Complete(out));
    }
    fn derivative4_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative4(input, immut_vars, mut_vars, None)?;
            let f = f_res.unwrap_tensor_mut();
            f.scalar_multiplication(self.weights[i]);
            if i == 0 {
                out = f.clone();
            } else {
                out = out.elementwise_addition(f);
            }
        }
        return Ok(OTFResult::Complete(out));
    }
}

/// Useful for less than 0 constraints
#[derive(Clone)]
pub struct OTFMaxZero;
impl OptimaTensorFunction for OTFMaxZero {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let x = input.unwrap_scalar();
        let val = x.max(0.0);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let x = input.unwrap_scalar();
        let val = if x > 0.0 { 1.0 }
        else { 0.0 };
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }
}

/// Useful for greater than 0 constraints
#[derive(Clone)]
pub struct OTFMinZero;
impl OptimaTensorFunction for OTFMinZero {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let x = input.unwrap_scalar();
        let val = x.min(0.0);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let x = input.unwrap_scalar();
        let val = if x < 0.0 { 1.0 }
        else { 0.0 };
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }
}

#[derive(Clone)]
pub struct OTFSin;
impl OptimaTensorFunction for OTFSin {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = -input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative3_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = -input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative4_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }
}

#[derive(Clone)]
pub struct OTFCos;
impl OptimaTensorFunction for OTFCos {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = -input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = -input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative3_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative4_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }
}

/// Also known as frobenius norm
#[derive(Clone)]
pub struct OTFTensorL2NormSquared;
impl OptimaTensorFunction for OTFTensorL2NormSquared {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut sum = 0.0;
        let vectorized_data = input.vectorized_data();
        for v in vectorized_data {
            sum += *v * *v;
        }
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(sum)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(input.dimensions());
        let vectorized_data = input.vectorized_data();
        let out_vectorized_data = out.vectorized_data_mut();
        for (i, v) in vectorized_data.iter().enumerate() {
            out_vectorized_data[i] = 2.0 * *v;
        }
        return Ok(OTFResult::Complete(out));
    }

    fn derivative2_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(input, 2));
        let vectorized_data = input.vectorized_data();
        let out_vectorized_data = out.vectorized_data_mut();
        for (i, _) in vectorized_data.iter().enumerate() {
            out_vectorized_data[i] = 2.0;
        }
        return Ok(OTFResult::Complete(out));
    }

    fn derivative3_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(input, 3));
        return Ok(OTFResult::Complete(out));
    }

    fn derivative4_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(input, 4));
        return Ok(OTFResult::Complete(out));
    }
}

/// coefficient*(x - horizontal_shift)^2
#[derive(Clone)]
pub struct OTFQuadraticLoss {
    pub coefficient: f64,
    pub horizontal_shift: f64
}
impl OptimaTensorFunction for OTFQuadraticLoss {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar();
        let out = self.coefficient * (val - self.horizontal_shift).powi(2);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar();
        let out = 2.0*self.coefficient*(val - self.horizontal_shift);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let out = 2.0*self.coefficient;
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }
}

#[derive(Clone)]
pub struct OTFAddScalar {
    pub scalar: f64,
}
impl OptimaTensorFunction for OTFAddScalar {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar();
        let out = val + self.scalar;
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(1.0)))
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }
}

#[derive(Clone)]
pub struct OTFMultiplyByScalar {
    pub scalar: f64,
}
impl OptimaTensorFunction for OTFMultiplyByScalar {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar();
        let out = val * self.scalar;
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(self.scalar)))
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }
}
