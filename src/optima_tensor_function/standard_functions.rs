use nalgebra::DVector;
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

#[derive(Clone)]
pub struct OTFTensorAveragedL2NormSquared;
impl OptimaTensorFunction for OTFTensorAveragedL2NormSquared {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.call(input, immut_vars, mut_vars).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative(input, immut_vars, mut_vars, None).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative2_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative2(input, immut_vars, mut_vars, None).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative3_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative3(input, immut_vars, mut_vars, None).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative4_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative4(input, immut_vars, mut_vars, None).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }
}

#[derive(Clone)]
pub struct OTFTensorL1Norm;
impl OptimaTensorFunction for OTFTensorL1Norm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out_sum = 0.0;
        for v in input.vectorized_data() { out_sum += v.abs(); }
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out_sum)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut out_tensor = input.clone();
        let input_vectorized_data = input.vectorized_data();
        for (idx, v) in out_tensor.vectorized_data_mut().iter_mut().enumerate() {
            let s = input_vectorized_data[idx];
            *v = if s.is_sign_positive() { 1.0 } else if s.is_sign_negative() { -1.0 } else { 0.0 }
        }
        return Ok(OTFResult::Complete(out_tensor));
    }
}

#[derive(Clone)]
pub struct OTFTensorAveragedL1Norm;
impl OptimaTensorFunction for OTFTensorAveragedL1Norm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL1Norm;
        let mut out_res = o.call(input, immut_vars, mut_vars).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL1Norm;
        let mut out_res = o.derivative(input, immut_vars, mut_vars, None).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }
}

#[derive(Clone)]
pub struct OTFTensorLinfNorm;
impl OptimaTensorFunction for OTFTensorLinfNorm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut max = -f64::INFINITY;
        for v in input.vectorized_data() {
            let a = v.abs();
            if a > max { max = a; }
        }
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(max)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut max = -f64::INFINITY;
        let mut max_idx = usize::MAX;
        for (idx, v) in input.vectorized_data().iter().enumerate() {
            let a = v.abs();
            if a > max { max = a; max_idx = idx; }
        }
        let mut out = OptimaTensor::new_zeros(input.dimensions());
        let val = input.vectorized_data()[max_idx];
        out.vectorized_data_mut()[max_idx] = if val > 0.0 { 1.0 } else if val < 0.0 { -1.0 } else { 0.0 };

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

/// When an objective term is added to a larger optimization model (e.g., a weighted sum),
/// it is often useful to consider each term on a normalized scale (e.g., roughly 0 - 1) such that
/// meaninful tradeoffs can be made between the terms in the model.  This `OTFNormalizer` functions
/// helps meet this goal as the normalization value serves as the new "1" value for the function.
#[derive(Clone)]
pub struct OTFNormalizer {
    normalization_value: f64,
    f: OTFMultiplyByScalar
}
impl OTFNormalizer {
    pub fn new(normalization_value: f64) -> Self {
        Self {
            normalization_value,
            f: OTFMultiplyByScalar { scalar: 1.0/ normalization_value }
        }
    }
}
impl OptimaTensorFunction for OTFNormalizer {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return self.f.call_raw(input, _immut_vars, _mut_vars, _session_key);
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return self.f.derivative_analytical_raw(_input, _immut_vars, _mut_vars, _session_key);
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return self.f.derivative2_analytical_raw(_input, _immut_vars, _mut_vars, _session_key);
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return self.f.derivative3_analytical_raw(_input, _immut_vars, _mut_vars, _session_key);
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return self.f.derivative4_analytical_raw(_input, _immut_vars, _mut_vars, _session_key);
    }
}

/// Function that just returns 0.  This is useful for optimization problems that only has constraints
/// and no cost.
#[derive(Clone)]
pub struct ZeroFunction;
impl OptimaTensorFunction for ZeroFunction {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }
}
