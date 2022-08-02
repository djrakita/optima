use nalgebra::DVector;
use crate::optima_tensor_function::{OptimaTensor, OptimaTensorFunction, OptimaTensorFunctionGenerics, OTFImmutVars, OTFMutVars, OTFMutVarsSessionKey, OTFResult};
use crate::utils::utils_console::{optima_print_multi_entry, OptimaDebug, OptimaPrintMultiEntry, OptimaPrintMultiEntryCollection, PrintColor};
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

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let g_res = self.g.call(input, _immut_vars, _mut_vars, debug.spawn_child()).expect("error");
        let g = g_res.unwrap_tensor();
        let f_res = self.f.call(g, _immut_vars, _mut_vars, debug.spawn_child()).expect("error");
        return Ok(f_res);
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let g_res = self.g.call(input, _immut_vars, _mut_vars, debug.spawn_child())?;
        let g = g_res.unwrap_tensor();
        let dg_dx_res = self.g.derivative(input, _immut_vars, _mut_vars, None, debug.spawn_child())?;
        let dg_dx = dg_dx_res.unwrap_tensor();
        let df_dg_res = self.f.derivative(g, _immut_vars, _mut_vars, None, debug.spawn_child())?;
        let df_dg = df_dg_res.unwrap_tensor();
        let df_dx = df_dg.dot(&dg_dx);
        if let OptimaDebug::True { num_indentation, num_indentation_history, .. } = &debug {
            let leading_marks = OptimaTensorFunctionGenerics::num_indentation_history_to_leading_marks(num_indentation_history);
            let mut m = OptimaPrintMultiEntryCollection::new_empty();
            m.add(OptimaPrintMultiEntry::new_from_str("Output will be df_dg.dot(&dg_dx))", PrintColor::None, false, false));
            m.add(OptimaPrintMultiEntry::new_from_string(format!("(df_dg: {:?}, dg_dx: {:?})", df_dg.to_string(), dg_dx.to_string()), PrintColor::Magenta, true, true));
            optima_print_multi_entry(m, *num_indentation, None, leading_marks.clone());
        }
        return Ok(OTFResult::Complete(df_dx));
    }

    fn to_string(&self) -> String {
        format!("OTFComposition< f: {}, g: {} >", self.f.to_string(), self.g.to_string())
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
    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.call(input, _immut_vars, _mut_vars, debug.spawn_child()).expect("error");
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
    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative(_input, _immut_vars, _mut_vars, None, debug.spawn_child())?;
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
    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative2(_input, _immut_vars, _mut_vars, None, debug.spawn_child())?;
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
    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative3(_input, _immut_vars, _mut_vars, None, debug.spawn_child())?;
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
    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(vec![]);
        for (i, function) in self.functions.iter().enumerate() {
            let mut f_res = function.derivative4(_input, _immut_vars, _mut_vars, None, debug.spawn_child())?;
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

    fn to_string(&self) -> String {
        let mut out_string = "OTFWeightedSum< ".to_string();
        for (i, f) in self.functions.iter().enumerate() {
            out_string += format!(" < f: {}, weight: {} >", f.to_string(), self.weights[i]).as_str();
        }

        out_string += " >";

        out_string
    }
}

/// Useful for less than 0 constraints
#[derive(Clone)]
pub struct OTFMaxZero;
impl OptimaTensorFunction for OTFMaxZero {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let x = input.unwrap_scalar();
        let val = x.max(0.0);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let x = _input.unwrap_scalar();
        let val = if x > 0.0 { 1.0 }
        else { 0.0 };
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn to_string(&self) -> String {
        "OTFMaxZero".to_string()
    }
}

/// Useful for greater than 0 constraints
#[derive(Clone)]
pub struct OTFMinZero;
impl OptimaTensorFunction for OTFMinZero {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let x = input.unwrap_scalar();
        let val = x.min(0.0);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let x = _input.unwrap_scalar();
        let val = if x < 0.0 { 1.0 }
        else { 0.0 };
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn to_string(&self) -> String {
        "OTFMinZero".to_string()
    }
}

#[derive(Clone)]
pub struct OTFSin;
impl OptimaTensorFunction for OTFSin {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().sin();
        let output = OptimaTensor::new_from_scalar(val);
        return Ok(OTFResult::Complete(output));
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = _input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = -_input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = -_input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = _input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn to_string(&self) -> String {
        "OTFSin".to_string()
    }
}

#[derive(Clone)]
pub struct OTFCos;
impl OptimaTensorFunction for OTFCos {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = -_input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = -_input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = _input.unwrap_scalar().sin();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = _input.unwrap_scalar().cos();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(val)));
    }

    fn to_string(&self) -> String {
        "OTFCos".to_string()
    }
}

/// Also known as frobenius norm
#[derive(Clone)]
pub struct OTFTensorL2NormSquared;
impl OptimaTensorFunction for OTFTensorL2NormSquared {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut sum = 0.0;
        let vectorized_data = input.vectorized_data();
        for v in vectorized_data {
            sum += *v * *v;
        }
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(sum)));
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(_input.dimensions());
        let vectorized_data = _input.vectorized_data();
        let out_vectorized_data = out.vectorized_data_mut();
        for (i, v) in vectorized_data.iter().enumerate() {
            out_vectorized_data[i] = 2.0 * *v;
        }
        return Ok(OTFResult::Complete(out));
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(_input, 2));
        let vectorized_data = _input.vectorized_data();
        let out_vectorized_data = out.vectorized_data_mut();
        for (i, _) in vectorized_data.iter().enumerate() {
            out_vectorized_data[i] = 2.0;
        }
        return Ok(OTFResult::Complete(out));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(_input, 3));
        return Ok(OTFResult::Complete(out));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(_input, 4));
        return Ok(OTFResult::Complete(out));
    }

    fn to_string(&self) -> String {
        "OTFTensorL2NormSquared".to_string()
    }
}

#[derive(Clone)]
pub struct OTFTensorAveragedL2NormSquared;
impl OptimaTensorFunction for OTFTensorAveragedL2NormSquared {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.call(input, _immut_vars, _mut_vars, _debug).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative(_input, _immut_vars, _mut_vars, None, _debug).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / _input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative2(_input, _immut_vars, _mut_vars, None, _debug).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / _input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative3(_input, _immut_vars, _mut_vars, None, _debug).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / _input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL2NormSquared;
        let mut out_res = o.derivative4(_input, _immut_vars, _mut_vars, None, _debug).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / _input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn to_string(&self) -> String {
        "OTFTensorAveragedL2NormSquared".to_string()
    }
}

#[derive(Clone)]
pub struct OTFTensorL1Norm;
impl OptimaTensorFunction for OTFTensorL1Norm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out_sum = 0.0;
        for v in input.vectorized_data() { out_sum += v.abs(); }
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out_sum)));
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut out_tensor = _input.clone();
        let input_vectorized_data = _input.vectorized_data();
        for (idx, v) in out_tensor.vectorized_data_mut().iter_mut().enumerate() {
            let s = input_vectorized_data[idx];
            *v = if s.is_sign_positive() { 1.0 } else if s.is_sign_negative() { -1.0 } else { 0.0 }
        }
        return Ok(OTFResult::Complete(out_tensor));
    }

    fn to_string(&self) -> String {
        "OTFTensorL1Norm".to_string()
    }
}

#[derive(Clone)]
pub struct OTFTensorAveragedL1Norm;
impl OptimaTensorFunction for OTFTensorAveragedL1Norm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL1Norm;
        let mut out_res = o.call(input, _immut_vars, _mut_vars, _debug).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let o = OTFTensorL1Norm;
        let mut out_res = o.derivative(_input, _immut_vars, _mut_vars, None, _debug).expect("error");
        let mut out = out_res.unwrap_tensor_mut();
        out.scalar_multiplication(1.0 / _input.vectorized_data().len() as f64);
        return Ok(out_res);
    }

    fn to_string(&self) -> String {
        "OTFTensorAveragedL1Norm".to_string()
    }
}

#[derive(Clone)]
pub struct OTFTensorPNorm { pub p: f64 }
impl OptimaTensorFunction for OTFTensorPNorm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let vectorized = input.vectorized_data();
        let mut out = 0.0;

        for v in vectorized { out += v.abs().powf(self.p); }
        out = out.powf(1.0 / self.p);

        assert!(!out.is_nan(), "call raw with an input of {:?} is NaN.", input);

        Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let start = instant::Instant::now();
        let vectorized = _input.vectorized_data();
        let mut out = OptimaTensor::new_zeros(_input.dimensions());
        let mut out_vectorized = out.vectorized_data_mut();

        let mut prod = 0.0;
        for v in vectorized { prod += v.abs().powf(self.p); }
        if prod != 0.0 { prod = prod.powf(1.0 / self.p - 1.0); }

        for (i, v) in vectorized.iter().enumerate() {
            let val = *v * v.abs().powf(self.p - 2.0) * prod;
            assert!(!val.is_nan(), "val with an input of {:?} is NaN.  prod is {:?}", _input, prod);
            out_vectorized[i] = val;
        }

        Ok(OTFResult::Complete(out))
    }

    fn to_string(&self) -> String {
        format!("OTFTensorPNorm_{}", self.p)
    }
}

#[derive(Clone)]
pub struct OTFTensorLinfNorm;
impl OptimaTensorFunction for OTFTensorLinfNorm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut max = -f64::INFINITY;
        for v in input.vectorized_data() {
            let a = v.abs();
            if a > max { max = a; }
        }
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(max)));
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut max = -f64::INFINITY;
        let mut max_idx = usize::MAX;
        for (idx, v) in _input.vectorized_data().iter().enumerate() {
            let a = v.abs();
            if a > max { max = a; max_idx = idx; }
        }
        let mut out = OptimaTensor::new_zeros(_input.dimensions());
        let val = _input.vectorized_data()[max_idx];
        out.vectorized_data_mut()[max_idx] = if val > 0.0 { 1.0 } else if val < 0.0 { -1.0 } else { 0.0 };

        return Ok(OTFResult::Complete(out));
    }

    fn to_string(&self) -> String {
        "OTFTensorLinfNorm".to_string()
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

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar();
        let out = self.coefficient * (val - self.horizontal_shift).powi(2);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = _input.unwrap_scalar();
        let out = 2.0*self.coefficient*(val - self.horizontal_shift);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let out = 2.0*self.coefficient;
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn to_string(&self) -> String {
        format!("OTFQuadraticLoss_{}_{}", self.horizontal_shift, self.coefficient)
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

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar();
        let out = val + self.scalar;
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(1.0)))
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn to_string(&self) -> String {
        format!("OTFAddScalar_{}", self.scalar)
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

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let val = input.unwrap_scalar();
        let out = val * self.scalar;
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)))
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(self.scalar)))
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)))
    }

    fn to_string(&self) -> String {
        format!("OTFMultiplyByScalar_{}", self.scalar)
    }
}

/// When an objective term is added to a larger optimization model (e.g., a weighted sum),
/// it is often useful to consider each term on a normalized scale (e.g., roughly 0 - 1) such that
/// meaninful tradeoffs can be made between the terms in the model.  This `OTFNormalizer` functions
/// helps meet this goal as the normalization value serves as the new "1" value for the function.
#[derive(Clone)]
pub struct OTFLinearNormalizer {
    normalization_value: f64,
    f: OTFMultiplyByScalar
}
impl OTFLinearNormalizer {
    pub fn new(normalization_value: f64) -> Self {
        Self {
            normalization_value,
            f: OTFMultiplyByScalar { scalar: 1.0/ normalization_value }
        }
    }
}
impl OptimaTensorFunction for OTFLinearNormalizer {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.call_raw(input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative2_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative3_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative4_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn to_string(&self) -> String {
        format!("OTFLinearNormalizer_{}", self.normalization_value)
    }
}

#[derive(Clone)]
pub struct OTFQuadraticNormalizer {
    normalization_value: f64,
    f: OTFQuadraticLoss
}
impl OTFQuadraticNormalizer {
    pub fn new(normalization_value: f64) -> Self {
        Self {
            normalization_value,
            f: OTFQuadraticLoss {
                coefficient: 1.0 / (normalization_value * normalization_value),
                horizontal_shift: 0.0
            }
        }
    }
}
impl OptimaTensorFunction for OTFQuadraticNormalizer {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.call_raw(input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative2_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative3_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return self.f.derivative4_analytical_raw(_input, _immut_vars, _mut_vars, _session_key, _debug);
    }

    fn to_string(&self) -> String {
        format!("OTFQuadraticNormalizer_{}", self.normalization_value)
    }
}

/// Function that just returns 0.  This is useful for optimization problems that only has constraints
/// and no cost.
#[derive(Clone)]
pub struct OTFZeroFunction;
impl OptimaTensorFunction for OTFZeroFunction {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(0.0)));
    }

    fn to_string(&self) -> String {
        "OTFZeroFunction".to_string()
    }
}
