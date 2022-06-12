use crate::optima_tensor_function::{OptimaTensor, OptimaTensorFunction, OTFImmutVars, OTFMutVars, OTFMutVarsSessionKey, OTFResult};
use crate::utils::utils_errors::OptimaError;

pub struct OTFComposition<F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction {
    input_dimensions: Vec<usize>,
    output_dimensions: Vec<usize>,
    f: F,
    g: G
}
impl<F, G> OTFComposition<F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction {
    pub fn new(outer_function: F, enveloped_function: G) -> Self {
        Self {
            input_dimensions: enveloped_function.input_dimensions(),
            output_dimensions: outer_function.output_dimensions(),
            f: outer_function,
            g: enveloped_function
        }
    }
}
impl<F, G> OptimaTensorFunction for OTFComposition<F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction {
    fn input_dimensions(&self) -> Vec<usize> {
        self.input_dimensions.clone()
    }

    fn output_dimensions(&self) -> Vec<usize> {
        self.output_dimensions.clone()
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

pub struct OTFSin;
impl OptimaTensorFunction for OTFSin {
    fn input_dimensions(&self) -> Vec<usize> { vec![] }

    fn output_dimensions(&self) -> Vec<usize> { vec![] }

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

pub struct OTFCos;
impl OptimaTensorFunction for OTFCos {
    fn input_dimensions(&self) -> Vec<usize> { vec![] }

    fn output_dimensions(&self) -> Vec<usize> { vec![] }

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