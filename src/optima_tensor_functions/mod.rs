use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_optima_tensor_functions::{OptimaTensor_, OptimaTensorFunction, OTFImmutVars, OTFMutVars, OTFMutVarsSessionKey, OTFResult};

/// f(g(x))
pub struct OTFComposition<F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction {
    input_dimensions: (Vec<usize>, Vec<usize>),
    output_dimensions: (Vec<usize>, Vec<usize>),
    f: F,
    g: G
}
impl<F, G> OTFComposition <F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction  {
    pub fn new(f: F, g: G) -> Self {
        if g.output_dimensions() != f.input_dimensions() {
            panic!("OTFComposition dimensions do not align. (envloped: {:?}, outer: {:?})", g.output_dimensions(), f.input_dimensions());
        }

        Self {
            input_dimensions: g.input_dimensions(),
            output_dimensions: f.output_dimensions(),
            f,
            g
        }
    }
}
impl <F, G> OptimaTensorFunction for OTFComposition<F, G>
    where F: OptimaTensorFunction,
          G: OptimaTensorFunction{
    fn call_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let g_res = self.g.call_raw(input, immut_vars, mut_vars, session_key)?;
        let g = g_res.unwrap_tensor();
        let f = self.f.call_raw(g, immut_vars, mut_vars, session_key)?;
        return Ok(f);
    }
    fn derivative_analytical_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let g_res = self.g.call_raw(input, immut_vars, mut_vars, session_key)?;
        let g = g_res.unwrap_tensor();
        let dg_dx_res = self.g.derivative(input, immut_vars, mut_vars, None)?;
        let dg_dx = dg_dx_res.unwrap_tensor();
        let df_dg_res = self.f.derivative(g, immut_vars, mut_vars, None)?;
        let df_dg = df_dg_res.unwrap_tensor();
        let df_dx = df_dg.dot(dg_dx);
        return Ok(OTFResult::Complete(df_dx));
    }
    fn input_dimensions(&self) -> (Vec<usize>, Vec<usize>) {
        self.input_dimensions.clone()
    }
    fn output_dimensions(&self) -> (Vec<usize>, Vec<usize>) {
        self.output_dimensions.clone()
    }
}



pub mod robotics_functions;

