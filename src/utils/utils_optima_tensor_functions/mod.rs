use std::fmt::Debug;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array2, ArrayD, Axis};
use ndarray_einsum_beta::tensordot;
use serde::{Serialize, Deserialize};
use crate::robot_set_modules::GetRobotSet;
use crate::utils::utils_console::{optima_print, optima_print_new_line, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_generic_data_structures::{AveragingFloat, EnumBinarySearchSignatureContainer, EnumHashMapSignatureContainer, EnumMapToSignature, EnumSignatureContainer, EnumSignatureContainerType};
use crate::utils::utils_sampling::SimpleSamplers;

#[allow(unused_variables)]
pub trait OptimaTensorFunction {
    fn call(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.call_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn call_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError>;

    fn derivative(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        self.derivative_generic(Self::derivative_analytical, Self::derivative_finite_difference, Self::derivative_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative_none_mode(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative(input, immut_vars, mut_vars, None);
    }
    fn derivative_analytical(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative_analytical_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative_finite_difference(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative_finite_difference_generic(Self::call,
                                                         Self::get_derivative_output_tensor,
                                                         input,
                                                         immut_vars,
                                                         mut_vars);
    }
    fn derivative_test(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    fn derivative2(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        self.derivative_generic(Self::derivative2_analytical, Self::derivative2_finite_difference, Self::derivative2_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative2_none_mode(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative2(input, immut_vars, mut_vars, None);
    }
    fn derivative2_analytical(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative2_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative2_analytical_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative2_finite_difference(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative_finite_difference_generic(Self::derivative_none_mode,
                                                         Self::get_derivative2_output_tensor,
                                                         input,
                                                         immut_vars,
                                                         mut_vars);
    }
    fn derivative2_test(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    fn derivative3(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        self.derivative_generic(Self::derivative3_analytical, Self::derivative3_finite_difference, Self::derivative3_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative3_none_mode(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative3(input, immut_vars, mut_vars, None);
    }
    fn derivative3_analytical(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative3_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative3_analytical_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative3_finite_difference(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative_finite_difference_generic(Self::derivative2_none_mode,
                                                         Self::get_derivative3_output_tensor,
                                                         input,
                                                         immut_vars,
                                                         mut_vars);
    }
    fn derivative3_test(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    fn derivative4(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        self.derivative_generic(Self::derivative4_analytical, Self::derivative4_finite_difference, Self::derivative4_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative4_none_mode(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative4(input, immut_vars, mut_vars, None);
    }
    fn derivative4_analytical(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative4_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative4_analytical_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative4_finite_difference(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative_finite_difference_generic(Self::derivative3_none_mode,
                                                         Self::get_derivative4_output_tensor,
                                                         input,
                                                         immut_vars,
                                                         mut_vars);
    }
    fn derivative4_test(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    fn derivative_generic<F1, F2, F3>(&self,
                                      analytical: F1,
                                      finite_difference: F2,
                                      test: F3,
                                      input: &OptimaTensor_,
                                      immut_vars: &OTFImmutVars,
                                      mut_vars: &mut OTFMutVars,
                                      mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError>
        where F1: Fn(&Self, &OptimaTensor_, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError>,
              F2: Fn(&Self, &OptimaTensor_, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError>,
              F3: Fn(&Self, &OptimaTensor_, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        match mode {
            None => {
                {
                    let res = analytical(self, input, immut_vars, mut_vars)?;
                    match res {
                        OTFResult::Unimplemented => { }
                        OTFResult::Complete(_) => { return Ok(res); }
                    }
                }

                {
                    let res = finite_difference(self, input, immut_vars, mut_vars)?;
                    match res {
                        OTFResult::Unimplemented => {}
                        OTFResult::Complete(_) => { return Ok(res); }
                    }
                }


                panic!("Called an Unimplemented Derivative on OTF.")

            }
            Some(mode) => {
                let res = match mode {
                    OTFDerivativeMode::Analytical => { analytical(self, input, immut_vars, mut_vars) }
                    OTFDerivativeMode::FiniteDifference => { finite_difference(self, input,  immut_vars, mut_vars) }
                    OTFDerivativeMode::Test => { test(self, input, immut_vars, mut_vars) }
                }?;
                return Ok(res);
            }
        }
    }
    fn derivative_finite_difference_generic<F1, F2>(&self, caller: F1, outputter: F2, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError>
        where F1: Fn(&Self, &OptimaTensor_, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError> ,
              F2: Fn(&Self, &OptimaTensor_, &OTFImmutVars) -> OptimaTensor_ {
        let mut output = outputter(self, input, immut_vars);

        let f0_result = caller(self, input, immut_vars, mut_vars)?;
        let f0 = match f0_result {
            OTFResult::Unimplemented => { unimplemented!() }
            OTFResult::Complete(f0) => { f0 }
        };

        let vectorized_input = input.vectorized_data().to_vec();

        for (vectorized_input_idx, vectorized_input_val) in vectorized_input.iter().enumerate() {
            let mut input_clone = input.clone();
            input_clone.vectorized_data_mut()[vectorized_input_idx] += FD_PERTURBATION;

            let fh_result = caller(self, &input_clone, immut_vars, mut_vars)?;
            let mut fh = match fh_result {
                OTFResult::Unimplemented => { unimplemented!() }
                OTFResult::Complete(fh) => { fh }
            };

            // tensor version of (f_h - f_0) / perturbation
            fh.elementwise_subtraction_inplace(&f0);
            fh.scalar_division_inplace(FD_PERTURBATION);

            output.assign_to_inner_dimensions_tensor_from_outer_dimensions_vectorized_idx(vectorized_input_idx, &fh);
        }

        Ok(OTFResult::Complete(output))
    }

    fn input_dimensions(&self) -> (Vec<usize>, Vec<usize>);
    fn output_dimensions(&self) -> (Vec<usize>, Vec<usize>);

    fn get_call_output_tensor(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars) -> OptimaTensor_ {
        let output_dimensions = self.output_dimensions();
        return OptimaTensor_::new_from_inner_and_outer_dimensions(output_dimensions.0, output_dimensions.1);
    }
    fn get_derivative_output_tensor(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars) -> OptimaTensor_ {
        let output_dimensions = self.output_dimensions();
        let inner_dimensions = [output_dimensions.0, output_dimensions.1].concat();
        return OptimaTensor_::new_from_inner_and_outer_dimensions(inner_dimensions, input.combined_dimensions.clone());
    }
    fn get_derivative2_output_tensor(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars) -> OptimaTensor_ {
        let output_dimensions = self.output_dimensions();
        let inner_dimensions = [output_dimensions.0, output_dimensions.1, input.combined_dimensions.clone()].concat();
        return OptimaTensor_::new_from_inner_and_outer_dimensions(inner_dimensions, input.combined_dimensions.clone());
    }
    fn get_derivative3_output_tensor(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars) -> OptimaTensor_ {
        let output_dimensions = self.output_dimensions();
        let inner_dimensions = [output_dimensions.0, output_dimensions.1, input.combined_dimensions.clone(), input.combined_dimensions.clone()].concat();
        return OptimaTensor_::new_from_inner_and_outer_dimensions(inner_dimensions, input.combined_dimensions.clone());
    }
    fn get_derivative4_output_tensor(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars) -> OptimaTensor_ {
        let output_dimensions = self.output_dimensions();
        let inner_dimensions = [output_dimensions.0, output_dimensions.1, input.combined_dimensions.clone(), input.combined_dimensions.clone(), input.combined_dimensions.clone()].concat();
        return OptimaTensor_::new_from_inner_and_outer_dimensions(inner_dimensions, input.combined_dimensions.clone());
    }

    fn diagnostics(&self, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) {
        optima_print_new_line();
        optima_print("Call Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.call_diagnostics(immut_vars, mut_vars, 1000);
        optima_print_new_line();
        optima_print("Derivative Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative_diagnostics(immut_vars, mut_vars, 1000);
        optima_print_new_line();
        optima_print("Derivative2 Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative2_diagnostics(immut_vars, mut_vars, 500);
        optima_print_new_line();
        optima_print("Derivative3 Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative3_diagnostics(immut_vars, mut_vars, 100);
        optima_print_new_line();
        optima_print("Derivative4 Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative4_diagnostics(immut_vars, mut_vars, 50);
    }
    fn call_diagnostics(&self, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        let input_dimensions = self.input_dimensions();

        let mut random_inputs = vec![];
        for _ in 0..num_inputs {
            random_inputs.push(OptimaTensor_::new_random_sample(input_dimensions.0.clone(), input_dimensions.1.clone()));
        }

        let mut call_time = AveragingFloat::new();

        for r in &random_inputs {
            let start = instant::Instant::now();
            self.call(r, immut_vars, mut_vars).expect("error");
            let duration = start.elapsed();
            call_time.add_new_value(duration.as_secs_f64());
        }

        let call_time_as_duration = instant::Duration::from_secs_f64(call_time.value());

        optima_print(&format!("Call time over {} inputs is ", num_inputs), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("{:?}", call_time_as_duration), PrintMode::Print, PrintColor::Green, true);
        optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);

    }
    fn derivative_diagnostics(&self, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        self.derivative_diagnostics_generic(Self::derivative, immut_vars, mut_vars, num_inputs);
    }
    fn derivative2_diagnostics(&self, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        self.derivative_diagnostics_generic(Self::derivative2, immut_vars, mut_vars, num_inputs);
    }
    fn derivative3_diagnostics(&self, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        self.derivative_diagnostics_generic(Self::derivative3, immut_vars, mut_vars, num_inputs);
    }
    fn derivative4_diagnostics(&self, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        self.derivative_diagnostics_generic(Self::derivative4, immut_vars, mut_vars, num_inputs);
    }

    fn derivative_diagnostics_generic<F>(&self, derivative_function: F, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize)
        where F: Fn(&Self, &OptimaTensor_, &OTFImmutVars, &mut OTFMutVars, Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        let input_dimensions = self.input_dimensions();

        let mut random_inputs = vec![];
        for _ in 0..num_inputs {
            random_inputs.push(OptimaTensor_::new_random_sample(input_dimensions.0.clone(), input_dimensions.1.clone()));
        }

        let mut finite_difference_time = AveragingFloat::new();
        let mut finite_difference_results = vec![];
        for r in &random_inputs {
            let start = instant::Instant::now();
            let res = derivative_function(self, r, immut_vars, mut_vars, Some(OTFDerivativeMode::FiniteDifference)).expect("error");
            let duration = start.elapsed();
            finite_difference_time.add_new_value(duration.as_secs_f64());
            match res {
                OTFResult::Unimplemented => { unreachable!() }
                OTFResult::Complete(tensor) => { finite_difference_results.push(tensor); }
            }
        }
        let finite_difference_time_as_duration = instant::Duration::from_secs_f64(finite_difference_time.value());

        let mut analytical_time = AveragingFloat::new();
        let mut analytical_diffs = AveragingFloat::new();
        let mut analytical_results = vec![];
        let mut analytical_unimplemented = false;
        for r in &random_inputs {
            let start = instant::Instant::now();
            let res = derivative_function(self, r, immut_vars, mut_vars, Some(OTFDerivativeMode::Analytical)).expect("error");
            let duration = start.elapsed();
            analytical_time.add_new_value(duration.as_secs_f64());
            match res {
                OTFResult::Unimplemented => { analytical_unimplemented = true; break; }
                OTFResult::Complete(tensor) => { analytical_results.push(tensor); }
            }
        }
        let analytical_time_as_duration = instant::Duration::from_secs_f64(analytical_time.value());

        let mut test_time = AveragingFloat::new();
        let mut test_diffs = AveragingFloat::new();
        let mut test_results = vec![];
        let mut test_unimplemented = false;
        for r in &random_inputs {
            let start = instant::Instant::now();
            let res = derivative_function(self, r, immut_vars, mut_vars, Some(OTFDerivativeMode::Test)).expect("error");
            let duration = start.elapsed();
            test_time.add_new_value(duration.as_secs_f64());
            match res {
                OTFResult::Unimplemented => { test_unimplemented = true; break; }
                OTFResult::Complete(tensor) => { test_results.push(tensor); }
            }
        }
        let test_time_as_duration = instant::Duration::from_secs_f64(test_time.value());

        if !analytical_unimplemented {
            for (i, finite_difference_res) in finite_difference_results.iter().enumerate() {
                let analytical_res = &analytical_results[i];
                let diff = finite_difference_res.average_difference(analytical_res);
                analytical_diffs.add_new_value(diff);
            }
        }
        if !test_unimplemented {
            for (i, finite_difference_res) in finite_difference_results.iter().enumerate() {
                let test_res = &test_results[i];
                let diff = finite_difference_res.average_difference(test_res);
                test_diffs.add_new_value(diff);
            }
        }

        optima_print(&format!("Finite Difference time over {} inputs is ", num_inputs), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("{:?}", finite_difference_time_as_duration), PrintMode::Print, PrintColor::Green, true);
        optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);

        if !analytical_unimplemented {
            optima_print(&format!("Analytical time over {} inputs is ", num_inputs), PrintMode::Print, PrintColor::None, false);
            optima_print(&format!("{:?}", analytical_time_as_duration), PrintMode::Print, PrintColor::Green, true);
            optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);
        }
        if !test_unimplemented {
            optima_print(&format!("Test time over {} inputs is ", num_inputs), PrintMode::Print, PrintColor::None, false);
            optima_print(&format!("{:?}", test_time_as_duration), PrintMode::Print, PrintColor::Green, true);
            optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);
        }

        optima_print_new_line();

        if !analytical_unimplemented {
            let analytical_diffs_value = analytical_diffs.value();
            let color = if analytical_diffs_value.abs() > 0.5 {
                PrintColor::Red
            } else if analytical_diffs_value.abs() > 0.05 {
                PrintColor::Yellow
            } else {
                PrintColor::Green
            };
            optima_print(&format!("Analytical difference from Finite Difference over {} inputs is ", num_inputs), PrintMode::Print, PrintColor::None, false);
            optima_print(&format!("{:?}", analytical_diffs_value), PrintMode::Print, color, true);
            optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);
        }
        if !test_unimplemented {
            let test_diffs_value = test_diffs.value();
            let color = if test_diffs_value.abs() > 0.5 {
                PrintColor::Red
            } else if test_diffs_value.abs() > 0.05 {
                PrintColor::Yellow
            } else {
                PrintColor::Green
            };
            optima_print(&format!("Test difference from Finite Difference over {} inputs is ", num_inputs), PrintMode::Print, PrintColor::None, false);
            optima_print(&format!("{:?}", test_diffs_value), PrintMode::Print, color, true);
            optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);
        }
    }
}

pub const FD_PERTURBATION: f64 = 0.000001;

pub struct OTFSin;
impl OptimaTensorFunction for OTFSin {
    fn call_raw(&self, input: &OptimaTensor_, _vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor_::new_from_scalar(input.get_value(vec![0, 0]).sin())));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor_, _vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor_::new_from_scalar(input.get_value(vec![0, 0]).cos())));
    }

    fn derivative2_analytical_raw(&self, input: &OptimaTensor_, _vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor_::new_from_scalar(-input.get_value(vec![0, 0]).sin())));
    }

    fn derivative3_analytical_raw(&self, input: &OptimaTensor_, _vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor_::new_from_scalar(-input.get_value(vec![0, 0]).cos())));
    }

    fn derivative4_analytical_raw(&self, input: &OptimaTensor_, _vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        return Ok(OTFResult::Complete(OptimaTensor_::new_from_scalar(input.get_value(vec![0, 0]).sin())));
    }

    fn input_dimensions(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![1], vec![1])
    }

    fn output_dimensions(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![1], vec![1])
    }
}

#[derive(Clone, Debug)]
pub enum OTFResult {
    Unimplemented, Complete(OptimaTensor_)
}
impl OTFResult {
    pub fn unwrap_tensor(&self) -> &OptimaTensor_ {
        match self {
            OTFResult::Unimplemented => { unimplemented!() }
            OTFResult::Complete(t) => { return t; }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OTFDerivativeMode {
    Analytical, FiniteDifference, Test
}

/// Creates an optima tensor that is (inner_dimensions) x (outer_dimensions).
/// Can be thought of as an outer_dimensions matrix of inner_dimensions matrices. 
#[derive(Clone, Debug)]
pub struct OptimaTensor_ {
    ndim: usize,
    tensor_nd: Option<ArrayD<f64>>,
    tensor_2d: Option<Array2<f64>>,
    inner_dimensions: Vec<usize>,
    outer_dimensions: Vec<usize>,
    combined_dimensions: Vec<usize>,
    inner_dimensions_axes: Vec<Axis>,
    outer_dimensions_axes: Vec<Axis>,
    inner_dimensions_strides: Vec<usize>,
    outer_dimensions_strides: Vec<usize>,
    combined_strides: Vec<usize>,
    id: i32
}
impl OptimaTensor_ {
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

        let inner_dimensions = Self::flatten_dimensions(inner_dimensions);
        let outer_dimensions = Self::flatten_dimensions(outer_dimensions);

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
        let mut inner_dimensions_strides = vec![0; inner_dimensions_len];
        let mut outer_dimensions_strides = vec![0; outer_dimensions_len];

        let tensor_2d = if combined_dimensions.len() == 2 {
            let tensor = Array2::zeros((inner_dimensions[0], outer_dimensions[0]));
            for s in tensor.strides() { combined_strides.push(*s as usize); }
            Some(tensor)
        } else {
            None
        };
        let tensor_nd = if combined_dimensions.len() == 2 {
            None
        } else {
            let tensor = Array::<f64, _>::zeros(combined_dimensions.clone());
            for s in tensor.strides() { combined_strides.push(*s as usize); }
            Some(tensor)
        };

        for (i, _) in inner_dimensions.iter().enumerate() {
            let mut total_number_of_elements = 1;
            for (j, d) in inner_dimensions.iter().enumerate() {
                if j > i {
                    total_number_of_elements *= d;
                }
            }
            inner_dimensions_strides[i] = total_number_of_elements;
        }
        for (i, _) in outer_dimensions.iter().enumerate() {
            let mut total_number_of_elements = 1;
            for (j, d) in outer_dimensions.iter().enumerate() {
                if j > i {
                    total_number_of_elements *= d;
                }
            }
            outer_dimensions_strides[i] = total_number_of_elements;
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
            combined_strides,
            id: SimpleSamplers::uniform_samples_i32(&vec![(i32::MIN, i32::MAX)])[0]
        }
    }
    pub fn new_from_scalar(val: f64) -> Self {
        let mut out = Self::new(vec![1,1]);
        out.assign_value(vec![0,0], val);
        out
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
    pub fn new_random_sample(inner_dimensions: Vec<usize>, outer_dimensions: Vec<usize>) -> Self {
        let mut out_self = Self::new_from_inner_and_outer_dimensions(inner_dimensions, outer_dimensions);
        let vectorized_data = out_self.vectorized_data_mut();
        let l = vectorized_data.len();
        let samples = SimpleSamplers::uniform_samples(&vec![(-2.0, 2.0); l]);
        for (i, s) in samples.iter().enumerate() { vectorized_data[i] = *s; }
        return out_self;
    }

    pub fn get_value(&self, indices: Vec<usize>) -> f64 {
        if self.ndim() != indices.len() {
            panic!("Wrong number of indices, should have been {}.", self.ndim);
        }

        for (i, dim) in self.combined_dimensions.iter().enumerate() {
            if indices[i] >= *dim {
                panic!("Index {} was too high (should be < {}).", i, dim);
            }
        }

        let vectorized_idx = self.combined_indices_to_vectorized_idx(indices);
        let vectorized_data = if let Some(tensor) = &self.tensor_2d {
            tensor.as_slice().unwrap()
        } else if let Some(tensor) = &self.tensor_nd {
            tensor.as_slice().unwrap()
        } else {
            unreachable!()
        };

        return vectorized_data[vectorized_idx];
    }
    pub fn assign_value(&mut self, indices: Vec<usize>, value: f64) {
        if self.ndim() != indices.len() {
            panic!("Wrong number of indices, should have been {}.", self.ndim);
        }

        for (i, dim) in self.combined_dimensions.iter().enumerate() {
            if indices[i] >= *dim {
                panic!("Index {} was too high (should be < {}).", i, dim);
            }
        }

        let vectorized_idx = self.combined_indices_to_vectorized_idx(indices);
        let vectorized_data = if let Some(tensor) = &mut self.tensor_2d {
            tensor.as_slice_mut().unwrap()
        } else if let Some(tensor) = &mut self.tensor_nd {
            tensor.as_slice_mut().unwrap()
        } else {
            unreachable!()
        };

        vectorized_data[vectorized_idx] = value;
    }

    #[allow(dead_code)]
    fn tensor_nd_mut(&mut self) -> &mut ArrayD<f64> {
        return match &mut self.tensor_nd {
            None => { panic!("tensor_nd is None.  Try tensor_2d instead.") }
            Some(t) => { t }
        }
    }
    #[allow(dead_code)]
    fn tensor_nd(&self) -> &ArrayD<f64> {
        return match &self.tensor_nd {
            None => { panic!("tensor_nd is None.  Try tensor_2d instead.") }
            Some(t) => { t }
        }
    }
    #[allow(dead_code)]
    fn tensor_2d_mut(&mut self) -> &mut Array2<f64> {
        return match &mut self.tensor_2d {
            None => { panic!("tensor_2d is None.  Try tensor_nd instead.") }
            Some(t) => { t }
        }
    }
    #[allow(dead_code)]
    fn tensor_2d(&self) -> &Array2<f64> {
        return match &self.tensor_2d {
            None => { panic!("tensor_2d is None.  Try tensor_nd instead.") }
            Some(t) => { t }
        }
    }
    fn vectorized_data(&self) -> &[f64] {
        if let Some(tensor) = &self.tensor_2d {
            return tensor.as_slice().unwrap()
        } else if let Some(tensor) = &self.tensor_nd {
            return tensor.as_slice().unwrap()
        }

        unreachable!()
    }
    fn vectorized_data_mut(&mut self) -> &mut [f64] {
        if let Some(tensor) = &mut self.tensor_2d {
            return tensor.as_slice_mut().unwrap()
        } else if let Some(tensor) = &mut self.tensor_nd {
            return tensor.as_slice_mut().unwrap()
        }

        unreachable!()
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
    pub fn convert_to_scalar(&self) -> f64 {
        if self.ndim > 2 {
            panic!("Cannot convert an optima tensor of dimension > 2 to a scalar.");
        }

        let tensor = self.tensor_2d();
        let shape = tensor.shape();

        if shape[0] > 1 || shape[1] > 1 {
            panic!("Cannot convert an optima tensor with shape of {:?} to scalar.", shape);
        }

        return tensor[(0,0)];
    }

    pub fn scalar_multiplication(&self, val: f64) -> OptimaTensor_ {
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
    pub fn scalar_addition(&self, val: f64) -> OptimaTensor_ {
        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            *tensor = val + tensor_self;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            *tensor = val + tensor_self;
        }

        out
    }
    pub fn scalar_addition_inplace(&mut self, val: f64) {
        if let Some(tensor) = &mut self.tensor_nd {
            *tensor += val;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            *tensor += val;
        }
    }
    pub fn scalar_division(&self, val: f64) -> OptimaTensor_ {
        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            *tensor = tensor_self / val;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            *tensor = tensor_self / val;
        }

        out
    }
    pub fn scalar_division_inplace(&mut self, val: f64) {
        if let Some(tensor) = &mut self.tensor_nd {
            *tensor /= val;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            *tensor /= val;
        }
    }
    pub fn scalar_subtraction(&self, val: f64) -> OptimaTensor_ {
        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            *tensor = tensor_self - val;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            *tensor = tensor_self - val;
        }

        out
    }
    pub fn scalar_subtraction_inplace(&mut self, val: f64) {
        if let Some(tensor) = &mut self.tensor_nd {
            *tensor -= val;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            *tensor -= val;
        }
    }

    pub fn elementwise_multiplication(&self, other: &OptimaTensor_) -> OptimaTensor_ {
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
    pub fn elementwise_multiplication_inplace(&mut self, other: &OptimaTensor_) {
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
    pub fn elementwise_addition(&self, other: &OptimaTensor_) -> OptimaTensor_ {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor = tensor_self + tensor_other;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor = tensor_self + tensor_other;
        }

        out
    }
    pub fn elementwise_addition_inplace(&mut self, other: &OptimaTensor_) {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        if let Some(tensor) = &mut self.tensor_nd {
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor += tensor_other;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor += tensor_other;
        }
    }
    pub fn elementwise_division(&self, other: &OptimaTensor_) -> OptimaTensor_ {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor = tensor_self / tensor_other;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor = tensor_self / tensor_other;
        }

        out
    }
    pub fn elementwise_division_inplace(&mut self, other: &OptimaTensor_) {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        if let Some(tensor) = &mut self.tensor_nd {
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor /= tensor_other;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor /= tensor_other;
        }
    }
    pub fn elementwise_subtraction(&self, other: &OptimaTensor_) -> OptimaTensor_ {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        let mut out = Self::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), self.outer_dimensions.clone());

        if let Some(tensor) = &mut out.tensor_nd {
            let tensor_self = self.tensor_nd.as_ref().unwrap();
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor = tensor_self - tensor_other;
        }

        if let Some(tensor) = &mut out.tensor_2d {
            let tensor_self = self.tensor_2d.as_ref().unwrap();
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor = tensor_self - tensor_other;
        }

        out
    }
    pub fn elementwise_subtraction_inplace(&mut self, other: &OptimaTensor_) {
        if self.inner_dimensions() != other.inner_dimensions() || self.outer_dimensions() != other.outer_dimensions() {
            panic!("OptimaTensor dimensions are incompatible for elementwise_multiplication_inplace.");
        }

        if let Some(tensor) = &mut self.tensor_nd {
            let tensor_other = other.tensor_nd.as_ref().unwrap();
            *tensor -= tensor_other;
        }

        if let Some(tensor) = &mut self.tensor_2d {
            let tensor_other = other.tensor_2d.as_ref().unwrap();
            *tensor -= tensor_other;
        }
    }

    pub fn average_difference(&self, other: &OptimaTensor_) -> f64 {
        let diff = self.elementwise_subtraction(other);
        let mut sum = 0.0;
        let mut count = 0.0;
        let vectorized_data = diff.vectorized_data();
        for d in vectorized_data {
            sum += *d;
            count += 1.0;
        }
        return sum / count;
    }

    pub fn dot(&self, other: &OptimaTensor_) -> OptimaTensor_ {
        if self.outer_dimensions() != other.inner_dimensions() {
            let error_str = format!("OptimaTensor dimensions are incompatible for dot (self.outer_dimensions: {:?} x other.inner_dimensions: {:?})", self.outer_dimensions(), other.inner_dimensions());
            panic!("{}", error_str);
        }

        let mut out = OptimaTensor_::new_from_inner_and_outer_dimensions(self.inner_dimensions.clone(), other.outer_dimensions.clone());

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
    pub fn dot_inplace(&mut self, other: &OptimaTensor_) {
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

    pub fn assign_to_inner_dimensions_tensor_from_outer_dimensions_vectorized_idx(&mut self, outer_dimensions_vectorized_idx: usize, inner_dimensions_tensor: &OptimaTensor_) {
        let outer_dimensions_indices = self.outer_dimensions_vectorized_idx_to_indices(outer_dimensions_vectorized_idx);
        self.assign_to_inner_dimensions_tensor_from_outer_dimensions_indices(outer_dimensions_indices, inner_dimensions_tensor);
    }
    pub fn assign_to_inner_dimensions_tensor_from_outer_dimensions_indices(&mut self, outer_dimensions_indices: Vec<usize>, inner_dimensions_tensor: &OptimaTensor_) {
        let c1 = Self::flatten_dimensions(inner_dimensions_tensor.combined_dimensions.clone());
        let c2 = Self::flatten_dimensions(self.inner_dimensions.clone());

        if c1 != c2 {
            panic!("Dimensions did not line up for assign_to_inner_dimensions_tensor_from_outer_dimensions_indices");
        }

        let inner_dimensions_tensor_vectorized = if let Some(array) = &inner_dimensions_tensor.tensor_2d {
            array.as_slice().unwrap().to_vec()
        } else if let Some(array) = &inner_dimensions_tensor.tensor_nd {
            array.as_slice().unwrap().to_vec()
        } else {
            unreachable!()
        };

        for (inner_dimensions_tensor_vectorized_idx, inner_dimensions_tensor_vectorized_value) in inner_dimensions_tensor_vectorized.iter().enumerate() {
            let inner_dimensions_indices = self.inner_dimensions_vectorized_idx_to_indices(inner_dimensions_tensor_vectorized_idx);
            let combined_indices = [inner_dimensions_indices, outer_dimensions_indices.clone()].concat();
            let combined_vectorized_idx = self.combined_indices_to_vectorized_idx(combined_indices);

            let combined_vectorized_data = if let Some(array) = &mut self.tensor_2d {
                array.as_slice_mut().unwrap()
            } else if let Some(array) = &mut self.tensor_nd {
                array.as_slice_mut().unwrap()
            } else {
                unreachable!()
            };

            combined_vectorized_data[combined_vectorized_idx] = *inner_dimensions_tensor_vectorized_value;
        }
    }

    #[allow(dead_code)]
    fn combined_vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        return self.vectorized_idx_to_indices(vectorized_idx, &self.combined_strides);
    }
    fn inner_dimensions_vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        return self.vectorized_idx_to_indices(vectorized_idx, &self.inner_dimensions_strides);
    }
    fn outer_dimensions_vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        return self.vectorized_idx_to_indices(vectorized_idx, &self.outer_dimensions_strides);
    }
    fn vectorized_idx_to_indices(&self, vectorized_idx: usize, strides: &Vec<usize>) -> Vec<usize> {
        let mut out_vec = vec![];
        let mut remainder = vectorized_idx;

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

    pub fn combined_indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        return self.indices_to_vectorized_idx(indices, &self.combined_strides);
    }
    pub fn inner_dimensions_indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        return self.indices_to_vectorized_idx(indices, &self.inner_dimensions_strides);
    }
    pub fn outer_dimensions_indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        return self.indices_to_vectorized_idx(indices, &self.outer_dimensions_strides);
    }
    fn indices_to_vectorized_idx(&self, indices: Vec<usize>, strides: &Vec<usize>) -> usize {
        let mut out = 0;

        for (i, idx) in indices.iter().enumerate() {
            out += strides[i] * *idx;
        }

        out
    }

    /// Takes out needless single unit dimensions.
    /// For example, will turn \[3,1,4,1] dimensions into \[3,4].
    fn flatten_dimensions(dimensions: Vec<usize>) -> Vec<usize> {
        if dimensions.len() == 1 { return dimensions.clone() }

        let mut out_vec = vec![];

        for dim in dimensions {
            if dim != 1 { out_vec.push(dim); }
        }

        if out_vec.is_empty() { out_vec.push(1); }

        out_vec
    }

    pub fn print_summary(&self) {
        if let Some(tensor) = &self.tensor_nd {
            println!("{}", tensor);
        }

        if let Some(tensor) = &self.tensor_2d {
            println!("{}", tensor);
        }
    }
}

#[derive(Clone, Debug)]
pub enum VectorType {
    Row, Column
}

pub struct OTFImmutVars {
    c: Box<dyn EnumSignatureContainer<OTFImmutVarsObject, OTFImmutVarsObjectSignature>>
}
impl OTFImmutVars {
    pub fn new(enum_signature_container_type: EnumSignatureContainerType) -> Self {
        Self {
            c: match enum_signature_container_type {
                EnumSignatureContainerType::BinarySearch => { Box::new(EnumBinarySearchSignatureContainer::new()) }
                EnumSignatureContainerType::HashMap => { Box::new(EnumHashMapSignatureContainer::new()) }
            }
        }
    }
    pub fn object_ref(&self, signature: &OTFImmutVarsObjectSignature) -> Option<&OTFImmutVarsObject> {
        self.c.object_ref(signature)
    }
    pub fn insert_or_replace(&mut self, object: OTFImmutVarsObject) {
        self.c.insert_or_replace_object(object);
    }
    pub fn insert_or_replace_get_robot_set<T: GetRobotSet + 'static>(&mut self, object: T) {
        self.insert_or_replace(OTFImmutVarsObject::GetRobotSet(Box::new(object)));
    }
}

pub enum OTFImmutVarsObject {
    Test,
    GetRobotSet(Box<dyn GetRobotSet>)
}
impl EnumMapToSignature<OTFImmutVarsObjectSignature> for OTFImmutVarsObject {
    fn map_to_signature(&self) -> OTFImmutVarsObjectSignature {
        match self {
            OTFImmutVarsObject::Test => { OTFImmutVarsObjectSignature::Test }
            OTFImmutVarsObject::GetRobotSet(_) => { OTFImmutVarsObjectSignature::GetRobotSet }
        }
    }
}
impl OTFImmutVarsObject {
    pub fn unwrap_get_robot_set(&self) -> &Box<dyn GetRobotSet> {
        match self {
            OTFImmutVarsObject::GetRobotSet(t) => { return t; }
            _ => { panic!("wrong type.") }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OTFImmutVarsObjectSignature {
    Test,
    GetRobotSet
}

/// var indices must be locked prior to getting keys
#[derive(Clone, Debug)]
pub struct OTFMutVars {
    enum_objects: Vec<OTFMutVarsObject>,
    signatures: Vec<OTFMutVarsObjectSignature>,
    vectorized_tensors: Vec<Vec<f64>>,
    registered_session_blocks: Vec<OTFMutVarsSessionBlock>,
    session_key_counter: usize
}
impl OTFMutVars {
    pub fn new() -> Self {
        Self {
            enum_objects: vec![],
            signatures: vec![],
            vectorized_tensors: vec![],
            registered_session_blocks: vec![],
            session_key_counter: 0
        }
    }
    pub fn register_session(&mut self, input: &OptimaTensor_) -> OTFMutVarsSessionKey {
        let out = OTFMutVarsSessionKey { key_idx: self.session_key_counter };
        let block = OTFMutVarsSessionBlock {
            key: out.clone(),
            input_tensor_id: input.id,
            already_gave_out_var_keys_this_session: false,
            registered_vars: vec![]
        };
        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(&out).unwrap());
        let idx = match binary_search_res {
            Ok(idx) => {idx}
            Err(idx) => {idx}
        };
        self.registered_session_blocks.insert(idx, block);
        self.session_key_counter += 1;
        return out;
    }
    pub fn close_session(&mut self, session_key: &OTFMutVarsSessionKey) {
        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(session_key).unwrap());
        match binary_search_res {
            Ok(idx) => { self.registered_session_blocks.remove(idx); }
            Err(_) => { panic!("Tried to close a session that does not exist.") }
        }
    }
    pub fn insert_base_variable(&mut self, object: OTFMutVarsObject, session_key: &OTFMutVarsSessionKey) {
        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(session_key).unwrap());
        let session_block_idx = match binary_search_res {
            Ok(idx) => {idx}
            Err(_) => { panic!("Tried to insert a var in a session that does not exist.") }
        };
        let session_block = &self.registered_session_blocks[session_block_idx];

        if session_block.already_gave_out_var_keys_this_session {
            panic!("Cannot insert a variable in a session that has already given out var keys.");
        }

        let signature = object.map_to_signature();
        let binary_search_res = self.signatures.binary_search_by(|x| x.partial_cmp(&signature).unwrap());
        match binary_search_res {
            Ok(idx) => {
                self.enum_objects[idx] = object;
            }
            Err(idx) => {
                self.enum_objects.insert(idx, object);
                self.signatures.insert(idx, signature);
                self.vectorized_tensors.insert(idx, vec![]);
            }
        };
    }
    pub fn register_vars(&mut self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, recompute_var_ifs: &Vec<RecomputeVarIf>, signatures: &Vec<OTFMutVarsObjectSignature>, session_key: &OTFMutVarsSessionKey) {
        if recompute_var_ifs.len() != signatures.len() {
            panic!("recompute_var_ifs len must equal signatures len");
        }

        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(session_key).unwrap());
        let session_block_idx = match binary_search_res {
            Ok(idx) => {idx}
            Err(_) => { panic!("Tried to register a var in a session that does not exist.") }
        };
        let session_block = &self.registered_session_blocks[session_block_idx];
        if session_block.already_gave_out_var_keys_this_session {
            panic!("Cannot register var in this session as it has already given out var keys.  All registers should be done before getting keys.");
        }

        if session_block.input_tensor_id != input.id {
            panic!("Tried to register a variable with an input tensor that does not match the given session.")
        }

        for (i, signature) in signatures.iter().enumerate() {
            let session_block = &self.registered_session_blocks[session_block_idx];
            let binary_search_res = session_block.registered_vars.binary_search_by(|x| x.cmp(signature));
            match binary_search_res {
                Ok(_) => { continue; }
                Err(_) => {
                    let binary_search_res = self.signatures.binary_search_by(|x| x.partial_cmp(signature).unwrap());
                    let (compute, idx) = match binary_search_res {
                        Ok(idx) => {
                            (recompute_var_ifs[i].recompute(input, &self.vectorized_tensors[idx], immut_vars), idx)
                        }
                        Err(idx) => {
                            (true, idx)
                        }
                    };

                    if compute {
                        let var_object = signature.compute_var(input, immut_vars, self);
                        self.signatures.insert(idx, signature.clone());
                        self.enum_objects.insert(idx, var_object);
                        self.vectorized_tensors.insert(idx, input.vectorized_data().to_vec());
                    }
                }
            }
            let session_block = &mut self.registered_session_blocks[session_block_idx];
            session_block.registered_vars.push(signature.clone());
        }
    }
    pub fn get_var_keys(&mut self, signatures: &Vec<OTFMutVarsObjectSignature>, input: &OptimaTensor_, session_key: &OTFMutVarsSessionKey) -> OTFMutVarsKeyContainer {
        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(session_key).unwrap());
        let session_block_idx = match binary_search_res {
            Ok(idx) => {idx}
            Err(_) => { panic!("Tried to register a var in a session that does not exist.") }
        };
        let session_block = &self.registered_session_blocks[session_block_idx];

        if session_block.input_tensor_id != input.id {
            panic!("Tried to get var keys with an input tensor that does not match the given session.")
        }

        let mut out_keys = OTFMutVarsKeyContainer { container: vec![] };

        for signature in signatures {
            if !session_block.registered_vars.contains(signature) {
                panic!("Tried to get var key from a variable that was not registered.");
            }
            let binary_search_res = self.signatures.binary_search_by(|x| x.partial_cmp(signature).unwrap());
            match binary_search_res {
                Ok(idx) => {
                    out_keys.container.push(OTFMutVarsKey { key_idx: idx });
                }
                Err(_) => { unreachable!("Should be unreachable since the variable should have been registered already.") }
            }
        }

        let session_block = &mut self.registered_session_blocks[session_block_idx];
        session_block.already_gave_out_var_keys_this_session = true;

        out_keys
    }
    pub fn unlock_object_mut_refs(&mut self, keys: &OTFMutVarsKeyContainer, input: &OptimaTensor_, session_key: &OTFMutVarsSessionKey) -> Vec<&mut OTFMutVarsObject> {
        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(session_key).unwrap());
        let session_block_idx = match binary_search_res {
            Ok(idx) => {idx}
            Err(_) => { panic!("Tried to register a var in a session that does not exist.") }
        };
        let session_block = &self.registered_session_blocks[session_block_idx];

        if session_block.input_tensor_id != input.id {
            panic!("Tried to get object refs with an input tensor that does not match the given session.")
        }

        let out = self.enum_objects.iter_mut()
            .enumerate()
            .filter(|(i, _)| keys.container.contains(&OTFMutVarsKey { key_idx: *i }))
            .map(|(_, o)| o)
            .collect();

        return out;
    }
}

#[derive(Clone, Debug)]
pub enum RecomputeVarIf {
    IsAnyNewInput,
    IsAnyNonFDPerturbationInput,
    InputInfNormIsGreaterThan(f64),
    Never
}
impl RecomputeVarIf {
    #[allow(unused_variables)]
    pub fn recompute(&self, input: &OptimaTensor_, previous_vectorized_tensor: &Vec<f64>, immut_vars: &OTFImmutVars) -> bool {
        match self {
            RecomputeVarIf::IsAnyNewInput => {
                let vectorized_tensor = input.vectorized_data();
                for (i, v) in vectorized_tensor.iter().enumerate() {
                    if *v != previous_vectorized_tensor[i] { return true; }
                }
                return false;
            }
            RecomputeVarIf::IsAnyNonFDPerturbationInput => {
                let vectorized_tensor = input.vectorized_data();
                for (i, v) in vectorized_tensor.iter().enumerate() {
                    let diff = (*v - previous_vectorized_tensor[i]).abs();
                    if diff > FD_PERTURBATION { return true; }
                }
                return false;
            }
            RecomputeVarIf::InputInfNormIsGreaterThan(val) => {
                let vectorized_tensor = input.vectorized_data();
                for (i, v) in vectorized_tensor.iter().enumerate() {
                    let diff = (*v - previous_vectorized_tensor[i]).abs();
                    if diff > *val { return true; }
                }
                return false;
            }
            RecomputeVarIf::Never => { return false; }
        }
    }
}

#[derive(Clone, Debug)]
pub enum OTFMutVarsObject {
    Test
}
impl EnumMapToSignature<OTFMutVarsObjectSignature> for OTFMutVarsObject {
    fn map_to_signature(&self) -> OTFMutVarsObjectSignature {
        match self {
            OTFMutVarsObject::Test => { OTFMutVarsObjectSignature::Test }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OTFMutVarsObjectSignature {
    Test
}
impl OTFMutVarsObjectSignature {
    pub fn compute_var(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> OTFMutVarsObject {
        let session_key = mut_vars.register_session(input);
        let out = self.compute_var_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        out
    }
    #[allow(unused_variables)]
    fn compute_var_raw(&self, input: &OptimaTensor_, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> OTFMutVarsObject {
        match self {
            OTFMutVarsObjectSignature::Test => { return OTFMutVarsObject::Test }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OTFMutVarsKey {
    key_idx: usize
}
#[derive(Clone, Debug)]
pub struct OTFMutVarsKeyContainer {
    container: Vec<OTFMutVarsKey>
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct OTFMutVarsSessionKey {
    key_idx: usize
}
#[derive(Clone, Debug)]
struct OTFMutVarsSessionBlock {
    key: OTFMutVarsSessionKey,
    input_tensor_id: i32,
    already_gave_out_var_keys_this_session: bool,
    registered_vars: Vec<OTFMutVarsObjectSignature>,
}


pub mod optima_tensor;

