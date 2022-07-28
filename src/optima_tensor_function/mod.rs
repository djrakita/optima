use std::ops::{Add, Div, Mul, Sub};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, ArrayD};
use serde::{Serialize, Deserialize};
use crate::optima_tensor_function::robotics_functions::RobotCollisionProximityBVHMode;
use crate::robot_set_modules::GetRobotSet;
use crate::robot_set_modules::robot_set::RobotSet;
use crate::robot_set_modules::robot_set_kinematics_module::{RobotSetFKDOFPerturbationsResult, RobotSetFKResult};
use crate::scenes::robot_geometric_shape_scene::{RobotGeometricShapeScene, RobotGeometricShapeSceneQuery};
use crate::utils::utils_console::{optima_print, optima_print_new_line, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_generic_data_structures::{AveragingFloat, EnumBinarySearchTypeContainer, EnumHashMapTypeContainer, EnumMapToType, EnumTypeContainer, EnumTypeContainerType, SimpleDataType, WindowMemoryContainer};
use crate::utils::utils_math::interpolation::{LinearInterpolationMode, SimpleInterpolationUtils};
use crate::utils::utils_robot::robot_generic_structures::{TimedGenericRobotJointStateWindowMemoryContainer};
use crate::utils::utils_robot::robot_set_link_specification::RobotLinkTransformGoalCollection;
use crate::utils::utils_sampling::SimpleSamplers;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
use crate::utils::utils_shape_geometry::geometric_shape::{BVHCombinableShapeAABB, LogCondition, StopCondition};
use crate::utils::utils_shape_geometry::shape_collection::{BVHVisit, ProximaEngine, ProximaSceneFilterOutput, ShapeCollectionBVH, SignedDistanceAggregator, WitnessPointsCollection};

pub trait OptimaTensorFunction: OptimaTensorFunctionClone {
    fn output_dimensions(&self) -> Vec<usize>;
    fn call(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.call_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError>;

    fn derivative(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        OptimaTensorFunctionGenerics::derivative_generic(self, Self::derivative_analytical, Self::derivative_finite_difference, Self::derivative_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative_none_mode(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative(input, immut_vars, mut_vars, None);
    }
    fn derivative_analytical(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return OptimaTensorFunctionGenerics::derivative_finite_difference_generic(self,
                                                                                  Self::call,
                                                                                  1,
                                                                                  input,
                                                                                  immut_vars,
                                                                                  mut_vars);
    }
    fn derivative_test(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    fn derivative2(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        OptimaTensorFunctionGenerics::derivative_generic(self, Self::derivative2_analytical, Self::derivative2_finite_difference, Self::derivative2_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative2_none_mode(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative2(input, immut_vars, mut_vars, None);
    }
    fn derivative2_analytical(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative2_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative2_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return OptimaTensorFunctionGenerics::derivative_finite_difference_generic(self,
                                                                                  Self::derivative_none_mode,
                                                                                  2,
                                                                                  input,
                                                                                  immut_vars,
                                                                                  mut_vars);
    }
    fn derivative2_test(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    fn derivative3(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        OptimaTensorFunctionGenerics::derivative_generic(self, Self::derivative3_analytical, Self::derivative3_finite_difference, Self::derivative3_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative3_none_mode(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative3(input, immut_vars, mut_vars, None);
    }
    fn derivative3_analytical(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative3_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative3_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return OptimaTensorFunctionGenerics::derivative_finite_difference_generic(self,
                                                                                  Self::derivative2_none_mode,
                                                                                  3,
                                                                                  input,
                                                                                  immut_vars,
                                                                                  mut_vars);
    }
    fn derivative3_test(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    fn derivative4(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {
        OptimaTensorFunctionGenerics::derivative_generic(self, Self::derivative4_analytical, Self::derivative4_finite_difference, Self::derivative4_test, input, immut_vars, mut_vars, mode)
    }
    fn derivative4_none_mode(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return self.derivative4(input, immut_vars, mut_vars, None);
    }
    fn derivative4_analytical(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let out = self.derivative4_analytical_raw(input, immut_vars, mut_vars, &session_key);
        mut_vars.close_session(&session_key);
        return out;
    }
    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }
    fn derivative4_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return OptimaTensorFunctionGenerics::derivative_finite_difference_generic(self, Self::derivative3_none_mode,
                                                                                  4,
                                                                                  input,
                                                                                  immut_vars,
                                                                                  mut_vars);
    }
    fn derivative4_test(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        Ok(OTFResult::Unimplemented)
    }

    /// derivative order of 0 here essentially means the "call" function output
    fn get_output_dimensions_from_derivative_order(&self, input: &OptimaTensor, derivative_order: usize) -> Vec<usize> {
        let input_dimensions = input.dimensions();
        let output_dimensions = self.output_dimensions();

        let mut combined_dimensions = output_dimensions.clone();
        for _ in 0..derivative_order {
            combined_dimensions = [combined_dimensions, input_dimensions.clone()].concat();
        }

        return combined_dimensions;
    }

    fn diagnostics(&self, input_sampling_mode: InputSamplingMode, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) {
        optima_print_new_line();
        optima_print("Call Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.call_diagnostics(input_sampling_mode.clone(), immut_vars, mut_vars, 1000);
        optima_print_new_line();
        optima_print("Derivative Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative_diagnostics(input_sampling_mode.clone(), immut_vars, mut_vars, 500);
        optima_print_new_line();
        optima_print("Derivative2 Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative2_diagnostics(input_sampling_mode.clone(), immut_vars, mut_vars, 50);
        optima_print_new_line();
        optima_print("Derivative3 Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative3_diagnostics(input_sampling_mode.clone(), immut_vars, mut_vars, 5);
        optima_print_new_line();
        optima_print("Derivative4 Diagnostics ---> ", PrintMode::Println, PrintColor::Blue, true);
        self.derivative4_diagnostics(input_sampling_mode.clone(), immut_vars, mut_vars, 5);
    }
    fn call_diagnostics(&self, input_sampling_mode: InputSamplingMode, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_calls: usize) {
        let mut call_time = AveragingFloat::new();

        let inputs = OptimaTensorFunctionGenerics::diagnostics_input_sampling(num_calls, input_sampling_mode);
        for input in inputs {
            let start = instant::Instant::now();
            self.call( &input, immut_vars, mut_vars).expect("error");
            let duration = start.elapsed();
            call_time.add_new_value(duration.as_secs_f64());
        }

        let call_time_as_duration = instant::Duration::from_secs_f64(call_time.value());

        optima_print(&format!("Call time over {} inputs is ", num_calls), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("{:?}", call_time_as_duration), PrintMode::Print, PrintColor::Green, true);
        optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);
    }
    fn derivative_diagnostics(&self, input_sampling_mode: InputSamplingMode, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        OptimaTensorFunctionGenerics::derivative_diagnostics_generic(self, Self::derivative, input_sampling_mode, immut_vars, mut_vars, num_inputs);
    }
    fn derivative2_diagnostics(&self, input_sampling_mode: InputSamplingMode, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        OptimaTensorFunctionGenerics::derivative_diagnostics_generic(self, Self::derivative2, input_sampling_mode, immut_vars, mut_vars, num_inputs);
    }
    fn derivative3_diagnostics(&self, input_sampling_mode: InputSamplingMode, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        OptimaTensorFunctionGenerics::derivative_diagnostics_generic(self, Self::derivative3, input_sampling_mode, immut_vars, mut_vars, num_inputs);
    }
    fn derivative4_diagnostics(&self, input_sampling_mode: InputSamplingMode, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_inputs: usize) {
        OptimaTensorFunctionGenerics::derivative_diagnostics_generic(self, Self::derivative4, input_sampling_mode, immut_vars, mut_vars, num_inputs);
    }

    fn empirical_convexity_or_concavity_check_via_hessian(&self, input_dimensions: Vec<usize>, num_checks: usize, c: ConvexOrConcave, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) {
        assert_eq!(self.output_dimensions().len(), 0, "can only check convexity or concavity with scalar functions.");

        for i in 0..num_checks {
            let r = OptimaTensor::new_random_sampling(OTFDimensions::Fixed(input_dimensions.clone()));
            let hessian_res = self.derivative2(&r, immut_vars, mut_vars, None).expect("error");
            let hessian = hessian_res.unwrap_tensor();
            let determinant = match hessian {
                OptimaTensor::Scalar(o) => { o.value[0] }
                OptimaTensor::Vector(_) => { panic!("wrong type.") }
                OptimaTensor::Matrix(o) => { o.matrix.determinant() }
                OptimaTensor::TensorND(_) => { panic!("wrong type.") }
            };
            let success = match &c {
                ConvexOrConcave::Convex => {
                    if determinant >= -0.000001 { true } else { false }
                }
                ConvexOrConcave::Concave => {
                    if determinant <= 0.000001 { true } else { false }
                }
            };

            match success {
                true => {
                    optima_print(&format!("Check {} was a success.  Determinant: {}", i, determinant), PrintMode::Println, PrintColor::Green, true);
                }
                false => {
                    match &c {
                        ConvexOrConcave::Convex => {
                            optima_print(&format!("Function is not convex.  Determinant: {}", determinant), PrintMode::Println, PrintColor::Red, true);
                        }
                        ConvexOrConcave::Concave => {
                            optima_print(&format!("Function is not concave.  Determinant: {}", determinant), PrintMode::Println, PrintColor::Red, true);
                        }
                    }
                    return;
                }
            }
        }

        match &c {
                ConvexOrConcave::Convex => {
                    optima_print(&format!("Based on this empirical test, the function appears to be convex."), PrintMode::Println, PrintColor::Green, true);
                }
                ConvexOrConcave::Concave => {
                    optima_print(&format!("Based on this empirical test, the function appears to be concave."), PrintMode::Println, PrintColor::Green, true);
                }
            }
    }
}
pub struct OptimaTensorFunctionGenerics;
impl OptimaTensorFunctionGenerics {
    pub fn derivative_generic<S: ?Sized, F1, F2, F3>(s: &S,
                                         analytical: F1,
                                         finite_difference: F2,
                                         test: F3,
                                         input: &OptimaTensor,
                                         immut_vars: &OTFImmutVars,
                                         mut_vars: &mut OTFMutVars,
                                         mode: Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError>
        where S: OptimaTensorFunction,
              F1: Fn(&S, &OptimaTensor, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError>,
              F2: Fn(&S, &OptimaTensor, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError>,
              F3: Fn(&S, &OptimaTensor, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError>{
        match mode {
            None => {
                {
                    let res = analytical(s, input, immut_vars, mut_vars)?;
                    match res {
                        OTFResult::Unimplemented => { }
                        OTFResult::Complete(_) => { return Ok(res); }
                    }
                }

                {
                    let res = finite_difference(s, input, immut_vars, mut_vars)?;
                    match res {
                        OTFResult::Unimplemented => { }
                        OTFResult::Complete(_) => { return Ok(res); }
                    }
                }

                panic!("Called an Unimplemented Derivative on OTF.")

            }
            Some(mode) => {
                let res = match mode {
                    OTFDerivativeMode::Analytical => { analytical(s, input, immut_vars, mut_vars) }
                    OTFDerivativeMode::FiniteDifference => { finite_difference(s, input, immut_vars, mut_vars) }
                    OTFDerivativeMode::Test => { test(s, input, immut_vars, mut_vars) }
                }?;
                return Ok(res);
            }
        }
    }
    pub fn derivative_finite_difference_generic<S: ?Sized, F>(s: &S, caller: F, derivative_order: usize, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError>
        where S: OptimaTensorFunction,
              F: Fn(&S, &OptimaTensor, &OTFImmutVars, &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        assert!(derivative_order > 0);

        let input_vectorized_data = input.vectorized_data();

        let output_dimensions = s.get_output_dimensions_from_derivative_order(input, derivative_order);
        let slice_dimensions = s.get_output_dimensions_from_derivative_order(input, derivative_order - 1);

        let mut output = OptimaTensor::new_zeros(output_dimensions);

        let f_0_res = caller(s, input, immut_vars, mut_vars)?;
        let f_0 = f_0_res.unwrap_tensor();

        for (vectorized_input_idx, _) in input_vectorized_data.iter().enumerate() {
            let mut input_clone = input.clone();
            input_clone.vectorized_data_mut()[vectorized_input_idx] += FD_PERTURBATION;

            let f_h_result = caller(s, &input_clone, immut_vars, mut_vars)?;
            let f_h = f_h_result.unwrap_tensor();

            let mut f_d = f_h.elementwise_subtraction(&f_0);
            f_d.scalar_division(FD_PERTURBATION);

            let input_indices = input.vectorized_idx_to_indices(vectorized_input_idx);

            let mut slice_scope = vec![];
            for _ in &slice_dimensions { slice_scope.push(OptimaTensorSliceScope::Free); }
            for i in input_indices { slice_scope.push(OptimaTensorSliceScope::Fixed(i)); }

            output.insert_slice(slice_scope, f_d);
        }

        return Ok(OTFResult::Complete(output));
    }
    pub fn derivative_diagnostics_generic<S: ?Sized, F>(s: &S, derivative_function: F, input_sampling_mode: InputSamplingMode, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, num_calls: usize)
        where S: OptimaTensorFunction,
              F: Fn(&S, &OptimaTensor, &OTFImmutVars, &mut OTFMutVars, Option<OTFDerivativeMode>) -> Result<OTFResult, OptimaError> {

        let mut rand_inputs = Self::diagnostics_input_sampling(num_calls, input_sampling_mode);

        let mut finite_difference_time = AveragingFloat::new();
        let mut finite_difference_results = vec![];
        for i in 0..num_calls {
            let start = instant::Instant::now();
            let res = derivative_function(s, &rand_inputs[i], immut_vars, mut_vars, Some(OTFDerivativeMode::FiniteDifference)).expect("error");
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
        for i in 0..num_calls {
            let start = instant::Instant::now();
            let res = derivative_function(s, &rand_inputs[i], immut_vars, mut_vars, Some(OTFDerivativeMode::Analytical)).expect("error");
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
        for i in 0..num_calls {
            let start = instant::Instant::now();
            let res = derivative_function(s, &rand_inputs[i], immut_vars, mut_vars, Some(OTFDerivativeMode::Test)).expect("error");
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

        optima_print(&format!("Finite Difference time over {} inputs is ", num_calls), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("{:?}", finite_difference_time_as_duration), PrintMode::Print, PrintColor::Green, true);
        optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);

        if !analytical_unimplemented {
            optima_print(&format!("Analytical time over {} inputs is ", num_calls), PrintMode::Print, PrintColor::None, false);
            optima_print(&format!("{:?}", analytical_time_as_duration), PrintMode::Print, PrintColor::Green, true);
            optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);
        }
        if !test_unimplemented {
            optima_print(&format!("Test time over {} inputs is ", num_calls), PrintMode::Print, PrintColor::None, false);
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
            optima_print(&format!("Analytical difference from Finite Difference over {} inputs is ", num_calls), PrintMode::Print, PrintColor::None, false);
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
            optima_print(&format!("Test difference from Finite Difference over {} inputs is ", num_calls), PrintMode::Print, PrintColor::None, false);
            optima_print(&format!("{:?}", test_diffs_value), PrintMode::Print, color, true);
            optima_print(&format!(" on average.\n"), PrintMode::Print, PrintColor::None, false);
        }
    }
    pub fn diagnostics_input_sampling(num_inputs: usize, input_sampling_mode: InputSamplingMode) -> Vec<OptimaTensor> {
        let mut out_vec = vec![];
        match input_sampling_mode {
            InputSamplingMode::SameInput { input } => {
                for _ in 0..num_inputs { out_vec.push( input.clone() ) }
            }
            InputSamplingMode::UniformRandom { input_dimensions } => {
                for _ in 0..num_inputs { out_vec.push( OptimaTensor::new_random_sampling(OTFDimensions::Fixed(input_dimensions.clone())) ) }
            }
            InputSamplingMode::RandomWalk { input_dimensions, step_size } => {
                let mut curr_state = OptimaTensor::new_random_sampling(OTFDimensions::Fixed(input_dimensions.clone()));
                for _ in 0..num_inputs {
                    out_vec.push(curr_state.clone());
                    let rand_push = OptimaTensor::new_random_sampling(OTFDimensions::Fixed(input_dimensions.clone()));
                    curr_state = curr_state + step_size * rand_push;
                }
            }
            InputSamplingMode::RandomWalkFromStartPoint { start_point, step_size } => {
                let input_dimensions = start_point.dimensions();
                let mut curr_state = start_point.clone();
                for _ in 0..num_inputs {
                    out_vec.push(curr_state.clone());
                    let rand_push = OptimaTensor::new_random_sampling(OTFDimensions::Fixed(input_dimensions.clone()));
                    curr_state = curr_state + step_size * rand_push;
                }
            }
        }
        out_vec
    }
}

pub trait OptimaTensorFunctionClone {
    fn clone_box(&self) -> Box<dyn OptimaTensorFunction>;
}
impl<T> OptimaTensorFunctionClone for T where T: 'static + OptimaTensorFunction + Clone {
    fn clone_box(&self) -> Box<dyn OptimaTensorFunction> {
        Box::new(self.clone())
    }
}
impl Clone for Box<dyn OptimaTensorFunction> {
    fn clone(&self) -> Box<dyn OptimaTensorFunction> {
        self.clone_box()
    }
}

pub const FD_PERTURBATION: f64 = 0.000001;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OTFDimensions {
    Any,
    PerDimension(Vec<OTFDimension>),
    Fixed(Vec<usize>)
}
impl OTFDimensions {
    pub fn num_dimensions(&self) -> usize {
        match self {
            OTFDimensions::Any => { panic!("cannot get number of dimensions from Any.") }
            OTFDimensions::PerDimension(d) => { d.len() }
            OTFDimensions::Fixed(d) => { d.len() }
        }
    }
    pub fn total_num_elements(&self) -> usize {
        match self {
            OTFDimensions::Any => { panic!("cannot get number of elements from Any.") }
            OTFDimensions::PerDimension(_) => { panic!("cannot get number of elements from PerDimension.")  }
            OTFDimensions::Fixed(d) => {
                let mut out = 1;

                for dd in d { out *= *dd; }

                return out;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OTFDimension {
    Any,
    Fixed(usize)
}

#[derive(Clone, Debug)]
pub enum OTFResult {
    Unimplemented, Complete(OptimaTensor)
}
impl OTFResult {
    pub fn unwrap_tensor(&self) -> &OptimaTensor {
        match self {
            OTFResult::Unimplemented => { unimplemented!() }
            OTFResult::Complete(t) => { return t; }
        }
    }
    pub fn unwrap_tensor_mut(&mut self) -> &mut OptimaTensor {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConvexOrConcave {
    Convex,
    Concave
}

#[derive(Clone, Debug)]
pub enum InputSamplingMode<'a> {
    SameInput { input: &'a OptimaTensor },
    UniformRandom { input_dimensions: Vec<usize> },
    RandomWalk { input_dimensions: Vec<usize>, step_size: f64 },
    RandomWalkFromStartPoint { start_point: &'a OptimaTensor, step_size: f64 }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub enum OptimaTensor {
    Scalar(OptimaTensor0D),
    Vector(OptimaTensor1D),
    Matrix(OptimaTensor2D),
    TensorND(OptimaTensorND)
}
impl OptimaTensor {
    pub fn new_from_scalar(value: f64) -> Self {
        Self::Scalar(OptimaTensor0D::new(value))
    }
    pub fn new_from_vector(vector: DVector<f64>) -> Self {
        let mut out = Self::Vector(OptimaTensor1D::new(vector));
        out.collapse_dimensions();
        out
    }
    pub fn new_from_matrix(matrix: DMatrix<f64>) -> Self {
        let mut out = Self::Matrix(OptimaTensor2D::new(matrix));
        out.collapse_dimensions();
        out
    }
    pub fn new_from_tensor(tensor: ArrayD<f64>) -> Self {
        let mut out = Self::TensorND(OptimaTensorND::new(tensor));
        out.collapse_dimensions();
        out
    }
    pub fn new_from_single_array(arr: &[f64]) -> Self {
        let l = arr.len();
        assert!(l > 0);
        return if l == 1 {
            Self::new_from_scalar(arr[0])
        } else {
            let vector = DVector::from_column_slice(arr);
            Self::new_from_vector(vector)
        }
    }
    pub fn new_zeros(dimensions: Vec<usize>) -> Self {
        let l = dimensions.len();
        let mut out = if l == 0 {
            Self::Scalar(OptimaTensor0D::new_zero())
        } else if l == 1 {
            Self::Vector(OptimaTensor1D::new_zeros(dimensions[0]))
        } else if l == 2 {
            Self::Matrix(OptimaTensor2D::new_zeros(dimensions[0], dimensions[1]))
        } else {
            Self::TensorND(OptimaTensorND::new_zeros(dimensions))
        };
        out.collapse_dimensions();
        return out;
    }
    pub fn new_random_sampling(dimensions: OTFDimensions) -> Self {
        let dimensions = match dimensions {
            OTFDimensions::Any => {
                let mut dimensions = vec![];
                let num_dimensions = SimpleSamplers::uniform_samples_i32(&vec![(1,5)])[0] as usize;
                for _ in 0..num_dimensions {
                    dimensions.push( SimpleSamplers::uniform_samples_i32(&vec![(1,7)])[0] as usize );
                }
                dimensions
            }
            OTFDimensions::PerDimension(d) => {
                let mut dimensions = vec![];

                for dd in d {
                    match dd {
                        OTFDimension::Any => { dimensions.push( SimpleSamplers::uniform_samples_i32(&vec![(1,7)])[0] as usize ); }
                        OTFDimension::Fixed(a) => { dimensions.push(a); }
                    }
                }

                dimensions
            }
            OTFDimensions::Fixed(d) => { d }
        };

        let mut out = Self::new_zeros(dimensions);
        let v = out.vectorized_data_mut();
        for vv in v {
            *vv = SimpleSamplers::uniform_sample((-5.0,5.0));
        }
        out
    }
    pub fn unwrap_scalar(&self) -> f64 {
        match self {
            OptimaTensor::Scalar(t) => { t.value[0] }
            _ => {
                panic!("wrong type.")
            }
        }
    }
    pub fn unwrap_vector(&self) -> &DVector<f64> {
        match self {
            OptimaTensor::Vector(t) => { &t.vector }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_matrix(&self) -> &DMatrix<f64> {
        match self {
            OptimaTensor::Matrix(t) => { &t.matrix }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_tensor_nd(&self) -> &ArrayD<f64> {
        match self {
            OptimaTensor::TensorND(t) => { &t.tensor }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn collapse_dimensions(&mut self) {
        let dimensions = self.dimensions();
        let mut true_dimensions = vec![];
        for d in &dimensions {
            if *d > 1 { true_dimensions.push(*d); }
        }

        if dimensions.len() == true_dimensions.len() { return; }

        let mut new_self = Self::new_zeros(true_dimensions);
        copy_vectorized_data(self.vectorized_data(), new_self.vectorized_data_mut());

        *self = new_self;
    }
    pub fn spawn_new_with_new_dimensions(&self, new_added_dimensions: Vec<usize>) -> OptimaTensor {
        let mut dimensions = self.dimensions();
        for d in new_added_dimensions {
            dimensions.push(d);
        }
        return Self::new_zeros(dimensions);
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
    pub fn linearly_interpolate(&self, other: &OptimaTensor, mode: &LinearInterpolationMode) -> Vec<OptimaTensor> {
        linearly_interpolate_optima_tensors(self, other, mode)
    }
    pub fn dot(&self, other: &OptimaTensor) -> OptimaTensor {
        let mut out = match self {
            OptimaTensor::Scalar(t) => { t.dot(other) }
            OptimaTensor::Vector(t) => { t.dot(other) }
            OptimaTensor::Matrix(t) => { t.dot(other) }
            OptimaTensor::TensorND(t) => { t.dot(other) }
        };
        out.collapse_dimensions();
        out
    }
    pub fn elementwise_addition(&self, other: &OptimaTensor) -> OptimaTensor {
        match self {
            OptimaTensor::Scalar(t) => { t.elementwise_addition(other) }
            OptimaTensor::Vector(t) => { t.elementwise_addition(other) }
            OptimaTensor::Matrix(t) => { t.elementwise_addition(other) }
            OptimaTensor::TensorND(t) => { t.elementwise_addition(other) }
        }
    }
    pub fn elementwise_subtraction(&self, other: &OptimaTensor) -> OptimaTensor {
        match self {
            OptimaTensor::Scalar(t) => { t.elementwise_subtraction(other) }
            OptimaTensor::Vector(t) => { t.elementwise_subtraction(other) }
            OptimaTensor::Matrix(t) => { t.elementwise_subtraction(other) }
            OptimaTensor::TensorND(t) => { t.elementwise_subtraction(other) }
        }
    }
    pub fn elementwise_multiplication(&self, other: &OptimaTensor) -> OptimaTensor {
        match self {
            OptimaTensor::Scalar(t) => { t.elementwise_multiplication(other) }
            OptimaTensor::Vector(t) => { t.elementwise_multiplication(other) }
            OptimaTensor::Matrix(t) => { t.elementwise_multiplication(other) }
            OptimaTensor::TensorND(t) => { t.elementwise_multiplication(other) }
        }
    }
    pub fn elementwise_division(&self, other: &OptimaTensor) -> OptimaTensor {
        match self {
            OptimaTensor::Scalar(t) => { t.elementwise_division(other) }
            OptimaTensor::Vector(t) => { t.elementwise_division(other) }
            OptimaTensor::Matrix(t) => { t.elementwise_division(other) }
            OptimaTensor::TensorND(t) => { t.elementwise_division(other) }
        }
    }
    pub fn scalar_multiplication(&mut self, value: f64) {
        let v = self.vectorized_data_mut();
        for vv in v { *vv *= value; }
    }
    pub fn scalar_addition(&mut self, value: f64) {
        let v = self.vectorized_data_mut();
        for vv in v { *vv += value; }
    }
    pub fn scalar_subtraction(&mut self, value: f64) {
        let v = self.vectorized_data_mut();
        for vv in v { *vv -= value; }
    }
    pub fn scalar_division(&mut self, value: f64) {
        let v = self.vectorized_data_mut();
        for vv in v { *vv /= value; }
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
    pub fn insert_slice(&mut self, slice_scope: Vec<OptimaTensorSliceScope>, slice: OptimaTensor) {
        let self_dimensions = self.dimensions();

        if self_dimensions.len() != slice_scope.len() {
            panic!("slice scope must be same length as dimensions.");
        }

        let mut combined_indices = vec![];

        let mut free_indices = vec![];
        let mut free_vectorized_len = 1;
        for (i, s) in slice_scope.iter().enumerate() {
            match s {
                OptimaTensorSliceScope::Free => {
                    free_vectorized_len *= self_dimensions[i];
                    free_indices.push(i);
                    combined_indices.push(0);
                }
                OptimaTensorSliceScope::Fixed(idx) => {
                    combined_indices.push(*idx);
                }
            }
        }

        let slice_vectorized_data = slice.vectorized_data();
        if free_vectorized_len != slice_vectorized_data.len() {
            panic!("slice data is incorrect length. {:?}, {:?}", free_vectorized_len, slice_vectorized_data.len())
        }

        for (i, s) in slice_vectorized_data.iter().enumerate() {
            let slice_indices = slice.vectorized_idx_to_indices(i);
            for (j, slice_idx) in slice_indices.iter().enumerate() {
                combined_indices[free_indices[j]] = *slice_idx;
            }
            self.set_value(combined_indices.clone(), *s);
        }

    }
    pub fn vectorized_data(&self) -> &[f64] {
        match self {
            OptimaTensor::Scalar(t) => { t.vectorized_data() }
            OptimaTensor::Vector(t) => { t.vectorized_data() }
            OptimaTensor::Matrix(t) => { t.vectorized_data() }
            OptimaTensor::TensorND(t) => { t.vectorized_data() }
        }
    }
    pub fn average_difference(&self, other: &OptimaTensor) -> f64 {
        let self_vectorized_data = self.vectorized_data();
        let other_vectorized_data = other.vectorized_data();

        let l1 = self_vectorized_data.len();
        let l2 = other_vectorized_data.len();

        assert_eq!(l1, l2);

        let mut a = AveragingFloat::new();

        for i in 0..l1 {
            a.add_new_value( self_vectorized_data[i] - other_vectorized_data[i] )
        }

        a.value()
    }
    pub fn id(&self) -> f64 {
        match self {
            OptimaTensor::Scalar(t) => { t.id }
            OptimaTensor::Vector(t) => { t.id }
            OptimaTensor::Matrix(t) => { t.id }
            OptimaTensor::TensorND(t) => { t.id }
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
    #[allow(dead_code)]
    fn indices_to_vectorized_idx(&self, indices: Vec<usize>) -> usize {
        return match self {
            OptimaTensor::Scalar(t) => { t.indices_to_vectorized_idx(indices) }
            OptimaTensor::Vector(t) => { t.indices_to_vectorized_idx(indices) }
            OptimaTensor::Matrix(t) => { t.indices_to_vectorized_idx(indices) }
            OptimaTensor::TensorND(t) => { t.indices_to_vectorized_idx(indices) }
        }
    }
}
impl EnumMapToType<OptimaTensorSignature> for OptimaTensor {
    fn map_to_type(&self) -> OptimaTensorSignature {
        match self {
            OptimaTensor::Scalar(_) => { OptimaTensorSignature::Scalar }
            OptimaTensor::Vector(_) => { OptimaTensorSignature::Vector }
            OptimaTensor::Matrix(_) => { OptimaTensorSignature::Matrix }
            OptimaTensor::TensorND(_) => { OptimaTensorSignature::TensorND }
        }
    }
}
impl Add for OptimaTensor {
    type Output = OptimaTensor;

    fn add(self, rhs: Self) -> Self::Output {
        self.elementwise_addition(&rhs)
    }
}
impl Add<&OptimaTensor> for &OptimaTensor {
    type Output = OptimaTensor;

    fn add(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_addition(rhs)
    }
}
impl Add<&OptimaTensor> for OptimaTensor {
    type Output = OptimaTensor;

    fn add(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_addition(rhs)
    }
}
impl Sub for OptimaTensor {
    type Output = OptimaTensor;

    fn sub(self, rhs: Self) -> Self::Output {
        self.elementwise_subtraction(&rhs)
    }
}
impl Sub<&OptimaTensor> for &OptimaTensor {
    type Output = OptimaTensor;

    fn sub(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_subtraction(rhs)
    }
}
impl Sub<&OptimaTensor> for OptimaTensor {
    type Output = OptimaTensor;

    fn sub(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_subtraction(rhs)
    }
}
impl Mul for OptimaTensor {
    type Output = OptimaTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        self.elementwise_multiplication(&rhs)
    }
}
impl Mul<&OptimaTensor> for &OptimaTensor {
    type Output = OptimaTensor;

    fn mul(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_multiplication(rhs)
    }
}
impl Mul<&OptimaTensor> for OptimaTensor {
    type Output = OptimaTensor;

    fn mul(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_multiplication(rhs)
    }
}
impl Div for OptimaTensor {
    type Output = OptimaTensor;

    fn div(self, rhs: Self) -> Self::Output {
        self.elementwise_division(&rhs)
    }
}
impl Div<&OptimaTensor> for &OptimaTensor {
    type Output = OptimaTensor;

    fn div(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_division(rhs)
    }
}
impl Div<&OptimaTensor> for OptimaTensor {
    type Output = OptimaTensor;

    fn div(self, rhs: &OptimaTensor) -> Self::Output {
        self.elementwise_division(rhs)
    }
}
impl Add<OptimaTensor> for f64 {
    type Output = OptimaTensor;

    fn add(self, rhs: OptimaTensor) -> Self::Output {
        let mut out = rhs.clone();
        out.scalar_addition(self);
        return out;
    }
}
impl Mul<OptimaTensor> for f64 {
    type Output = OptimaTensor;

    fn mul(self, rhs: OptimaTensor) -> Self::Output {
        let mut out = rhs.clone();
        out.scalar_multiplication(self);
        return out;
    }
}
impl Add<&OptimaTensor> for f64 {
    type Output = OptimaTensor;

    fn add(self, rhs: &OptimaTensor) -> Self::Output {
        let mut out = rhs.clone();
        out.scalar_addition(self);
        return out;
    }
}
impl Mul<&OptimaTensor> for f64 {
    type Output = OptimaTensor;

    fn mul(self, rhs: &OptimaTensor) -> Self::Output {
        let mut out = rhs.clone();
        out.scalar_multiplication(self);
        return out;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptimaTensorSignature {
    Scalar,
    Vector,
    Matrix,
    TensorND
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimaTensor0D {
    value: [f64; 1],
    id: f64
}
impl OptimaTensor0D {
    pub fn new_zero() -> Self {
        Self::new(0.0)
    }
    pub fn new(value: f64) -> Self {
        Self {
            value: [value],
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
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
    pub fn dot(&self, other: &OptimaTensor) -> OptimaTensor {
        return match other {
            OptimaTensor::Scalar(t) => {
                let val = self.value[0] * t.value[0];
                OptimaTensor::new_from_scalar(val)
            }
            OptimaTensor::Vector(t) => {
                let out = t.vector.clone() * self.value[0];
                OptimaTensor::new_from_vector(out)
            }
            OptimaTensor::Matrix(t) => {
                let out = t.matrix.clone() * self.value[0];
                OptimaTensor::new_from_matrix(out)
            }
            OptimaTensor::TensorND(t) => {
                let out = t.tensor.clone() * self.value[0];
                OptimaTensor::new_from_tensor(out)
            }
        }
    }
    pub fn elementwise_addition(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Scalar(v) => {
                let out_val = &self.value[0] + &v.value[0];
                OptimaTensor::new_from_scalar(out_val)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_addition(other)
            }
        }
    }
    pub fn elementwise_subtraction(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Scalar(v) => {
                let out_val = &self.value[0] - &v.value[0];
                OptimaTensor::new_from_scalar(out_val)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_subtraction(other)
            }
        }
    }
    pub fn elementwise_multiplication(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Scalar(v) => {
                let out_val = &self.value[0] * &v.value[0];
                OptimaTensor::new_from_scalar(out_val)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_multiplication(other)
            }
        }
    }
    pub fn elementwise_division(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Scalar(v) => {
                let out_val = &self.value[0] / &v.value[0];
                OptimaTensor::new_from_scalar(out_val)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_division(other)
            }
        }
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
impl Clone for OptimaTensor0D {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimaTensor1D {
    vector: DVector<f64>,
    id: f64
}
impl OptimaTensor1D {
    pub fn new_zeros(len: usize) -> Self {
        Self::new(DVector::zeros(len))
    }
    pub fn new(vector: DVector<f64>) -> Self {
        Self {
            vector,
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
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
    pub fn dot(&self, other: &OptimaTensor) -> OptimaTensor {
        return match other {
            OptimaTensor::Vector(t) => {
                let out_sum = self.vector.dot(&t.vector);
                OptimaTensor::new_from_scalar(out_sum)
            }
            OptimaTensor::Scalar(t) => {
                let out_vector = t.value[0] * self.vector.clone();
                OptimaTensor::new_from_vector(out_vector)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.dot(other)
            }
        }
    }
    pub fn elementwise_addition(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Vector(v) => {
                let out_vec = &self.vector + &v.vector;
                OptimaTensor::new_from_vector(out_vec)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_addition(other)
            }
        }
    }
    pub fn elementwise_subtraction(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Vector(v) => {
                let out_vec = &self.vector - &v.vector;
                OptimaTensor::new_from_vector(out_vec)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_subtraction(other)
            }
        }
    }
    pub fn elementwise_multiplication(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Vector(v) => {
                assert_eq!(self.vectorized_data().len(), other.vectorized_data().len());
                let out_vec: Vec<f64> = self.vectorized_data()
                    .iter()
                    .zip(other.vectorized_data().iter())
                    .map(|(a, b)| *a * *b)
                    .collect();
                OptimaTensor::new_from_vector(DVector::from_vec(out_vec))
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_multiplication(other)
            }
        }
    }
    pub fn elementwise_division(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Vector(v) => {
                assert_eq!(self.vectorized_data().len(), other.vectorized_data().len());
                let out_vec: Vec<f64> = self.vectorized_data()
                    .iter()
                    .zip(other.vectorized_data().iter())
                    .map(|(a, b)| *a / *b)
                    .collect();
                OptimaTensor::new_from_vector(DVector::from_vec(out_vec))
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_division(other)
            }
        }
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
impl Clone for OptimaTensor1D {
    fn clone(&self) -> Self {
        Self {
            vector: self.vector.clone(),
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimaTensor2D {
    matrix: DMatrix<f64>,
    id: f64
}
impl OptimaTensor2D {
    pub fn new_zeros(nrows: usize, ncols: usize) -> Self {
        let mat = DMatrix::zeros(nrows, ncols);
        return Self::new(mat);
    }
    pub fn new(matrix: DMatrix<f64>) -> Self {
        Self {
            matrix,
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
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
                let mut num_non_one = 0;
                for d in dimensions {
                    if d != 1 { num_non_one += 1 }
                }
                if num_non_one > 1 {
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
    pub fn dot(&self, other: &OptimaTensor) -> OptimaTensor {
        return match other {
            OptimaTensor::Matrix(t) => {
                let out_mat = if self.matrix.ncols() == t.matrix.nrows() {
                    &self.matrix * &t.matrix
                } else if self.matrix.nrows() == t.matrix.nrows() {
                    &self.matrix.transpose() * &t.matrix
                } else if self.matrix.ncols() == t.matrix.ncols() {
                    &self.matrix * &t.matrix.transpose()
                } else {
                    panic!("dimensions do not align.")
                };

                OptimaTensor::new_from_matrix(out_mat)
            }
            OptimaTensor::Scalar(t) => {
                let out = self.matrix.clone() * t.value[0];
                OptimaTensor::new_from_matrix(out)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.dot(other)
            }
        }
    }
    pub fn elementwise_addition(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Matrix(v) => {
                let out_mat = &self.matrix + &v.matrix;
                OptimaTensor::new_from_matrix(out_mat)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_addition(other)
            }
        }
    }
    pub fn elementwise_subtraction(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Matrix(v) => {
                let out_mat = &self.matrix - &v.matrix;
                OptimaTensor::new_from_matrix(out_mat)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_subtraction(other)
            }
        }
    }
    pub fn elementwise_multiplication(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Matrix(v) => {
                let v1 = self.vectorized_data();
                let v2 = v.vectorized_data();
                assert_eq!(v1.len(), v2.len());
                let mut out = self.clone();
                let mut out_vec = out.vectorized_data_mut();
                v1.iter().zip(v2.iter()).enumerate().for_each(|(idx, (a, b))| out_vec[idx] = *a * *b);
                OptimaTensor::Matrix(out)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_multiplication(other)
            }
        }
    }
    pub fn elementwise_division(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::Matrix(v) => {
                let v1 = self.vectorized_data();
                let v2 = v.vectorized_data();
                assert_eq!(v1.len(), v2.len());
                let mut out = self.clone();
                let mut out_vec = out.vectorized_data_mut();
                v1.iter().zip(v2.iter()).enumerate().for_each(|(idx, (a, b))| out_vec[idx] = *a / *b);
                OptimaTensor::Matrix(out)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_division(other)
            }
        }
    }
    fn vectorized_data(&self) -> &[f64] { self.matrix.as_slice() }
    fn vectorized_data_mut(&mut self) -> &mut [f64] { self.matrix.as_mut_slice() }
    fn vectorized_idx_to_indices(&self, vectorized_idx: usize) -> Vec<usize> {
        if vectorized_idx >= self.vectorized_data().len() {
            panic!("invalid vectorized idx.");
        }

        let mut out_vec = vec![];
        let mut remainder = vectorized_idx;

        let strides = vec![self.matrix.ncols(), 1];

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
impl Clone for OptimaTensor2D {
    fn clone(&self) -> Self {
        Self {
            matrix: self.matrix.clone(),
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
        }
    }
}

#[derive(Debug)]
pub struct OptimaTensorND {
    tensor: ArrayD<f64>,
    id: f64
}
impl OptimaTensorND {
    pub fn new_zeros(dimensions: Vec<usize>) -> Self {
        let tensor = Array::<f64, _>::zeros(dimensions);
        return Self::new(tensor);
    }
    pub fn new(tensor: ArrayD<f64>) -> Self {
        Self {
            tensor,
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
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
                let mut num_non_one = 0;
                for d in dimensions {
                    if d != 1 { num_non_one += 1 }
                }
                if num_non_one > 1 {
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
            self.cell_mut_refn(indices)
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
            self.cell_refn(indices)
        };

        return *val;
    }
    pub fn dot(&self, other: &OptimaTensor) -> OptimaTensor {
        return match other {
            OptimaTensor::TensorND(_) => {
                todo!("This hasn't been implemented yet.  If you encounter this error, it's probably time to implement it.")
                // For reference: let res = tensordot(&arr, &arr2, &vec![Axis(1)], &vec![Axis(0)]);
            }
            OptimaTensor::Scalar(t) => {
                let out = self.tensor.clone() * t.value[0];
                OptimaTensor::new_from_tensor(out)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.dot(other)
            }
        }
    }
    pub fn elementwise_addition(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::TensorND(v) => {
                let out_tensor = &self.tensor + &v.tensor;
                OptimaTensor::new_from_tensor(out_tensor)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_addition(other)
            }
        }
    }
    pub fn elementwise_subtraction(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::TensorND(v) => {
                let out_tensor = &self.tensor - &v.tensor;
                OptimaTensor::new_from_tensor(out_tensor)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_subtraction(other)
            }
        }
    }
    pub fn elementwise_multiplication(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::TensorND(v) => {
                let v1 = self.vectorized_data();
                let v2 = v.vectorized_data();
                assert_eq!(v1.len(), v2.len());
                let mut out = self.clone();
                let mut out_vec = out.vectorized_data_mut();
                v1.iter().zip(v2.iter()).enumerate().for_each(|(idx, (a, b))| out_vec[idx] = *a * *b);
                OptimaTensor::TensorND(out)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_multiplication(other)
            }
        }
    }
    pub fn elementwise_division(&self, other: &OptimaTensor) -> OptimaTensor {
        match other {
            OptimaTensor::TensorND(v) => {
                let v1 = self.vectorized_data();
                let v2 = v.vectorized_data();
                assert_eq!(v1.len(), v2.len());
                let mut out = self.clone();
                let mut out_vec = out.vectorized_data_mut();
                v1.iter().zip(v2.iter()).enumerate().for_each(|(idx, (a, b))| out_vec[idx] = *a / *b);
                OptimaTensor::TensorND(out)
            }
            _ => {
                let c = self.convert(&other.map_to_type());
                c.elementwise_division(other)
            }
        }
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
    fn cell_mut_refn(&mut self, indices: Vec<usize>) -> &mut f64 {
        let vectorized_idx = self.indices_to_vectorized_idx(indices);
        let vectorized_data = self.vectorized_data_mut();
        &mut vectorized_data[vectorized_idx]
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
    fn cell_refn(&self, indices: Vec<usize>) -> &f64 {
        let vectorized_idx = self.indices_to_vectorized_idx(indices);
        let vectorized_data = self.vectorized_data();
        &vectorized_data[vectorized_idx]
    }
}
impl Clone for OptimaTensorND {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
        }
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

#[derive(Clone, Debug)]
pub enum OptimaTensorSliceScope {
    Free,
    Fixed(usize)
}

pub fn linearly_interpolate_optima_tensors(a: &OptimaTensor, b: &OptimaTensor, mode: &LinearInterpolationMode) -> Vec<OptimaTensor> {
    let a_vectorized = a.vectorized_data();
    let b_vectorized = b.vectorized_data();

    assert_eq!(a_vectorized.len(), b_vectorized.len());

    let a_dvec = DVector::from_column_slice(a_vectorized);
    let b_dvec = DVector::from_column_slice(b_vectorized);

    let dvecs = SimpleInterpolationUtils::linear_interpolation(&a_dvec, &b_dvec, mode);

    let mut out_vec = vec![];

    for d in &dvecs {
        let mut o = OptimaTensor::new_zeros(a.dimensions());
        let v = o.vectorized_data_mut();
        for (i, dd) in d.iter().enumerate() { v[i] = *dd; }
        out_vec.push(o);
    }

    out_vec
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct OTFImmutVars {
    c: Box<dyn EnumTypeContainer<OTFImmutVarsObject, OTFImmutVarsObjectType>>
}
impl OTFImmutVars {
    pub fn new() -> Self {
        Self::new_with_enum_signature_container_type(EnumTypeContainerType::default())
    }
    pub fn new_with_enum_signature_container_type(enum_signature_container_type: EnumTypeContainerType) -> Self {
        Self {
            c: match enum_signature_container_type {
                EnumTypeContainerType::BinarySearch => { Box::new(EnumBinarySearchTypeContainer::new()) }
                EnumTypeContainerType::HashMap => { Box::new(EnumHashMapTypeContainer::new()) }
            }
        }
    }
    pub fn object_ref(&self, signature: &OTFImmutVarsObjectType) -> Option<&OTFImmutVarsObject> {
        self.c.object_ref(signature)
    }
    pub fn object_ref_mut(&mut self, signature: &OTFImmutVarsObjectType) -> Option<&mut OTFImmutVarsObject> {
        self.c.object_mut_ref(signature)
    }
    pub fn insert_or_replace(&mut self, object: OTFImmutVarsObject) {
        self.c.insert_or_replace_object(object);
    }
    pub fn insert_or_replace_get_robot_set<T: GetRobotSet + 'static>(&mut self, object: T) {
        self.insert_or_replace(OTFImmutVarsObject::GetRobotSet(Box::new(object)));
    }

    pub fn ref_robot_set(&self) -> &RobotSet {
        let object = self.object_ref(&OTFImmutVarsObjectType::GetRobotSet);
        if let Some(object) = object {
            let get_robot_set = object.unwrap_get_robot_set();
            let robot_set = get_robot_set.get_robot_set();
            return robot_set;
        }
        let object = self.object_ref(&OTFImmutVarsObjectType::RobotGeometricShapeScene);
        if let Some(object) = object {
            let robot_geometric_shape_scene = object.unwrap_robot_geometric_shape_scene();
            let robot_set = robot_geometric_shape_scene.get_robot_set();
            return robot_set;
        }

        panic!("Could not recover a robot set here.");
    }
    pub fn ref_robot_geometric_shape_scene(&self) -> &RobotGeometricShapeScene {
        let object = self.object_ref(&OTFImmutVarsObjectType::RobotGeometricShapeScene).expect("needs RobotGeometricShapeScene");
        let robot_geometric_shape_scene = object.unwrap_robot_geometric_shape_scene();
        return robot_geometric_shape_scene;
    }
}

#[derive(Clone)]
pub enum OTFImmutVarsObject {
    GetRobotSet(Box<dyn GetRobotSet>),
    RobotLinkTransformGoalCollection(RobotLinkTransformGoalCollection),
    OptimaTensorWindowMemoryContainer(OptimaTensorWindowMemoryContainer),
    TimedGenericRobotJointStateWindowMemoryContainer(TimedGenericRobotJointStateWindowMemoryContainer),
    GenericRobotJointStateCurrTime(f64),
    RobotGeometricShapeScene(RobotGeometricShapeScene)
}
impl EnumMapToType<OTFImmutVarsObjectType> for OTFImmutVarsObject {
    fn map_to_type(&self) -> OTFImmutVarsObjectType {
        match self {
            OTFImmutVarsObject::GetRobotSet(_) => { OTFImmutVarsObjectType::GetRobotSet }
            OTFImmutVarsObject::RobotLinkTransformGoalCollection(_) => { OTFImmutVarsObjectType::RobotLinkTransformGoalCollection }
            OTFImmutVarsObject::OptimaTensorWindowMemoryContainer(_) => { OTFImmutVarsObjectType::OptimaTensorWindowMemoryContainer }
            OTFImmutVarsObject::TimedGenericRobotJointStateWindowMemoryContainer(_) => { OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer }
            OTFImmutVarsObject::GenericRobotJointStateCurrTime(_) => { OTFImmutVarsObjectType::GenericRobotJointStateCurrTime }
            OTFImmutVarsObject::RobotGeometricShapeScene(_) => { OTFImmutVarsObjectType::RobotGeometricShapeScene }
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
    pub fn unwrap_robot_link_transform_specification_collection(&self) -> &RobotLinkTransformGoalCollection {
        return match self {
            OTFImmutVarsObject::RobotLinkTransformGoalCollection(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_robot_link_transform_specification_collection_mut(&mut self) -> &mut RobotLinkTransformGoalCollection {
        return match self {
            OTFImmutVarsObject::RobotLinkTransformGoalCollection(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_optima_tensor_window_memory_container(&self) -> &OptimaTensorWindowMemoryContainer {
        return match self {
            OTFImmutVarsObject::OptimaTensorWindowMemoryContainer(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_optima_tensor_window_memory_container_mut(&mut self) -> &mut OptimaTensorWindowMemoryContainer {
        return match self {
            OTFImmutVarsObject::OptimaTensorWindowMemoryContainer(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_timed_generic_robot_joint_state_window_memory_container(&self) -> &TimedGenericRobotJointStateWindowMemoryContainer {
        return match self {
            OTFImmutVarsObject::TimedGenericRobotJointStateWindowMemoryContainer(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_timed_generic_robot_joint_state_window_memory_container_mut(&mut self) -> &mut TimedGenericRobotJointStateWindowMemoryContainer {
        return match self {
            OTFImmutVarsObject::TimedGenericRobotJointStateWindowMemoryContainer(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_generic_robot_joint_state_curr_time(&self) -> &f64 {
        return match self {
            OTFImmutVarsObject::GenericRobotJointStateCurrTime(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_generic_robot_joint_state_curr_time_mut(&mut self) -> &mut f64 {
        return match self {
            OTFImmutVarsObject::GenericRobotJointStateCurrTime(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_robot_geometric_shape_scene(&self) -> &RobotGeometricShapeScene {
        return match self {
            OTFImmutVarsObject::RobotGeometricShapeScene(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_robot_geometric_shape_scene_mut(&mut self) -> &mut RobotGeometricShapeScene {
        return match self {
            OTFImmutVarsObject::RobotGeometricShapeScene(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OTFImmutVarsObjectType {
    GetRobotSet,
    RobotLinkTransformGoalCollection,
    OptimaTensorWindowMemoryContainer,
    TimedGenericRobotJointStateWindowMemoryContainer,
    GenericRobotJointStateCurrTime,
    RobotGeometricShapeScene
}

/// var indices must be locked prior to getting keys
#[derive(Clone, Debug)]
pub struct OTFMutVars {
    enum_objects: Vec<OTFMutVarsObject>,
    signatures: Vec<OTFMutVarsObjectType>,
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
    pub fn get_vars(&mut self, signatures: &Vec<OTFMutVarsObjectType>, params: &Vec<OTFMutVarsParams>, recompute_var_ifs: &Vec<RecomputeVarIf>, input: &OptimaTensor, immut_vars: &OTFImmutVars, session_key: &OTFMutVarsSessionKey) -> Vec<&mut OTFMutVarsObject> {
        self.register_vars(input, immut_vars, recompute_var_ifs, signatures, params, session_key);
        let var_keys = self.get_var_keys(signatures, input, session_key);
        return self.unlock_object_mut_refs(&var_keys, input, session_key);
    }
    pub fn register_session(&mut self, input: &OptimaTensor) -> OTFMutVarsSessionKey {
        let out = OTFMutVarsSessionKey { key_idx: self.session_key_counter };
        let block = OTFMutVarsSessionBlock {
            key: out.clone(),
            is_base_session: false,
            input_tensor_id: input.id(),
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
    pub fn register_base_session(&mut self) -> OTFMutVarsSessionKey {
        let out = OTFMutVarsSessionKey { key_idx: self.session_key_counter };
        let block = OTFMutVarsSessionBlock {
            key: out.clone(),
            is_base_session: true,
            input_tensor_id: f64::MIN,
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

        assert!(session_block.is_base_session, "must insert base variable in a base session.");

        if session_block.already_gave_out_var_keys_this_session {
            panic!("Cannot insert a variable in a session that has already given out var keys.");
        }

        let signature = object.map_to_type();
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
    fn register_vars(&mut self, input: &OptimaTensor, immut_vars: &OTFImmutVars, recompute_var_ifs: &Vec<RecomputeVarIf>, signatures: &Vec<OTFMutVarsObjectType>, params: &Vec<OTFMutVarsParams>, session_key: &OTFMutVarsSessionKey) {
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

        assert!(!session_block.is_base_session, "cannot register general vars with base session key.");

        if session_block.input_tensor_id != input.id() {
            panic!("Tried to register a variable with an input tensor that does not match the given session.")
        }

        for (i, signature) in signatures.iter().enumerate() {
            let session_block = &self.registered_session_blocks[session_block_idx];
            let binary_search_res = session_block.registered_vars.binary_search_by(|x| x.cmp(signature));
            match binary_search_res {
                Ok(_) => { continue; }
                Err(_) => {
                    let binary_search_res = self.signatures.binary_search_by(|x| x.partial_cmp(signature).unwrap());
                    let compute= match binary_search_res {
                        Ok(idx) => {
                            recompute_var_ifs[i].recompute(input, &self.vectorized_tensors[idx], immut_vars)
                        }
                        Err(_) => {
                            true
                        }
                    };

                    if compute {
                        match binary_search_res {
                            Ok(idx) => {
                                let var_object = signature.compute_var(input, immut_vars, self, params);
                                self.enum_objects[idx] = var_object;
                                self.vectorized_tensors[idx] = input.vectorized_data().to_vec();
                            }
                            Err(idx) => {
                                let var_object = signature.compute_var(input, immut_vars, self, params);
                                self.signatures.insert(idx, signature.clone());
                                self.enum_objects.insert(idx, var_object);
                                self.vectorized_tensors.insert(idx, input.vectorized_data().to_vec());
                            }
                        }
                    }
                }
            }
            let session_block = &mut self.registered_session_blocks[session_block_idx];
            session_block.registered_vars.push(signature.clone());
        }
    }
    fn get_var_keys(&mut self, signatures: &Vec<OTFMutVarsObjectType>, input: &OptimaTensor, session_key: &OTFMutVarsSessionKey) -> OTFMutVarsKeyContainer {
        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(session_key).unwrap());
        let session_block_idx = match binary_search_res {
            Ok(idx) => {idx}
            Err(_) => { panic!("Tried to register a var in a session that does not exist.") }
        };
        let session_block = &self.registered_session_blocks[session_block_idx];

        if session_block.input_tensor_id != input.id() {
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
    fn unlock_object_mut_refs(&mut self, keys: &OTFMutVarsKeyContainer, input: &OptimaTensor, session_key: &OTFMutVarsSessionKey) -> Vec<&mut OTFMutVarsObject> {
        let binary_search_res = self.registered_session_blocks.binary_search_by(|x| x.key.partial_cmp(session_key).unwrap());
        let session_block_idx = match binary_search_res {
            Ok(idx) => {idx}
            Err(_) => { panic!("Tried to register a var in a session that does not exist.") }
        };
        let session_block = &self.registered_session_blocks[session_block_idx];

        if session_block.input_tensor_id != input.id() {
            panic!("Tried to get object refs with an input tensor that does not match the given session.")
        }

        let out = self.enum_objects.iter_mut()
            .enumerate()
            .filter(|(i, _)| keys.container.contains(&OTFMutVarsKey { key_idx: *i }))
            .map(|(_, o)| o)
            .collect();

        return out;
    }
    pub fn print_num_objects(&self) {
        println!("{:?}", self.enum_objects.len());
    }
}

#[derive(Clone, Debug)]
pub enum RecomputeVarIf {
    Always,
    IsAnyNewInput,
    IsAnyNonFDPerturbationInput,
    InputInfNormIsGreaterThan(f64),
    Never
}
impl RecomputeVarIf {
    #[allow(unused_variables)]
    pub fn recompute(&self, input: &OptimaTensor, previous_vectorized_tensor: &Vec<f64>, immut_vars: &OTFImmutVars) -> bool {
        return match self {
            RecomputeVarIf::Always => { true }
            RecomputeVarIf::IsAnyNewInput => {
                let vectorized_tensor = input.vectorized_data();
                for (i, v) in vectorized_tensor.iter().enumerate() {
                    if *v != previous_vectorized_tensor[i] { return true; }
                }
                false
            }
            RecomputeVarIf::IsAnyNonFDPerturbationInput => {
                let vectorized_tensor = input.vectorized_data();
                for (i, v) in vectorized_tensor.iter().enumerate() {
                    let diff = (*v - previous_vectorized_tensor[i]).abs();
                    if diff > FD_PERTURBATION { return true; }
                }
                false
            }
            RecomputeVarIf::InputInfNormIsGreaterThan(val) => {
                let vectorized_tensor = input.vectorized_data();
                for (i, v) in vectorized_tensor.iter().enumerate() {
                    let diff = (*v - previous_vectorized_tensor[i]).abs();
                    if diff > *val { return true; }
                }
                false
            }
            RecomputeVarIf::Never => { false }
        }
    }
}

#[derive(Clone, Debug)]
pub enum OTFMutVarsParams {
    None,
    SimpleDataType(SimpleDataType),
    RobotCollisionProximityBVHMode(RobotCollisionProximityBVHMode),
    VecOfParams(Vec<OTFMutVarsParams>)
}
impl OTFMutVarsParams {
    pub fn unwrap_simple_data_type(&self) -> &SimpleDataType {
        return match self {
            OTFMutVarsParams::SimpleDataType(r) => { r }
            _ => { panic!("wrong type") }
        }
    }
    pub fn unwrap_robot_collision_proximity_bvh_mode(&self) -> &RobotCollisionProximityBVHMode {
        return match self {
            OTFMutVarsParams::RobotCollisionProximityBVHMode(r) => { r }
            _ => { panic!("wrong type") }
        }
    }
    pub fn unwrap_vec_of_params(&self) -> &Vec<OTFMutVarsParams> {
        return match self {
            OTFMutVarsParams::VecOfParams(r) => { r }
            _ => { panic!("wrong type") }
        }
    }
}

#[derive(Clone, Debug)]
pub enum OTFMutVarsObject {
    None,
    RobotSetFKResult(RobotSetFKResult),
    RobotSetFKDOFPerturbationsResult(RobotSetFKDOFPerturbationsResult),
    ProximaEngine(ProximaEngine),
    BVHAABB(ShapeCollectionBVH<BVHCombinableShapeAABB>),
}
impl OTFMutVarsObject {
    pub fn unwrap_robot_set_fk_result(&self) -> &RobotSetFKResult {
        return match self {
            OTFMutVarsObject::RobotSetFKResult(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_robot_set_fk_dof_perturbations_result(&self) -> &RobotSetFKDOFPerturbationsResult {
        return match self {
            OTFMutVarsObject::RobotSetFKDOFPerturbationsResult(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_proxima_engine(&self) -> &ProximaEngine {
        return match self {
            OTFMutVarsObject::ProximaEngine(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_proxima_engine_mut(&mut self) -> &mut ProximaEngine {
        return match self {
            OTFMutVarsObject::ProximaEngine(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_bvh_aabb(&self) -> &ShapeCollectionBVH<BVHCombinableShapeAABB> {
        return match self {
            OTFMutVarsObject::BVHAABB(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_bvh_aabb_mut(&mut self) -> &mut ShapeCollectionBVH<BVHCombinableShapeAABB> {
        return match self {
            OTFMutVarsObject::BVHAABB(r) => { r }
            _ => { panic!("wrong type.") }
        }
    }
}
impl EnumMapToType<OTFMutVarsObjectType> for OTFMutVarsObject {
    fn map_to_type(&self) -> OTFMutVarsObjectType {
        match self {
            OTFMutVarsObject::None => { OTFMutVarsObjectType::None }
            OTFMutVarsObject::RobotSetFKResult(_) => { OTFMutVarsObjectType::RobotSetFKResult }
            OTFMutVarsObject::RobotSetFKDOFPerturbationsResult(_) => { OTFMutVarsObjectType::RobotSetFKDOFPerturbationsResult }
            OTFMutVarsObject::ProximaEngine(_) => { OTFMutVarsObjectType::ProximaEngine }
            OTFMutVarsObject::BVHAABB(_) => { OTFMutVarsObjectType::BVHAABB }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OTFMutVarsObjectType {
    None,
    RobotSetFKResult,
    RobotSetFKDOFPerturbationsResult,
    ProximaEngine,
    AbstractBVH,
    BVHAABB,
}
impl OTFMutVarsObjectType {
    pub fn compute_var(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, params: &Vec<OTFMutVarsParams>) -> OTFMutVarsObject {
        let session_key = mut_vars.register_session(input);
        let out = self.compute_var_raw(input, immut_vars, mut_vars, params, &session_key);
        mut_vars.close_session(&session_key);
        out
    }
    #[allow(unused_variables)]
    fn compute_var_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, params: &Vec<OTFMutVarsParams>, session_key: &OTFMutVarsSessionKey) -> OTFMutVarsObject {
        match self {
            OTFMutVarsObjectType::RobotSetFKResult => {
                // let robot_set_object = immut_vars.object_ref(&OTFImmutVarsObjectType::GetRobotSet).expect("error");
                // let robot_set = robot_set_object.unwrap_get_robot_set().get_robot_set();
                let robot_set = immut_vars.ref_robot_set();

                let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

                let res = robot_set.robot_set_kinematics_module().compute_fk(&robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");

                OTFMutVarsObject::RobotSetFKResult(res)
            }
            OTFMutVarsObjectType::RobotSetFKDOFPerturbationsResult => {
                let robot_set = immut_vars.ref_robot_set();

                let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

                let res = robot_set.robot_set_kinematics_module().compute_fk_dof_perturbations(&robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion, Some(FD_PERTURBATION)).expect("error");

                OTFMutVarsObject::RobotSetFKDOFPerturbationsResult(res)
            }
            OTFMutVarsObjectType::ProximaEngine => {
                let object = immut_vars.object_ref(&OTFImmutVarsObjectType::RobotGeometricShapeScene).expect("needs RobotGeometricShapeScene");
                let robot_geometric_shape_scene = object.unwrap_robot_geometric_shape_scene();
                let proxima_engine = robot_geometric_shape_scene.spawn_proxima_engine(None);
                OTFMutVarsObject::ProximaEngine(proxima_engine)
            }
            OTFMutVarsObjectType::BVHAABB => {
                let object = immut_vars.object_ref(&OTFImmutVarsObjectType::RobotGeometricShapeScene).expect("needs RobotGeometricShapeScene");
                let robot_geometric_shape_scene = object.unwrap_robot_geometric_shape_scene();
                let robot_set = robot_geometric_shape_scene.robot_set();
                let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");
                let bvh = robot_geometric_shape_scene.spawn_bvh::<BVHCombinableShapeAABB>(&robot_set_joint_state, None, 2);
                OTFMutVarsObject::BVHAABB(bvh)
            }
            OTFMutVarsObjectType::None => { OTFMutVarsObject::None }
            OTFMutVarsObjectType::AbstractBVH => {
                let robot_collision_proximity_bvh_mode = params[0].unwrap_robot_collision_proximity_bvh_mode();
                return match robot_collision_proximity_bvh_mode {
                    RobotCollisionProximityBVHMode::None => {
                        OTFMutVarsObject::None
                    }
                    RobotCollisionProximityBVHMode::AABB => {
                        Self::BVHAABB.compute_var_raw(input, immut_vars, mut_vars, params, session_key)
                    }
                }
            }
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
    is_base_session: bool,
    input_tensor_id: f64,
    already_gave_out_var_keys_this_session: bool,
    registered_vars: Vec<OTFMutVarsObjectType>,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct OptimaTensorWindowMemoryContainer {
    pub c: WindowMemoryContainer<OptimaTensor>
}
impl OptimaTensorWindowMemoryContainer {
    pub fn new(window_size: usize, init: OptimaTensor) -> Self {
        Self {
            c: WindowMemoryContainer::new(window_size, init)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

pub mod robotics_functions;
pub mod standard_functions;
/*
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_optima_tensor_functions::{OptimaTensor_, OptimaTensorFunction_, OTFImmutVars, OTFMutVars, OTFMutVarsSessionKey, OTFResult};

/// f(g(x))
pub struct OTFComposition<F, G>
    where F: OptimaTensorFunction_,
          G: OptimaTensorFunction_ {
    input_dimensions: (Vec<usize>, Vec<usize>),
    output_dimensions: (Vec<usize>, Vec<usize>),
    f: F,
    g: G
}
impl<F, G> OTFComposition <F, G>
    where F: OptimaTensorFunction_,
          G: OptimaTensorFunction_ {
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
impl <F, G> OptimaTensorFunction_ for OTFComposition<F, G>
    where F: OptimaTensorFunction_,
          G: OptimaTensorFunction_ {
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

*/

