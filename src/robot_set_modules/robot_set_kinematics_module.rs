#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_kinematics_module::{RobotKinematicsModule, RobotFKResult, FloatingLinkInput, JacobianEndPoint, JacobianMode};
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::robot_set_modules::robot_set_joint_state_module::{RobotSetJointState, RobotSetJointStateModule};
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::load_object_from_json_string;
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
use crate::utils::utils_traits::{SaveAndLoadable, ToAndFromRonString};

/// RobotSet analogue of the `RobotKinematicsModule`.  The same concepts apply, just on a set of possibly
/// multiple robots.
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotSetKinematicsModule {
    robot_set_joint_state_module: RobotSetJointStateModule,
    robot_kinematics_modules: Vec<RobotKinematicsModule>
}
impl RobotSetKinematicsModule {
    pub fn new(robot_set_configuration_module: &RobotSetConfigurationModule) -> Self {
        let mut robot_fk_modules = vec![];
        for r in robot_set_configuration_module.robot_configuration_modules() {
            let robot_fk_module = RobotKinematicsModule::new(r.clone());
            robot_fk_modules.push(robot_fk_module);
        }
        Self {
            robot_set_joint_state_module: RobotSetJointStateModule::new(robot_set_configuration_module),
            robot_kinematics_modules: robot_fk_modules
        }
    }
    pub fn new_from_set_name(set_name: &str) -> Result<Self, OptimaError> {
        let robot_set_configuration_module = RobotSetConfigurationModule::new_from_set_name(set_name)?;
        return Ok(Self::new(&robot_set_configuration_module));
    }
    pub fn compute_fk(&self, set_joint_state: &RobotSetJointState, t: &OptimaSE3PoseType) -> Result<RobotSetFKResult, OptimaError> {
        let mut out_vec = vec![];

        let joint_states = &self.robot_set_joint_state_module.split_robot_set_joint_state_into_robot_joint_states(set_joint_state)?;

        for (i, joint_state) in joint_states.iter().enumerate() {
            let fk_res = self.robot_kinematics_modules.get(i).unwrap().compute_fk(joint_state, t)?;
            out_vec.push(fk_res);
        }

        Ok(RobotSetFKResult {
            robot_fk_results: out_vec
        })
    }
    pub fn compute_fk_on_subset_of_robots(&self, robot_idxs: Vec<usize>, set_joint_state: &RobotSetJointState, t: &OptimaSE3PoseType) -> Result<RobotSetFKResult, OptimaError> {
        let mut out_vec = vec![];
        let num_robots = self.robot_kinematics_modules.len();

        let joint_states = &self.robot_set_joint_state_module.split_robot_set_joint_state_into_robot_joint_states(set_joint_state)?;

        for robot_idx in 0..num_robots {
            if robot_idxs.contains(&robot_idx) {
                let fk_res = self.robot_kinematics_modules.get(robot_idx).unwrap().compute_fk(joint_states.get(robot_idx).unwrap(), t)?;
                out_vec.push(fk_res);
            } else {
                out_vec.push(RobotFKResult::new_empty(self.robot_kinematics_modules.get(robot_idx).unwrap()))
            }
        }

        Ok(RobotSetFKResult {
            robot_fk_results: out_vec
        })
    }
    pub fn compute_fk_floating_chain(&self, set_joint_state: &RobotSetJointState, t: &OptimaSE3PoseType, floating_link_inputs: Vec<Option<FloatingLinkInput>>) -> Result<RobotSetFKResult, OptimaError> {
        let mut out_vec = vec![];

        let joint_states = &self.robot_set_joint_state_module.split_robot_set_joint_state_into_robot_joint_states(set_joint_state)?;

        for (i, joint_state) in joint_states.iter().enumerate() {
            let floating_link_input = floating_link_inputs.get(i).unwrap();
            match floating_link_input {
                None => {
                    out_vec.push(RobotFKResult::new_empty(self.robot_kinematics_modules.get(i).unwrap()));
                }
                Some(floating_link_input) => {
                    let fk_res = self.robot_kinematics_modules.get(i).unwrap().compute_fk_floating_chain(joint_state, t, floating_link_input)?;
                    out_vec.push(fk_res);
                }
            }
        }

        Ok(RobotSetFKResult {
            robot_fk_results: out_vec
        })
    }
    pub fn compute_fk_dof_perturbations(&self, joint_state: &RobotSetJointState, t: &OptimaSE3PoseType, perturbation: Option<f64>) -> Result<RobotSetFKDOFPerturbationsResult, OptimaError> {
        let perturbation = match perturbation {
            None => { 0.00001 }
            Some(p) => { p }
        };

        let dof_joint_state = self.robot_set_joint_state_module.convert_state_to_dof_state(joint_state)?;

        let central_fk_result = self.compute_fk(&dof_joint_state, t)?;

        let mut fk_dof_perturbation_results = vec![];

        let len = dof_joint_state.concatenated_state().len();
        for i in 0..len {
            let mut dof_joint_state_copy = dof_joint_state.clone();
            dof_joint_state_copy[i] += perturbation;
            let fk_res = self.compute_fk(&dof_joint_state_copy, t)?;
            fk_dof_perturbation_results.push(fk_res);
        }

        Ok(RobotSetFKDOFPerturbationsResult {
            perturbation,
            central_fk_result,
            fk_dof_perturbation_results
        })
    }
    pub fn compute_jacobian(&self,
                            joint_state: &RobotSetJointState,
                            robot_idx_in_set: usize,
                            start_link_idx: Option<usize>,
                            end_link_idx: usize,
                            robot_jacobian_end_point: &JacobianEndPoint,
                            start_link_pose: Option<OptimaSE3Pose>,
                            jacobian_mode: JacobianMode) -> Result<DMatrix<f64>, OptimaError> {
        if robot_idx_in_set >= self.robot_kinematics_modules.len() {
            return Err(OptimaError::new_idx_out_of_bound_error(robot_idx_in_set, self.robot_kinematics_modules.len(), file!(), line!()));
        }

        let num_dofs = self.robot_set_joint_state_module.num_dofs();
        let num_rows = match jacobian_mode {
            JacobianMode::Full => { 6 }
            JacobianMode::Translational => { 3 }
            JacobianMode::Rotational => { 3 }
        };
        let mut jacobian = DMatrix::zeros(num_rows, num_dofs);

        let mut start_idx = 0;
        for i in 0..robot_idx_in_set {
            start_idx += self.robot_set_joint_state_module.robot_joint_state_modules().get(i).unwrap().num_dofs();
        }

        let robot_states = self.robot_set_joint_state_module.split_robot_set_joint_state_into_robot_joint_states(joint_state)?;
        let robot_state = robot_states.get(robot_idx_in_set).unwrap();
        let robot_jacobian = self.robot_kinematics_modules.get(robot_idx_in_set).unwrap().compute_jacobian(robot_state, start_link_idx, end_link_idx, robot_jacobian_end_point, start_link_pose, jacobian_mode)?;

        let num_columns = robot_state.len();
        for i in 0..num_columns {
            let col = i + start_idx;
            for row in 0..num_rows {
                jacobian[(row, col)] = robot_jacobian[(row, i)];
            }
        }

        Ok(jacobian)
    }
}
impl SaveAndLoadable for RobotSetKinematicsModule {
    type SaveType = (String, String);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_set_joint_state_module.get_serialization_string(), self.robot_kinematics_modules.get_serialization_string())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let robot_set_joint_state_module = RobotSetJointStateModule::load_from_json_string(&load.0)?;
        let robot_kinematics_modules  = Vec::load_from_json_string(&load.1)?;

        Ok(Self {
            robot_set_joint_state_module,
            robot_kinematics_modules
        })
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotSetKinematicsModule {
    #[new]
    pub fn new_from_set_name_py(set_name: &str) -> Self {
        Self::new_from_set_name(set_name).expect("error")
    }
    #[staticmethod]
    pub fn new_py(robot_set_configuration_module: &RobotSetConfigurationModule) -> Self {
        Self::new(robot_set_configuration_module)
    }
    #[args(pose_type = "\"ImplicitDualQuaternion\"")]
    pub fn compute_fk_py(&self, joint_state: Vec<f64>, pose_type: &str) -> RobotSetFKResult {
        let robot_joint_state = self.robot_set_joint_state_module.spawn_robot_set_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        return self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::from_ron_string(pose_type).expect("error")).expect("error");
    }
}

/// RobotSet analogue of the `RobotSetFKResult`.  The same concepts apply, just on a set of possibly
/// multiple robots.  Just contains a vector of individual `RobotFKResult` structs corresponding to
/// the possibly multiple robots in the set.
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotSetFKResult {
    robot_fk_results: Vec<RobotFKResult>
}
impl RobotSetFKResult {
    pub fn robot_fk_results(&self) -> &Vec<RobotFKResult> {
        &self.robot_fk_results
    }
    pub fn robot_fk_result(&self, robot_idx_in_set: usize) -> Result<&RobotFKResult, OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(robot_idx_in_set, self.robot_fk_results.len(), file!(), line!())?;

        return Ok(&self.robot_fk_results[robot_idx_in_set]);
    }
    pub fn print_summary(&self) {
        for (i, robot_fk_result) in self.robot_fk_results.iter().enumerate() {
            optima_print(&format!("Robot {} ---> ", i), PrintMode::Println, PrintColor::Cyan, true);
            robot_fk_result.print_summary();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RobotSetFKResult {
    pub fn get_fk_result(&self, idx: usize) -> RobotFKResult {
        self.robot_fk_results.get(idx).unwrap().clone()
    }
    pub fn num_fk_results(&self) -> usize {
        self.robot_fk_results.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetFKDOFPerturbationsResult {
    perturbation: f64,
    central_fk_result: RobotSetFKResult,
    fk_dof_perturbation_results: Vec<RobotSetFKResult>
}