#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::ops::{Add, Index, IndexMut, Mul};
use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_joint_state_module::{RobotJointState, RobotJointStateModule, RobotJointStateType};
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::load_object_from_json_string;
use crate::utils::utils_traits::{SaveAndLoadable, ToAndFromRonString};

/// RobotSet analogue of the `RobotJointStateModule`.  The same concepts apply, just on a set of possibly
/// multiple robots.
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotSetJointStateModule {
    num_dofs: usize,
    num_axes: usize,
    robot_joint_state_modules: Vec<RobotJointStateModule>,
}
impl RobotSetJointStateModule {
    pub fn new(robot_set_configuration_module: &RobotSetConfigurationModule) -> Self {
        let mut num_dofs = 0;
        let mut num_axes = 0;
        let mut robot_joint_state_modules = vec![];
        for r in robot_set_configuration_module.robot_configuration_modules() {
            let ja = RobotJointStateModule::new(r.clone());
            num_dofs += ja.num_dofs();
            num_axes += ja.num_axes();
            robot_joint_state_modules.push(ja.clone());
        }

        Self {
            num_dofs,
            num_axes,
            robot_joint_state_modules
        }
    }
    pub fn new_from_set_name(set_name: &str) -> Result<Self, OptimaError> {
        let robot_set_configuration_module = RobotSetConfigurationModule::new_from_set_name(set_name)?;
        return Ok(Self::new(&robot_set_configuration_module));
    }
    pub fn convert_state_to_full_state(&self, robot_set_joint_state: &RobotSetJointState) -> Result<RobotSetJointState, OptimaError> {
        if &robot_set_joint_state.robot_set_joint_state_type == &RobotSetJointStateType::Full {
            return Ok(robot_set_joint_state.clone());
        }

        let mut out_dvec = DVector::zeros(self.num_axes);

        let joint_states = &self.split_robot_set_joint_state_into_robot_joint_states(robot_set_joint_state)?;
        let mut curr_idx = 0;
        for (i, r) in self.robot_joint_state_modules.iter().enumerate() {
            let converted_robot_joint_state = r.convert_joint_state_to_full_state(&joint_states[i])?;
            let dv = converted_robot_joint_state.joint_state();
            let dv_len = dv.len();
            for j in 0..dv_len {
                out_dvec[curr_idx] = dv[j];
                curr_idx += 1;
            }
        }

        return Ok(RobotSetJointState {
            robot_set_joint_state_type: RobotSetJointStateType::Full,
            concatenated_state: out_dvec
        });
    }
    pub fn convert_state_to_dof_state(&self, robot_set_joint_state: &RobotSetJointState) -> Result<RobotSetJointState, OptimaError> {
        if &robot_set_joint_state.robot_set_joint_state_type == &RobotSetJointStateType::DOF {
            return Ok(robot_set_joint_state.clone());
        }

        let mut out_dvec = DVector::zeros(self.num_dofs);

        let joint_states = &self.split_robot_set_joint_state_into_robot_joint_states(robot_set_joint_state)?;
        let mut curr_idx = 0;
        for (i, r) in self.robot_joint_state_modules.iter().enumerate() {
            let converted_robot_joint_state = r.convert_joint_state_to_dof_state(&joint_states[i])?;
            let dv = converted_robot_joint_state.joint_state();
            let dv_len = dv.len();
            for j in 0..dv_len {
                out_dvec[curr_idx] = dv[j];
                curr_idx += 1;
            }
        }

        return Ok(RobotSetJointState {
            robot_set_joint_state_type: RobotSetJointStateType::DOF,
            concatenated_state: out_dvec
        });
    }
    pub fn spawn_robot_set_joint_state(&self, concatenated_state: DVector<f64>, robot_set_joint_state_type: RobotSetJointStateType) -> Result<RobotSetJointState, OptimaError> {
        let correct_length = match robot_set_joint_state_type {
            RobotSetJointStateType::DOF => { self.num_dofs }
            RobotSetJointStateType::Full => { self.num_axes }
        };

        if concatenated_state.len() != correct_length {
            return Err(OptimaError::new_robot_state_vec_wrong_size_error("spawn_robot_set_joint_state", concatenated_state.len(), correct_length, file!(), line!()));
        }

        Ok(RobotSetJointState {
            robot_set_joint_state_type,
            concatenated_state
        })
    }
    pub fn spawn_robot_set_joint_state_try_auto_type(&self, concatenated_state: DVector<f64>) -> Result<RobotSetJointState, OptimaError> {
        if concatenated_state.len() != self.num_dofs || concatenated_state.len() != self.num_axes {
            return Err(OptimaError::new_generic_error_str(&format!("Could not successfully make an auto \
            RobotSetJointState in try_new_auto_type().  The given state length was {} while either {} or {} was required.",
                                                            concatenated_state.len(), self.num_axes, self.num_dofs),
                                                   file!(),
                                                   line!()));
        }

        return if concatenated_state.len() == self.num_axes {
            self.spawn_robot_set_joint_state(concatenated_state, RobotSetJointStateType::Full)
        } else {
            self.spawn_robot_set_joint_state(concatenated_state, RobotSetJointStateType::DOF)
        }
    }
    pub fn spawn_zeros_robot_set_joint_state(&self, robot_set_state_type: RobotSetJointStateType) -> RobotSetJointState {
        return match robot_set_state_type {
            RobotSetJointStateType::DOF => {
                RobotSetJointState {
                    robot_set_joint_state_type: RobotSetJointStateType::DOF,
                    concatenated_state: DVector::from_vec(vec![0.0; self.num_dofs])
                }
            }
            RobotSetJointStateType::Full => {
                RobotSetJointState {
                    robot_set_joint_state_type: RobotSetJointStateType::Full,
                    concatenated_state: DVector::from_vec(vec![0.0; self.num_axes])
                }
            }
        }
    }
    pub fn get_joint_state_bounds(&self, t: &RobotSetJointStateType) -> Vec<(f64, f64)> {
        let mut out_vec = vec![];
        for r in &self.robot_joint_state_modules {
            let joint_state_bounds = r.get_joint_state_bounds(&t.map_to_robot_joint_state_type());
            for j in joint_state_bounds { out_vec.push(j); }
        }
        out_vec
    }
    pub fn sample_set_joint_state(&self, t: &RobotSetJointStateType) -> RobotSetJointState {
        let mut out_dvec = match t {
            RobotSetJointStateType::DOF => { DVector::zeros(self.num_dofs) }
            RobotSetJointStateType::Full => { DVector::zeros(self.num_axes) }
        };

        let mut curr_idx = 0;
        for r in &self.robot_joint_state_modules {
            let joint_state = r.sample_joint_state(&t.map_to_robot_joint_state_type());

            let dv = joint_state.joint_state();
            let l = dv.len();
            for i in 0..l {
                out_dvec[curr_idx] = dv[i];
                curr_idx += 1;
            }
        }

        RobotSetJointState {
            robot_set_joint_state_type: t.clone(),
            concatenated_state: out_dvec
        }
    }
    pub fn split_robot_set_joint_state_into_robot_joint_states(&self, robot_set_joint_state: &RobotSetJointState) -> Result<Vec<RobotJointState>, OptimaError> {
        let split = self.split_concatenated_dvec_into_separate_robot_dvecs(&robot_set_joint_state.concatenated_state, &robot_set_joint_state.robot_set_joint_state_type)?;

        let mut out_states = vec![];
        for (i, s) in split.iter().enumerate() {
            out_states.push( self.robot_joint_state_modules[i].spawn_robot_joint_state(s.clone(), robot_set_joint_state.robot_set_joint_state_type().map_to_robot_joint_state_type())? )
        }

        Ok(out_states)
    }
    fn split_concatenated_dvec_into_separate_robot_dvecs(&self, concatenated_state: &DVector<f64>, robot_set_joint_state_type: &RobotSetJointStateType) -> Result<Vec<DVector<f64>>, OptimaError> {
        match robot_set_joint_state_type {
            RobotSetJointStateType::DOF => {
                if concatenated_state.len() != self.num_dofs {
                    return Err(OptimaError::new_robot_state_vec_wrong_size_error("split_concatenated_dvec_into_separate_robot_dvecs", concatenated_state.len(), self.num_dofs, file!(), line!()));
                }
            }
            RobotSetJointStateType::Full => {
                if concatenated_state.len() != self.num_axes {
                    return Err(OptimaError::new_robot_state_vec_wrong_size_error("split_concatenated_dvec_into_separate_robot_dvecs", concatenated_state.len(), self.num_axes, file!(), line!()));
                }
            }
        }

        let mut out_vec = vec![];

        let mut curr_idx = 0;
        for r in &self.robot_joint_state_modules {
            let a = match robot_set_joint_state_type {
                RobotSetJointStateType::DOF => { r.num_dofs() }
                RobotSetJointStateType::Full => { r.num_axes() }
            };
            let mut dv = DVector::zeros(a);
            for i in 0..a {
                dv[i] = concatenated_state[curr_idx];
                curr_idx += 1;
            }
            out_vec.push(dv);
        }

        Ok(out_vec)
    }
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }
    pub fn num_axes(&self) -> usize {
        self.num_axes
    }
    pub fn robot_joint_state_modules(&self) -> &Vec<RobotJointStateModule> {
        &self.robot_joint_state_modules
    }
}
impl SaveAndLoadable for RobotSetJointStateModule {
    type SaveType = (usize, usize, String);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.num_dofs, self.num_axes, self.robot_joint_state_modules.get_serialization_string())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;

        let robot_joint_state_modules = Vec::load_from_json_string(&load.2)?;

        Ok(Self {
            num_dofs: load.0,
            num_axes: load.1,
            robot_joint_state_modules
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotSetJointStateModule {
    #[new]
    pub fn new_from_set_name_py(set_name: &str) -> Self {
        Self::new_from_set_name(set_name).expect("error")
    }
    #[staticmethod]
    pub fn new_py(robot_set_configuration_module: &RobotSetConfigurationModule) -> Self {
        Self::new(robot_set_configuration_module)
    }
    pub fn convert_state_to_full_state_py(&self, robot_set_joint_state: Vec<f64>) -> Vec<f64> {
        let out = self.spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let out = self.convert_state_to_full_state(&out).expect("error");
        let out: &Vec<f64> =  out.concatenated_state.data.as_vec();
        return out.clone();
    }
    pub fn convert_state_to_dof_state_py(&self, robot_set_joint_state: Vec<f64>) -> Vec<f64> {
        let out = self.spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let out = self.convert_state_to_dof_state(&out).expect("error");
        let out: &Vec<f64> =  out.concatenated_state.data.as_vec();
        return out.clone();
    }
    #[args(robot_set_joint_state_type = "\"DOF\"")]
    pub fn spawn_zeros_robot_set_joint_state_py(&self, robot_set_joint_state_type: &str) -> Vec<f64> {
        let s = self.spawn_zeros_robot_set_joint_state(RobotSetJointStateType::from_ron_string(robot_set_joint_state_type).expect("error"));
        let v: &Vec<f64> = s.concatenated_state.data.as_vec();
        return v.clone()
    }
    pub fn num_dofs_py(&self) -> usize {
        self.num_dofs()
    }
    pub fn num_axes_py(&self) -> usize {
        self.num_axes()
    }
    pub fn robot_joint_state_modules_py(&self) -> Vec<RobotJointStateModule> {
        self.robot_joint_state_modules.clone()
    }
    pub fn split_robot_set_joint_state_into_robot_joint_states_py(&self, robot_set_joint_state: Vec<f64>) -> Vec<Vec<f64>> {
        let d = DVector::from_vec(robot_set_joint_state);
        let state = self.spawn_robot_set_joint_state_try_auto_type(d).expect("error");
        let res = self.split_robot_set_joint_state_into_robot_joint_states(&state).expect("error");

        let mut out_vec = vec![];

        for r in &res {
            let v: &Vec<f64> = r.joint_state().data.as_vec();
            out_vec.push(v.clone());
        }

        out_vec
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotSetJointStateModule {
    #[wasm_bindgen(constructor)]
    pub fn new_from_set_name_wasm(set_name: &str) -> Self {
        Self::new_from_set_name(set_name).expect("error")
    }
    pub fn convert_state_to_full_state_wasm(&self, robot_set_joint_state: Vec<f64>) -> Vec<f64> {
        let out = self.spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let out = self.convert_state_to_full_state(&out).expect("error");
        let out: &Vec<f64> =  out.concatenated_state.data.as_vec();
        return out.clone();
    }
    pub fn convert_state_to_dof_state_wasm(&self, robot_set_joint_state: Vec<f64>) -> Vec<f64> {
        let out = self.spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let out = self.convert_state_to_dof_state(&out).expect("error");
        let out: &Vec<f64> =  out.concatenated_state.data.as_vec();
        return out.clone();
    }
}

/// RobotSet analogue of the `RobotJointState`.  The same concepts apply, just on a set of possibly
/// multiple robots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetJointState {
    robot_set_joint_state_type: RobotSetJointStateType,
    concatenated_state: DVector<f64>
}
impl RobotSetJointState {
    pub fn robot_set_joint_state_type(&self) -> &RobotSetJointStateType {
        &self.robot_set_joint_state_type
    }
    pub fn concatenated_state(&self) -> &DVector<f64> {
        &self.concatenated_state
    }
}
impl Add for RobotSetJointState {
    type Output = Result<RobotSetJointState, OptimaError>;
    fn add(self, rhs: Self) -> Self::Output {
        if &self.robot_set_joint_state_type != &rhs.robot_set_joint_state_type {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to add robot set states of different types ({:?} + {:?}).", self.robot_set_joint_state_type(), rhs.robot_set_joint_state_type()), file!(), line!()));
        }

        if self.concatenated_state.len() != rhs.concatenated_state.len() {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to add robot set states of different lengths ({:?} + {:?}).", self.concatenated_state.len(), rhs.concatenated_state.len()), file!(), line!()));
        }

        return Ok(RobotSetJointState {
            robot_set_joint_state_type: self.robot_set_joint_state_type.clone(),
            concatenated_state: self.concatenated_state + rhs.concatenated_state
        })
    }
}
impl Mul<RobotSetJointState> for f64 {
    type Output = RobotSetJointState;

    fn mul(self, rhs: RobotSetJointState) -> Self::Output {
        let mut output = rhs.clone();
        output.concatenated_state *= self;
        output
    }
}
impl Index<usize> for RobotSetJointState {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.concatenated_state[index];
    }
}
impl IndexMut<usize> for RobotSetJointState {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.concatenated_state[index]
    }
}

/// RobotSet analogue of the `RobotJointStateType`.  The same concepts apply, just on a set of possibly
/// multiple robots.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobotSetJointStateType {
    DOF,
    Full
}
impl RobotSetJointStateType {
    pub fn map_to_robot_joint_state_type(&self) -> RobotJointStateType {
        match self {
            RobotSetJointStateType::DOF => { RobotJointStateType::DOF }
            RobotSetJointStateType::Full => { RobotJointStateType::Full }
        }
    }
}