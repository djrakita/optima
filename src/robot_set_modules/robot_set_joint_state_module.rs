use std::ops::{Add, Index, Mul};
use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_joint_state_module::{RobotJointState, RobotJointStateModule, RobotJointStateType};
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::utils::utils_errors::OptimaError;

/// RobotSet analogue of the `RobotJointStateModule`.  The same concepts apply, just on a set of possibly
/// multiple robots.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
        let mut out_joint_states = vec![];

        let joint_states = &robot_set_joint_state.robot_joint_states;
        let mut curr_idx = 0;
        for (i, r) in self.robot_joint_state_modules.iter().enumerate() {
            let converted_robot_joint_state = r.convert_joint_state_to_full_state(&joint_states[i])?;
            let dv = converted_robot_joint_state.joint_state();
            let dv_len = dv.len();
            for j in 0..dv_len {
                out_dvec[curr_idx] = dv[j];
                curr_idx += 1;
            }
            out_joint_states.push(converted_robot_joint_state);
        }

        return Ok(RobotSetJointState {
            robot_set_joint_state_type: RobotSetJointStateType::Full,
            concatenated_state: out_dvec,
            robot_joint_states: out_joint_states
        });
    }
    pub fn convert_state_to_dof_state(&self, robot_set_joint_state: &RobotSetJointState) -> Result<RobotSetJointState, OptimaError> {
        if &robot_set_joint_state.robot_set_joint_state_type == &RobotSetJointStateType::DOF {
            return Ok(robot_set_joint_state.clone());
        }

        let mut out_dvec = DVector::zeros(self.num_dofs);
        let mut out_joint_states = vec![];

        let joint_states = &robot_set_joint_state.robot_joint_states;
        let mut curr_idx = 0;
        for (i, r) in self.robot_joint_state_modules.iter().enumerate() {
            let converted_robot_joint_state = r.convert_joint_state_to_dof_state(&joint_states[i])?;
            let dv = converted_robot_joint_state.joint_state();
            let dv_len = dv.len();
            for j in 0..dv_len {
                out_dvec[curr_idx] = dv[j];
                curr_idx += 1;
            }
            out_joint_states.push(converted_robot_joint_state);
        }

        return Ok(RobotSetJointState {
            robot_set_joint_state_type: RobotSetJointStateType::DOF,
            concatenated_state: out_dvec,
            robot_joint_states: out_joint_states
        });
    }
    pub fn spawn_robot_set_joint_state(&self, concatenated_state: DVector<f64>, robot_set_joint_state_type: RobotSetJointStateType) -> Result<RobotSetJointState, OptimaError> {
        let split = self.split_concatenated_dvec_into_separate_robot_dvecs(&concatenated_state, &robot_set_joint_state_type)?;
        let mut robot_joint_states = vec![];
        for (i, r) in self.robot_joint_state_modules.iter().enumerate() {
            robot_joint_states.push(r.spawn_robot_joint_state(split[i].clone(), robot_set_joint_state_type.map_to_robot_joint_state_type())?);
        }

        Ok(RobotSetJointState {
            robot_set_joint_state_type,
            concatenated_state,
            robot_joint_states
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

        let mut out_joint_states = vec![];

        let mut curr_idx = 0;
        for r in &self.robot_joint_state_modules {
            let joint_state = r.sample_joint_state(&t.map_to_robot_joint_state_type());

            let dv = joint_state.joint_state();
            let l = dv.len();
            for i in 0..l {
                out_dvec[curr_idx] = dv[i];
                curr_idx += 1;
            }

            out_joint_states.push(joint_state);
        }

        RobotSetJointState {
            robot_set_joint_state_type: t.clone(),
            concatenated_state: out_dvec,
            robot_joint_states: out_joint_states
        }
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
}

/// RobotSet analogue of the `RobotJointState`.  The same concepts apply, just on a set of possibly
/// multiple robots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetJointState {
    robot_set_joint_state_type: RobotSetJointStateType,
    concatenated_state: DVector<f64>,
    robot_joint_states: Vec<RobotJointState>
}
impl RobotSetJointState {
    pub fn robot_set_joint_state_type(&self) -> &RobotSetJointStateType {
        &self.robot_set_joint_state_type
    }
    pub fn concatenated_state(&self) -> &DVector<f64> {
        &self.concatenated_state
    }
    pub fn robot_joint_states(&self) -> &Vec<RobotJointState> {
        &self.robot_joint_states
    }
}
impl Add for RobotSetJointState {
    type Output = Result<RobotSetJointState, OptimaError>;
    fn add(self, rhs: Self) -> Self::Output {
        if &self.robot_set_joint_state_type != &rhs.robot_set_joint_state_type {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to add robot set states of different types ({:?} + {:?}).", self.robot_set_joint_state_type(), rhs.robot_set_joint_state_type()), file!(), line!()));
        }

        if self.robot_joint_states.len() != rhs.robot_joint_states.len() {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to add robot set states from different robot sets."), file!(), line!()));
        }

        let added_dvec = &self.concatenated_state + &rhs.concatenated_state;
        let mut out_robot_joint_states = vec![];
        let l = self.robot_joint_states.len();
        for i in 0..l {
            out_robot_joint_states.push( (self.robot_joint_states[i].clone() + rhs.robot_joint_states[i].clone())? )
        }

        Ok(RobotSetJointState {
            robot_set_joint_state_type: self.robot_set_joint_state_type.clone(),
            concatenated_state: added_dvec,
            robot_joint_states: out_robot_joint_states
        })
    }
}
impl Mul<RobotSetJointState> for f64 {
    type Output = RobotSetJointState;

    fn mul(self, rhs: RobotSetJointState) -> Self::Output {
        let multiplied_dvec = self * rhs.concatenated_state;
        let mut out_joint_states = vec![];
        for j in &rhs.robot_joint_states {
            out_joint_states.push( self * j.clone() )
        }

        RobotSetJointState {
            robot_set_joint_state_type: rhs.robot_set_joint_state_type.clone(),
            concatenated_state: multiplied_dvec,
            robot_joint_states: out_joint_states
        }
    }
}
impl Index<usize> for RobotSetJointState {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.concatenated_state[index];
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