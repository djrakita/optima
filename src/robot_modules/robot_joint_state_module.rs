#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use std::ops::{Add, Index, IndexMut, Mul};
use crate::robot_modules::robot_configuration_module::{RobotConfigurationModule};
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string};
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_robot::joint::JointAxis;
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_sampling::SimpleSamplers;
use crate::utils::utils_traits::SaveAndLoadable;

/// The `RobotJointStateModule` organizes and operates over robot states.  "Robot joint states" are vectors
/// that contain scalar joint values for each joint axis in the robot model.
/// These objects are sometimes referred to as robot configurations or robot poses in the robotics literature,
/// but in this library, we will stick to the convention of referring to them as robot joint states.
///
/// The `RobotJointStateModule` has two primary fields:
/// - `ordered_dof_joint_axes`
/// - `ordered_joint_axes`
///
/// The `ordered_dof_joint_axes` field is a vector of `JointAxis` objects corresponding to the robot's
/// degrees of freedom (DOFs).  Note that this does NOT include any joint axes that have fixed values.
/// The number of robot degrees of freedom (DOFs) for a robot configuration is, thus, the length of
/// the `ordered_dof_joint_axes` vector (also accessible via the `num_dofs` field).
///
/// The `ordered_joint_axes` field is a vector of `JointAxis` objects corresponding to all axes
/// in the robot configuration.  Note that this DOES include joint axes, even if they have a fixed value.
/// The number of axes available in the robot configuration is accessible via the `num_axes` field.
/// Note that `num_dofs` <= `num_axes` for any robot configuration.
///
/// Note that neither the `ordered_dof_joint_axes` nor `ordered_joint_axes` vectors will include
/// joint axis objects on any joint that is listed as not present in the robot configuration.
///
/// These two differing views of joint axis lists (either including fixed axes or not) suggest two different
/// variants of robot joint states:
/// - A dof joint state
/// - A full joint state
///
/// A dof joint state only contains values for joint values that are free (not fixed), while the full joint state
/// includes joint values for ALL present joint axes (even if they are fixed).  A dof joint state is important
/// for operations such as optimization where only the free values are decision variables,
/// while a full joint state is important for operations such as forward kinematics where all present joint
/// axes need to somehow contribute to the model.
///
/// A dof joint state can be converted to a full joint state via the function `convert_dof_state_to_full_state`.
/// A full joint state can be converted to a dof joint state via the function `convert_full_state_to_dof_state`.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotJointStateModule {
    num_dofs: usize,
    num_axes: usize,
    ordered_dof_joint_axes: Vec<JointAxis>,
    ordered_joint_axes: Vec<JointAxis>,
    robot_configuration_module: RobotConfigurationModule,
    joint_idx_to_dof_state_idxs_mapping: Vec<Vec<usize>>,
    joint_idx_to_full_state_idxs_mapping: Vec<Vec<usize>>,
}
impl RobotJointStateModule {
    pub fn new(robot_configuration_module: RobotConfigurationModule) -> Self {
        let mut out_self = Self {
            num_dofs: 0,
            num_axes: 0,
            ordered_dof_joint_axes: vec![],
            ordered_joint_axes: vec![],
            robot_configuration_module,
            joint_idx_to_dof_state_idxs_mapping: vec![],
            joint_idx_to_full_state_idxs_mapping: vec![]
        };

        out_self.set_ordered_joint_axes();
        out_self.initialize_joint_idx_to_full_state_idxs();
        out_self.initialize_joint_idx_to_dof_state_idxs();
        out_self.num_dofs = out_self.ordered_dof_joint_axes.len();
        out_self.num_axes = out_self.ordered_joint_axes.len();

        return out_self;
    }
    pub fn new_from_names(robot_names: RobotNames) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_names)?;
        return Ok(Self::new(robot_configuration_module));
    }
    fn set_ordered_joint_axes(&mut self) {
        for j in self.robot_configuration_module.robot_model_module().joints() {
            if j.active() {
                let joint_axes = j.joint_axes();
                for ja in joint_axes {
                    self.ordered_joint_axes.push(ja.clone());
                    if !ja.is_fixed() {
                        self.ordered_dof_joint_axes.push(ja.clone());
                    }
                }
            }
        }
    }
    fn initialize_joint_idx_to_dof_state_idxs(&mut self) {
        let mut out_vec = vec![];
        let num_joints = self.robot_configuration_module.robot_model_module().joints().len();
        for _ in 0..num_joints { out_vec.push(vec![]); }

        for (i, ja) in self.ordered_dof_joint_axes.iter().enumerate() {
            out_vec[ja.joint_idx()].push(i);
        }

        self.joint_idx_to_dof_state_idxs_mapping = out_vec;
    }
    fn initialize_joint_idx_to_full_state_idxs(&mut self) {
        let mut out_vec = vec![];
        let num_joints = self.robot_configuration_module.robot_model_module().joints().len();
        for _ in 0..num_joints { out_vec.push(vec![]); }

        for (i, ja) in self.ordered_joint_axes.iter().enumerate() {
            out_vec[ja.joint_idx()].push(i);
        }

        self.joint_idx_to_full_state_idxs_mapping = out_vec;
    }
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }
    pub fn num_axes(&self) -> usize {
        self.num_axes
    }
    /// Returns joint axes in order (excluding fixed axes, thus only corresponding to degrees of freedom).
    pub fn ordered_dof_joint_axes(&self) -> &Vec<JointAxis> {
        &self.ordered_dof_joint_axes
    }
    /// Returns all joint axes in order (included fixed axes).
    pub fn ordered_joint_axes(&self) -> &Vec<JointAxis> {
        &self.ordered_joint_axes
    }
    /// Converts a joint state to a full state.
    pub fn convert_joint_state_to_full_state(&self, joint_state: &RobotJointState) -> Result<RobotJointState, OptimaError> {
        if joint_state.len() != self.num_dofs {
            return Err(OptimaError::new_robot_state_vec_wrong_size_error("convert_dof_state_to_full_state", joint_state.len(), self.num_dofs, file!(), line!()))
        }

        if joint_state.robot_joint_state_type() == &RobotJointStateType::DOF { return Ok(joint_state.clone()); }

        let mut out_robot_state_vector = DVector::zeros(self.num_axes);

        let mut bookmark = 0 as usize;

        for (i, a) in self.ordered_joint_axes.iter().enumerate() {
            if a.is_fixed() {
                out_robot_state_vector[i] = a.fixed_value().unwrap();
            } else {
                out_robot_state_vector[i] = joint_state[bookmark];
                bookmark += 1;
            }
        }

        return Ok(RobotJointState::new(out_robot_state_vector, RobotJointStateType::Full, self)?);
    }
    /// Converts a joint state to a dof joint state.
    pub fn convert_joint_state_to_dof_state(&self, joint_state: &RobotJointState) -> Result<RobotJointState, OptimaError> {
        if joint_state.len() != self.num_axes() {
            return Err(OptimaError::new_robot_state_vec_wrong_size_error("convert_full_state_to_dof_state", joint_state.len(), self.num_axes, file!(), line!()))
        }

        if joint_state.robot_joint_state_type() == &RobotJointStateType::Full { return Ok(joint_state.clone()); }

        let mut out_robot_state_vector = DVector::zeros(self.num_dofs);

        let mut bookmark = 0 as usize;

        for (i, a) in self.ordered_joint_axes.iter().enumerate() {
            if !a.is_fixed() {
                out_robot_state_vector[bookmark] = joint_state[i];
                bookmark += 1;
            }
        }

        return Ok(RobotJointState::new(out_robot_state_vector, RobotJointStateType::DOF, self)?);
    }
    pub fn map_joint_idx_to_joint_state_idxs(&self, joint_idx: usize, joint_state_type: &RobotJointStateType) -> Result<&Vec<usize>, OptimaError> {
        match joint_state_type {
            RobotJointStateType::DOF => {
                if joint_idx >= self.joint_idx_to_dof_state_idxs_mapping.len() {
            return Err(OptimaError::new_idx_out_of_bound_error(joint_idx, self.joint_idx_to_dof_state_idxs_mapping.len(), file!(), line!()));
        }

                return Ok(&self.joint_idx_to_dof_state_idxs_mapping[joint_idx]);
            }
            RobotJointStateType::Full => {
                if joint_idx >= self.joint_idx_to_full_state_idxs_mapping.len() {
                    return Err(OptimaError::new_idx_out_of_bound_error(joint_idx, self.joint_idx_to_full_state_idxs_mapping.len(), file!(), line!()));
                }

                return Ok(&self.joint_idx_to_full_state_idxs_mapping[joint_idx]);
            }
        }
    }
    pub fn map_joint_idx_and_sub_dof_idx_to_joint_state_idx(&self, joint_idx: usize, joint_sub_dof_idx: usize, joint_state_type: &RobotJointStateType) -> Result<usize, OptimaError> {
        let idxs = self.map_joint_idx_to_joint_state_idxs(joint_idx, joint_state_type)?;
        if joint_sub_dof_idx >= idxs.len() {
            return Err(OptimaError::new_idx_out_of_bound_error(joint_sub_dof_idx, idxs.len(), file!(), line!()));
        }
        return Ok(idxs[joint_sub_dof_idx]);
    }
    pub fn spawn_robot_joint_state(&self, joint_state: DVector<f64>, robot_joint_state_type: RobotJointStateType) -> Result<RobotJointState, OptimaError> {
        return RobotJointState::new(joint_state, robot_joint_state_type, self);
    }
    pub fn spawn_robot_joint_state_try_auto_type(&self, joint_state: DVector<f64>) -> Result<RobotJointState, OptimaError> {
        return RobotJointState::new_try_auto_type(joint_state, self);
    }
    pub fn spawn_zeros_robot_joint_state(&self, robot_state_type: RobotJointStateType) -> RobotJointState {
        let mut out_joint_state = match robot_state_type {
            RobotJointStateType::DOF => { DVector::zeros(self.num_dofs) }
            RobotJointStateType::Full => { DVector::zeros(self.num_axes) }
        };

        return match robot_state_type {
            RobotJointStateType::DOF => {
                RobotJointState::new_unchecked(out_joint_state, robot_state_type.clone())
            }
            RobotJointStateType::Full => {
                for (i, axis) in self.ordered_joint_axes.iter().enumerate() {
                    if axis.is_fixed() {
                        let fixed_value = axis.fixed_value().unwrap();
                        out_joint_state[i] = fixed_value;
                    }
                }
                RobotJointState::new_unchecked(out_joint_state, robot_state_type.clone())
            }
        }
    }
    pub fn inject_joint_value_into_robot_joint_state(&self, robot_joint_state: &mut RobotJointState, joint_idx: usize, joint_sub_dof_idx: usize, joint_value: f64) -> Result<(), OptimaError> {
        let idx = self.map_joint_idx_and_sub_dof_idx_to_joint_state_idx(joint_idx, joint_sub_dof_idx, &robot_joint_state.robot_joint_state_type)?;
        robot_joint_state.joint_state[idx] = joint_value;

        return Ok(());
    }
    pub fn get_joint_state_bounds(&self, t: &RobotJointStateType) -> Vec<(f64, f64)> {
        let axes = match t {
            RobotJointStateType::DOF => { &self.ordered_dof_joint_axes }
            RobotJointStateType::Full => { &self.ordered_joint_axes }
        };

        let mut out_vec = vec![];

        for axis in axes {
            let fixed_value = axis.fixed_value();
            match fixed_value {
                None => { out_vec.push( axis.bounds() ) }
                Some(fixed_value) => { out_vec.push( (fixed_value, fixed_value) ); }
            }
        }

        out_vec
    }
    pub fn sample_joint_state(&self, t: &RobotJointStateType) -> RobotJointState {
        let axes = match t {
            RobotJointStateType::DOF => { &self.ordered_dof_joint_axes }
            RobotJointStateType::Full => { &self.ordered_joint_axes }
        };

        let mut out_dvec = DVector::zeros(axes.len());

        for (i, axis) in axes.iter().enumerate() {
            let fixed_value = axis.fixed_value();
            match fixed_value {
                None => {
                    let sample = SimpleSamplers::uniform_samples(&vec![axis.bounds()]);
                    out_dvec[i] = sample[0];
                }
                Some(fixed_value) => {
                    out_dvec[i] = fixed_value
                }
            }
        }

        return RobotJointState::new(out_dvec, t.clone(), self).expect("error");
    }
    pub fn print_robot_joint_state_summary(&self, robot_joint_state: &RobotJointState)  {
        let joint_axes = match robot_joint_state.robot_joint_state_type {
            RobotJointStateType::DOF => { &self.ordered_dof_joint_axes }
            RobotJointStateType::Full => { &self.ordered_joint_axes }
        };

        for (i, joint_axis) in joint_axes.iter().enumerate() {
            optima_print(&format!("Joint state index {} ---> ", i), PrintMode::Println, PrintColor::Blue, true);
            optima_print(&format!("   > joint name: {}", self.robot_configuration_module.robot_model_module().joints()[joint_axis.joint_idx()].name()), PrintMode::Println, PrintColor::None, false);
            optima_print(&format!("   > joint index: {}", joint_axis.joint_idx()), PrintMode::Println, PrintColor::None, false);
            optima_print(&format!("   > joint sub dof index: {}", joint_axis.joint_sub_dof_idx()), PrintMode::Println, PrintColor::None, false);
            optima_print(&format!("   > axis: {:?}", joint_axis.axis()), PrintMode::Println, PrintColor::None, false);
            optima_print(&format!("   > joint value: {}", robot_joint_state[i]), PrintMode::Println, PrintColor::None, false);
        }
    }
    pub fn robot_name(&self) -> &str {
        return self.robot_configuration_module.robot_name()
    }
}
impl SaveAndLoadable for RobotJointStateModule {
    type SaveType = String;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        self.robot_configuration_module.get_serialization_string()
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let robot_configuration_module = RobotConfigurationModule::load_from_json_string(&load)?;
        return Ok(RobotJointStateModule::new(robot_configuration_module));
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotJointStateModule {
    #[new]
    pub fn new_py(robot_name: &str, configuration_name: Option<&str>) -> RobotJointStateModule {
        return Self::new_from_names(RobotNames::new(robot_name, configuration_name)).expect("error");
    }
    pub fn convert_joint_state_to_full_state_py(&self, joint_state: Vec<f64>) -> Vec<f64> {
        let robot_state = self.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let res = self.convert_joint_state_to_full_state(&robot_state).expect("error");
        return NalgebraConversions::dvector_to_vec(&res.joint_state);
    }
    pub fn convert_joint_state_to_dof_state_py(&self, joint_state: Vec<f64>) -> Vec<f64> {
        let robot_state = self.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let res = self.convert_joint_state_to_dof_state(&robot_state).expect("error");
        return NalgebraConversions::dvector_to_vec(&res.joint_state);
    }
    pub fn num_dofs_py(&self) -> usize { self.num_dofs() }
    pub fn num_axes_py(&self) -> usize {
        self.num_axes()
    }
}

/// WASM implementations.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotJointStateModule {
    #[wasm_bindgen(constructor)]
    pub fn new_wasm(robot_name: String, configuration_name: Option<String>) -> RobotJointStateModule {
        return match configuration_name {
            None => { Self::new_from_names(RobotNames::new(&robot_name, None)).expect("error") }
            Some(c) => { Self::new_from_names(RobotNames::new(&robot_name, Some(&c))).expect("error") }
        }
    }
    pub fn convert_joint_state_to_full_state_wasm(&self, joint_state: Vec<f64>) -> Vec<f64> {
        let robot_state = self.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let res = self.convert_joint_state_to_full_state(&robot_state).expect("error");
        return NalgebraConversions::dvector_to_vec(&res.joint_state);
    }
    pub fn convert_joint_state_to_dof_state_wasm(&self, joint_state: Vec<f64>) -> Vec<f64> {
        let robot_state = self.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let res = self.convert_joint_state_to_dof_state(&robot_state).expect("error");
        return NalgebraConversions::dvector_to_vec(&res.joint_state);
    }
    pub fn num_dofs_wasm(&self) -> usize { self.num_dofs() }
    pub fn num_axes_wasm(&self) -> usize {
        self.num_axes()
    }
}

/// "Robot states" are vectors that contain scalar joint values for each joint axis in the robot model.
/// These objects are sometimes referred to as robot configurations or robot poses in the robotics literature,
/// but in this library, we will stick to the convention of referring to them as robot states.
///
/// A `RobotJointState` object contains the vector of joint angles in the field `joint_state`, as well as a
/// state type (either DOF or Full).
///
/// A DOF joint state only contains values for joint values that are free (not fixed), while the Full joint state
/// includes joint values for ALL present joint axes (even if they are fixed).  A dof joint state is important
/// for operations such as optimization where only the free values are decision variables,
/// while a full joint state is important for operations such as forward kinematics where all present joint
/// axes need to somehow contribute to the model.
///
/// The library will ensure that mathematical operations (additions, scalar multiplication, etc) can
/// only occur over robot states of the same type.  Conversions between DOF and Full states can be done
/// via the `RobotJointStateModule`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotJointState {
    joint_state: DVector<f64>,
    robot_joint_state_type: RobotJointStateType
}
impl RobotJointState {
    fn new(joint_state: DVector<f64>, robot_state_type: RobotJointStateType, robot_state_module: &RobotJointStateModule) -> Result<Self, OptimaError> {
        match robot_state_type {
            RobotJointStateType::DOF => {
                if robot_state_module.num_dofs() != joint_state.len() {
                    return Err(OptimaError::new_robot_state_vec_wrong_size_error("RobotJointState::new", joint_state.len(), robot_state_module.num_dofs(), file!(), line!()));
                }
            }
            RobotJointStateType::Full => {
                if robot_state_module.num_axes() != joint_state.len() {
                    return Err(OptimaError::new_robot_state_vec_wrong_size_error("RobotJointState::new", joint_state.len(), robot_state_module.num_axes(), file!(), line!()));
                }
            }
        }

        Ok(Self {
            joint_state,
            robot_joint_state_type: robot_state_type
        })
    }
    fn new_try_auto_type(joint_state: DVector<f64>, robot_state_module: &RobotJointStateModule) -> Result<Self, OptimaError> {
        return if robot_state_module.num_axes() == joint_state.len() {
            Ok(Self::new_unchecked(joint_state, RobotJointStateType::Full))
        } else if robot_state_module.num_dofs() == joint_state.len() {
            Ok(Self::new_unchecked(joint_state, RobotJointStateType::DOF))
        } else {
            Err(OptimaError::new_generic_error_str(&format!("Could not successfully make an auto \
            RobotJointState in try_new_auto_type().  The given state length was {} while either {} or {} was required.",
                                                            joint_state.len(), robot_state_module.num_axes(), robot_state_module.num_dofs()),
                                                   file!(),
                                                   line!()))
        }
    }
    fn new_unchecked(joint_state: DVector<f64>, robot_state_type: RobotJointStateType) -> Self {
        Self {
            joint_state,
            robot_joint_state_type: robot_state_type
        }
    }
    pub fn joint_state(&self) -> &DVector<f64> {
        &self.joint_state
    }
    pub fn robot_joint_state_type(&self) -> &RobotJointStateType {
        &self.robot_joint_state_type
    }
    pub fn len(&self) -> usize {
        return self.joint_state.len();
    }
}
impl Add for RobotJointState {
    type Output = Result<RobotJointState, OptimaError>;

    fn add(self, rhs: Self) -> Self::Output {
        if &self.robot_joint_state_type != &rhs.robot_joint_state_type {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to add robot states of different types ({:?} + {:?}).", self.robot_joint_state_type(), rhs.robot_joint_state_type()), file!(), line!()));
        }
        return Ok(RobotJointState::new_unchecked(self.joint_state() + rhs.joint_state(), self.robot_joint_state_type.clone()))
    }
}
impl Mul<RobotJointState> for f64 {
    type Output = RobotJointState;

    fn mul(self, rhs: RobotJointState) -> Self::Output {
        return RobotJointState::new_unchecked(self * rhs.joint_state(), rhs.robot_joint_state_type.clone());
    }
}
impl Index<usize> for RobotJointState {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.joint_state[index];
    }
}
impl IndexMut<usize> for RobotJointState {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.joint_state[index]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobotJointStateType {
    DOF,
    Full
}

