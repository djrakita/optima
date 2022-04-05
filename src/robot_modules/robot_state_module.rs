#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_generator_module::RobotConfigurationGeneratorModule;
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_robot::joint::JointAxis;

/// The `RobotStateModule` organizes and operates over robot states.  "Robot states" are vectors
/// that contain scalar transformation values for each joint axis in the robot model.  These objects
/// are sometimes referred to as robot configurations or robot poses in the robotics literature,
/// but in this library, we will stick to the convention of referring to them as robot states.
///
/// The `RobotStateModule` has two primary fields:
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
/// variants of robot states:
/// - A dof state
/// - A full state
///
/// A dof state only contains values for joint values that are free (not fixed), while the full state
/// includes joint values for ALL present joint axes (even if they are fixed).  A dof state is important
/// for operations such as optimization where only the free values are decision variables,
/// while a full state is important for operations such as forward kinematics where all present joint
/// axes need to somehow contribute to the model.
///
/// A dof state can be converted to a full state via the function `convert_dof_state_to_full_state`.
/// A full state can be converted to a dof state via the function `convert_full_state_to_dof_state`.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotStateModule {
    num_dofs: usize,
    num_axes: usize,
    ordered_dof_joint_axes: Vec<JointAxis>,
    ordered_joint_axes: Vec<JointAxis>,
    robot_configuration_module: RobotConfigurationModule
}
impl RobotStateModule {
    pub fn new(robot_configuration_module: RobotConfigurationModule) -> Self {
        let mut out_self = Self {
            num_dofs: 0,
            num_axes: 0,
            ordered_dof_joint_axes: vec![],
            ordered_joint_axes: vec![],
            robot_configuration_module
        };

        out_self.set_ordered_joint_axes();
        out_self.num_dofs = out_self.ordered_dof_joint_axes.len();
        out_self.num_axes = out_self.ordered_joint_axes.len();

        return out_self;
    }

    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationGeneratorModule::new(robot_name)?.generate_configuration(configuration_name)?;
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

    /// Converts a dof state to a full state.
    pub fn convert_dof_state_to_full_state(&self, dof_state: &DVector<f64>) -> Result<DVector<f64>, OptimaError> {
        if dof_state.len() != self.num_dofs {
            return Err(OptimaError::new_robot_state_vec_wrong_size_error("convert_dof_state_to_full_state", dof_state.len(), self.num_dofs, file!(), line!()))
        }

        if self.num_axes == self.num_dofs {
            return Ok(dof_state.clone());
        }

        let mut out_robot_state_vector = DVector::zeros(self.num_axes);

        let mut bookmark = 0 as usize;

        for (i, a) in self.ordered_joint_axes.iter().enumerate() {
            if a.is_fixed() {
                out_robot_state_vector[i] = a.fixed_value().unwrap();
            } else {
                out_robot_state_vector[i] = dof_state[bookmark];
                bookmark += 1;
            }
        }

        return Ok(out_robot_state_vector);
    }

    /// Converts a full state to a dof state.
    pub fn convert_full_state_to_dof_state(&self, full_state: &DVector<f64>) -> Result<DVector<f64>, OptimaError> {
        if full_state.len() != self.num_axes() {
            return Err(OptimaError::new_robot_state_vec_wrong_size_error("convert_full_state_to_dof_state", full_state.len(), self.num_axes, file!(), line!()))
        }

        if self.num_axes == self.num_dofs {
            return Ok(full_state.clone());
        }

        let mut out_robot_state_vector = DVector::zeros(self.num_dofs);

        let mut bookmark = 0 as usize;

        for (i, a) in self.ordered_joint_axes.iter().enumerate() {
            if !a.is_fixed() {
                out_robot_state_vector[bookmark] = full_state[i];
                bookmark += 1;
            }
        }

        return Ok(out_robot_state_vector);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotStateModule {
    #[new]
    pub fn new_py(robot_name: &str, configuration_name: Option<&str>) -> RobotStateModule {
        return Self::new_from_names(robot_name, configuration_name).expect("error");
    }

    pub fn convert_dof_state_to_full_state_py(&self, dof_state: Vec<f64>) -> Vec<f64> {
        let res = self.convert_dof_state_to_full_state(&NalgebraConversions::vec_to_dvector(&dof_state)).expect("error");
        return NalgebraConversions::dvector_to_vec(&res);
    }

    pub fn convert_full_state_to_dof_state_py(&self, full_state: Vec<f64>) -> Vec<f64> {
        let res = self.convert_full_state_to_dof_state(&NalgebraConversions::vec_to_dvector(&full_state)).expect("error");
        return NalgebraConversions::dvector_to_vec(&res);
    }

    pub fn num_dofs_py(&self) -> usize { self.num_dofs() }

    pub fn num_axes_py(&self) -> usize {
        self.num_axes()
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotStateModule {
    #[wasm_bindgen(constructor)]
    pub fn new_wasm(robot_name: String, configuration_name: Option<String>) -> RobotStateModule {
        return match configuration_name {
            None => { Self::new_from_names(&robot_name, None).expect("error") }
            Some(c) => { Self::new_from_names(&robot_name, Some(&c)).expect("error") }
        }
    }

    pub fn convert_dof_state_to_full_state_wasm(&self, dof_state: Vec<f64>) -> Vec<f64> {
        let res = self.convert_dof_state_to_full_state(&NalgebraConversions::vec_to_dvector(&dof_state)).expect("error");
        return NalgebraConversions::dvector_to_vec(&res);
    }

    pub fn convert_full_state_to_dof_state_wasm(&self, full_state: Vec<f64>) -> Vec<f64> {
        let res = self.convert_full_state_to_dof_state(&NalgebraConversions::vec_to_dvector(&full_state)).expect("error");
        return NalgebraConversions::dvector_to_vec(&res);
    }

    pub fn num_dofs_wasm(&self) -> usize { self.num_dofs() }

    pub fn num_axes_wasm(&self) -> usize {
        self.num_axes()
    }
}