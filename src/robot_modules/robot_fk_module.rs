#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_joint_state_module::{RobotJointState, RobotJointStateModule, RobotJointStateType};
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_robot::joint::JointAxisPrimitiveType;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

/// The `RobotFKModule` computes the forward kinematics of a robot configuration.  Forward kinematics
/// takes as input a robot joint_state and outputs the SE(3) poses of all links on the robot.
///
/// # Example
/// ```
/// use nalgebra::DVector;
/// use optima::robot_modules::robot_fk_module::RobotFKModule;
/// use optima::robot_modules::robot_joint_state_module::RobotJointStateModule;
/// use optima::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
///
/// let robot_joint_state_module = RobotJointStateModule::new_from_names("ur5", None).expect("error");
/// let robot_joint_state = robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::zeros(6)).expect("error");
///
/// let robot_fk_module = RobotFKModule::new_from_names("ur5", None).expect("error");
/// let fk_res = robot_fk_module.compute_fk(&robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
/// fk_res.print_summary();
///
/// // Output:
/// // Link 0 base_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.0, 0.0]], is_identity: true, rot_is_identity: true, translation_is_zeros: true }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.0, 0.0]]
/// // Link 1 shoulder_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.0, 0.089159]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.0, 0.089159]]
/// // Link 2 upper_arm_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.13585, 0.089159]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.13585, 0.089159]]
/// // Link 3 forearm_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.016149999999999998, 0.514159]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.016149999999999998, 0.514159]]
/// // Link 4 wrist_1_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.016149999999999998, 0.906409]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.016149999999999998, 0.906409]]
/// // Link 5 wrist_2_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.10915, 0.906409]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.10915, 0.906409]]
/// // Link 6 wrist_3_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.10915, 1.001059]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.10915, 1.001059]]
/// // Link 7 ee_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.7071067805519557, 0.7071067818211392], translation: [[0.0, 0.19145, 1.001059]], is_identity: false, rot_is_identity: false, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 1.5707963250000003]]
/// //    > Pose Translation: [[0.0, 0.19145, 1.001059]]
/// // Link 8 base --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, -1.0, 1.7948965149208059e-9], translation: [[0.0, 0.0, 0.0]], is_identity: false, rot_is_identity: false, translation_is_zeros: true }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, 0.0, -3.14159265]]
/// //    > Pose Translation: [[0.0, 0.0, 0.0]]
/// // Link 9 tool0 --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [-0.7071067805519557, 0.0, 0.0, 0.7071067818211392], translation: [[0.0, 0.19145, 1.001059]], is_identity: false, rot_is_identity: false, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[-1.5707963250000003, 0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.19145, 1.001059]]
/// // Link 10 world --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.0, 0.0]], is_identity: true, rot_is_identity: true, translation_is_zeros: true }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.0, 0.0]]
/// //
/// ```
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFKModule {
    robot_configuration_module: RobotConfigurationModule,
    robot_joint_state_module: RobotJointStateModule,
    starter_result: RobotFKResult
}
impl RobotFKModule {
    pub fn new(robot_configuration_module: RobotConfigurationModule, robot_joint_state_module: RobotJointStateModule) -> Self {
        let mut starter_result = RobotFKResult { link_entries: vec![] };
        let links = robot_configuration_module.robot_model_module().links();
        for (i, link) in links.iter().enumerate() {
            starter_result.link_entries.push( RobotFKResultLinkEntry {
                link_idx: i,
                link_name: link.name().to_string(),
                pose: None
            } )
        }

        Self {
            robot_configuration_module,
            robot_joint_state_module,
            starter_result
        }
    }
    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_name, configuration_name)?;
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        return Ok(Self::new(robot_configuration_module, robot_joint_state_module));
    }
    pub fn compute_fk(&self, joint_state: &RobotJointState, t: &OptimaSE3PoseType) -> Result<RobotFKResult, OptimaError> {
        let joint_state = self.robot_joint_state_module.convert_joint_state_to_full_state(joint_state)?;
        let mut output = self.starter_result.clone();

        let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();

        for link_tree_traversal_layer in link_tree_traversal_layers {
            for link_idx in link_tree_traversal_layer {
                self.compute_fk_on_single_link(&joint_state, *link_idx, t, &mut output)?;
            }
        }

        return Ok(output);
    }
    /// This function computes the forward kinematics for some part of the whole robot configuration.
    /// It provides three primary arguments over the standard `compute_fk` function:
    /// - start_link_idx: An optional link index that will serve as the beginning of the partial
    /// floating chain.  If this is None, the start of the chain will be the default world base link.
    /// - end_link_idx: An optional link index that will serve as the end of the partial floating chain.
    /// If this is None, the whole forward kinematics process will play out from the start_link_idx on.
    /// - start_link_pose: An optional SE(3) pose used to situate the beginning link (start_link_idx) in the chain.
    /// If this is None, this pose will be the default pose just based on the given `RobotState`.
    pub fn compute_fk_floating_chain(&self, joint_state: &RobotJointState, t: &OptimaSE3PoseType, start_link_idx: Option<usize>, end_link_idx: Option<usize>, start_link_pose: Option<OptimaSE3Pose>) -> Result<RobotFKResult, OptimaError> {
        let num_links = self.robot_configuration_module.robot_model_module().links().len();
        if let Some(start_link_idx) = start_link_idx {
            if start_link_idx >= num_links {
            return Err(OptimaError::new_idx_out_of_bound_error(start_link_idx, num_links, file!(), line!()));
        }
        }
        if let Some(end_link_idx) = end_link_idx {
            if end_link_idx >= num_links {
                return Err(OptimaError::new_idx_out_of_bound_error(end_link_idx, num_links, file!(), line!()));
            }
        }

        let joint_state = self.robot_joint_state_module.convert_joint_state_to_full_state(joint_state)?;
        let mut output = self.starter_result.clone();

        let start_link_idx = match start_link_idx {
            None => { self.robot_configuration_module.robot_model_module().robot_base_link_idx() }
            Some(start_link_idx) => { start_link_idx }
        };

        match &start_link_pose {
            None => {
                let mut tmp_output = self.starter_result.clone();
                let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();

                for link_tree_traversal_layer in link_tree_traversal_layers {
                    for link_idx in link_tree_traversal_layer {
                        self.compute_fk_on_single_link(&joint_state, *link_idx, t, &mut tmp_output)?;
                        if *link_idx == start_link_idx {
                            output.link_entries[start_link_idx].pose = tmp_output.link_entries[start_link_idx].pose.clone();
                            break;
                        }
                    }
                }
            }
            Some(s) => {
                if s.map_to_pose_type() != t {
                    return Err(OptimaError::new_generic_error_str(&format!("Given start link pose was not the correct type ({:?} instead of {:?})", s.map_to_pose_type(), t), file!(), line!()));
                }
                output.link_entries[start_link_idx].pose = Some(s.clone());
            }
        }

        let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();

        let links = self.robot_configuration_module.robot_model_module().links();

        for link_tree_traversal_layer in link_tree_traversal_layers {
            for link_idx in link_tree_traversal_layer {
                let predecessor_link_idx_option = links[*link_idx].preceding_link_idx();
                if predecessor_link_idx_option.is_none() { continue; }
                let predecessor_link_idx = predecessor_link_idx_option.unwrap();
                if output.link_entries[predecessor_link_idx].pose.is_some() {
                    self.compute_fk_on_single_link(&joint_state, *link_idx, t, &mut output)?;
                    if let Some(end_link_idx_) = end_link_idx {
                        if end_link_idx_ == *link_idx {
                            return Ok(output);
                        }
                    }
                }
            }
        }

        if end_link_idx.is_some() {
            return Err(OptimaError::new_generic_error_str(&format!("No valid link chain found between link {} and {} in compute_fk_partial_chain", start_link_idx, end_link_idx.unwrap()), file!(), line!()));
        }

        return Ok(output);
    }
    fn compute_fk_on_single_link(&self, joint_state: &RobotJointState, link_idx: usize, t: &OptimaSE3PoseType, output: &mut RobotFKResult) -> Result<(), OptimaError> {
        let link = self.robot_configuration_module.robot_model_module().get_link_by_idx(link_idx)?;
        let preceding_link_option = link.preceding_link_idx();
        if preceding_link_option.is_none() {
            output.link_entries[link_idx].pose = Some(OptimaSE3Pose::new_from_euler_angles(0.,0.,0.,0.,0.,0., t));
            return Ok(());
        }

        let preceding_link_idx = preceding_link_option.unwrap();

        let preceding_joint_option = link.preceding_joint_idx();
        if preceding_joint_option.is_none() {
            output.link_entries[link_idx].pose = output.link_entries[preceding_link_idx].pose.clone();
            return Ok(());
        }

        let preceding_joint_idx = preceding_joint_option.unwrap();
        let preceding_joint = &self.robot_configuration_module.robot_model_module().joints()[preceding_joint_idx];

        let full_state_idxs = self.robot_joint_state_module.map_joint_idx_to_joint_state_idxs(preceding_joint_idx, &RobotJointStateType::Full)?;

        let mut out_pose = output.link_entries[preceding_link_idx].pose.clone().expect("error");

        let offset_pose_all = preceding_joint.origin_offset_pose();
        out_pose = out_pose.multiply(offset_pose_all.get_pose_by_type(t), false)?;

        let joint_axes = preceding_joint.joint_axes();

        for (i, full_state_idx) in full_state_idxs.iter().enumerate() {
            let joint_axis = &joint_axes[i];
            let joint_value = joint_state[*full_state_idx];

            let axis_pose = match joint_axis.axis_primitive_type() {
                JointAxisPrimitiveType::Rotation => {
                    let axis = &joint_axis.axis_as_unit();
                    OptimaSE3Pose::new_from_axis_angle(axis, joint_value, 0.,0.,0., t)
                }
                JointAxisPrimitiveType::Translation => {
                    let axis = joint_value * &joint_axis.axis();
                    OptimaSE3Pose::new_from_euler_angles(0.,0.,0., axis[0], axis[1], axis[2], t)
                }
            };

            out_pose = out_pose.multiply(&axis_pose, false)?;
        }

        output.link_entries[link_idx].pose = Some(out_pose);

        Ok(())
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotFKModule {
    #[new]
    pub fn new_py(robot_name: &str, configuration_name: Option<&str>) -> RobotFKModule {
        return Self::new_from_names(robot_name, configuration_name).expect("error");
    }
    #[args(pose_type = "\"ImplicitDualQuaternion\"")]
    pub fn compute_fk_py(&self, joint_state: Vec<f64>, pose_type: &str) -> RobotFKResult {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        if pose_type == "ImplicitDualQuaternion" {
            return self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
        } else if pose_type == "UnitQuaternionAndTranslation" {
            return self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::UnitQuaternionAndTranslation).expect("error");
        } else if pose_type == "RotationMatrixAndTranslation" {
            return self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::RotationMatrixAndTranslation).expect("error");
        } else if pose_type == "HomogeneousMatrix" {
            return self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::HomogeneousMatrix).expect("error");
        } else if pose_type == "EulerAnglesAndTranslation" {
            return self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::EulerAnglesAndTranslation).expect("error");
        } else {
            panic!("{} is not a valid pose_type in compute_fk_py()", pose_type)
        }
    }
}

/// WASM implementations.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotFKModule {
    #[wasm_bindgen(constructor)]
    pub fn new_wasm(robot_name: String, configuration_name: Option<String>) -> RobotFKModule {
        return match &configuration_name {
            None => {
                RobotFKModule::new_from_names(&robot_name, None).expect("error")
            }
            Some(c) => {
                RobotFKModule::new_from_names(&robot_name, Some(c)).expect("error")
            }
        }
    }
    pub fn compute_fk_wasm(&self, joint_state: Vec<f64>, pose_type: Option<String>) -> JsValue {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let pose_type_ = match pose_type {
            None => { "ImplicitDualQuaternion".to_string() }
            Some(p) => { p }
        };
        if pose_type_ == "ImplicitDualQuaternion" {
            return JsValue::from_serde(&self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error")).unwrap()
        } else if pose_type_ == "UnitQuaternionAndTranslation" {
            return JsValue::from_serde(&self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::UnitQuaternionAndTranslation).expect("error")).unwrap()
        } else if pose_type_ == "RotationMatrixAndTranslation" {
            return JsValue::from_serde(&self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::RotationMatrixAndTranslation).expect("error")).unwrap()
        } else if pose_type_ == "HomogeneousMatrix" {
            return JsValue::from_serde(&self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::HomogeneousMatrix).expect("error")).unwrap()
        } else if pose_type_ == "EulerAnglesAndTranslation" {
            return JsValue::from_serde(&self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::EulerAnglesAndTranslation).expect("error")).unwrap()
        } else {
            panic!("{} is not a valid pose_type in compute_fk_py()", pose_type_)
        }
    }
}

/// The output of a forward kinematics computation.
/// The primary field in this object is `link_entries`.  This is a list of `RobotFKResultLinkEntry`
/// objects.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFKResult {
    link_entries: Vec<RobotFKResultLinkEntry>
}
impl RobotFKResult {
    /// Returns a reference to the results link entries.
    pub fn link_entries(&self) -> &Vec<RobotFKResultLinkEntry> {
        &self.link_entries
    }
    /// Prints a summary of the forward kinematics result.
    pub fn print_summary(&self) {
        for e in self.link_entries() {
            optima_print(&format!("Link {} {} ---> ", e.link_idx, e.link_name), PrintMode::Println, PrintColor::Blue, true);
            optima_print(&format!("   > Pose: {:?}", e.pose), PrintMode::Println, PrintColor::None, false);
            if e.pose.is_some() {
                let euler_angles = e.pose.as_ref().unwrap().to_euler_angles_and_translation();
                optima_print(&format!("   > Pose Euler Angles: {:?}", euler_angles.0), PrintMode::Println, PrintColor::None, false);
                optima_print(&format!("   > Pose Translation: {:?}", euler_angles.1), PrintMode::Println, PrintColor::None, false);
            }
        }
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotFKResult {
    pub fn print_summary_py(&self) {
        self.print_summary();
    }
    pub fn get_robot_fk_result_link_entry(&self, link_idx: usize) -> RobotFKResultLinkEntry {
        return self.link_entries[link_idx].clone();
    }
    pub fn get_link_pose(&self, link_idx: usize) -> Option<Vec<Vec<f64>>> {
        let e = &self.link_entries[link_idx];
        return e.pose_py();
    }
}

/// A `RobotFKResultLinkEntry` specifies information about one particular link in the forward kinematics
/// process.  It provides the link index, the link's name, and the pose of the link.
/// If the link is NOT included in the FK computation (the link is not present in the model, etc)
/// the pose will be None.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFKResultLinkEntry {
    link_idx: usize,
    link_name: String,
    pose: Option<OptimaSE3Pose>
}
impl RobotFKResultLinkEntry {
    pub fn link_idx(&self) -> usize {
        self.link_idx
    }
    pub fn link_name(&self) -> &str {
        &self.link_name
    }
    pub fn pose(&self) -> &Option<OptimaSE3Pose> {
        &self.pose
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotFKResultLinkEntry {
    pub fn link_idx_py(&self) -> usize { self.link_idx }
    pub fn link_name_py(&self) -> String { self.link_name.clone() }
    pub fn pose_py(&self) -> Option<Vec<Vec<f64>>> {
        return match &self.pose {
            None => {
                None
            }
            Some(pose) => {
                Some(pose.to_vec_representation())
            }
        }
    }
}
