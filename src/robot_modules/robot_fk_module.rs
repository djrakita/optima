#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_generator_module::RobotConfigurationGeneratorModule;
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_state_module::{RobotState, RobotStateModule};
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_robot::joint::JointAxisPrimitiveType;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFKModule {
    robot_configuration_module: RobotConfigurationModule,
    robot_state_module: RobotStateModule,
    starter_out_vec: Vec<Option<OptimaSE3Pose>>,
    starter_result: RobotFKResult
}
impl RobotFKModule {
    pub fn new(robot_configuration_module: RobotConfigurationModule, robot_state_module: RobotStateModule) -> Self {
        let mut starter_out_vec = vec![];
        let l = robot_configuration_module.robot_model_module().links().len();
        for _ in 0..l { starter_out_vec.push(None); }

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
            robot_state_module,
            starter_out_vec,
            starter_result
        }
    }
    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationGeneratorModule::new(robot_name)?.generate_configuration(configuration_name)?;
        let robot_state_module = RobotStateModule::new(robot_configuration_module.clone());
        return Ok(Self::new(robot_configuration_module, robot_state_module));
    }
    pub fn compute_fk(&self, state: &RobotState, t: &OptimaSE3PoseType) -> Result<RobotFKResult, OptimaError> {
        let state = self.robot_state_module.convert_state_to_full_state(state)?;
        let mut output = self.starter_result.clone();

        let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();

        for link_tree_traversal_layer in link_tree_traversal_layers {
            for link_idx in link_tree_traversal_layer {
                self.compute_fk_on_single_link(&state, *link_idx, t, &mut output)?;
            }
        }

        return Ok(output);
    }
    fn compute_fk_on_single_link(&self, state: &RobotState, link_idx: usize, t: &OptimaSE3PoseType, output: &mut RobotFKResult) -> Result<(), OptimaError> {
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

        let full_state_idxs = self.robot_state_module.map_joint_idx_to_full_state_idxs(preceding_joint_idx)?;

        let mut out_pose = output.link_entries[preceding_link_idx].pose.clone().expect("error");

        let offset_pose_all = preceding_joint.origin_offset_pose();
        out_pose = out_pose.multiply(offset_pose_all.get_pose_by_type(t), false)?;

        let joint_axes = preceding_joint.joint_axes();

        for (i, full_state_idx) in full_state_idxs.iter().enumerate() {
            let joint_axis = &joint_axes[i];
            let joint_value = state[*full_state_idx];

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

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotFKModule {
    #[new]
    pub fn new_py(robot_name: &str, configuration_name: Option<&str>) -> RobotFKModule {
        return Self::new_from_names(robot_name, configuration_name).expect("error");
    }
    #[args(pose_type = "\"ImplicitDualQuaternion\"")]
    pub fn compute_fk_py(&self, state: Vec<f64>, pose_type: &str) -> RobotFKResult {
        let robot_state = self.robot_state_module.spawn_robot_state_try_auto_type(NalgebraConversions::vec_to_dvector(&state)).expect("error");
        if pose_type == "ImplicitDualQuaternion" {
            return self.compute_fk(&robot_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
        } else if pose_type == "UnitQuaternionAndTranslation" {
            return self.compute_fk(&robot_state, &OptimaSE3PoseType::UnitQuaternionAndTranslation).expect("error");
        } else if pose_type == "RotationMatrixAndTranslation" {
            return self.compute_fk(&robot_state, &OptimaSE3PoseType::RotationMatrixAndTranslation).expect("error");
        } else if pose_type == "HomogeneousMatrix" {
            return self.compute_fk(&robot_state, &OptimaSE3PoseType::HomogeneousMatrix).expect("error");
        } else {
            panic!("{} is not a valid pose_type in compute_fk_py()", pose_type)
        }
    }
}

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

    pub fn compute_fk_wasm(&self, state: Vec<f64>, pose_type: &str) -> JsValue {
        let robot_state = self.robot_state_module.spawn_robot_state_try_auto_type(NalgebraConversions::vec_to_dvector(&state)).expect("error");
        if pose_type == "ImplicitDualQuaternion" {
            return JsValue::from_serde(&self.compute_fk(&robot_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error")).unwrap()
        } else if pose_type == "UnitQuaternionAndTranslation" {
            return JsValue::from_serde(&self.compute_fk(&robot_state, &OptimaSE3PoseType::UnitQuaternionAndTranslation).expect("error")).unwrap()
        } else if pose_type == "RotationMatrixAndTranslation" {
            return JsValue::from_serde(&self.compute_fk(&robot_state, &OptimaSE3PoseType::RotationMatrixAndTranslation).expect("error")).unwrap()
        } else if pose_type == "HomogeneousMatrix" {
            return JsValue::from_serde(&self.compute_fk(&robot_state, &OptimaSE3PoseType::HomogeneousMatrix).expect("error")).unwrap()
        } else {
            panic!("{} is not a valid pose_type in compute_fk_py()", pose_type)
        }


        // let res = self.compute_fk(&robot_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
        // JsValue::from_serde(&res).unwrap()
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFKResult {
    link_entries: Vec<RobotFKResultLinkEntry>
}
impl RobotFKResult {
    pub fn link_entries(&self) -> &Vec<RobotFKResultLinkEntry> {
        &self.link_entries
    }
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
    pub fn get_link_pose_euler_angles_and_translation(&self, link_idx: usize) -> Option<Vec<Vec<f64>>> {
        let e = &self.link_entries[link_idx];
        return e.pose_euler_angles_and_translation_py();
    }
}

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
    pub fn pose_euler_angles_and_translation_py(&self) -> Option<Vec<Vec<f64>>> {
        return match &self.pose {
            None => {
                None
            }
            Some(pose) => {
                let euler_angles_and_translation = pose.to_euler_angles_and_translation();
                let mut out_vec = vec![];
                let e = &euler_angles_and_translation.0;
                let t = &euler_angles_and_translation.1;
                out_vec.push(vec![ e[0], e[1], e[2] ]);
                out_vec.push(vec![ t[0], t[1], t[2] ]);
                return Some(out_vec);
            }
        }
    }
}
