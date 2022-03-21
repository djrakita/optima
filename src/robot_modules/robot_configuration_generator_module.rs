use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;

#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::{RobotDirUtils, RobotModuleJsonType};

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotConfigurationGeneratorModule {
    robot_configuration_infos: HashMap<String, RobotConfigurationInfo>,
    robot_model_module: RobotModelModule
}
impl RobotConfigurationGeneratorModule {
    pub fn new(robot_name: &str) -> Result<Self, OptimaError> {
        let path = RobotDirUtils::get_path_to_robot_module_json(robot_name, RobotModuleJsonType::ConfigurationGeneratorModule)?;
        if path.exists() {
            todo!()
        } else {
            let robot_model_module = RobotModelModule::new(robot_name)?;
            let mut out_self = Self {
                robot_configuration_infos: Default::default(),
                robot_model_module
            };
            out_self.save_to_json();
            Ok(out_self)
        }
    }

    pub fn save_to_json(&self) {

    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotConfigurationGeneratorModule {

}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotConfigurationGeneratorModule {

}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotConfigurationInfo {
    configuration_name: String,
    description: Option<String>,
    dead_end_link_idxs: Vec<usize>,
    inactive_joint_idxs: Vec<usize>,
    base_offset: OptimaSE3Pose,
    mobile_base_mode: MobileBaseMode
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InactiveJointInfo {
    joint_idx: usize,
    num_joint_dofs: usize,
    fixed_joint_values: Vec<f64>
}
impl InactiveJointInfo {
    pub fn joint_idx(&self) -> usize {
        self.joint_idx
    }
    pub fn num_joint_dofs(&self) -> usize {
        self.num_joint_dofs
    }
    pub fn fixed_joint_values(&self) -> &Vec<f64> {
        &self.fixed_joint_values
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MobileBaseMode {
    Static,
    Floating { x_bounds: (f64, f64), y_bounds: (f64, f64), z_bounds: (f64, f64), xr_bounds: (f64, f64), yr_bounds: (f64, f64), zr_bounds: (f64, f64) },
    PlanarTranslation { x_bounds: (f64, f64), y_bounds: (f64, f64) },
    PlanarRotation { zr_bounds: (f64, f64) },
    PlanarTranslationAndRotation { x_bounds: (f64, f64), y_bounds: (f64, f64), zr_bounds: (f64, f64) }
}
impl MobileBaseMode {
    pub fn get_bounds(&self) -> Vec<(f64, f64)> {
        match self {
            MobileBaseMode::Static => {
                vec![]
            }
            MobileBaseMode::Floating { x_bounds, y_bounds, z_bounds, xr_bounds, yr_bounds, zr_bounds } => {
                vec![x_bounds.clone(), y_bounds.clone(), z_bounds.clone(), xr_bounds.clone(), yr_bounds.clone(), zr_bounds.clone()]
            }
            MobileBaseMode::PlanarTranslation { x_bounds, y_bounds } => {
                vec![ x_bounds.clone(), y_bounds.clone() ]
            }
            MobileBaseMode::PlanarRotation { zr_bounds } => {
                vec![ zr_bounds.clone() ]
            }
            MobileBaseMode::PlanarTranslationAndRotation { x_bounds, y_bounds, zr_bounds } => {
                vec![ x_bounds.clone(), y_bounds.clone(), zr_bounds.clone() ]
            }
        }
    }
}

