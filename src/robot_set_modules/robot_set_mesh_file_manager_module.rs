#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_mesh_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::load_object_from_json_string;
use crate::utils::utils_traits::SaveAndLoadable;

/// RobotSet analogue of the `RobotSetMeshFileManagerModule`.  The same concepts apply, just on a set of possibly
/// multiple robots.
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotSetMeshFileManagerModule {
    robot_mesh_file_manager_modules: Vec<RobotMeshFileManagerModule>
}
impl RobotSetMeshFileManagerModule {
    pub fn new(robot_set_configuration_module: &RobotSetConfigurationModule) -> Result<Self, OptimaError> {
        let mut out_vec = vec![];

        for r in robot_set_configuration_module.robot_configuration_modules() {
            let robot_mesh_file_manager_module = RobotMeshFileManagerModule::new(r.robot_model_module())?;
            out_vec.push(robot_mesh_file_manager_module);
        }

        Ok(Self {
            robot_mesh_file_manager_modules: out_vec
        })
    }
    pub fn new_from_set_name(set_name: &str) -> Result<Self, OptimaError> {
        let robot_set_configuration_module = RobotSetConfigurationModule::new_from_set_name(set_name)?;
        return Self::new(&robot_set_configuration_module);
    }
    pub fn robot_mesh_file_manager_modules(&self) -> &Vec<RobotMeshFileManagerModule> {
        &self.robot_mesh_file_manager_modules
    }
}
impl SaveAndLoadable for RobotSetMeshFileManagerModule {
    type SaveType = String;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        self.robot_mesh_file_manager_modules.get_serialization_string()
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let robot_mesh_file_manager_modules = Vec::load_from_json_string(&load)?;
        Ok(Self {
            robot_mesh_file_manager_modules
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotSetMeshFileManagerModule {
    #[new]
    pub fn new_from_set_name_py(set_name: &str) -> Self {
        Self::new_from_set_name(set_name).expect("error")
    }
    #[staticmethod]
    pub fn new_py(robot_set_configuration_module: &RobotSetConfigurationModule) -> Self {
        Self::new(robot_set_configuration_module).expect("error")
    }
    pub fn robot_mesh_file_manager_modules_py(&self) -> Vec<RobotMeshFileManagerModule> {
        self.robot_mesh_file_manager_modules.clone()
    }
}