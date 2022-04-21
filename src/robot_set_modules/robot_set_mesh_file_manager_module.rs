use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_mesh_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::utils::utils_errors::OptimaError;

#[derive(Clone, Debug, Serialize, Deserialize)]
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