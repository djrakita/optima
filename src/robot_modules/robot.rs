use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_modules::robot_fk_module::RobotFKModule;
use crate::robot_modules::robot_joint_state_module::RobotJointStateModule;
use crate::utils::utils_errors::OptimaError;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Robot {
    robot_configuration_module: RobotConfigurationModule,
    robot_file_manager_module: RobotMeshFileManagerModule,
    robot_joint_state_module: RobotJointStateModule,
    robot_fk_module: RobotFKModule
}
impl Robot {
    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_name, configuration_name)?;
        let robot_file_manager_module = RobotMeshFileManagerModule::new(robot_configuration_module.robot_model_module())?;
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        let robot_fk_module = RobotFKModule::new(robot_configuration_module.clone(), robot_joint_state_module.clone());

        Ok(Self {
            robot_configuration_module,
            robot_file_manager_module,
            robot_joint_state_module,
            robot_fk_module
        })
    }
    pub fn robot_configuration_module(&self) -> &RobotConfigurationModule {
        &self.robot_configuration_module
    }
    pub fn robot_file_manager_module(&self) -> &RobotMeshFileManagerModule {
        &self.robot_file_manager_module
    }
    pub fn robot_joint_state_module(&self) -> &RobotJointStateModule {
        &self.robot_joint_state_module
    }
    pub fn robot_fk_module(&self) -> &RobotFKModule {
        &self.robot_fk_module
    }
}