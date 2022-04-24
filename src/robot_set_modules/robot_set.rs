use serde::{Serialize, Deserialize};
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::robot_set_modules::robot_set_kinematics_module::RobotSetKinematicsModule;
use crate::robot_set_modules::robot_set_geometric_shape_module::RobotSetGeometricShapeModule;
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointStateModule;
use crate::robot_set_modules::robot_set_mesh_file_manager_module::RobotSetMeshFileManagerModule;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::load_object_from_json_string;
use crate::utils::utils_traits::SaveAndLoadable;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSet {
    robot_set_configuration_module: RobotSetConfigurationModule,
    robot_set_joint_state_module: RobotSetJointStateModule,
    robot_set_mesh_file_manager: RobotSetMeshFileManagerModule,
    robot_set_kinematics_module: RobotSetKinematicsModule,
    robot_set_geometric_shape_module: RobotSetGeometricShapeModule
}
impl RobotSet {
    pub fn new_from_robot_set_configuration_module(r: RobotSetConfigurationModule) -> Result<Self, OptimaError> {
        let robot_set_joint_state_module = RobotSetJointStateModule::new(&r);
        let robot_set_mesh_file_manager = RobotSetMeshFileManagerModule::new(&r)?;
        let robot_set_kinematics_module = RobotSetKinematicsModule::new(&r);
        let robot_set_geometric_shape_module = RobotSetGeometricShapeModule::new(&r)?;

        Ok(Self {
            robot_set_configuration_module: r,
            robot_set_joint_state_module,
            robot_set_mesh_file_manager,
            robot_set_kinematics_module,
            robot_set_geometric_shape_module
        })
    }
    pub fn new_from_set_name(set_name: &str) -> Result<Self, OptimaError> {
        let r = RobotSetConfigurationModule::new_from_set_name(set_name)?;
        return Self::new_from_robot_set_configuration_module(r);
    }
    pub fn robot_set_configuration_module(&self) -> &RobotSetConfigurationModule {
        &self.robot_set_configuration_module
    }
    pub fn robot_set_joint_state_module(&self) -> &RobotSetJointStateModule {
        &self.robot_set_joint_state_module
    }
    pub fn robot_set_mesh_file_manager(&self) -> &RobotSetMeshFileManagerModule {
        &self.robot_set_mesh_file_manager
    }
    pub fn robot_set_kinematics_module(&self) -> &RobotSetKinematicsModule {
        &self.robot_set_kinematics_module
    }
    pub fn robot_set_geometric_shape_module(&self) -> &RobotSetGeometricShapeModule {
        &self.robot_set_geometric_shape_module
    }
}
impl SaveAndLoadable for RobotSet {
    type SaveType = (String, String, String, String, String);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_set_configuration_module.get_serialization_string(),
         self.robot_set_joint_state_module.get_serialization_string(),
         self.robot_set_mesh_file_manager.get_serialization_string(),
         self.robot_set_kinematics_module.get_serialization_string(),
         self.robot_set_geometric_shape_module.get_serialization_string())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;

        let robot_set_configuration_module = RobotSetConfigurationModule::load_from_json_string(&load.0)?;
        let robot_set_joint_state_module = RobotSetJointStateModule::load_from_json_string(&load.1)?;
        let robot_set_mesh_file_manager = RobotSetMeshFileManagerModule::load_from_json_string(&load.2)?;
        let robot_set_kinematics_module = RobotSetKinematicsModule::load_from_json_string(&load.3)?;
        let robot_set_geometric_shape_module = RobotSetGeometricShapeModule::load_from_json_string(&load.4)?;

        Ok(Self {
            robot_set_configuration_module,
            robot_set_joint_state_module,
            robot_set_mesh_file_manager,
            robot_set_kinematics_module,
            robot_set_geometric_shape_module
        })
    }
}