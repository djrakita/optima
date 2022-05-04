#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_mesh_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_modules::robot_kinematics_module::RobotKinematicsModule;
use crate::robot_modules::robot_geometric_shape_module::RobotGeometricShapeModule;
use crate::robot_modules::robot_joint_state_module::{RobotJointState, RobotJointStateModule};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::load_object_from_json_string;
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_traits::SaveAndLoadable;

/// An aggregation of many robot modules.  In most cases, applications in Optima will use
/// the more general `RobotSet` struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Robot {
    robot_configuration_module: RobotConfigurationModule,
    robot_mesh_file_manager_module: RobotMeshFileManagerModule,
    robot_joint_state_module: RobotJointStateModule,
    robot_kinematics_module: RobotKinematicsModule
}
impl Robot {
    pub fn new_from_names(robot_name: RobotNames) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_name.clone())?;
        let robot_mesh_file_manager_module = RobotMeshFileManagerModule::new(robot_configuration_module.robot_model_module())?;
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        let robot_fk_module = RobotKinematicsModule::new(robot_configuration_module.clone());

        Ok(Self {
            robot_configuration_module,
            robot_mesh_file_manager_module,
            robot_joint_state_module,
            robot_kinematics_module: robot_fk_module
        })
    }
    pub fn new_from_robot_configuration_module(robot_configuration_module: RobotConfigurationModule) -> Result<Self, OptimaError> {
        let robot_mesh_file_manager_module = RobotMeshFileManagerModule::new(robot_configuration_module.robot_model_module())?;
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        let robot_fk_module = RobotKinematicsModule::new(robot_configuration_module.clone());

        Ok(Self {
            robot_configuration_module,
            robot_mesh_file_manager_module,
            robot_joint_state_module,
            robot_kinematics_module: robot_fk_module
        })
    }
    pub fn robot_configuration_module(&self) -> &RobotConfigurationModule {
        &self.robot_configuration_module
    }
    pub fn robot_mesh_file_manager_module(&self) -> &RobotMeshFileManagerModule {
        &self.robot_mesh_file_manager_module
    }
    pub fn robot_joint_state_module(&self) -> &RobotJointStateModule {
        &self.robot_joint_state_module
    }
    pub fn robot_kinematics_module(&self) -> &RobotKinematicsModule {
        &self.robot_kinematics_module
    }
    pub fn generate_robot_geometric_shape_module(&self) -> Result<RobotGeometricShapeModule, OptimaError> {
        return RobotGeometricShapeModule::new(self.robot_configuration_module.clone(), false);
    }
    pub fn spawn_robot_joint_state(&self, v: DVector<f64>) -> Result<RobotJointState, OptimaError> {
        self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(v)
    }
}
impl SaveAndLoadable for Robot {
    type SaveType = (String, String, String, String);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_configuration_module.get_serialization_string(),
         self.robot_mesh_file_manager_module.get_serialization_string(),
         self.robot_joint_state_module.get_serialization_string(),
         self.robot_kinematics_module.get_serialization_string())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let robot_configuration_module = RobotConfigurationModule::load_from_json_string(&load.0)?;
        let robot_mesh_file_manager_module = RobotMeshFileManagerModule::load_from_json_string(&load.1)?;
        let robot_joint_state_module = RobotJointStateModule::load_from_json_string(&load.2)?;
        let robot_kinematics_module = RobotKinematicsModule::load_from_json_string(&load.3)?;

        Ok(Self {
            robot_configuration_module,
            robot_mesh_file_manager_module,
            robot_joint_state_module,
            robot_kinematics_module
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pyclass]
pub struct RobotPy {
    #[pyo3(get)]
    robot_configuration_module: Py<RobotConfigurationModule>,
    #[pyo3(get)]
    robot_mesh_file_manager_module: Py<RobotMeshFileManagerModule>,
    #[pyo3(get)]
    robot_joint_state_module: Py<RobotJointStateModule>,
    #[pyo3(get)]
    robot_kinematics_module: Py<RobotKinematicsModule>,
    phantom_robot: Robot
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotPy {
    #[new]
    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>, py: Python) -> Self {
        let robot_names = RobotNames::new(robot_name, configuration_name);
        let r = Robot::new_from_names(robot_names).expect("error");
        
        Self {
            robot_configuration_module: Py::new(py, r.robot_configuration_module.clone()).expect("error"),
            robot_mesh_file_manager_module: Py::new(py, r.robot_mesh_file_manager_module.clone()).expect("error"),
            robot_joint_state_module: Py::new(py, r.robot_joint_state_module.clone()).expect("error"),
            robot_kinematics_module: Py::new(py, r.robot_kinematics_module.clone()).expect("error"),
            phantom_robot: r,
        }
    }
    pub fn generate_robot_geometric_shape_module_py(&self) -> RobotGeometricShapeModule {
        self.phantom_robot.generate_robot_geometric_shape_module().expect("error")
    }
}
