#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::robot_set_modules::GetRobotSet;
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::robot_set_modules::robot_set_kinematics_module::RobotSetKinematicsModule;
use crate::robot_set_modules::robot_set_geometric_shape_module::RobotSetGeometricShapeModule;
use crate::robot_set_modules::robot_set_joint_state_module::{RobotSetJointState, RobotSetJointStateModule};
use crate::robot_set_modules::robot_set_mesh_file_manager_module::RobotSetMeshFileManagerModule;
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::load_object_from_json_string;
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
use crate::utils::utils_traits::SaveAndLoadable;

/// An aggregation of many robot set modules.  This is a central, important struct that is
/// used throughout the the Optima library.
///
/// # Example
/// ```
/// use optima::robot_set_modules::robot_set::RobotSet;
/// use optima::robot_set_modules::robot_set_joint_state_module::RobotSetJointStateType;
/// use optima::utils::utils_robot::robot_module_utils::RobotNames;
/// use optima::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
///
/// // This example shows the initialization of a RobotSet of a ur5 and sawyer robot,
/// // spawning of a robot_set_joint_state, and printing thet result of a forward kinematics computation
/// // on the joint state.
/// let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5"), RobotNames::new_base("sawyer")]).expect("error");
/// let robot_set_joint_state = robot_set.robot_set_joint_state_module().spawn_zeros_robot_set_joint_state(RobotSetJointStateType::DOF);
/// let fk_res = robot_set.robot_set_kinematics_module().compute_fk(&robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
/// fk_res.print_summary();
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSet {
    robot_set_configuration_module: RobotSetConfigurationModule,
    robot_set_joint_state_module: RobotSetJointStateModule,
    robot_set_mesh_file_manager_module: RobotSetMeshFileManagerModule,
    robot_set_kinematics_module: RobotSetKinematicsModule
}
impl RobotSet {
    pub fn new_from_robot_set_configuration_module(r: RobotSetConfigurationModule) -> Self {
        let robot_set_joint_state_module = RobotSetJointStateModule::new(&r);
        let robot_set_mesh_file_manager_module = RobotSetMeshFileManagerModule::new(&r).expect("error");
        let robot_set_kinematics_module = RobotSetKinematicsModule::new(&r);

        Self {
            robot_set_configuration_module: r,
            robot_set_joint_state_module,
            robot_set_mesh_file_manager_module,
            robot_set_kinematics_module
        }
    }
    /// Initializes a `RobotSet` using the set name that is found in the optima_assets/optima_robot_sets/
    /// directory.  `RobotSet` objects can be saved using the `RobotSetConfigurationModule` struct.
    pub fn new_from_set_name(set_name: &str) -> Self {
        let r = RobotSetConfigurationModule::new_from_set_name(set_name).expect("error");
        return Self::new_from_robot_set_configuration_module(r);
    }
    pub fn new_from_robot_names(robot_names: Vec<RobotNames>) -> Self {
        let mut r = RobotSetConfigurationModule::new_empty();
        for rn in &robot_names { r.add_robot_configuration_from_names(rn.clone()).expect("error"); }
        return Self::new_from_robot_set_configuration_module(r);
    }
    pub fn new_single_robot(robot_name: &str, configuration_name: Option<&str>) -> Self {
        Self::new_from_robot_names(vec![RobotNames::new(robot_name, configuration_name)])
    }
    pub fn robot_set_configuration_module(&self) -> &RobotSetConfigurationModule {
        &self.robot_set_configuration_module
    }
    pub fn robot_set_joint_state_module(&self) -> &RobotSetJointStateModule {
        &self.robot_set_joint_state_module
    }
    pub fn robot_set_mesh_file_manager(&self) -> &RobotSetMeshFileManagerModule {
        &self.robot_set_mesh_file_manager_module
    }
    pub fn robot_set_kinematics_module(&self) -> &RobotSetKinematicsModule {
        &self.robot_set_kinematics_module
    }
    pub fn generate_robot_set_geometric_shape_module(&self) -> Result<RobotSetGeometricShapeModule, OptimaError> {
        return RobotSetGeometricShapeModule::new(&self.robot_set_configuration_module);
    }
    pub fn spawn_robot_set_joint_state(&self, v: DVector<f64>) -> Result<RobotSetJointState, OptimaError> {
        self.robot_set_joint_state_module.spawn_robot_set_joint_state_try_auto_type(v)
    }
    pub fn print_summary(&self) {
        let num_robots = self.robot_set_configuration_module.robot_configuration_modules().len();
        optima_print(&format!("{} robots.", num_robots), PrintMode::Println, PrintColor::Blue, true);
        for (i, robot_configuration) in self.robot_set_configuration_module.robot_configuration_modules().iter().enumerate() {
            optima_print(&format!(" Robot {} ---> {:?}", i, robot_configuration.robot_name()), PrintMode::Println, PrintColor::Blue, false);
            optima_print(&format!("   Base Offset: {:?}", robot_configuration.robot_configuration_info().base_offset().get_pose_by_type(&OptimaSE3PoseType::EulerAnglesAndTranslation)), PrintMode::Println, PrintColor::None, false );
            // optima_print(&format!("   Mobile Base Mode: {:?}", robot_configuration.robot_configuration_info().mobile_base_mode()), PrintMode::Println, PrintColor::None, false );
        }
    }
}
impl SaveAndLoadable for RobotSet {
    type SaveType = (String, String, String, String);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_set_configuration_module.get_serialization_string(),
         self.robot_set_joint_state_module.get_serialization_string(),
         self.robot_set_mesh_file_manager_module.get_serialization_string(),
         self.robot_set_kinematics_module.get_serialization_string())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;

        let robot_set_configuration_module = RobotSetConfigurationModule::load_from_json_string(&load.0)?;
        let robot_set_joint_state_module = RobotSetJointStateModule::load_from_json_string(&load.1)?;
        let robot_set_mesh_file_manager = RobotSetMeshFileManagerModule::load_from_json_string(&load.2)?;
        let robot_set_kinematics_module = RobotSetKinematicsModule::load_from_json_string(&load.3)?;

        Ok(Self {
            robot_set_configuration_module,
            robot_set_joint_state_module,
            robot_set_mesh_file_manager_module: robot_set_mesh_file_manager,
            robot_set_kinematics_module
        })
    }
}
impl GetRobotSet for RobotSet {
    fn get_robot_set(&self) -> &RobotSet {
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pyclass]
#[derive(Clone)]
pub struct RobotSetPy {
    #[pyo3(get)]
    robot_set_configuration_module: Py<RobotSetConfigurationModule>,
    #[pyo3(get)]
    robot_set_joint_state_module: Py<RobotSetJointStateModule>,
    #[pyo3(get)]
    robot_set_mesh_file_manager_module: Py<RobotSetMeshFileManagerModule>,
    #[pyo3(get)]
    robot_set_kinematics_module: Py<RobotSetKinematicsModule>,
    phantom_robot_set: RobotSet
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotSetPy {
    #[new]
    pub fn new_from_set_name(set_name: &str, py: Python) -> Self {
        let r = RobotSet::new_from_set_name(set_name);

        Self {
            robot_set_configuration_module: Py::new(py, r.robot_set_configuration_module.clone()).expect("error"),
            robot_set_joint_state_module: Py::new(py, r.robot_set_joint_state_module.clone()).expect("error"),
            robot_set_mesh_file_manager_module: Py::new(py, r.robot_set_mesh_file_manager_module.clone()).expect("error"),
            robot_set_kinematics_module: Py::new(py, r.robot_set_kinematics_module.clone()).expect("error"),
            phantom_robot_set: r
        }
    }
    #[staticmethod]
    pub fn new(robot_set_configuration_module: &RobotSetConfigurationModule, py: Python) -> Self {
        let r = RobotSet::new_from_robot_set_configuration_module(robot_set_configuration_module.clone());

        Self {
            robot_set_configuration_module: Py::new(py, r.robot_set_configuration_module.clone()).expect("error"),
            robot_set_joint_state_module: Py::new(py, r.robot_set_joint_state_module.clone()).expect("error"),
            robot_set_mesh_file_manager_module: Py::new(py, r.robot_set_mesh_file_manager_module.clone()).expect("error"),
            robot_set_kinematics_module: Py::new(py, r.robot_set_kinematics_module.clone()).expect("error"),
            phantom_robot_set: r
        }
    }
    #[staticmethod]
    pub fn new_single_robot(robot_name: &str, configuration_name: Option<&str>, py: Python) -> Self {
        let mut robot_set_configuration_module = RobotSetConfigurationModule::new_empty();
        robot_set_configuration_module.add_robot_configuration_from_names(RobotNames::new(robot_name, configuration_name)).expect("error");
        Self::new(&robot_set_configuration_module, py)
    }
    pub fn generate_robot_set_geometric_shape_module(&self) -> RobotSetGeometricShapeModule {
        self.phantom_robot_set.generate_robot_set_geometric_shape_module().expect("error")
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl RobotSetPy {
    pub fn get_robot_set(&self) -> &RobotSet {
        &self.phantom_robot_set
    }
}