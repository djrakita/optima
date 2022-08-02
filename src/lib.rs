
//! Optima is an easy to set up and easy to use toolbox for applied planning and optimization.
//! Its primary use-case is robot motion generation (e.g., motion planning, optimization-based inverse kinematics, etc),
//! though its underlying structures are general and can apply to many problem spaces.
//! The core library is written in Rust, though high quality ports to high-level languages such as
//! Python and Javascript are available via PyO3 and WebAssembly, respectively.

pub mod inverse_kinematics;
pub mod nonlinear_optimization;
pub mod optima_tensor_function;
pub mod robot_modules;
pub mod robot_set_modules;
pub mod scenes;
pub mod utils;

#[cfg(not(target_arch = "wasm32"))]
use pyo3::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
#[pymodule]
fn optima(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<scenes::robot_geometric_shape_scene::RobotGeometricShapeScenePy>()?;

    m.add_class::<inverse_kinematics::OptimaIKPy>()?;
    m.add_class::<utils::utils_robot::robot_set_link_specification::RobotLinkTFGoalPy>()?;
    m.add_class::<utils::utils_robot::robot_set_link_specification::RobotLinkTFSpecPy>()?;
    m.add_class::<utils::utils_robot::robot_set_link_specification::RobotLinkTFSpecCollectionPy>()?;
    m.add_class::<utils::utils_robot::robot_set_link_specification::RobotLinkTFSpecAndAllowableErrorPy>()?;
    m.add_class::<utils::utils_robot::robot_set_link_specification::RobotLinkTFSpecAndAllowableErrorCollectionPy>()?;
    m.add_class::<utils::utils_robot::robot_set_link_specification::RobotLinkTFAllowableErrorPy>()?;

    m.add_class::<robot_set_modules::robot_set::RobotSetPy>()?;
    m.add_class::<robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule>()?;
    m.add_class::<robot_set_modules::robot_set_joint_state_module::RobotSetJointStateModule>()?;
    m.add_class::<robot_set_modules::robot_set_kinematics_module::RobotSetKinematicsModule>()?;
    m.add_class::<robot_set_modules::robot_set_geometric_shape_module::RobotSetGeometricShapeModule>()?;
    m.add_class::<robot_set_modules::robot_set_mesh_file_manager_module::RobotSetMeshFileManagerModule>()?;

    m.add_class::<robot_modules::robot::RobotPy>()?;
    m.add_class::<robot_modules::robot_model_module::RobotModelModule>()?;
    m.add_class::<robot_modules::robot_configuration_module::RobotConfigurationModulePy>()?;
    m.add_class::<robot_modules::robot_joint_state_module::RobotJointStateModule>()?;
    m.add_class::<robot_modules::robot_kinematics_module::RobotKinematicsModule>()?;
    m.add_class::<robot_modules::robot_geometric_shape_module::RobotGeometricShapeModule>()?;
    m.add_class::<robot_modules::robot_mesh_file_manager_module::RobotMeshFileManagerModule>()?;

    m.add_class::<utils::utils_se3::optima_se3_pose::OptimaSE3PosePy>()?;
    m.add_class::<utils::utils_se3::optima_rotation::OptimaRotationPy>()?;
    Ok(())
}

