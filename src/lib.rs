
//! Optima is an easy to set up and easy to use toolbox for applied planning and optimization.
//! Its primary use-case is robot motion generation (e.g., motion planning, optimization-based inverse kinematics, etc),
//! though its underlying structures are general and can apply to many problem spaces.
//! The core library is written in Rust, though high quality ports to high-level languages such as
//! Python and Javascript are available via PyO3 and WebAssembly, respectively.

pub mod robot_modules;
pub mod robot_set_modules;
pub mod scenes;
pub mod utils;

#[cfg(not(target_arch = "wasm32"))]
use pyo3::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
#[pymodule]
fn optima(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<robot_modules::robot_model_module::RobotModelModule>()?;
    m.add_class::<robot_modules::robot_configuration_module::RobotConfigurationModule>()?;
    m.add_class::<robot_modules::robot_joint_state_module::RobotJointStateModule>()?;
    m.add_class::<robot_modules::robot_kinematics_module::RobotKinematicsModule>()?;
    m.add_class::<robot_modules::robot_mesh_file_manager_module::RobotMeshFileManagerModule>()?;
    Ok(())
}

#[macro_export]
macro_rules! dvector {
    ($v: expr) => {
        {
            DVector::from_vec($v)
        };
    }
}