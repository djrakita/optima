
//! # Introduction
//!
//! Optima is an easy to set up and easy to use robotics toolbox.  Its primary use-case is robot motion generation (e.g., trajectory optimization, motion planning, optimization-based inverse kinematics, etc), though its underlying structures are general and can apply to many planning and optimization problem domains.  The core library is written in Rust, though high quality support for other targets such as Python and Webassembly are afforded via Rust's flexible compiler.
//!
//! Optima features an extensive suite of efficient and flexible robotics subroutines, including forward kinematics, jacobian computations, etc.  Optima also features implementations of state-of-the-art robotics algorithms related to planning, optimization, and proximity computations, such as RelaxedIK, CollisionIK, and Proxima (with more algorithms on the way).  These robotics features are designed to be easy to set up and access.  Models of many popular robots such as UR5, Franka Emika Panda, Kinova Gen3, etc., are included out-of-the-box and can be instantiated with a single line of code.  If your robot model is not included in our provided assets, a robot can be added to the library in just minutes.
//!
//!
//!
//! ## High-level Features
//!
//! - The underlying non-linear optimization engine has several backend options, such as the [Open Engine](https://alphaville.github.io/optimization-engine/docs/open-intro) and [NLopt](https://nlopt.readthedocs.io/en/latest/).  All algorithms support a non-linear objective function, non-linear equality constraints, and non-linear inequality constraints.  Most provided algorithms are local solvers, though several global optimization algorithms are provided via NLopt as well.  These optimization algorithm options can be swapped in and out by changing a single parameter, making testing and benchmarking over optimization algorithms easy to conduct.
//!
//! - The library presents a powerful Rust trait called `OptimaTensorFunction`; any function that implements this trait can automatically compute derivatives (up to the fourth derivative) using a derivative composition graph that automatically selects either analytical derivatives, when provided, or finite-difference approximate derivatives, by default, through its traversal.  Thus, instead of depending on low-level automatic differentiation libraries that often fail on important subroutines, this approach provides a highly efficient and flexible way to perform derivatives for any implemented function.  High order derivatives are even computable on multi-dimensional functions as the library supports tensors of any arbitrary dimension.  For instance, a function \\( \mathit{f}: \mathbb{R}^{3 \times 5} \rightarrow {R}^{4 \times 4} \\) implemented as an `OptimaTensorFunction` can automatically provide derivatives \\( \frac{\partial f}{\partial \mathbf{X}} \in {R}^{4 \times 4 \times 3 \times 5} \\), \\( \frac{\partial^2 f}{\partial \mathbf{X}^2} \in {R}^{4 \times 4 \times 3 \times 5 \times 3 \times 5} \\), etc.
//!
//! - Optima uses flexible transform and rotation implementations that make SE(3) computations a breeze.  For instance, any computations over transforms can use homogeneous matrices, implicit dual quaternions, a rotation matrix + vector, unit quaternion + vector, euler angles + vector, etc.  No matter what SE(3) representation is chosen, the library will automatically ensure that all conversions and computations are handled correctly under the hood.
//!
//! - In just a few lines of code, individual robot models can be combined to form a "robot set", allowing for easy planning and optimization over several robots all at once.  For instance, a UR5 (6DOF) can be easily combined with two Rethink Sawyer (7DOF) robots to form a robot set with 20 total DOF.  In addition, any robot in the library (including those in a robot set) can be easily supplemented with a mobile base of numerous types (e.g., floating base, planar base, etc.).  The extra degrees of freedom accompanying these mobile base options are automatically added to the robot (or robot set) model.
//!
//! ## Documentation
//! Further documentation including setup instructions, tutorials, etc. for Optima can be found at [https://djrakita.github.io/optima_toolbox/](https://djrakita.github.io/optima_toolbox/).

pub mod inverse_kinematics;
pub mod optimization;
pub mod optima_tensor_function;
pub mod robot_modules;
pub mod robot_set_modules;
pub mod scenes;
pub mod utils;

#[cfg(feature = "optima_bevy")]
pub mod optima_bevy;

#[cfg(not(target_arch = "wasm32"))]
use pyo3::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
#[pymodule]
fn optima(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<scenes::robot_geometric_shape_scene::RobotGeometricShapeScenePy>()?;

    // m.add_class::<inverse_kinematics::OptimaIKPy>()?;
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

/////////////////////////////////////////////

