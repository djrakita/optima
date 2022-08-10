extern crate optima;

use nalgebra::DVector;
use optima::robot_modules::robot_kinematics_module::{JacobianEndPoint, JacobianMode};
use optima::robot_set_modules::robot_set::RobotSet;
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main() {
    // Create a base configuration ur5.
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5")]);

    // Spawn joint state with zeros for all degrees of freedom.
    let joint_state = robot_set.spawn_robot_set_joint_state(DVector::from_vec(vec![0.0; 6])).expect("error");

    // Computes the full jacobian matrix with end link idx of 9.  The `robot_jacobian_end_point` in
    // this case is simply the link center.
    let full_jacobian_matrix = robot_set.robot_set_kinematics_module().compute_jacobian(&joint_state, 0, None, 9, &JacobianEndPoint::Link, None, JacobianMode::Full).expect("error");

    println!("{}", full_jacobian_matrix);

    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Computes the 3 x N (N = 6 for the ur5 robot) translation jacobian matrix with end link idx of 9.
    // The `robot_jacobian_end_point` in this case is simply the link center.
    let translational_jacobian_matrix = robot_set.robot_set_kinematics_module().compute_jacobian(&joint_state, 0, None, 9, &JacobianEndPoint::Link, None, JacobianMode::Translational).expect("error");

    println!("{}", translational_jacobian_matrix);

    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Load the test configuration created in the previous tutorial.
    // Make sure that you ran example 3_robot_set_configurations prior to starting this tutorial.
    let robot_set = RobotSet::new_from_set_name("test_configuration");

    // Spawn joint state with zeros for all degrees of freedom.
    let joint_state = robot_set.spawn_robot_set_joint_state(DVector::from_vec(vec![0.0; 16])).expect("error");

    // Computes the 6 x N (N = 16 for this configuration) full jacobian matrix with end link idx of 18 on robot 1
    // (sawyer, in this case of this `RobotSet`).
    // The `robot_jacobian_end_point` in this case is the inertial origin of the link.
    let full_jacobian_matrix = robot_set.robot_set_kinematics_module().compute_jacobian(&joint_state, 1, None, 18, &JacobianEndPoint::InertialOrigin, None, JacobianMode::Full).expect("error");

    println!("{}", full_jacobian_matrix);

    println!("////////////////////////////////////////////////////////////////////////////////////");
}