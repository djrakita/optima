extern crate optima;

use nalgebra::DVector;
use optima::robot_set_modules::robot_set::RobotSet;
use optima::utils::utils_robot::robot_module_utils::RobotNames;
use optima::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;

fn main() {
    // Create a base configuration ur5.
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5")]);

    // Spawn joint state with zeros for all degrees of freedom.
    let joint_state = robot_set.spawn_robot_set_joint_state(DVector::from_vec(vec![0.0; 6])).expect("error");

    // Compute forward kinematics.  The second argument in `compute_fk` is the SE(3) representation that should
    // be used to compute FK.  All options are `ImplicitDualQuaternion`, `EulerAnglesAndTranslation`,
    // `RotationMatrixAndTranslation`, `UnitQuaternionAndTranslation`, and `HomogeneousMatrix`.
    // The result is an `RobotSetFKResult`.
    let fk_res = robot_set.robot_set_kinematics_module().compute_fk(&joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");

    // Print summary of the fk result.
    fk_res.print_summary();

    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Load the test configuration created in the previous tutorial.
    // Make sure that you ran example 3_robot_set_configurations prior to starting this tutorial.
    let robot_set = RobotSet::new_from_set_name("test_configuration");

    // Spawn joint state with zeros for all degrees of freedom.
    let joint_state = robot_set.spawn_robot_set_joint_state(DVector::from_vec(vec![0.0; 16])).expect("error");

    // Compute forward kinematics.  The second argument in `compute_fk` is the SE(3) representation that should
    // be used to compute FK.  All options are `ImplicitDualQuaternion`, `EulerAnglesAndTranslation`,
    // `RotationMatrixAndTranslation`, `UnitQuaternionAndTranslation`, and `HomogeneousMatrix`.
    // The result is an `RobotSetFKResult`.
    let fk_res = robot_set.robot_set_kinematics_module().compute_fk(&joint_state, &OptimaSE3PoseType::HomogeneousMatrix).expect("error");

    // Print summary of the fk result.
    fk_res.print_summary();

    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Access the pose of the just computed fk result.
    // The arguments in `get_pose_from_idxs` here are `robot_idx_in_set` (in this case,
    // the sawyer robot, the "second" robot in the set) and `link_idx_in_robot` (in this case,
    // link 18 "right hand" on the sawyer robot).
    let pose = fk_res.get_pose_from_idxs(1, 18);
    println!("{:?}", pose);
}