extern crate optima;

use nalgebra::DVector;
use optima::robot_modules::robot_configuration_module::RobotConfigurationModule;
use optima::robot_set_modules::robot_set::RobotSet;
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main() {
    // Let's start this demonstration by initializing a base configuration of the ur5 robot
    // and fixing its 0th joint (should_pan_joint, in the case of the ur5 robot) to -1.0.
    let mut robot_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("ur5")).expect("error");
    robot_configuration.set_fixed_joint(0, 0, -1.0).expect("error");
    let robot_set = RobotSet::new_from_robot_configuration_modules(vec![robot_configuration]);

    // Use the robot_set_joint_state_module to spawn a `RobotSetJointState`.  In this case, the
    // resulting `RobotSetJointState` will be of type DOF because the model only has five
    // degrees of freedom (one is fixed), and five values are provided.
    let dof_joint_state = robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(vec![0., 1., 2., 3., 4.])).expect("error");
    println!("DOF joint state 1: {:?}", dof_joint_state);

    // This converts the DOF `RobotSetJointState` to a Full `RobotSetJointState`, i.e., it places the
    // fixed value of -1.0 in the right place within the full state.
    let full_joint_state = robot_set.robot_set_joint_state_module().convert_state_to_full_state(&dof_joint_state).expect("error");
    println!("Full joint state 1: {:?}", full_joint_state);

    // We can use the robot_set_joint_state_module to print a summary of the joint state.
    robot_set.robot_set_joint_state_module().print_robot_joint_state_summary(&full_joint_state);

    println!(); ////////////////////////////////////////////////////////////////////////////////////

    // Let's now try a different configuration, where we set a dead end link at link index 4.
    // This will remove four links (wrist_2_link, wrist_3_link, ee_link, and tool0) from the model,
    // and will also remove connecting joints between these links as well.
    let mut robot_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("ur5")).expect("error");
    robot_configuration.set_dead_end_link(5).expect("error");
    let robot_set = RobotSet::new_from_robot_configuration_modules(vec![robot_configuration]);

    // Use the robot_set_joint_state_module to spawn a `RobotSetJointState`.  In this case, the
    // resulting `RobotSetJointState` will automatically be of type Full because the model only has five
    // degrees of freedom, and five values are provided.
    let full_joint_state = robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(vec![0., 1., 2., 3., 4.])).expect("error");
    println!("Full joint state 2: {:?}", full_joint_state);

    let dof_joint_state = robot_set.robot_set_joint_state_module().convert_state_to_dof_state(&dof_joint_state).expect("error");
    println!("DOF joint state 2: {:?}", dof_joint_state);

    // Again, we can use the robot_set_joint_state_module to print a summary of the joint state.
    robot_set.robot_set_joint_state_module().print_robot_joint_state_summary(&dof_joint_state);

    println!(); ////////////////////////////////////////////////////////////////////////////////////

    // Lastly, let's show how joint states work for robot sets with multiple robots.  For simplicity,
    // we will use a ur5 base model (6DOF) and a sawyer base model (8DOF, including the screen).
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5"), RobotNames::new_base("sawyer")]);

    // The joint state here will have 14 degrees of freedom (6 from ur5 and 8 from sawyer) instantiated
    // using a concatenated vector of 14 values.
    // Because no joints are fixed or removed, the DOF and Full joint state types here will be
    // identical (by default, the underlying type will be Full in these situations).
    let joint_state = robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(vec![0.0; 14])).expect("error");
    println!("joint state 3: {:?}", joint_state);

    // For a last time, we can use the robot_set_joint_state_module to print a summary of the joint state.
    robot_set.robot_set_joint_state_module().print_robot_joint_state_summary(&joint_state);
}
