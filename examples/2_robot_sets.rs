extern crate optima;

use optima::robot_set_modules::robot_set::RobotSet;
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main() {
    // Loads a robot set with two robots (a ur5 and sawyer).
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new("ur5", None), RobotNames::new("sawyer", None)]);

    // prints a summary of the robot set configuration
    robot_set.robot_set_configuration_module().print_summary();

    // prints a summary of the robot set's degrees of freedom.
    robot_set.robot_set_joint_state_module().print_dof_summary();
}