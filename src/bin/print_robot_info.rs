extern crate optima;

use std::env;
use optima::robot_modules::robot::Robot;
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main () {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Argument must be the given robot's name");

    let robot_name = args[1].as_str();

    // load the given robot
    let robot = Robot::new_from_names(RobotNames::new(robot_name, None));

    // prints a summary of the robot configuration
    robot.robot_configuration_module().print_summary();

    // prints a summary of the robot's degrees of freedom
    robot.robot_joint_state_module().print_dof_summary();
}