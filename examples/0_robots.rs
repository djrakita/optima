extern crate optima;

use optima::robot_modules::robot::Robot;
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main () {
    // load the given robot
    let robot = Robot::new_from_names(RobotNames::new("ur5", None));

    // prints a summary of the robot configuration
    robot.robot_configuration_module().print_summary();

    // prints a summary of the robot's degrees of freedom
    robot.robot_joint_state_module().print_dof_summary();
}