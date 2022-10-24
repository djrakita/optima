use std::env;
use optima::optima_bevy::scripts::bevy_robot_self_collisions_calibrator;
use optima::robot_modules::robot::Robot;
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Argument must be the given robot's name");

    let robot_name = args[1].as_str();
    let robot = Robot::new_from_names(RobotNames::new_base(robot_name));

    bevy_robot_self_collisions_calibrator(&robot);
}