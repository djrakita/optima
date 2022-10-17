extern crate optima;

use std::env;
use optima::robot_modules::robot_preprocessing_module::RobotPreprocessingModule;

fn main () {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Argument must be the given robot's name");

    let robot_name = args[1].as_str();

    RobotPreprocessingModule::preprocess_robot_from_console_input(robot_name).expect("error");
}