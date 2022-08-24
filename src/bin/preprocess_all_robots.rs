extern crate optima;

use optima::robot_modules::robot_preprocessing_module::RobotPreprocessingModule;

fn main() {
    RobotPreprocessingModule::preprocess_all_robots_from_console_input().expect("error");
}