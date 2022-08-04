extern crate optima;

use optima::robot_modules::robot_configuration_module::RobotConfigurationModule;
use optima::robot_set_modules::robot_set::RobotSet;
use optima::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main() {
    // Initialize `RobotSetConfigurationModule`
    let mut robot_set_configuration_module = RobotSetConfigurationModule::new_empty();

    // Load a base robot configuration module
    let mut ur5_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("ur5")).expect("error");



}