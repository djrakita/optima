extern crate optima;

use optima::robot_modules::robot_configuration_module::{ContiguousChainMobilityMode, RobotConfigurationModule};
use optima::robot_set_modules::robot_set::RobotSet;
use optima::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use optima::utils::utils_robot::robot_module_utils::RobotNames;
use optima::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

fn main() {
    // Initialize `RobotSetConfigurationModule`
    let mut robot_set_configuration_module = RobotSetConfigurationModule::new_empty();

    // Load a base robot configuration module
    let mut ur5_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("ur5")).expect("error");
    // Set the ur5 to have a planar translation mobile base.
    ur5_configuration.set_mobile_base(ContiguousChainMobilityMode::PlanarTranslation {
        x_bounds: (-2.0, 2.0),
        y_bounds: (-2.0, 2.0)
    }).expect("error");

    // Add the ur5 configuration to the robot_set_configuration
    robot_set_configuration_module.add_robot_configuration(ur5_configuration).expect("error");

    let mut sawyer_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("sawyer")).expect("error");
    // Move the sawyer robot over 1 meter on the y axis so it is not overlapping with the ur5 robot.
    sawyer_configuration.set_base_offset(&OptimaSE3Pose::new_from_euler_angles(0.,0.,0., 0., 1., 0., &OptimaSE3PoseType::ImplicitDualQuaternion)).expect("error");
    // Remove the pedestal from the sawyer model.
    sawyer_configuration.set_dead_end_link(2).expect("error");

    // Add the sawyer configuration to the robot_set_configuration
    robot_set_configuration_module.add_robot_configuration(sawyer_configuration).expect("error");

    // Similar to a `RobotConfigurationModule`, a `RobotSetConfigurationModule` can be saved to a file
    // to allow for easy loading of a model at a later time.  This command will save a file
    // `optima_toolbox/optima_assets/optima_robot_sets/test_configuration.JSON`
    robot_set_configuration_module.save_robot_set_configuration_module("test_configuration").expect("error");

    // We will now load the just saved configuration
    let loaded_robot_set_configuration_module = RobotSetConfigurationModule::new_from_set_name("test_configuration").expect("error");

    // This loaded robot set configuration will now be used to initialize a `RobotSet`.
    let robot_set = RobotSet::new_from_robot_set_configuration_module(loaded_robot_set_configuration_module);

    // When we now print information about the robot set, the changes we made to the configurations
    // are reflected in the combined model.
    robot_set.print_summary();
    robot_set.robot_set_configuration_module().print_summary();
    robot_set.robot_set_joint_state_module().print_dof_summary();
}