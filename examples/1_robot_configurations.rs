extern crate optima;

use optima::robot_modules::robot::Robot;
use optima::robot_modules::robot_configuration_module::{ContiguousChainMobilityMode, RobotConfigurationModule};
use optima::utils::utils_robot::robot_module_utils::RobotNames;

fn main() {
    // Initialize a new robot configuration for the ur5 robot.
    let mut robot_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("ur5")).expect("error");

    // Sets the 0 joint idx (should_pan_joint, in the case of the ur5 robot) with joint_sub_idx 0 (the
    // default sub-idx for joints with only 1 degree of freedom) to a fixed value of 1.2.  From here on
    // out, this joint will no longer be considered a degree of freedom and, instead,
    // will remain locked at 1.2.
    robot_configuration.set_fixed_joint(0, 0, 1.2);

    // Sets link with idx 5 (wrist_2_link, in the case of the ur5 robot) as a "dead end link".  A
    // dead-end link means that this link, as well as all predecessor links in the kinematic chain,
    // will be removed from the model.  Note that this will also remove all joints that connect
    // these removed links, thus possibly reducing the number of degrees of freedom of the model.
    // Links and joints that are removed in this way are said to be not "present" in the model.
    robot_configuration.set_dead_end_link(5);

    // Sets the base of the robot to be mobile.  In this case, the base is floating (not
    // realistic for the ur5 robot, but is very useful, for example, for the hips
    // of humanoid robots).  This base will automatically add 6 degrees of freedom
    // to the model, 3 for translation (x, y, z) and 3 for rotation (rx, ry, rz).
    // All lower and upper bounds are set for each of these degrees of freedom.
    // All mobility modes are PlanarTranslation (x and y translation), PlanarRotation (rz rotation),
    // PlanarTranslationAndRotation (x and y translation + z rotation),
    // Static (i.e., 0 DOFs), and Floating (x, y, z translation + rx, ry, rz rotation).
    robot_configuration.set_mobile_base(ContiguousChainMobilityMode::Floating {
        x_bounds: (-2.0, 2.0),
        y_bounds: (-2.0, 2.0),
        z_bounds: (-2.0, 2.0),
        xr_bounds: (-3.14, 3.14),
        yr_bounds: (-3.14, 3.14),
        zr_bounds: (-3.14, 3.14)
    });

    // Saves the robot configuration for later use.  This will add a file to
    // `optima_toolbox/optima_assets/optima_robots/ur5/configurations/test_configuration.JSON`.
    // NOTE: if you run this example multiple times and the test_configuration file is already
    // present, a prompt will show up in the console asking if you would like to save over
    // the already saved file.
    robot_configuration.save("test_configuration");

    // For illustrative purposes, let's now load in our robot configuration from the file that
    // was just saved.  The second parameter in the `RobotNames` struct is now an option of
    // `Some` with the name of our recently saved configuration.
    let loaded_robot_configuration = RobotConfigurationModule::new_from_names(RobotNames::new("ur5", Some("test_configuration"))).expect("error");

    // This configuration can now be used to instantiate a `Robot`.
    let robot = Robot::new_from_robot_configuration_module(loaded_robot_configuration);

    // when we print information about the robot, we see that the information that we
    // inputted above is reflected in the printed output.
    robot.robot_configuration_module().print_summary();
    robot.robot_joint_state_module().print_dof_summary();
}