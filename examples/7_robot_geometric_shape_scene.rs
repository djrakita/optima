extern crate optima;

use optima::robot_modules::robot_geometric_shape_module::RobotLinkShapeRepresentation;
use optima::robot_set_modules::robot_set::RobotSet;
use optima::scenes::robot_geometric_shape_scene::{EnvObjPoseConstraint, EnvObjShapeRepresentation, EnvObjInfoBlock, RobotGeometricShapeScene};
use optima::utils::utils_robot::robot_module_utils::RobotNames;
use optima::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

fn main() {
    // Initialize a `RobotSet`.  In this case, it is just a base model ur5 robot.
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5")]);

    // We use the `RobotSet` to initialize a `RobotGeometricShapeScene`.  By default, the
    // environment is empty.
    let mut robot_geometric_shape_scene = RobotGeometricShapeScene::new(robot_set, RobotLinkShapeRepresentation::ConvexShapes, None, vec![]).expect("error");

    // This line adds a "sphere" object to the environment, scaled down to a size of 0.1.
    // Note that any mesh directory in `optima_toolbox/optima_assets/optima_scenes/mesh_files`
    // can be loaded in.
    let env_obj_idx = robot_geometric_shape_scene.add_environment_object(EnvObjInfoBlock::new("sphere", Some(0.1), Some(EnvObjShapeRepresentation::BestFitConvexShape), None, None), false).expect("error");

    // Adding the object to the scene above returns the index of the added environment object.
    // Indices are assigned in order, so the first returned index will be 0.
    assert_eq!(env_obj_idx, 0);

    // This line can update the rotation and translation of the given environment object in the scene.
    // In this case, we are moving the sphere such that its position will be at (0.5,0.5,0).
    // Note that we can also set the transform of an object such that it is relative to another
    // shape in the environment.
    robot_geometric_shape_scene.set_curr_single_env_obj_pose_constraint(env_obj_idx, EnvObjPoseConstraint::Absolute(OptimaSE3Pose::new_from_euler_angles(0., 0., 0., 0.5, 0.5, 0., &OptimaSE3PoseType::ImplicitDualQuaternion))).expect("error");

    // This prints a summary of the whole scene.
    robot_geometric_shape_scene.print_summary();
}