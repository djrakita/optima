extern crate optima;

use nalgebra::{DVector, Vector3};
use optima::robot_modules::robot_geometric_shape_module::RobotLinkShapeRepresentation;
use optima::robot_set_modules::robot_set::RobotSet;
use optima::scenes::robot_geometric_shape_scene::{EnvObjInfoBlock, EnvObjPoseConstraint, RobotGeometricShapeScene, RobotGeometricShapeSceneQuery};
use optima::utils::utils_robot::robot_module_utils::RobotNames;
use optima::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
use optima::utils::utils_shape_geometry::geometric_shape::{LogCondition, StopCondition};

fn main() {
    // Let's start by initializing the same `RobotGeometricShapeScene` from the previous example.
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5")]);
    let mut robot_geometric_shape_scene = RobotGeometricShapeScene::new(robot_set, None).expect("error");
    let env_obj_idx = robot_geometric_shape_scene.add_environment_object(EnvObjInfoBlock {
        asset_name: "sphere".to_string(),
        scale: Vector3::new(0.1, 0.1, 0.1),
        ..Default::default()
    }, false).expect("error");
    robot_geometric_shape_scene.set_curr_single_env_obj_pose_constraint(env_obj_idx, EnvObjPoseConstraint::Absolute(OptimaSE3Pose::new_from_euler_angles(0., 0., 0., 0.5, 0.5, 0., &OptimaSE3PoseType::ImplicitDualQuaternion))).expect("error");

    let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(DVector::from_vec(vec![0.,0.,0.,0.,0.,0.])).expect("error");

    let input = RobotGeometricShapeSceneQuery::IntersectionTest {
        robot_set_joint_state: Some(&robot_set_joint_state),
        env_obj_pose_constraint_group_input: None,
        inclusion_list: &None
    };

    let res = robot_geometric_shape_scene.shape_collection_query(&input, RobotLinkShapeRepresentation::ConvexShapes, StopCondition::Intersection, LogCondition::LogAll, true).expect("error");

    println!("{:?}", res);
}