extern crate optima;

use optima::inverse_kinematics::OptimaIK;
use optima::optima_tensor_function::robotics_functions::RobotCollisionProximityGenericParams;
use optima::optimization::{NonlinearOptimizerType, OptimizerParameters};
use optima::robot_modules::robot_geometric_shape_module::RobotLinkShapeRepresentation;
use optima::robot_set_modules::robot_set::RobotSet;
use optima::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;
use optima::utils::utils_console::OptimaDebug;
use optima::utils::utils_robot::robot_module_utils::RobotNames;
use optima::utils::utils_robot::robot_set_link_specification::{RobotLinkTFAllowableError, RobotLinkTFGoal, RobotLinkTFSpec, RobotLinkTFSpecAndAllowableError, RobotLinkTFSpecAndAllowableErrorCollection};
use optima::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

fn main() {
    // Initialize `RobotSet` with base ur5 configuration.
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5")]);

    // Initialize a `RobotGeometricShapeScene`.
    let robot_geometric_shape_scene = RobotGeometricShapeScene::new(robot_set, RobotLinkShapeRepresentation::ConvexShapes, , vec![]).expect("error");

    // Initialize "static" IK solver.
    // The `NonlinearOptimizerType` enum many options.  Feel free to check them all out; here we
    // will just use the OpEn (Optimization Engine) option.
    // This solver can optionally exhibit a collision avoidance objective.  This is controlled
    // by the `RobotCollisionProximityGenericParams` enum.  Here, it is None (i.e., no collision
    // avoidance objective will be present).
    let mut optima_ik = OptimaIK::new_static_ik(robot_geometric_shape_scene, NonlinearOptimizerType::OpEn, RobotCollisionProximityGenericParams::None);

    // Initialize a `RobotLinkTFSpecAndAllowableErrorCollection`.  This will be used to specify
    // which links should be at what location or orientation, as well as what errors will be
    // tolerable, in our final solution.
    let mut robot_link_tf_spec_and_allowable_error_collection = RobotLinkTFSpecAndAllowableErrorCollection::new_empty();

    // Creates a `RobotLinkTFSpecAndAllowableError` object.  This specifies that the ur5's link 9
    // (its end effector link) should exhibit an identity orientation (euler angles 0,0,0) and
    // a location of x: 0.5, y: 0.1, z: 0.4.  also, the translation errors along all axes whould
    // be within 1 millimenter, while rotation errors along all axes should be within 0.01 radians.
    let robot_link_tf_spec_and_allowable_error = RobotLinkTFSpecAndAllowableError::new(RobotLinkTFSpec::Absolute {
        goal: RobotLinkTFGoal::LinkSE3PoseGoal {
            robot_idx_in_set: 0,
            link_idx_in_robot: 9,
            goal: OptimaSE3Pose::new_from_euler_angles(0., 0., 0., 0.5, 0.1, 0.4, &OptimaSE3PoseType::ImplicitDualQuaternion),
            weight: None
        }
    }, Some(RobotLinkTFAllowableError {
        rx: 0.01,
        ry: 0.01,
        rz: 0.01,
        x: 0.001,
        y: 0.001,
        z: 0.001
    }));

    // This adds the robot_link_tf_spec_and_allowable_error to the collection.  Note that the collection
    // can accept numerous entries and the optimization will try its best to match all of them,
    // if possible.
    robot_link_tf_spec_and_allowable_error_collection.add(robot_link_tf_spec_and_allowable_error);

    // This calls the static ik solver with the `RobotLinkTFSpecAndAllowableErrorCollection` as
    // first input.  This function can optionally take an initial condition (in this case it is `None`).
    // Also, I set a maximum number of tries of 100 here; if an acceptable solution within the error
    // bounds cannot be found in this number of tries, the solver will return None.
    // However, if a valid solution is found, it will be returned as Some(RobotSetJointState).
    // We also specified here that the solver should reject any solution that is in a collision;
    // If a solution is found in a collision, it will throw away the result and try again (up to the
    // maximum number of tries).
    let res = optima_ik.solve_static_ik(robot_link_tf_spec_and_allowable_error_collection, &OptimizerParameters::default(), None, Some(100), true, OptimaDebug::False);

    // This prints the result.  Because, by default, the static ik solver in `OptimaIK` uses
    // random initial conditions, the result will be different each time this script is
    // run.
    println!("{:?}", res);
}

