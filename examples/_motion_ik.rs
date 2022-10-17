extern crate optima;

use nalgebra::DVector;
use optima::inverse_kinematics::OptimaIK;
use optima::optima_tensor_function::OTFImmutVars;
use optima::optima_tensor_function::robotics_functions::{RobotCollisionProximityGenericParams};
use optima::optimization::{NonlinearOptimizerType, OptimizerParameters};
use optima::robot_set_modules::robot_set::RobotSet;
use optima::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;
use optima::utils::utils_console::OptimaDebug;
use optima::utils::utils_robot::robot_module_utils::RobotNames;
use optima::utils::utils_robot::robot_set_link_specification::{RobotLinkTFGoal, RobotLinkTFSpec, RobotLinkTFSpecCollection};
use optima::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

fn main() {
    // Initialize `RobotSet` with base ur5 configuration.
    let robot_set = RobotSet::new_from_robot_names(vec![RobotNames::new_base("ur5")]);

    // Initialize a `RobotGeometricShapeScene`.
    let robot_geometric_shape_scene = RobotGeometricShapeScene::new(robot_set, None).expect("error");

    // Spawn the robot joint state that the robot's motion will start at.
    let init_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(DVector::from_vec(vec![0.1; 6])).expect("error");

    // Initialize the "motion" version of `OptimaIK`.
    // This is using the OpEn Nonlinear optimizer (same as previous example).
    // This solver can optionally exhibit a collision avoidance objective.  This is controlled
    // by the `RobotCollisionProximityGenericParams` enum.  Here, it is None (i.e., no collision
    // avoidance objective will be present).
    // We also provide the initial joint state and initial time.  In most cases, the initial time
    // will be 0.0.
    let mut immut_vars = OTFImmutVars::new();
    let mut optima_ik = OptimaIK::new_motion_ik(&mut immut_vars, robot_geometric_shape_scene, NonlinearOptimizerType::OpEn, RobotCollisionProximityGenericParams::None, &init_state, 0.0);

    // This specifies a robot link transform that the robot should try to achieve.  Note that, in this
    // example, we only provide this single goal, and the solver will exhibit a sequence of
    // state/ time tuples that will, over time, meet this goal.  Alternatively, we could also
    // provide new transform goal for links at each call of `solve_motion_ik` below.  This would
    // be useful for something like shared control or telemanipulation where an operator is
    // providing a new transform goal at each update.
    let mut robot_link_tf_spec_collection = RobotLinkTFSpecCollection::new();
    robot_link_tf_spec_collection.add(RobotLinkTFSpec::Absolute {
        goal: RobotLinkTFGoal::LinkSE3PoseGoal {
            robot_idx_in_set: 0,
            link_idx_in_robot: 9,
            goal: OptimaSE3Pose::new_from_euler_angles(0., 0., 0., 0.5, 0.1, 0.4, &OptimaSE3PoseType::ImplicitDualQuaternion),
            weight: None
        }
    });

    // Optimizer parameters.  We'll just use the default in this case.
    let mut params = OptimizerParameters::default();
    params.set_open_tolerance(0.00001);

    for i in 1..10000 {
        // Solves the motion ik problem at each given time.  The time here advances 1 millisecond
        // on each loop.  Thus, all 5000 loops will result in 5 seconds of outputted motion.
        // The output of each `solve_motion_ik` call will be a `TimedGenericRobotJointState` that
        // will contain a `RobotSetJointState` as well as a time value.
        let res = optima_ik.solve_motion_ik(&mut immut_vars, robot_link_tf_spec_collection.clone(), i as f64 * 0.001, &params, OptimaDebug::False);
        println!("{:?}", res);
    }
}