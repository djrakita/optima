use std::env;
use optima::optima_bevy::scripts::bevy_robot_sliders;
use optima::robot_set_modules::robot_set::RobotSet;
use optima::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2, "Argument must be the given robot's name");

    let robot_name = args[1].as_str();
    let robot_geometric_shape_scene = RobotGeometricShapeScene::new(RobotSet::new_single_robot(robot_name, None), None).expect("error");

    bevy_robot_sliders(&robot_geometric_shape_scene);
}