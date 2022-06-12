use crate::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;

pub trait GetRobotGeometricShapeScene {
    fn get_robot_geometric_shape_scene(&self) -> &RobotGeometricShapeScene;
}

pub mod robot_geometric_shape_scene;
