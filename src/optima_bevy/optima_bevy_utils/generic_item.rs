use crate::optima_bevy::optima_bevy_utils::materials::OptimaBevyMaterialComponent;
use bevy::ecs::component::Component;

#[derive(Component, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum GenericItemSignature {
    RobotLink { robot_set_idx: usize, robot_idx_in_set: usize, link_idx_in_robot: usize },
    EnvObj { robot_geometric_shape_scene_idx: usize, env_obj_idx: usize }
}