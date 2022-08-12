use bevy::app::App;
use crate::optima_bevy::optima_bevy_utils::lights::LightSystems;
use crate::optima_bevy::optima_bevy_utils::robots::RobotSystems;
use crate::optima_bevy::optima_bevy_utils::viewport_visuals::ViewportVisualsSystems;
use crate::optima_bevy::plugins::{OptimaBasePlugin, PanOrbitCameraPlugin};
use crate::robot_set_modules::robot_set::RobotSet;

pub fn bevy_display_robot_set(robot_set: &RobotSet) {
    App::new()
        .add_plugin(OptimaBasePlugin)
        .add_plugin(PanOrbitCameraPlugin)
        .add_startup_system(LightSystems::starter_point_lights)
        .add_startup_system(ViewportVisualsSystems::system_draw_robotics_grid)
        .insert_resource(robot_set.clone())
        .add_system(RobotSystems::system_spawn_robot_set)
        .run()
}