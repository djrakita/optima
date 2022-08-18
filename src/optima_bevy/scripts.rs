use bevy::app::App;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext, EguiPlugin};
use bevy_egui::egui::{emath, Pos2, Rect};
use crate::optima_bevy::optima_bevy_utils::egui::{EguiSystems, EguiWindowStateContainer};
use crate::optima_bevy::optima_bevy_utils::lights::LightSystems;
use crate::optima_bevy::optima_bevy_utils::robots::RobotSystems;
use crate::optima_bevy::optima_bevy_utils::viewport_visuals::ViewportVisualsSystems;
use crate::optima_bevy::plugins::{OptimaBasePlugin, OptimaEguiPlugin, OptimaPanOrbitCameraPlugin};
use crate::robot_set_modules::robot_set::RobotSet;
use crate::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;

pub fn bevy_display_robot_set(robot_set: &RobotSet) {
    App::new()
        .add_plugin(OptimaBasePlugin)
        .add_plugin(OptimaPanOrbitCameraPlugin)
        .add_startup_system(LightSystems::starter_point_lights)
        .add_startup_system(ViewportVisualsSystems::system_draw_robotics_grid)
        .insert_resource(robot_set.clone())
        .add_system(RobotSystems::system_spawn_robot_set)
        .run()
}

pub fn bevy_display_robot_geometric_shape_scene(robot_geometric_shape_scene: &RobotGeometricShapeScene) {
    App::new()
        .add_plugin(OptimaBasePlugin)
        .add_plugin(OptimaPanOrbitCameraPlugin)
        .add_startup_system(LightSystems::starter_point_lights)
        .add_startup_system(ViewportVisualsSystems::system_draw_robotics_grid)
        .insert_resource(robot_geometric_shape_scene.clone())
        .add_system(RobotSystems::system_spawn_robot_geometric_shape_scene)
        .run()
}

pub fn bevy_egui_test() {
    App::new()
        .add_plugin(OptimaBasePlugin)
        .add_plugin(OptimaPanOrbitCameraPlugin)
        .add_plugin(EguiPlugin)
        .add_plugin(OptimaEguiPlugin)
        .add_system(EguiSystems::system_egui_test.label("gui"))
        .insert_resource(RobotSet::new_single_robot("ur5", None))
        .add_startup_system(LightSystems::starter_point_lights)
        .add_startup_system(ViewportVisualsSystems::system_draw_robotics_grid)
        .run()
}



