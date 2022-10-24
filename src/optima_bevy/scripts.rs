use bevy::app::App;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiPlugin};
use bevy_prototype_debug_lines::DebugLinesPlugin;
use crate::optima_bevy::optima_bevy_utils::camera::CameraSystems;
use crate::optima_bevy::optima_bevy_utils::egui::{EguiSelectionBlockContainer, EguiSystems, EguiWindowStateContainer};
use crate::optima_bevy::optima_bevy_utils::engine::{EngineSystems, FrameCount};
use crate::optima_bevy::optima_bevy_utils::gui::{GuiGlobalInfo, GuiSystems};
use crate::optima_bevy::optima_bevy_utils::lights::LightSystems;
use crate::optima_bevy::optima_bevy_utils::materials::{MaterialChangeRequestContainer, MaterialSystems};
use crate::optima_bevy::optima_bevy_utils::robot_scenes::{RobotLinkInfoVars, RobotSceneSystems};
use crate::optima_bevy::optima_bevy_utils::viewport_visuals::ViewportVisualsSystems;
use crate::optima_tensor_function::{OTFImmutVars, OTFImmutVarsObject};
use crate::robot_modules::robot::Robot;
use crate::robot_set_modules::robot_set::RobotSet;
use crate::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;

pub fn bevy_display_robot_geometric_shape_scene(robot_geometric_shape_scene: &RobotGeometricShapeScene) {
    let mut app = App::new();
    optima_bevy_base(&mut app);
    optima_bevy_starter_lights(&mut app);
    optima_bevy_pan_orbit_camera(&mut app);
    optima_bevy_robotics_scene_visuals_starter(&mut app);
    optima_bevy_egui_starter(&mut app);
    optima_bevy_debug_lines(&mut app, false);
    optima_bevy_spawn_robot_geometric_shape_scene(&mut app, robot_geometric_shape_scene);

    app.run();
}

pub fn bevy_robot_sliders(robot_geometric_shape_scene: &RobotGeometricShapeScene) {
    let mut app = App::new();
    optima_bevy_base(&mut app);
    optima_bevy_starter_lights(&mut app);
    optima_bevy_pan_orbit_camera(&mut app);
    optima_bevy_robotics_scene_visuals_starter(&mut app);
    optima_bevy_egui_starter(&mut app);
    optima_bevy_debug_lines(&mut app, false);
    optima_bevy_spawn_robot_geometric_shape_scene(&mut app, robot_geometric_shape_scene);

    app.add_system(RobotSceneSystems::system_robot_set_joint_sliders_egui.label("gui").label("joint_sliders"));
    app.add_system(RobotSceneSystems::system_robot_set_link_info_egui.label("gui").before("joint_sliders"));

    app.run();
}

pub fn bevy_robot_jacobian_visualization(robot: &Robot) {
    let mut app = App::new();
    optima_bevy_base(&mut app);
    optima_bevy_starter_lights(&mut app);
    optima_bevy_pan_orbit_camera(&mut app);
    optima_bevy_robotics_scene_visuals_starter(&mut app);
    optima_bevy_egui_starter(&mut app);
    optima_bevy_debug_lines(&mut app, false);
    optima_bevy_spawn_robot_and_robot_geometric_shape_module(&mut app, robot);

    app.add_system(RobotSceneSystems::system_robot_set_joint_sliders_egui.label("gui").label("joint_sliders"));
    app.add_system(RobotSceneSystems::system_robot_set_link_info_egui.label("gui").before("joint_sliders"));
    app.add_system(RobotSceneSystems::system_robot_jacobian_visualization_egui.label("gui").after("joint_sliders"));

    app.run();
}

pub fn bevy_robot_self_collisions_calibrator(robot: &Robot) {
    let mut app = App::new();
    optima_bevy_base(&mut app);
    optima_bevy_starter_lights(&mut app);
    optima_bevy_pan_orbit_camera(&mut app);
    optima_bevy_robotics_scene_visuals_starter(&mut app);
    optima_bevy_egui_starter(&mut app);
    optima_bevy_debug_lines(&mut app, false);
    optima_bevy_spawn_robot_and_robot_geometric_shape_module(&mut app, robot);

    app.add_system(RobotSceneSystems::system_robot_set_joint_sliders_egui.label("gui").label("joint_sliders"));
    app.add_system(RobotSceneSystems::system_robot_self_collision_calibrator_egui.label("gui").after("joint_sliders"));

    app.run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

fn optima_bevy_base(app: &mut App) {
    app
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(WindowDescriptor {
        title: "OPTIMA".to_string(),
        ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(bevy_stl::StlPlugin)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(FrameCount(0))
        .insert_resource(GuiGlobalInfo::new())
        .insert_resource(MaterialChangeRequestContainer::new())
        .add_system_to_stage(CoreStage::Last, MaterialSystems::update_optima_bevy_material_components_from_change_requests.label("materials").label("materials1"))
        .add_system_to_stage(CoreStage::Last, MaterialSystems::update_optima_bevy_material_components_from_auto_update.label("materials").label("materials2").after("materials1"))
        .add_system_to_stage(CoreStage::Last, MaterialSystems::update_materials.label("materials").label("materials3").after("materials2"))
        .add_system_to_stage(CoreStage::Last, EngineSystems::system_frame_counter)
        .add_system_to_stage(CoreStage::Last, GuiSystems::system_reset_cursor_over_gui_window);

}
fn optima_bevy_starter_lights(app: &mut App) {
    app
        .add_startup_system(LightSystems::starter_point_lights);
}
fn optima_bevy_pan_orbit_camera(app: &mut App) {
    app
        .add_startup_system(CameraSystems::system_spawn_pan_orbit_camera)
        .add_system(CameraSystems::system_pan_orbit_camera.after("gui"));
}
fn optima_bevy_egui_starter(app: &mut App) {
    app.add_startup_system(EguiSystems::init_egui_system);
    app.add_plugin(EguiPlugin);
    app.insert_resource(EguiWindowStateContainer::new());
    app.insert_resource(EguiSelectionBlockContainer::new());
}
fn optima_bevy_debug_lines(app: &mut App, with_depth_test: bool) {
    app.add_plugin(DebugLinesPlugin::with_depth_test(with_depth_test));
}
fn optima_bevy_robotics_scene_visuals_starter(app: &mut App) {
    app
        .add_startup_system(ViewportVisualsSystems::system_draw_robotics_grid);
}
fn optima_bevy_load_robot_geometric_shape_scene(app: &mut App, robot_geometric_shape_scene: &RobotGeometricShapeScene) {
    let mut immut_vars = OTFImmutVars::new();
    immut_vars.insert_or_replace(OTFImmutVarsObject::RobotGeometricShapeScene(robot_geometric_shape_scene.clone()));

    app.insert_resource(immut_vars);
    app.insert_resource(RobotLinkInfoVars::new(robot_geometric_shape_scene.robot_set()));
    app.add_system(RobotSceneSystems::system_set_robot_set_joint_states);
}
fn optima_bevy_spawn_robot_geometric_shape_scene(app: &mut App, robot_geometric_shape_scene: &RobotGeometricShapeScene) {
    optima_bevy_load_robot_geometric_shape_scene(app, robot_geometric_shape_scene);
    app.add_system(RobotSceneSystems::system_spawn_standard_robot_geometric_shape_scene);
}
fn optima_bevy_spawn_robot_and_robot_geometric_shape_module(app: &mut App, robot: &Robot) {
    let robot_geometric_shape_module = robot.generate_robot_geometric_shape_module().expect("error");

    app.insert_resource(robot.clone());
    app.insert_resource(robot_geometric_shape_module.clone());

    let robot_set = RobotSet::new_from_robot_configuration_modules(vec![robot.robot_configuration_module().clone()]);
    let robot_geometric_shape_scene = RobotGeometricShapeScene::new(robot_set, None).expect("error");

    optima_bevy_spawn_robot_geometric_shape_scene(app, &robot_geometric_shape_scene);
}

pub enum OptimaBevyStageLabels {
    GUI
}