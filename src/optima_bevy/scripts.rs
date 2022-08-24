use bevy::app::App;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext, EguiPlugin};
use bevy_prototype_debug_lines::DebugLinesPlugin;
use crate::optima_bevy::optima_bevy_utils::camera::CameraSystems;
use crate::optima_bevy::optima_bevy_utils::egui::{EguiEngine, EguiSystems, EguiWindowStateContainer};
use crate::optima_bevy::optima_bevy_utils::engine::{EngineSystems, FrameCount};
use crate::optima_bevy::optima_bevy_utils::generic_item::GenericItemSignature;
use crate::optima_bevy::optima_bevy_utils::gui::{GuiGlobalInfo, GuiSystems};
use crate::optima_bevy::optima_bevy_utils::lights::LightSystems;
use crate::optima_bevy::optima_bevy_utils::materials::{MaterialChangeRequest, MaterialChangeRequestContainer, MaterialChangeRequestType, MaterialSystems, OptimaBevyMaterial};
use crate::optima_bevy::optima_bevy_utils::robot_scenes::{RobotLinkInfoVars, RobotSceneActions, RobotSceneSystems, RobotSetJointStateBevyComponent};
use crate::optima_bevy::optima_bevy_utils::viewport_visuals::ViewportVisualsSystems;
use crate::optima_tensor_function::{OTFImmutVars, OTFImmutVarsObject};
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

    app.add_system(debug);

    app.run();
}

fn debug(mut m: ResMut<MaterialChangeRequestContainer>,
         keys: Res<Input<KeyCode>>) {
    if keys.pressed(KeyCode::Space) {
        m.add_request(MaterialChangeRequest::new(GenericItemSignature::RobotLink {
            robot_set_idx: 0,
            robot_idx_in_set: 0,
            link_idx_in_robot: 3
        }, 0, MaterialChangeRequestType::ChangeButResetInNFrames { material: OptimaBevyMaterial::Color(Color::rgba(0.,0.,1., 0.3)), n: 1 }));
    }
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
    app.insert_resource(EguiEngine::new());
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

pub enum OptimaBevyStageLabels {
    GUI
}