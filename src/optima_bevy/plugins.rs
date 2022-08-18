use bevy::app::{App, CoreStage};
use bevy::DefaultPlugins;
use bevy::prelude::{Msaa, ParallelSystemDescriptorCoercion, Plugin, WindowDescriptor};
use crate::optima_bevy::optima_bevy_utils::camera::{CameraSystems};
use crate::optima_bevy::optima_bevy_utils::egui::{EguiEngine, EguiSystems};
use crate::optima_bevy::optima_bevy_utils::engine::{EngineSystems, FrameCount};
use crate::optima_bevy::optima_bevy_utils::gui::{GuiGlobalInfo, GuiSystems};

pub struct OptimaBasePlugin;
impl Plugin for OptimaBasePlugin {
    fn build(&self, app: &mut App) {
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
            .add_system_to_stage(CoreStage::Last, EngineSystems::system_frame_counter)
            .add_system_to_stage(CoreStage::Last, GuiSystems::system_reset_cursor_over_gui_window);
    }
}

pub struct OptimaPanOrbitCameraPlugin;
impl Plugin for OptimaPanOrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_startup_system(CameraSystems::system_spawn_pan_orbit_camera)
            .add_system(CameraSystems::system_pan_orbit_camera.after("gui"));
    }
}

pub struct OptimaEguiPlugin;
impl Plugin for OptimaEguiPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(EguiSystems::init_egui_system)
            .insert_resource(EguiEngine::new());
    }
}