use bevy::app::{App, CoreStage};
use bevy::DefaultPlugins;
use bevy::prelude::{Msaa, Plugin, WindowDescriptor};
use crate::optima_bevy::optima_bevy_utils::camera::{CameraSystems};
use crate::optima_bevy::optima_bevy_utils::engine::{EngineSystems, FrameCount};

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
            .add_system_to_stage(CoreStage::Last, EngineSystems::system_frame_counter);
    }
}

pub struct PanOrbitCameraPlugin;
impl Plugin for PanOrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_startup_system(CameraSystems::system_spawn_pan_orbit_camera)
            .add_system(CameraSystems::system_pan_orbit_camera);
    }
}