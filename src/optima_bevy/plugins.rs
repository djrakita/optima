use bevy::app::App;
use bevy::DefaultPlugins;
use bevy::math::{Quat, Vec3};
use bevy::pbr::{PointLight, PointLightBundle};
use bevy::prelude::{Assets, AssetServer, Color, Commands, Mesh, Msaa, PbrBundle, Res, ResMut, StandardMaterial, Transform, WindowDescriptor};
use bevy::utils::default;
use crate::optima_bevy::optima_bevy_utils::camera::{CameraActions, CameraSystems};
use crate::optima_bevy::optima_bevy_utils::viewport_visuals::ViewportVisualsActions;

pub fn draw_test() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(WindowDescriptor {
            title: "OPTIMA".to_string(),
            ..Default::default()
         })
        .add_plugins(DefaultPlugins)
        .add_plugin(bevy_stl::StlPlugin)
        .add_startup_system(setup)
        .add_startup_system(test_system)
        .add_system(CameraSystems::system_pan_orbit_camera)
        .run();
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.spawn_bundle(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    /*
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..default()
    });
    */

    CameraActions::action_spawn_pan_orbit_camera(&mut commands, Vec3::new(3.0, 0.3, 1.0));
}

fn test_system(mut commands: Commands,
               mut meshes: ResMut<Assets<Mesh>>,
               mut materials: ResMut<Assets<StandardMaterial>>,
               asset_server: Res<AssetServer>) {
    ViewportVisualsActions::action_draw_robotics_grid(&mut commands, &mut meshes, &mut materials);

    commands
        .spawn_bundle(PbrBundle {
            mesh: asset_server.load("2.stl"),
            material: materials.add(Color::rgb(0.9, 0.4, 0.3).into()),
            transform: Transform::from_rotation(Quat::from_rotation_z(0.0)),
            ..Default::default()
        });
}

// fn draw_line(mut commands: Commands) {

// }