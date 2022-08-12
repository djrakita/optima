use bevy::prelude::{Commands, default, PointLight, PointLightBundle, Transform};

pub struct LightSystems;
impl LightSystems {
    pub fn starter_point_lights(mut commands: Commands) {
        commands.spawn_bundle(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 4.0, 4.0),
        ..default()
    });
        commands.spawn_bundle(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            ..default()
        },
        transform: Transform::from_xyz(1.0, 2.0, -4.0),
        ..default()
    });
    }
}