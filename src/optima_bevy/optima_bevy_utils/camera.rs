use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::Vec3;
use bevy::prelude::*;
use bevy::render::camera::Projection;
use crate::optima_bevy::optima_bevy_utils::gui::GuiGlobalInfo;
use crate::optima_bevy::optima_bevy_utils::transform::TransformUtils;
use crate::optima_bevy::optima_bevy_utils::window::WindowUtils;

pub struct CameraActions;
impl CameraActions {
    pub fn action_spawn_pan_orbit_camera(commands: &mut Commands, location: Vec3) {
        let translation = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(location);
        let radius = translation.length();

        commands.spawn_bundle(Camera3dBundle {
            transform: Transform::from_translation(translation)
                .looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        }).insert(PanOrbitCamera {
            radius,
            ..Default::default()
        });
    }
}

pub struct CameraSystems;
impl CameraSystems {
    pub fn system_spawn_pan_orbit_camera(mut commands: Commands) {
        CameraActions::action_spawn_pan_orbit_camera(&mut commands, Vec3::new(5.0, 0.8, 1.5));
    }
    pub fn system_pan_orbit_camera(
        windows: Res<Windows>,
        mut ev_motion: EventReader<MouseMotion>,
        mut ev_scroll: EventReader<MouseWheel>,
        input_mouse: Res<Input<MouseButton>>,
        input_keyboard: Res<Input<KeyCode>>,
        mut query: Query<(&mut PanOrbitCamera, &mut Transform, &Projection)>,
        gui_global_info: Res<GuiGlobalInfo>) {

            if gui_global_info.is_hovering_over_gui { return; }

            let orbit_button = MouseButton::Left;
            // let pan_button = MouseButton::Middle;

            let mut pan = Vec2::ZERO;
            let mut rotation_move = Vec2::ZERO;
            let mut scroll = 0.0;
            let mut orbit_button_changed = false;

            if input_mouse.pressed(orbit_button) && (input_keyboard.pressed(KeyCode::RShift) || input_keyboard.pressed(KeyCode::LShift)) {
                // Pan only if we're not rotating at the moment
                for ev in ev_motion.iter() {
                    pan += ev.delta;
                }
            } else if input_mouse.pressed(orbit_button) && (input_keyboard.pressed(KeyCode::LAlt) || input_keyboard.pressed(KeyCode::RAlt)) {
                for ev in ev_motion.iter() {
                    rotation_move += ev.delta;
                }
            }
            for ev in ev_scroll.iter() {
                scroll += 0.05 * ev.y;
            }
            if input_mouse.just_released(orbit_button) || input_mouse.just_pressed(orbit_button) {
                orbit_button_changed = true;
            }

            for (mut pan_orbit, mut transform, projection) in query.iter_mut() {
                if orbit_button_changed {
                    // only check for upside down when orbiting started or ended this frame
                    // if the camera is "upside" down, panning horizontally would be inverted, so invert the input to make it correct
                    let up = transform.rotation * Vec3::Y;
                    pan_orbit.upside_down = up.y <= 0.0;
                }

                let mut any = false;
                if rotation_move.length_squared() > 0.0 {
                    any = true;
                    let window = WindowUtils::util_get_primary_window_size(&windows);
                    let delta_x = {
                        let delta = rotation_move.x / window.x * std::f32::consts::PI * 2.0;
                        if pan_orbit.upside_down { -delta } else { delta }
                    };
                    let delta_y = rotation_move.y / window.y * std::f32::consts::PI;
                    let yaw = Quat::from_rotation_y(-delta_x);
                    let pitch = Quat::from_rotation_x(-delta_y);
                    transform.rotation = yaw * transform.rotation; // rotate around global y axis
                    transform.rotation = transform.rotation * pitch; // rotate around local x axis
                } else if pan.length_squared() > 0.0 {
                    any = true;
                    // make panning distance independent of resolution and FOV,
                    let window = WindowUtils::util_get_primary_window_size(&windows);
                    if let Projection::Perspective(projection) = projection {
                        pan *= Vec2::new(projection.fov * projection.aspect_ratio, projection.fov) / window;
                    }
                    // translate by local axes
                    let right = transform.rotation * Vec3::X * -pan.x;
                    let up = transform.rotation * Vec3::Y * pan.y;
                    // make panning proportional to distance away from focus point
                    let translation = (right + up) * pan_orbit.radius;
                    pan_orbit.focus += translation;
                } else if scroll.abs() > 0.0 {
                    any = true;
                    pan_orbit.radius -= scroll * pan_orbit.radius * 0.2;
                    // dont allow zoom to reach zero or you get stuck
                    pan_orbit.radius = f32::max(pan_orbit.radius, 0.05);
                }

                if any {
                    // emulating parent/child to make the yaw/y-axis rotation behave like a turntable
                    // parent = x and y rotation
                    // child = z-offset
                    let rot_matrix = Mat3::from_quat(transform.rotation);
                    transform.translation = pan_orbit.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, pan_orbit.radius));
                }
            }
    }
}

#[derive(Component)]
pub struct PanOrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub upside_down: bool,
}
impl Default for PanOrbitCamera {
    fn default() -> Self {
        PanOrbitCamera {
            focus: Vec3::ZERO,
            radius: 5.0,
            upside_down: false,
        }
    }
}