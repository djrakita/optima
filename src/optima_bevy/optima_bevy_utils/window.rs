use bevy::math::Vec2;
use bevy::prelude::Res;
use bevy::window::Windows;

pub struct WindowUtils;
impl WindowUtils {
    pub fn util_get_primary_window_size(windows: &Res<Windows>) -> Vec2 {
        let window = windows.get_primary().unwrap();
        let size = Vec2::new(window.width() as f32, window.height() as f32);
        size
    }
}