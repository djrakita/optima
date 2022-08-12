use bevy::prelude::ResMut;

pub struct EngineSystems;
impl EngineSystems {
    pub fn system_frame_counter(mut frame_count: ResMut<FrameCount>) {
        frame_count.0 += 1;
    }
}

pub struct FrameCount(pub usize);