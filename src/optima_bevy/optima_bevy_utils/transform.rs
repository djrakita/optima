use bevy::math::{EulerRot, Vec3};
use bevy::prelude::{Quat, Transform};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose};

pub struct TransformUtils;
impl TransformUtils {
    pub fn util_convert_se3_pose_to_y_up_bevy_transform(pose: &OptimaSE3Pose) -> Transform {
        let pose_new = OptimaSE3Pose::new_from_euler_angles(-std::f64::consts::FRAC_PI_2, 0., 0., 0., 0., 0., pose.map_to_pose_type()).multiply(pose, false).expect("error");
        let translation = pose_new.translation();
        let rotation = pose_new.rotation().to_euler_angles();

        let t = Transform {
            translation: Vec3::new(translation.x as f32, translation.y as f32, translation.z as f32),
            rotation: Quat::from_euler(EulerRot::XYZ, rotation.x as f32, rotation.y as f32, rotation.z as f32),
            ..Default::default()
        };

        return t;
    }

    pub fn util_convert_z_up_vec3_to_y_up_bevy_vec3(vec: Vec3) -> Vec3 {
        return Vec3::new(vec.x, vec.z, -vec.y);
    }

    pub fn util_convert_bevy_y_up_vec3_to_z_up_vec3(vec: Vec3) -> Vec3 {
        return Vec3::new(vec.x, -vec.z, vec.y);
    }
}