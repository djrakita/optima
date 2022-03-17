use nalgebra::{UnitQuaternion, Vector3};
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::homogeneous_matrix::HomogeneousMatrix;
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationType};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
use crate::utils::utils_se3::rotation_and_translation::RotationAndTranslation;

/// A representation for an SE(3) transform (translation and rotation) proposed by Neil Dantam.
/// For more details, see the IJRR paper Robust and Efficient Forward, Differential, and Inverse
/// Kinematics using Dual Quaternions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImplicitDualQuaternion {
    rotation: UnitQuaternion<f64>,
    translation: Vector3<f64>,
    is_identity: bool,
    rot_is_identity: bool,
    translation_is_zeros: bool
}
impl ImplicitDualQuaternion {
    pub fn new(rotation: UnitQuaternion<f64>, translation: Vector3<f64>) -> Self {
        let mut out_self = Self {
            rotation,
            translation,
            is_identity: false,
            rot_is_identity: false,
            translation_is_zeros: false
        };
        out_self.decide_if_identity();

        return out_self;
    }
    pub fn new_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        let r= UnitQuaternion::from_euler_angles(rx, ry, rz);
        let t = Vector3::new(x, y, z);
        return Self::new(r, t);
    }
    pub fn new_identity() -> Self {
        // return Self::new_from_euler_angles(0.,0.,0., Vector3::zeros());
        todo!()
    }
    fn decide_if_identity(&mut self) {
        if (self.rotation.w.abs() - 1.0).abs() < 0.000001 && self.rotation.i.abs() < 0.000001 && self.rotation.j.abs() < 0.000001 && self.rotation.k.abs() < 0.000001 {
            self.rot_is_identity = true;
        } else {
            self.rot_is_identity = false;
        }

        if self.translation[0].abs() < 0.000001 && self.translation[1].abs() < 0.000001 && self.translation[2] < 0.000001 {
            self.translation_is_zeros = true;
        } else {
            self.translation_is_zeros = false;
        }

        if self.rot_is_identity && self.translation_is_zeros {
            self.is_identity = true;
        } else {
            self.is_identity = false;
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// Returns a reference to the rotation component of the implicit dual quaternion.
    pub fn rotation(&self) -> &UnitQuaternion<f64> { &self.rotation }
    /// Returns a reference to the translation component of the implicit dual quaternion.
    pub fn translation(&self) -> &Vector3<f64> {
        &self.translation
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    pub fn multiply(&self, other: &ImplicitDualQuaternion, rotation_conversion_if_necessary: bool) -> Result<ImplicitDualQuaternion, OptimaError> {
        let mut out_rot = self.rotation * &other.rotation;
        let mut out_translation = self.rotation * &other.translation + &self.translation;
        return Ok(ImplicitDualQuaternion::new(out_rot, out_translation));
    }
    pub fn multiply_shortcircuit(&self, other: &ImplicitDualQuaternion) -> ImplicitDualQuaternion {
        if self.is_identity { return other.clone(); }
        if other.is_identity { return self.clone(); }

        let mut out_rot = self.rotation.clone();
        if self.rot_is_identity { out_rot = other.rotation().clone(); }
        else if other.rot_is_identity {
            // do nothing
        }
        else {
            // out_rot *= &other.rotation;
        }

        let mut out_translation = self.translation.clone();
        if self.rot_is_identity && !other.translation_is_zeros {
            out_translation += &other.translation;
        } else if self.rot_is_identity && other.translation_is_zeros {
            // do nothing
        } else {
            out_translation += &self.rotation * &other.translation;
        }

        return ImplicitDualQuaternion::new(out_rot, out_translation);
    }
    pub fn multiply_by_vector3(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return self.rotation * point + self.translation;
    }
    pub fn multiply_by_vector3_shortcircuit(&self, point: &Vector3<f64>) -> Vector3<f64> {
        if self.is_identity { return point.clone(); }
        if self.rot_is_identity { return point + self.translation; }
        if self.translation_is_zeros { return self.rotation * point; }
        return self.rotation * point + self.translation;
    }
    pub fn inverse(&self) -> ImplicitDualQuaternion {
        let mut new_quat = self.rotation.inverse();
        let mut new_translation = &new_quat * -self.translation.clone();
        return ImplicitDualQuaternion::new(new_quat, new_translation);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    pub fn convert(&self, target_type: &OptimaSE3PoseType) -> OptimaSE3Pose {
        return match target_type {
            OptimaSE3PoseType::ImplicitDualQuaternion => {
                OptimaSE3Pose::new_implicit_dual_quaternion(self.clone())
            }
            OptimaSE3PoseType::HomogeneousMatrix => {
                let rotation = OptimaRotation::new_unit_quaternion(self.rotation().clone());
                let matrix = HomogeneousMatrix::rotation_and_translation_to_homogeneous_matrix(&rotation, self.translation());
                let homogeneous_matrix = HomogeneousMatrix::new(matrix);
                OptimaSE3Pose::new_homogeneous_matrix(homogeneous_matrix)
            }
            OptimaSE3PoseType::RotationAndTranslation => {
                let rotation = OptimaRotation::new_unit_quaternion(self.rotation().clone());
                OptimaSE3Pose::new_rotation_and_translation(RotationAndTranslation::new(rotation, self.translation().clone()))
            }
        }
    }
}