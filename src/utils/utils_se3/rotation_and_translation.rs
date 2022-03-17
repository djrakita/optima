use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::homogeneous_matrix::HomogeneousMatrix;
use crate::utils::utils_se3::implicit_dual_quaternion::ImplicitDualQuaternion;
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationType};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

/// A representation for an SE(3) transform composed of just rotation and translation components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotationAndTranslation {
    rotation: OptimaRotation,
    translation: Vector3<f64>
}
impl RotationAndTranslation {
    pub fn new(rotation: OptimaRotation, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation
        }
    }
    pub fn new_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64, rotation_type: &OptimaRotationType) -> Self {
        let rotation = match rotation_type {
            OptimaRotationType::RotationMatrix => { OptimaRotation::new_rotation_matrix_from_euler_angles(rx, ry, rz) }
            OptimaRotationType::UnitQuaternion => { OptimaRotation::new_unit_quaternion_from_euler_angles(rx, ry, rz) }
        };

        return Self::new(rotation, Vector3::new(x, y, z));
    }
    /// Returns the rotation component of the object.
    pub fn rotation(&self) -> &OptimaRotation {
        &self.rotation
    }
    /// Returns the translation component of the object.
    pub fn translation(&self) -> &Vector3<f64> {
        &self.translation
    }
    /// Multiplication
    pub fn multiply(&self, other: &RotationAndTranslation, conversion_if_necessary: bool) -> Result<RotationAndTranslation, OptimaError> {
        let rotation = self.rotation.multiply(&other.rotation, conversion_if_necessary)?;
        let translation = self.rotation.multiply_by_point(&other.translation) + &self.translation;
        return Ok(Self::new(rotation, translation));
    }
    /// Multiplication by a point.
    pub fn multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return self.rotation().multiply_by_point(point) + self.translation();
    }
    /// Inverse multiplies by the given point.  inverse multiplication is useful for placing the
    /// given point in the transform's local coordinate system.
    pub fn inverse_multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return self.rotation.inverse().multiply_by_point(&(point - &self.translation));
    }
    /// The inverse transform such that T * T^-1 = I.
    pub fn inverse(&self) -> RotationAndTranslation {
        let rotation = self.rotation.inverse();
        let translation = rotation.multiply_by_point(&-self.translation);
        return Self::new(rotation, translation);
    }
    /// The displacement transform such that T_self * T_disp = T_other.
    pub fn displacement(&self, other: &RotationAndTranslation, conversion_if_necessary: bool) -> Result<RotationAndTranslation, OptimaError> {
        return self.inverse().multiply(&other, conversion_if_necessary);
    }
    /// Provides an approximate distance between two objects.  This is not an
    /// official distance metric, but should still work in some optimization procedures.
    pub fn approximate_distance(&self, other: &RotationAndTranslation, conversion_if_necessary: bool) -> Result<f64, OptimaError> {
        let angle_between = self.rotation().angle_between(other.rotation(), conversion_if_necessary)?;
        let translation_between = (self.translation() - other.translation()).norm();
        return Ok(angle_between + translation_between);
    }
    /// Converts the internal rotation type of the object to another provided rotation type.
    pub fn convert_rotation_type(&mut self, target_type: &OptimaRotationType) {
        self.rotation = self.rotation.convert(target_type);
    }
    /// Converts the SE(3) pose to other supported pose types.
    pub fn convert(&self, target_type: &OptimaSE3PoseType) -> OptimaSE3Pose {
        return match target_type {
            OptimaSE3PoseType::ImplicitDualQuaternion => {
                let unit_quaternion = self.rotation().convert(&OptimaRotationType::UnitQuaternion).unwrap_unit_quaternion().expect("error").clone();
                let data = ImplicitDualQuaternion::new(unit_quaternion, self.translation().clone());
                OptimaSE3Pose::new_implicit_dual_quaternion(data)
            }
            OptimaSE3PoseType::HomogeneousMatrix => {
                let matrix = HomogeneousMatrix::rotation_and_translation_to_homogeneous_matrix(self.rotation(), self.translation());
                let data = HomogeneousMatrix::new(matrix);
                OptimaSE3Pose::new_homogeneous_matrix(data)
            }
            OptimaSE3PoseType::RotationAndTranslation => {
                OptimaSE3Pose::new_rotation_and_translation(self.clone())
            }
        }
    }
}