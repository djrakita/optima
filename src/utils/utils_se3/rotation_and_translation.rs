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
    pub fn rotation(&self) -> &OptimaRotation {
        &self.rotation
    }
    pub fn translation(&self) -> &Vector3<f64> {
        &self.translation
    }
    pub fn multiply(&self, other: &RotationAndTranslation, conversion_if_necessary: bool) -> Result<RotationAndTranslation, OptimaError> {
        let rotation = self.rotation.multiply(&other.rotation, conversion_if_necessary)?;
        let translation = self.rotation.multiply_by_point(&other.translation) + &self.translation;
        return Ok(Self::new(rotation, translation));
    }
    pub fn inverse(&self) -> RotationAndTranslation {
        let rotation = self.rotation.inverse();
        let translation = rotation.multiply_by_point(&-self.translation);
        return Self::new(rotation, translation);
    }
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