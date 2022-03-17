use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::homogeneous_matrix::HomogeneousMatrix;
use crate::utils::utils_se3::implicit_dual_quaternion::ImplicitDualQuaternion;
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationType};
use crate::utils::utils_se3::rotation_and_translation::RotationAndTranslation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimaSE3Pose {
    ImplicitDualQuaternion { data: ImplicitDualQuaternion, pose_type: OptimaSE3PoseType },
    HomogeneousMatrix { data: HomogeneousMatrix, pose_type: OptimaSE3PoseType },
    RotationAndTranslation { data: RotationAndTranslation, pose_type: OptimaSE3PoseType }
}
impl OptimaSE3Pose {
    pub fn new_implicit_dual_quaternion(data: ImplicitDualQuaternion) -> Self {
        Self::ImplicitDualQuaternion { data, pose_type: OptimaSE3PoseType::ImplicitDualQuaternion }
    }
    pub fn new_homogeneous_matrix(data: HomogeneousMatrix) -> Self {
        Self::HomogeneousMatrix { data, pose_type: OptimaSE3PoseType::HomogeneousMatrix }
    }
    pub fn new_rotation_and_translation(data: RotationAndTranslation) -> Self {
        Self::RotationAndTranslation { data, pose_type: OptimaSE3PoseType::RotationAndTranslation }
    }
    pub fn new_implicit_dual_quaternion_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_implicit_dual_quaternion(ImplicitDualQuaternion::new_from_euler_angles(rx, ry, rz, x, y, z))
    }
    pub fn new_homogeneous_matrix_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_homogeneous_matrix(HomogeneousMatrix::new_from_euler_angles(rx, ry, rz, x, y, z))
    }
    pub fn new_rotation_and_translation_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64, rotation_type: &OptimaRotationType) -> Self {
        Self::new_rotation_and_translation(RotationAndTranslation::new_from_euler_angles(rx, ry, rz, x, y, z, rotation_type))
    }
    pub fn convert(&self, target_type: &OptimaSE3PoseType) -> OptimaSE3Pose {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.convert(target_type) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.convert(target_type) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.convert(target_type) }
        }
    }
    pub fn inverse(&self) -> OptimaSE3Pose {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { Self::new_implicit_dual_quaternion(data.inverse()) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { Self::new_homogeneous_matrix(data.inverse()) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { Self::new_rotation_and_translation(data.inverse()) }
        }
    }
    pub fn multiply(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<OptimaSE3Pose, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_pose_type());
                self.multiply(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_string_descriptor_error("incompatible pose types in multiply."))
            }
        }

        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                        Ok(OptimaSE3Pose::new_implicit_dual_quaternion(data0.multiply_shortcircuit(data)))
                    }
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible pose types in multiply.")) }
                }
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                        Ok(OptimaSE3Pose::new_homogeneous_matrix(data0.multiply(data)))
                    }
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible pose types in multiply.")) }
                }
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                        Ok(OptimaSE3Pose::new_rotation_and_translation(data0.multiply(data, conversion_if_necessary)?))
                    }
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible pose types in multiply.")) }
                }
            }
        }
    }
    pub fn unwrap_homogeneous_matrix(&self) -> Result<&HomogeneousMatrix, OptimaError> {
        return match self {
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                Ok(data)
            }
            _ => {
                Err(OptimaError::new_string_descriptor_error("tried to unwrap homogenous matrix on incompatible type."))
            }
        }
    }
    pub fn unwrap_implicit_dual_quaternion(&self) -> Result<&ImplicitDualQuaternion, OptimaError> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                Ok(data)
            }
            _ => {
                Err(OptimaError::new_string_descriptor_error("tried to unwrap implicit dual quaternion on incompatible type."))
            }
        }
    }
    pub fn unwrap_rotation_and_translation(&self) -> Result<&RotationAndTranslation, OptimaError> {
        return match self {
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                Ok(data)
            }
            _ => {
                Err(OptimaError::new_string_descriptor_error("tried to unwrap rotation and translation on incompatible type."))
            }
        }
    }
    fn are_types_compatible(a: &OptimaSE3Pose, b: &OptimaSE3Pose) -> bool {
        return if a.get_pose_type() == b.get_pose_type() { true } else { false }
    }
    fn get_pose_type(&self) -> &OptimaSE3PoseType {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, pose_type } => { pose_type }
            OptimaSE3Pose::HomogeneousMatrix { data, pose_type } => { pose_type }
            OptimaSE3Pose::RotationAndTranslation { data, pose_type } => { pose_type }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum OptimaSE3PoseType {
    ImplicitDualQuaternion,
    HomogeneousMatrix,
    RotationAndTranslation
}
