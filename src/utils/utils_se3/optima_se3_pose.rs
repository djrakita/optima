#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use nalgebra::{Isometry3, Matrix3, Matrix4, Quaternion, Rotation3, Unit, UnitQuaternion, Vector3};
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::homogeneous_matrix::HomogeneousMatrix;
use crate::utils::utils_se3::implicit_dual_quaternion::ImplicitDualQuaternion;
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationType};
use crate::utils::utils_se3::rotation_and_translation::RotationAndTranslation;
#[cfg(target_arch = "wasm32")]
use crate::utils::utils_wasm::JsMatrix;
#[cfg(target_arch = "wasm32")]
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};

/// An enum used to represent a rotation or orientation.  The enum affords easy conversion between
/// rotation types and functions over singular or pairs of rotations.
/// This is the main object that should be used for representing an SE(3) pose due to its
/// flexibility and interoperability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimaSE3Pose {
    ImplicitDualQuaternion { data: ImplicitDualQuaternion, pose_type: OptimaSE3PoseType },
    HomogeneousMatrix { data: HomogeneousMatrix, pose_type: OptimaSE3PoseType },
    RotationAndTranslation { data: RotationAndTranslation, pose_type: OptimaSE3PoseType },
    EulerAnglesAndTranslation { euler_angles: Vector3<f64>, translation: Vector3<f64>, phantom_data: ImplicitDualQuaternion, pose_type: OptimaSE3PoseType }
}
impl OptimaSE3Pose {
    pub fn new_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64, t: &OptimaSE3PoseType) -> Self {
        match t {
            OptimaSE3PoseType::ImplicitDualQuaternion => {
                Self::new_implicit_dual_quaternion_from_euler_angles(rx, ry, rz, x, y, z)
            }
            OptimaSE3PoseType::HomogeneousMatrix => {
                Self::new_homogeneous_matrix_from_euler_angles(rx, ry, rz, x, y, z)
            }
            OptimaSE3PoseType::UnitQuaternionAndTranslation => {
                Self::new_unit_quaternion_and_translation_from_euler_angles(rx, ry, rz, x, y, z)
            }
            OptimaSE3PoseType::RotationMatrixAndTranslation => {
                Self::new_rotation_matrix_and_translation_from_euler_angles(rx, ry, rz, x, y, z)
            }
            OptimaSE3PoseType::EulerAnglesAndTranslation => {
                let phantom_data = ImplicitDualQuaternion::new_from_euler_angles(rx, ry, rz, x, y, z);
                let euler_angles_and_translation = phantom_data.to_euler_angles_and_translation();
                Self::EulerAnglesAndTranslation {
                    euler_angles: euler_angles_and_translation.0,
                    translation: euler_angles_and_translation.1,
                    phantom_data,
                    pose_type: OptimaSE3PoseType::EulerAnglesAndTranslation
                }
            }
        }
    }
    pub fn new_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64, x: f64, y: f64, z: f64, t: &OptimaSE3PoseType) -> Self {
        match t {
            OptimaSE3PoseType::ImplicitDualQuaternion => {
                Self::new_implicit_dual_quaternion_from_axis_angle(axis, angle, x, y, z)
            }
            OptimaSE3PoseType::HomogeneousMatrix => {
                Self::new_homogeneous_matrix_from_axis_angle(axis, angle, x, y, z)
            }
            OptimaSE3PoseType::UnitQuaternionAndTranslation => {
                Self::new_unit_quaternion_and_translation_from_axis_angle(axis, angle, x, y, z)
            }
            OptimaSE3PoseType::RotationMatrixAndTranslation => {
                Self::new_rotation_matrix_and_translation_from_axis_angle(axis, angle, x, y, z)
            }
            OptimaSE3PoseType::EulerAnglesAndTranslation => {
                let phantom_data = ImplicitDualQuaternion::new_from_axis_angle(axis, angle, x, y, z);
                let euler_angles_and_translation = phantom_data.to_euler_angles_and_translation();
                Self::EulerAnglesAndTranslation {
                    euler_angles: euler_angles_and_translation.0,
                    translation: euler_angles_and_translation.1,
                    phantom_data,
                    pose_type: OptimaSE3PoseType::EulerAnglesAndTranslation
                }
            }
        }
    }
    pub fn new_identity() -> Self {
        Self::new_from_euler_angles(0.,0.,0.,0.,0.,0., &OptimaSE3PoseType::ImplicitDualQuaternion)
    }

    pub fn new_homogeneous_matrix(data: HomogeneousMatrix) -> Self {
        Self::HomogeneousMatrix { data, pose_type: OptimaSE3PoseType::HomogeneousMatrix }
    }
    pub fn new_homogeneous_matrix_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_homogeneous_matrix(HomogeneousMatrix::new_from_euler_angles(rx, ry, rz, x, y, z))
    }
    pub fn new_homogeneous_matrix_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_homogeneous_matrix(HomogeneousMatrix::new_from_axis_angle(axis, angle, x, y, z))
    }

    pub fn new_rotation_and_translation(data: RotationAndTranslation) -> Self {
        match data.rotation() {
            OptimaRotation::RotationMatrix { .. } => {
                Self::RotationAndTranslation { data, pose_type: OptimaSE3PoseType::RotationMatrixAndTranslation }
            }
            OptimaRotation::UnitQuaternion { .. } => {
                Self::RotationAndTranslation { data, pose_type: OptimaSE3PoseType::UnitQuaternionAndTranslation }
            }
        }
    }
    pub fn new_rotation_and_translation_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64, rotation_type: &OptimaRotationType) -> Self {
        Self::new_rotation_and_translation(RotationAndTranslation::new_from_euler_angles(rx, ry, rz, x, y, z, rotation_type))
    }
    pub fn new_rotation_and_translation_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64, x: f64, y: f64, z: f64, rotation_type: &OptimaRotationType) -> Self {
        Self::new_rotation_and_translation(RotationAndTranslation::new_from_axis_angle(axis, angle, x, y, z, rotation_type))
    }

    pub fn new_unit_quaternion_and_translation(q: UnitQuaternion<f64>, t: Vector3<f64>) -> Self {
        Self::new_rotation_and_translation(RotationAndTranslation::new(OptimaRotation::new_unit_quaternion(q), t))
    }
    pub fn new_unit_quaternion_and_translation_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_rotation_and_translation_from_euler_angles(rx, ry, rz, x, y, z, &OptimaRotationType::UnitQuaternion)
    }
    pub fn new_unit_quaternion_and_translation_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_rotation_and_translation_from_axis_angle(axis, angle, x, y, z, &OptimaRotationType::UnitQuaternion)
    }

    pub fn new_rotation_matrix_and_translation(m: Rotation3<f64>, t: Vector3<f64>) -> Self {
        Self::new_rotation_and_translation(RotationAndTranslation::new(OptimaRotation::new_rotation_matrix(m), t))
    }
    pub fn new_rotation_matrix_and_translation_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_rotation_and_translation_from_euler_angles(rx, ry, rz, x, y, z, &OptimaRotationType::RotationMatrix)
    }
    pub fn new_rotation_matrix_and_translation_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_rotation_and_translation_from_axis_angle(axis, angle, x, y, z, &OptimaRotationType::RotationMatrix)
    }

    pub fn new_implicit_dual_quaternion(data: ImplicitDualQuaternion) -> Self {
        Self::ImplicitDualQuaternion { data, pose_type: OptimaSE3PoseType::ImplicitDualQuaternion }
    }
    pub fn new_implicit_dual_quaternion_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_implicit_dual_quaternion(ImplicitDualQuaternion::new_from_euler_angles(rx, ry, rz, x, y, z))
    }
    pub fn new_implicit_dual_quaternion_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new_implicit_dual_quaternion(ImplicitDualQuaternion::new_from_axis_angle(axis, angle, x, y, z))
    }

    pub fn new_euler_angles_and_translation(data: ImplicitDualQuaternion) -> Self {
        let euler_angles_and_translation = data.to_euler_angles_and_translation();
        Self::EulerAnglesAndTranslation {
            euler_angles: euler_angles_and_translation.0,
            translation: euler_angles_and_translation.1,
            phantom_data: data,
            pose_type: OptimaSE3PoseType::EulerAnglesAndTranslation
        }
    }

    /// Converts the SE(3) pose to other supported pose types.
    pub fn convert(&self, target_type: &OptimaSE3PoseType) -> OptimaSE3Pose {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.convert(target_type) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.convert(target_type) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.convert(target_type) }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles: _, translation: _, phantom_data, pose_type: _ } => {
                Self::new_implicit_dual_quaternion(phantom_data.clone()).convert(target_type)
            }
        }
    }
    /// The inverse transform such that T * T^-1 = I.
    pub fn inverse(&self) -> OptimaSE3Pose {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { Self::new_implicit_dual_quaternion(data.inverse()) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { Self::new_homogeneous_matrix(data.inverse()) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { Self::new_rotation_and_translation(data.inverse()) }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                Self::new_implicit_dual_quaternion(phantom_data.inverse()).convert(&OptimaSE3PoseType::EulerAnglesAndTranslation)
            }
        }
    }
    /// Transform multiplication.
    pub fn multiply(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<OptimaSE3Pose, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.map_to_pose_type());
                self.multiply(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible pose types in multiply.", file!(), line!()))
            }
        }

        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                        Ok(OptimaSE3Pose::new_implicit_dual_quaternion(data0.multiply_shortcircuit(data)))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in multiply.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                        Ok(OptimaSE3Pose::new_homogeneous_matrix(data0.multiply(data)))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in multiply.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                        Ok(OptimaSE3Pose::new_rotation_and_translation(data0.multiply(data, conversion_if_necessary)?))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in multiply.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                let data0 = phantom_data;
                match other {
                    OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                        Ok(OptimaSE3Pose::new_implicit_dual_quaternion(data0.multiply_shortcircuit(phantom_data)).convert(&OptimaSE3PoseType::EulerAnglesAndTranslation))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in distance function.", file!(), line!())) }
                }
            }
        }
    }
    /// Multiplication by a point.
    pub fn multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.multiply_by_point(point) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.multiply_by_point(point) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.multiply_by_point(point) }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => { phantom_data.inverse_multiply_by_point_shortcircuit(point) }
        }
    }
    /// Inverse multiplies by the given point.  inverse multiplication is useful for placing the
    /// given point in the transform's local coordinate system.
    pub fn inverse_multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.inverse_multiply_by_point_shortcircuit(point) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.inverse_multiply_by_point(point) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.inverse_multiply_by_point(point) }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => { phantom_data.inverse_multiply_by_point_shortcircuit(point) }
        }
    }
    /// The displacement transform such that T_self * T_disp = T_other.
    pub fn displacement(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<OptimaSE3Pose, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.map_to_pose_type());
                self.displacement(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible pose types in displacement.", file!(), line!()))
            }
        }

        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                        Ok(OptimaSE3Pose::new_implicit_dual_quaternion(data0.displacement(data)))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                        Ok(OptimaSE3Pose::new_homogeneous_matrix(data0.displacement(data)))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                        Ok(OptimaSE3Pose::new_rotation_and_translation(data0.displacement(data, conversion_if_necessary)?))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                let data0 = phantom_data;
                match other {
                    OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                        Ok(OptimaSE3Pose::new_implicit_dual_quaternion(data0.displacement(phantom_data)).convert(&OptimaSE3PoseType::EulerAnglesAndTranslation))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement.", file!(), line!())) }
                }
            }
        }
    }

    pub fn displacement_separate_rotation_and_translation(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<(OptimaRotation, Vector3<f64>), OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.map_to_pose_type());
                self.displacement_separate_rotation_and_translation(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible pose types in displacement_separate_rotation_and_translation.", file!(), line!()))
            }
        }

        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                        Ok(data0.displacement_separate_rotation_and_translation(data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement_separate_rotation_and_translation.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                        Ok(data0.displacement_separate_rotation_and_translation(data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement_separate_rotation_and_translation.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                        Ok(data0.displacement_separate_rotation_and_translation(data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement_separate_rotation_and_translation.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                let data0 = phantom_data;
                match other {
                    OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                        Ok(data0.displacement_separate_rotation_and_translation(phantom_data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in displacement_separate_rotation_and_translation.", file!(), line!())) }
                }
            }
        }
    }

    pub fn slerp(&self, other: &OptimaSE3Pose, t: f64, conversion_if_necessary: bool) -> Result<OptimaSE3Pose, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.map_to_pose_type());
                self.slerp(&new_operand, t, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible pose types in slerp.", file!(), line!()))
            }
        }

        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                        Ok(Self::new_implicit_dual_quaternion(data0.slerp(data, t)))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in slerp.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                        Ok(Self::new_homogeneous_matrix(data0.slerp(data, t)))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in slerp.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                        Ok(Self::new_rotation_and_translation(data0.slerp(data, t, conversion_if_necessary)?))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in slerp.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                let data0 = phantom_data;
                match other {
                    OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                        Ok(Self::new_euler_angles_and_translation( data0.slerp(phantom_data, t) ))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in slerp.", file!(), line!())) }
                }
            }
        }
    }
    /// Distance function between transforms.  This may be approximate.
    /// In the case of the implicit dual quaternion, this is smooth, differentiable, and exact (one
    /// of the benefits of that representation).
    pub fn distance_function(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<f64, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.map_to_pose_type());
                self.distance_function(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible pose types in distance function.", file!(), line!()))
            }
        }

        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                        Ok(data0.displacement(data).ln_l2_magnitude())
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in distance function.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                        Ok(data0.approximate_distance(&data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in distance function.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                let data0 = data;
                match other {
                    OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                        data0.approximate_distance(&data, conversion_if_necessary)
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in distance function.", file!(), line!())) }
                }
            }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                let data0 = phantom_data;
                match other {
                    OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles:_, translation:_, phantom_data, pose_type:_ } => {
                        Ok(data0.displacement(phantom_data).ln_l2_magnitude())
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible pose types in distance function.", file!(), line!())) }
                }
            }
        }
    }
    /// Unwraps homogeneous matrix.  Returns error if the underlying representation is not homogeneous matrix.
    pub fn unwrap_homogeneous_matrix(&self) -> Result<&HomogeneousMatrix, OptimaError> {
        return match self {
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                Ok(data)
            }
            _ => {
                Err(OptimaError::new_generic_error_str("tried to unwrap homogenous matrix on incompatible type.", file!(), line!()))
            }
        }
    }
    /// Unwraps implicit dual quaternion.  Returns error if the underlying representation is not IDQ.
    pub fn unwrap_implicit_dual_quaternion(&self) -> Result<&ImplicitDualQuaternion, OptimaError> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                Ok(data)
            }
            _ => {
                Err(OptimaError::new_generic_error_str("tried to unwrap implicit dual quaternion on incompatible type.", file!(), line!()))
            }
        }
    }
    /// Unwraps rotation and translation.  Returns error if the underlying representation is not R&T.
    pub fn unwrap_rotation_and_translation(&self) -> Result<&RotationAndTranslation, OptimaError> {
        return match self {
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                Ok(data)
            }
            _ => {
                Err(OptimaError::new_generic_error_str("tried to unwrap rotation and translation on incompatible type.", file!(), line!()))
            }
        }
    }
    /// Returns an euler angle and vector representation of the SE(3) pose.
    pub fn to_euler_angles_and_translation(&self) -> (Vector3<f64>, Vector3<f64>) {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => {
                data.to_euler_angles_and_translation()
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                data.to_euler_angles_and_translation()
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                data.to_euler_angles_and_translation()
            }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles, translation, .. } => {
                return (euler_angles.clone(), translation.clone())
            }
        }
    }
    /// Returns an axis angle and translation representation of the SE(3) pose.
    pub fn to_axis_angle_and_translation(&self) -> (Vector3<f64>, f64, Vector3<f64>) {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.to_axis_angle_and_translation() }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.to_axis_angle_and_translation() }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.to_axis_angle_and_translation() }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles: _, translation: _, phantom_data, pose_type: _ } => { phantom_data.to_axis_angle_and_translation() }
        }
    }
    /// Outputs an Isometry3 object in the Nalgebra library that corresponds to the SE(3) pose.
    pub fn to_nalgebra_isometry(&self) -> Isometry3<f64> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.to_nalgebra_isometry() }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.to_nalgebra_isometry() }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.to_nalgebra_isometry() }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles: _, translation: _, phantom_data, pose_type: _ } => { phantom_data.to_nalgebra_isometry() }
        }
    }
    /// Converts to vector representation.
    pub fn to_vec_representation(&self) -> Vec<Vec<f64>> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.to_vec_representation() }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.to_vec_representation() }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.to_vec_representation() }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles, translation, .. } => {
                let e = euler_angles;
                let t = translation;
                return vec![ vec![e[0], e[1], e[2]], vec![t[0], t[1], t[2]] ];
            }
        }
    }
    pub fn map_to_pose_type(&self) -> &OptimaSE3PoseType {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data: _, pose_type } => { pose_type }
            OptimaSE3Pose::HomogeneousMatrix { data: _, pose_type } => { pose_type }
            OptimaSE3Pose::RotationAndTranslation { data: _, pose_type } => { pose_type }
            OptimaSE3Pose::EulerAnglesAndTranslation { euler_angles: _, translation: _, phantom_data: _, pose_type } => { pose_type }
        }
    }
    fn are_types_compatible(a: &OptimaSE3Pose, b: &OptimaSE3Pose) -> bool {
        return if a.map_to_pose_type() == b.map_to_pose_type() { true } else { false }
    }
}
impl Default for OptimaSE3Pose {
    fn default() -> Self {
        Self::new_identity()
    }
}

/// An Enum that encodes a pose type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimaSE3PoseType {
    ImplicitDualQuaternion,
    HomogeneousMatrix,
    UnitQuaternionAndTranslation,
    RotationMatrixAndTranslation,
    EulerAnglesAndTranslation
}

/// A container object that holds all OPtimaSE3 types.  This is useful for functions that may need
/// to handle many pose types, so this allows all to be initialized and saved at once to avoid
/// many transform conversions at run-time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaSE3PoseAll {
    implicit_dual_quaternion: OptimaSE3Pose,
    homogeneous_matrix: OptimaSE3Pose,
    unit_quaternion_and_translation: OptimaSE3Pose,
    rotation_matrix_and_translation: OptimaSE3Pose,
    euler_angles_and_translation: OptimaSE3Pose
}
impl OptimaSE3PoseAll {
    pub fn new(p: &OptimaSE3Pose) -> Self {
        Self {
            implicit_dual_quaternion: p.convert(&OptimaSE3PoseType::ImplicitDualQuaternion),
            homogeneous_matrix: p.convert(&OptimaSE3PoseType::HomogeneousMatrix),
            unit_quaternion_and_translation: p.convert(&OptimaSE3PoseType::UnitQuaternionAndTranslation),
            rotation_matrix_and_translation: p.convert(&OptimaSE3PoseType::RotationMatrixAndTranslation),
            euler_angles_and_translation: p.convert(&OptimaSE3PoseType::EulerAnglesAndTranslation)
        }
    }

    pub fn new_identity() -> Self {
        return Self::new(&OptimaSE3Pose::new_unit_quaternion_and_translation_from_euler_angles(0.,0.,0.,0.,0.,0.));
    }

    pub fn get_pose_by_type(&self, t: &OptimaSE3PoseType) -> &OptimaSE3Pose {
        return match t {
            OptimaSE3PoseType::ImplicitDualQuaternion => { &self.implicit_dual_quaternion }
            OptimaSE3PoseType::HomogeneousMatrix => { &self.homogeneous_matrix }
            OptimaSE3PoseType::UnitQuaternionAndTranslation => { &self.unit_quaternion_and_translation }
            OptimaSE3PoseType::RotationMatrixAndTranslation => { &self.rotation_matrix_and_translation }
            OptimaSE3PoseType::EulerAnglesAndTranslation => { &self.euler_angles_and_translation }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct OptimaSE3PosePy {
    pose: OptimaSE3Pose
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl OptimaSE3PosePy {
    #[staticmethod]
    pub fn new_implicit_dual_quaternion_from_euler_angles_and_translation_py(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_implicit_dual_quaternion_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    #[staticmethod]
    pub fn new_homogeneous_matrix_from_euler_angles_and_translation_py(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_homogeneous_matrix_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    #[staticmethod]
    pub fn new_unit_quaternion_and_translation_from_euler_angles_and_translation_py(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_unit_quaternion_and_translation_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    #[staticmethod]
    pub fn new_rotation_matrix_and_translation_from_euler_angles_and_translation_py(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_rotation_matrix_and_translation_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    #[staticmethod]
    pub fn new_euler_angles_and_translation_py(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_euler_angles_and_translation(ImplicitDualQuaternion::new_from_euler_angles(rx, ry, rz, x, y, z))
        }
    }
    #[staticmethod]
    pub fn new_implicit_dual_quaternion_py(q: [f64; 4], t: [f64; 3]) -> Self {
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2]));
        let translation = Vector3::new(t[0], t[1], t[2]);
        return Self {
            pose: OptimaSE3Pose::new_implicit_dual_quaternion(ImplicitDualQuaternion::new(rotation, translation))
        }
    }
    #[staticmethod]
    pub fn new_homogeneous_matrix_py(m: [[f64; 4]; 4]) -> Self {
        let mat = Matrix4::new(
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3]);

        let h = HomogeneousMatrix::new(mat);
        return Self {
            pose: OptimaSE3Pose::new_homogeneous_matrix(h)
        }
    }
    #[staticmethod]
    pub fn new_unit_quaternion_and_translation_py(q: [f64; 4], t: [f64; 3]) -> Self {
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2]));
        let translation = Vector3::new(t[0], t[1], t[2]);
        return Self {
            pose: OptimaSE3Pose::new_unit_quaternion_and_translation(rotation, translation)
        }
    }
    #[staticmethod]
    pub fn new_rotation_matrix_and_translation_py(r: [[f64; 3]; 3], t: [f64; 3]) -> Self {
        let rotation = Rotation3::from_matrix(&Matrix3::new(
            r[0][0], r[0][1], r[0][2],
            r[1][0], r[1][1], r[1][2],
            r[2][0], r[2][1], r[2][2]));
        let translation = Vector3::new(t[0], t[1], t[2]);
        return Self {
            pose: OptimaSE3Pose::new_rotation_matrix_and_translation(rotation, translation)
        }
    }

    pub fn get_euler_angles_and_translation(&self) -> (Vec<f64>, Vec<f64>) {
        let euler_angles_and_translation = self.pose.to_euler_angles_and_translation();
        let e = euler_angles_and_translation.0;
        let t = euler_angles_and_translation.1;
        return (vec![e[0], e[1], e[2]], vec![t[0], t[1], t[2]])
    }

    /// \[i, j, k, w\]
    pub fn get_unit_quaternion_and_translation(&self) -> (Vec<f64>, Vec<f64>) {
        let unit_quat_and_translation_pose = self.pose.convert(&OptimaSE3PoseType::UnitQuaternionAndTranslation);
        let unit_quat_and_translation = unit_quat_and_translation_pose.unwrap_rotation_and_translation().expect("error");
        let q = unit_quat_and_translation.rotation().unwrap_unit_quaternion().expect("error");
        let t = unit_quat_and_translation.translation();
        return (vec![q.i, q.j, q.k, q.w], vec![t[0], t[1], t[2]])
    }

    pub fn get_rotation_matrix_and_translation(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let rot_mat_and_translation_pose = self.pose.convert(&OptimaSE3PoseType::RotationMatrixAndTranslation);
        let rot_mat_and_translation = rot_mat_and_translation_pose.unwrap_rotation_and_translation().expect("error");
        let r = rot_mat_and_translation.rotation().unwrap_rotation_matrix().expect("error");
        let t = rot_mat_and_translation.translation();
        let mat = vec![
            vec![r[(0,0)], r[(0,1)], r[(0,2)]],
            vec![r[(1,0)], r[(1,1)], r[(1,2)]],
            vec![r[(2,0)], r[(2,1)], r[(2,2)]]
        ];
        return (mat, vec![t[0], t[1], t[2]])
    }

    pub fn print_summary_py(&self) {
        println!("{:?}", self);
    }
}
impl OptimaSE3PosePy {
    pub fn pose(&self) -> &OptimaSE3Pose { &self.pose }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct OptimaSE3PoseWASM {
    pose: OptimaSE3Pose
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl OptimaSE3PoseWASM {
    pub fn new_implicit_dual_quaternion_from_euler_angles_wasm(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_implicit_dual_quaternion_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    pub fn new_homogeneous_matrix_from_euler_angles_wasm(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_homogeneous_matrix_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    pub fn new_unit_quaternion_and_translation_from_euler_angles_wasm(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_unit_quaternion_and_translation_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    pub fn new_rotation_matrix_and_translation_from_euler_angles_wasm(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_rotation_matrix_and_translation_from_euler_angles(rx, ry, rz, x, y, z)
        }
    }
    pub fn new_euler_angles_and_translation_wasm(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            pose: OptimaSE3Pose::new_euler_angles_and_translation(ImplicitDualQuaternion::new_from_euler_angles(rx, ry, rz, x, y, z))
        }
    }
    pub fn new_implicit_dual_quaternion_wasm(q: Vec<f64>, t: Vec<f64>) -> Self {
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2]));
        let translation = Vector3::new(t[0], t[1], t[2]);
        return Self {
            pose: OptimaSE3Pose::new_implicit_dual_quaternion(ImplicitDualQuaternion::new(rotation, translation))
        }
    }
    pub fn new_unit_quaternion_and_translation_wasm(q: Vec<f64>, t: Vec<f64>) -> Self {
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2]));
        let translation = Vector3::new(t[0], t[1], t[2]);
        return Self {
            pose: OptimaSE3Pose::new_unit_quaternion_and_translation(rotation, translation)
        }
    }
    pub fn new_homogeneous_matrix_wasm(input_mat: JsValue) -> Self {
        let mat: JsMatrix = input_mat.into_serde().unwrap();
        let m = mat.matrix();
        let mat = Matrix4::new(
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3]);

        let h = HomogeneousMatrix::new(mat);
        return Self {
            pose: OptimaSE3Pose::new_homogeneous_matrix(h)
        }
    }
    pub fn new_rotation_matrix_and_translation_wasm(r: JsValue, t: Vec<f64>) -> Self {
        let mat: JsMatrix = r.into_serde().unwrap();
        let m = mat.matrix();
        let rotation = Rotation3::from_matrix(&Matrix3::new(
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2]));
        let translation = Vector3::new(t[0], t[1], t[2]);
        return Self {
            pose: OptimaSE3Pose::new_rotation_matrix_and_translation(rotation, translation)
        }
    }

    pub fn serialized_pose(&self) -> JsValue {
        JsValue::from_serde(&self).unwrap()
    }
    pub fn print_summary_wasm(&self) {
        optima_print(&format!("{:?}", self), PrintMode::Println, PrintColor::None, true);
    }
}
impl OptimaSE3PoseWASM {
    pub fn pose(&self) -> &OptimaSE3Pose { &self.pose }
}
