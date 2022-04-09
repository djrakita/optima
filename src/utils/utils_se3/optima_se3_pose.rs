use nalgebra::{Rotation3, Unit, UnitQuaternion, Vector3};
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::homogeneous_matrix::HomogeneousMatrix;
use crate::utils::utils_se3::implicit_dual_quaternion::ImplicitDualQuaternion;
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationType};
use crate::utils::utils_se3::rotation_and_translation::RotationAndTranslation;

/// An enum used to represent a rotation or orientation.  The enum affords easy conversion between
/// rotation types and functions over singular or pairs of rotations.
/// This is the main object that should be used for representing an SE(3) pose due to its
/// flexibility and interoperability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimaSE3Pose {
    ImplicitDualQuaternion { data: ImplicitDualQuaternion, pose_type: OptimaSE3PoseType },
    HomogeneousMatrix { data: HomogeneousMatrix, pose_type: OptimaSE3PoseType },
    RotationAndTranslation { data: RotationAndTranslation, pose_type: OptimaSE3PoseType }
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
        }
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

    /// Converts the SE(3) pose to other supported pose types.
    pub fn convert(&self, target_type: &OptimaSE3PoseType) -> OptimaSE3Pose {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.convert(target_type) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.convert(target_type) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.convert(target_type) }
        }
    }
    /// The inverse transform such that T * T^-1 = I.
    pub fn inverse(&self) -> OptimaSE3Pose {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { Self::new_implicit_dual_quaternion(data.inverse()) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { Self::new_homogeneous_matrix(data.inverse()) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { Self::new_rotation_and_translation(data.inverse()) }
        }
    }
    /// Transform multiplication.
    pub fn multiply(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<OptimaSE3Pose, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_pose_type());
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
        }
    }
    /// Multiplication by a point.
    pub fn multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.inverse_multiply_by_point_shortcircuit(point) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.multiply_by_point(point) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.multiply_by_point(point) }
        }
    }
    /// Inverse multiplies by the given point.  inverse multiplication is useful for placing the
    /// given point in the transform's local coordinate system.
    pub fn inverse_multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.inverse_multiply_by_point_shortcircuit(point) }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.inverse_multiply_by_point(point) }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.inverse_multiply_by_point(point) }
        }
    }
    /// The displacement transform such that T_self * T_disp = T_other.
    pub fn displacement(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<OptimaSE3Pose, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_pose_type());
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
        }
    }
    /// Distance function between transforms.  This may be approximate.
    /// In the case of the implicit dual quaternion, this is smooth, differentiable, and exact (one
    /// of the benefits of that representation).
    pub fn distance_function(&self, other: &OptimaSE3Pose, conversion_if_necessary: bool) -> Result<f64, OptimaError> {
        let c = Self::are_types_compatible(self, other);
        if !c {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_pose_type());
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
                data.to_euler_angles_and_vector()
            }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => {
                data.to_euler_angles_and_vector()
            }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => {
                data.to_euler_angles_and_vector()
            }
        }
    }
    /// Converts to vector representation.
    pub fn to_vec_representation(&self) -> Vec<Vec<f64>> {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data, .. } => { data.to_vec_representation() }
            OptimaSE3Pose::HomogeneousMatrix { data, .. } => { data.to_vec_representation() }
            OptimaSE3Pose::RotationAndTranslation { data, .. } => { data.to_vec_representation() }
        }
    }

    fn are_types_compatible(a: &OptimaSE3Pose, b: &OptimaSE3Pose) -> bool {
        return if a.get_pose_type() == b.get_pose_type() { true } else { false }
    }
    fn get_pose_type(&self) -> &OptimaSE3PoseType {
        return match self {
            OptimaSE3Pose::ImplicitDualQuaternion { data: _, pose_type } => { pose_type }
            OptimaSE3Pose::HomogeneousMatrix { data: _, pose_type } => { pose_type }
            OptimaSE3Pose::RotationAndTranslation { data: _, pose_type } => { pose_type }
        }
    }
}

/// An Enum that encodes a pose type.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum OptimaSE3PoseType {
    ImplicitDualQuaternion,
    HomogeneousMatrix,
    UnitQuaternionAndTranslation,
    RotationMatrixAndTranslation
}

/// A container object that holds all OPtimaSE3 types.  This is useful for functions that may need
/// to handle many pose types, so this allows all to be initialized and saved at once to avoid
/// many transform conversions at run-time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimaSE3PoseAll {
    implicit_dual_quaternion: OptimaSE3Pose,
    homogeneous_matrix: OptimaSE3Pose,
    unit_quaternion_and_translation: OptimaSE3Pose,
    rotation_matrix_and_translation: OptimaSE3Pose
}
impl OptimaSE3PoseAll {
    pub fn new(p: &OptimaSE3Pose) -> Self {
        Self {
            implicit_dual_quaternion: p.convert(&OptimaSE3PoseType::ImplicitDualQuaternion),
            homogeneous_matrix: p.convert(&OptimaSE3PoseType::HomogeneousMatrix),
            unit_quaternion_and_translation: p.convert(&OptimaSE3PoseType::UnitQuaternionAndTranslation),
            rotation_matrix_and_translation: p.convert(&OptimaSE3PoseType::RotationMatrixAndTranslation)
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
        }
    }
}