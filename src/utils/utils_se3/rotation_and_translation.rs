use f64ad_core::f64ad::f64ad;
use nalgebra::{Isometry3, Unit, Vector3};
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::homogeneous_matrix::{HomogeneousMatrix, HomogeneousMatrixAD};
use crate::utils::utils_se3::implicit_dual_quaternion::{ImplicitDualQuaternion, ImplicitDualQuaternionAD};
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationAD, OptimaRotationType};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseAD, OptimaSE3PoseType};

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
    pub fn new_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64, x: f64, y: f64, z: f64, rotation_type: &OptimaRotationType) -> Self {
        let rotation = match rotation_type {
            OptimaRotationType::RotationMatrix => { OptimaRotation::new_rotation_matrix_from_axis_angle(axis, angle) }
            OptimaRotationType::UnitQuaternion => { OptimaRotation::new_unit_quaternion_from_axis_angle(axis, angle) }
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
    pub fn displacement_separate_rotation_and_translation(&self, other: &RotationAndTranslation) -> (OptimaRotation, Vector3<f64>) {
        let disp_rotation = self.rotation.displacement(other.rotation(), true).expect("error");
        let disp_translation = other.translation - self.translation;
        return (disp_rotation, disp_translation);
    }
    pub fn slerp(&self, other: &RotationAndTranslation, t: f64, conversion_if_necessary: bool) -> Result<RotationAndTranslation, OptimaError> {
        let new_rot = self.rotation.slerp(&other.rotation, t, conversion_if_necessary)?;
        let new_translation = (1.0 - t) * self.translation + t * other.translation;
        return Ok(Self::new(new_rot, new_translation));
    }
    /// Provides an approximate distance between two objects.  This is not an
    /// official distance metric, but should still work in some optimization procedures.
    pub fn approximate_distance(&self, other: &RotationAndTranslation, conversion_if_necessary: bool) -> Result<f64, OptimaError> {
        let angle_between = self.rotation().angle_between(other.rotation(), conversion_if_necessary)?;
        let translation_between = (self.translation() - other.translation()).norm();
        return Ok(angle_between + translation_between);
    }
    /// Returns an euler angle and vector representation of the SE(3) pose.
    pub fn to_euler_angles_and_translation(&self) -> (Vector3<f64>, Vector3<f64>) {
        let rotation = self.rotation();
        let euler_angles = rotation.to_euler_angles();
        return (euler_angles, self.translation.clone());
    }
    /// Returns an axis angle and translation representation of the SE(3) pose.
    pub fn to_axis_angle_and_translation(&self) -> (Vector3<f64>, f64, Vector3<f64>) {
        let axis_angle = match &self.rotation {
            OptimaRotation::RotationMatrix { data, .. } => { data.axis_angle() }
            OptimaRotation::UnitQuaternion { data, .. } => { data.axis_angle() }
        };
        let (axis, angle) = match axis_angle {
            None => { (Vector3::new(0.,0.,0.), 0.0) }
            Some(axis_angle) => { (Vector3::new(axis_angle.0[0], axis_angle.0[1], axis_angle.0[2]), axis_angle.1) }
        };
        return (axis, angle, self.translation().clone());
    }
    /// Outputs an Isometry3 object in the Nalgebra library that corresponds to the SE(3) pose.
    pub fn to_nalgebra_isometry(&self) -> Isometry3<f64> {
        let axis = match &self.rotation {
            OptimaRotation::RotationMatrix { data, .. } => { data.scaled_axis() }
            OptimaRotation::UnitQuaternion { data, .. } => { data.scaled_axis() }
        };
        return Isometry3::new(self.translation.clone(), axis);
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
            OptimaSE3PoseType::UnitQuaternionAndTranslation => {
                return match self.rotation() {
                    OptimaRotation::RotationMatrix { .. } => {
                        let mut out_rt = self.clone();
                        out_rt.convert_rotation_type(&OptimaRotationType::UnitQuaternion);
                        OptimaSE3Pose::new_rotation_and_translation(out_rt)
                    }
                    OptimaRotation::UnitQuaternion { .. } => {
                        OptimaSE3Pose::new_rotation_and_translation(self.clone())
                    }
                }
            }
            OptimaSE3PoseType::RotationMatrixAndTranslation => {
                return match self.rotation() {
                    OptimaRotation::RotationMatrix { .. } => {
                        OptimaSE3Pose::new_rotation_and_translation(self.clone())
                    }
                    OptimaRotation::UnitQuaternion { .. } => {
                        let mut out_rt = self.clone();
                        out_rt.convert_rotation_type(&OptimaRotationType::RotationMatrix);
                        OptimaSE3Pose::new_rotation_and_translation(out_rt)
                    }
                }
            }
            OptimaSE3PoseType::EulerAnglesAndTranslation => {
                let euler_angles_and_rotation = self.to_euler_angles_and_translation();
                let e = &euler_angles_and_rotation.0;
                let t = &euler_angles_and_rotation.1;
                return OptimaSE3Pose::EulerAnglesAndTranslation {
                    euler_angles: e.clone(),
                    translation: t.clone(),
                    phantom_data: self.convert(&OptimaSE3PoseType::ImplicitDualQuaternion).unwrap_implicit_dual_quaternion().expect("error").clone(),
                    pose_type: OptimaSE3PoseType::EulerAnglesAndTranslation
                }
            }
        }
    }
    /// Converts to vector representation
    ///
    /// If quaternion and translation: [[q_i, q_j, q_k, q_w], [x, y, z]]
    ///
    /// If rotation matrix and translation: [[r_00, r_01, r_02], [r_10, r_11, r_12], [r_20, r_21, r_22], [x, y, z]]
    pub fn to_vec_representation(&self) -> Vec<Vec<f64>> {
        let rotation = self.rotation();
        let mut out_vec = rotation.to_vec_representation();
        let t = self.translation();
        out_vec.push(vec![t[0], t[1], t[2]]);
        out_vec
    }
}

/// A representation for an SE(3) transform composed of just rotation and translation components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotationAndTranslationAD {
    rotation: OptimaRotationAD,
    translation: Vector3<f64ad>
}
impl RotationAndTranslationAD {
    pub fn new(rotation: OptimaRotationAD, translation: Vector3<f64ad>) -> Self {
        Self {
            rotation,
            translation
        }
    }
    pub fn new_from_euler_angles(rx: f64ad, ry: f64ad, rz: f64ad, x: f64ad, y: f64ad, z: f64ad, rotation_type: &OptimaRotationType) -> Self {
        let rotation = match rotation_type {
            OptimaRotationType::RotationMatrix => { OptimaRotationAD::new_rotation_matrix_from_euler_angles(rx, ry, rz) }
            OptimaRotationType::UnitQuaternion => { OptimaRotationAD::new_unit_quaternion_from_euler_angles(rx, ry, rz) }
        };

        return Self::new(rotation, Vector3::new(x, y, z));
    }
    pub fn new_from_axis_angle(axis: &Unit<Vector3<f64ad>>, angle: f64ad, x: f64ad, y: f64ad, z: f64ad, rotation_type: &OptimaRotationType) -> Self {
        let rotation = match rotation_type {
            OptimaRotationType::RotationMatrix => { OptimaRotationAD::new_rotation_matrix_from_axis_angle(axis, angle) }
            OptimaRotationType::UnitQuaternion => { OptimaRotationAD::new_unit_quaternion_from_axis_angle(axis, angle) }
        };

        return Self::new(rotation, Vector3::new(x, y, z));
    }
    /// Returns the rotation component of the object.
    pub fn rotation(&self) -> &OptimaRotationAD {
        &self.rotation
    }
    /// Returns the translation component of the object.
    pub fn translation(&self) -> &Vector3<f64ad> {
        &self.translation
    }
    /// Multiplication
    pub fn multiply(&self, other: &RotationAndTranslationAD, conversion_if_necessary: bool) -> Result<RotationAndTranslationAD, OptimaError> {
        let rotation = self.rotation.multiply(&other.rotation, conversion_if_necessary)?;
        let translation = self.rotation.multiply_by_point(&other.translation) + &self.translation;
        return Ok(Self::new(rotation, translation));
    }
    /// Multiplication by a point.
    pub fn multiply_by_point(&self, point: &Vector3<f64ad>) -> Vector3<f64ad> {
        return self.rotation().multiply_by_point(point) + self.translation();
    }
    /// Inverse multiplies by the given point.  inverse multiplication is useful for placing the
    /// given point in the transform's local coordinate system.
    pub fn inverse_multiply_by_point(&self, point: &Vector3<f64ad>) -> Vector3<f64ad> {
        return self.rotation.inverse().multiply_by_point(&(point - &self.translation));
    }
    /// The inverse transform such that T * T^-1 = I.
    pub fn inverse(&self) -> RotationAndTranslationAD {
        let rotation = self.rotation.inverse();
        let translation = rotation.multiply_by_point(&-self.translation);
        return Self::new(rotation, translation);
    }
    /// The displacement transform such that T_self * T_disp = T_other.
    pub fn displacement(&self, other: &RotationAndTranslationAD, conversion_if_necessary: bool) -> Result<RotationAndTranslationAD, OptimaError> {
        return self.inverse().multiply(&other, conversion_if_necessary);
    }
    pub fn displacement_separate_rotation_and_translation(&self, other: &RotationAndTranslationAD) -> (OptimaRotationAD, Vector3<f64ad>) {
        let disp_rotation = self.rotation.displacement(other.rotation(), true).expect("error");
        let disp_translation = other.translation - self.translation;
        return (disp_rotation, disp_translation);
    }
    pub fn slerp(&self, other: &RotationAndTranslationAD, t: f64ad, conversion_if_necessary: bool) -> Result<RotationAndTranslationAD, OptimaError> {
        let new_rot = self.rotation.slerp(&other.rotation, t, conversion_if_necessary)?;
        let new_translation = (1.0 - t) * self.translation + t * other.translation;
        return Ok(Self::new(new_rot, new_translation));
    }
    /// Provides an approximate distance between two objects.  This is not an
    /// official distance metric, but should still work in some optimization procedures.
    pub fn approximate_distance(&self, other: &RotationAndTranslationAD, conversion_if_necessary: bool) -> Result<f64ad, OptimaError> {
        let angle_between = self.rotation().angle_between(other.rotation(), conversion_if_necessary)?;
        let translation_between = (self.translation() - other.translation()).norm();
        return Ok(angle_between + translation_between);
    }
    /// Returns an euler angle and vector representation of the SE(3) pose.
    pub fn to_euler_angles_and_translation(&self) -> (Vector3<f64ad>, Vector3<f64ad>) {
        let rotation = self.rotation();
        let euler_angles = rotation.to_euler_angles();
        return (euler_angles, self.translation.clone());
    }
    /// Returns an axis angle and translation representation of the SE(3) pose.
    pub fn to_axis_angle_and_translation(&self) -> (Vector3<f64ad>, f64ad, Vector3<f64ad>) {
        let axis_angle = match &self.rotation {
            OptimaRotationAD::RotationMatrix { data, .. } => { data.axis_angle() }
            OptimaRotationAD::UnitQuaternion { data, .. } => { data.axis_angle() }
        };
        let (axis, angle) = match axis_angle {
            None => { (Vector3::new(f64ad::f64(0.),f64ad::f64(0.),f64ad::f64(0.)), f64ad::f64(0.)) }
            Some(axis_angle) => { (Vector3::new(axis_angle.0[0], axis_angle.0[1], axis_angle.0[2]), axis_angle.1) }
        };
        return (axis, angle, self.translation().clone());
    }
    /// Outputs an Isometry3 object in the Nalgebra library that corresponds to the SE(3) pose.
    pub fn to_nalgebra_isometry(&self) -> Isometry3<f64ad> {
        let axis = match &self.rotation {
            OptimaRotationAD::RotationMatrix { data, .. } => { data.scaled_axis() }
            OptimaRotationAD::UnitQuaternion { data, .. } => { data.scaled_axis() }
        };
        return Isometry3::new(self.translation.clone(), axis);
    }
    /// Converts the internal rotation type of the object to another provided rotation type.
    pub fn convert_rotation_type(&mut self, target_type: &OptimaRotationType) {
        self.rotation = self.rotation.convert(target_type);
    }

    /// Converts the SE(3) pose to other supported pose types.
    pub fn convert(&self, target_type: &OptimaSE3PoseType) -> OptimaSE3PoseAD {
        return match target_type {
            OptimaSE3PoseType::ImplicitDualQuaternion => {
                let unit_quaternion = self.rotation().convert(&OptimaRotationType::UnitQuaternion).unwrap_unit_quaternion().expect("error").clone();
                let data = ImplicitDualQuaternionAD::new(unit_quaternion, self.translation().clone());
                OptimaSE3PoseAD::new_implicit_dual_quaternion(data)
            }
            OptimaSE3PoseType::HomogeneousMatrix => {
                let matrix = HomogeneousMatrixAD::rotation_and_translation_to_homogeneous_matrix(self.rotation(), self.translation());
                let data = HomogeneousMatrixAD::new(matrix);
                OptimaSE3PoseAD::new_homogeneous_matrix(data)
            }
            OptimaSE3PoseType::UnitQuaternionAndTranslation => {
                return match self.rotation() {
                    OptimaRotationAD::RotationMatrix { .. } => {
                        let mut out_rt = self.clone();
                        out_rt.convert_rotation_type(&OptimaRotationType::UnitQuaternion);
                        OptimaSE3PoseAD::new_rotation_and_translation(out_rt)
                    }
                    OptimaRotationAD::UnitQuaternion { .. } => {
                        OptimaSE3PoseAD::new_rotation_and_translation(self.clone())
                    }
                }
            }
            OptimaSE3PoseType::RotationMatrixAndTranslation => {
                return match self.rotation() {
                    OptimaRotationAD::RotationMatrix { .. } => {
                        OptimaSE3PoseAD::new_rotation_and_translation(self.clone())
                    }
                    OptimaRotationAD::UnitQuaternion { .. } => {
                        let mut out_rt = self.clone();
                        out_rt.convert_rotation_type(&OptimaRotationType::RotationMatrix);
                        OptimaSE3PoseAD::new_rotation_and_translation(out_rt)
                    }
                }
            }
            OptimaSE3PoseType::EulerAnglesAndTranslation => {
                let euler_angles_and_rotation = self.to_euler_angles_and_translation();
                let e = &euler_angles_and_rotation.0;
                let t = &euler_angles_and_rotation.1;
                return OptimaSE3PoseAD::EulerAnglesAndTranslation {
                    euler_angles: e.clone(),
                    translation: t.clone(),
                    phantom_data: self.convert(&OptimaSE3PoseType::ImplicitDualQuaternion).unwrap_implicit_dual_quaternion().expect("error").clone(),
                    pose_type: OptimaSE3PoseType::EulerAnglesAndTranslation
                }
            }
        }
    }

    /// Converts to vector representation
    ///
    /// If quaternion and translation: [[q_i, q_j, q_k, q_w], [x, y, z]]
    ///
    /// If rotation matrix and translation: [[r_00, r_01, r_02], [r_10, r_11, r_12], [r_20, r_21, r_22], [x, y, z]]
    pub fn to_vec_representation(&self) -> Vec<Vec<f64ad>> {
        let rotation = self.rotation();
        let mut out_vec = rotation.to_vec_representation();
        let t = self.translation();
        out_vec.push(vec![t[0], t[1], t[2]]);
        out_vec
    }
}