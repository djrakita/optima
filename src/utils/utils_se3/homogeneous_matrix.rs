use nalgebra::{Matrix3, Matrix4, Rotation, Rotation3, UnitQuaternion, Vector3, Vector4};
use serde::{Serialize, Deserialize};
use crate::utils::utils_se3::implicit_dual_quaternion::ImplicitDualQuaternion;
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationType};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
use crate::utils::utils_se3::rotation_and_translation::RotationAndTranslation;

/// A representation for an SE(3) transform composed of a 4x4 homogeneous transformation matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HomogeneousMatrix {
    matrix: Matrix4<f64>
}
impl HomogeneousMatrix {
    pub fn new(matrix: Matrix4<f64>) -> Self {
        Self {
            matrix
        }
    }
    pub fn new_from_euler_angles(rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64) -> Self {
        let rotation = OptimaRotation::new_rotation_matrix_from_euler_angles(rx, ry, rz);
        let translation = Vector3::new(x, y, z);
        let matrix = Self::rotation_and_translation_to_homogeneous_matrix(&rotation, &translation);
        return Self::new(matrix);
    }
    /// Returns the rotation component of the homogeneous matrix.
    pub fn rotation(&self) -> Rotation3<f64> {
        let mut mat3 = Matrix3::zeros();

        mat3[(0,0)] = self.matrix[(0,0)];
        mat3[(0,1)] = self.matrix[(0,1)];
        mat3[(0,2)] = self.matrix[(0,2)];

        mat3[(1,0)] = self.matrix[(1,0)];
        mat3[(1,1)] = self.matrix[(1,1)];
        mat3[(1,2)] = self.matrix[(1,2)];

        mat3[(2,0)] = self.matrix[(2,0)];
        mat3[(2,1)] = self.matrix[(2,1)];
        mat3[(2,2)] = self.matrix[(2,2)];

        return Rotation3::from_matrix(&mat3);
    }
    /// Returns the translation component of the homogeneous matrix.
    pub fn translation(&self) -> Vector3<f64> {
        let mut out_vec = Vector3::new(self.matrix[(0,3)], self.matrix[(1,3)], self.matrix[(2,3)]);
        return out_vec;
    }
    /// multiplication
    pub fn multiply(&self, other: &HomogeneousMatrix) -> HomogeneousMatrix {
        let matrix = self.matrix * &other.matrix;
        return Self::new(matrix);
    }
    /// multiplication by a point
    pub fn multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        let four_point = Vector4::new(point[0], point[1], point[2], 1.0);
        let result_point = self.matrix * &four_point;
        return Vector3::new(result_point[0], result_point[1], result_point[2]);
    }
    /// Inverse multiplies by the given point.  inverse multiplication is useful for placing the
    /// given point in the transform's local coordinate system.
    pub fn inverse_multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return self.inverse().multiply_by_point(&point);
    }
    /// The inverse transform such that T * T^-1 = I.
    pub fn inverse(&self) -> Self {
        let mut matrix = Matrix4::zeros();
        let rot_mat = self.rotation();
        let rot_mat_transpose = rot_mat.transpose();
        let translation = self.translation();
        let new_translation = rot_mat_transpose * &translation;

        matrix[(0,0)] = rot_mat_transpose[(0,0)];
        matrix[(0,1)] = rot_mat_transpose[(0,1)];
        matrix[(0,2)] = rot_mat_transpose[(0,2)];

        matrix[(1,0)] = rot_mat_transpose[(1,0)];
        matrix[(1,1)] = rot_mat_transpose[(1,1)];
        matrix[(1,2)] = rot_mat_transpose[(1,2)];

        matrix[(2,0)] = rot_mat_transpose[(2,0)];
        matrix[(2,1)] = rot_mat_transpose[(2,1)];
        matrix[(2,2)] = rot_mat_transpose[(2,2)];

        matrix[(0,3)] = -new_translation[0];
        matrix[(1,3)] = -new_translation[1];
        matrix[(2,3)] = -new_translation[2];

        matrix[(3,3)] = 1.0;

        return Self::new(matrix);
    }
    /// The displacement transform such that T_self * T_disp = T_other.
    pub fn displacement(&self, other: &HomogeneousMatrix) -> HomogeneousMatrix {
        return self.inverse().multiply(&other);
    }
    /// Provides an approximate distance between two homogeneous matrices.  This is not an
    /// official distance metric, but should still work in some optimization procedures.
    pub fn approximate_distance(&self, other: &HomogeneousMatrix) -> f64 {
        let angle_between = self.rotation().angle_to(&other.rotation());
        let translation_between = (self.translation() - other.translation()).norm();
        return angle_between + translation_between;
    }
    /// Convenience function for mapping rotation and translation components to a 4x4 matrix.
    pub fn rotation_and_translation_to_homogeneous_matrix(rotation: &OptimaRotation, translation: &Vector3<f64>) -> Matrix4<f64> {
        let rot_mat = rotation.convert(&OptimaRotationType::RotationMatrix)
            .unwrap_rotation_matrix()
            .expect("error")
            .clone();
        let mut out_mat = Matrix4::zeros();
        out_mat[(0,0)] = rot_mat[(0,0)];
        out_mat[(0,1)] = rot_mat[(0,1)];
        out_mat[(0,2)] = rot_mat[(0,2)];

        out_mat[(1,0)] = rot_mat[(1,0)];
        out_mat[(1,1)] = rot_mat[(1,1)];
        out_mat[(1,2)] = rot_mat[(1,2)];

        out_mat[(2,0)] = rot_mat[(2,0)];
        out_mat[(2,1)] = rot_mat[(2,1)];
        out_mat[(2,2)] = rot_mat[(2,2)];

        out_mat[(0,3)] = translation[0];
        out_mat[(1,3)] = translation[1];
        out_mat[(2,3)] = translation[2];

        out_mat[(3,3)] = 1.0;

        return out_mat;
    }
    /// Convenience function for mapping a 4x4 matrix to rotation and translation components.
    pub fn homogeneous_matrix_to_rotation_and_translation(mat: &Matrix4<f64>) -> (OptimaRotation, Vector3<f64>) {
        let mut mat3 = Matrix3::zeros();

        mat3[(0,0)] = mat[(0,0)];
        mat3[(0,1)] = mat[(0,1)];
        mat3[(0,2)] = mat[(0,2)];

        mat3[(1,0)] = mat[(1,0)];
        mat3[(1,1)] = mat[(1,1)];
        mat3[(1,2)] = mat[(1,2)];

        mat3[(2,0)] = mat[(2,0)];
        mat3[(2,1)] = mat[(2,1)];
        mat3[(2,2)] = mat[(2,2)];

        let mut rot3 = Rotation3::from_matrix(&mat3);
        let mut quat = UnitQuaternion::from_rotation_matrix(&rot3);
        let rotation = OptimaRotation::new_unit_quaternion(quat);
        let translation = Vector3::new(mat[(0,3)], mat[(1,3)], mat[(2,3)]);
        return (rotation, translation);
    }
    /// Converts the SE(3) pose to other supported pose types.
    pub fn convert(&self, target_type: &OptimaSE3PoseType) -> OptimaSE3Pose {
        return match target_type {
            OptimaSE3PoseType::ImplicitDualQuaternion => {
                let rotation = UnitQuaternion::from_rotation_matrix(&self.rotation());
                let translation = self.translation();
                let data = ImplicitDualQuaternion::new(rotation, translation);
                OptimaSE3Pose::new_implicit_dual_quaternion(data)
            }
            OptimaSE3PoseType::HomogeneousMatrix => {
                OptimaSE3Pose::new_homogeneous_matrix(self.clone())
            }
            OptimaSE3PoseType::RotationAndTranslation => {
                let rotation = OptimaRotation::new_unit_quaternion(UnitQuaternion::from_rotation_matrix(&self.rotation()));
                let translation = self.translation();
                let data = RotationAndTranslation::new(rotation, translation);
                OptimaSE3Pose::new_rotation_and_translation(data)
            }
        }
    }
}