use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector6};
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
    /// Creates new implicit dual quaternion by exponentiating the natural logarithm vector (the
    /// vector returned by self.ln()
    pub fn new_from_exp(ln_vec: &Vector6<f64>) -> Self {
        let w = Vector3::new(ln_vec[0], ln_vec[1], ln_vec[2]);
        let v = Vector3::new(ln_vec[3], ln_vec[4], ln_vec[5]);

        let phi = w.norm();
        let s = phi.sin();
        let c = phi.cos();
        let gamma = w.dot(&v);

        let mut mu_r = 0.0;
        let mut mu_d = 0.0;

        if phi < 0.00000001 {
            mu_r = 1.0 - phi.powi(2)/6.0 + phi.powi(4) / 120.0;
            mu_d = 4.0/3.0 - 4.0*phi.powi(2)/15.0 + 8.0*phi.powi(4)/315.0;
        } else {
            mu_r = s / phi;
            mu_d = (2.0 - c*(2.0*mu_r)) / phi.powi(2);
        }

        let h_v: Vector3<f64> = mu_r * w;
        let mut quat_ = Quaternion::new(c, h_v[0], h_v[1], h_v[2]);
        let mut rotation = UnitQuaternion::from_quaternion(quat_);

        let mut translation = 2.0 * mu_r * (&h_v.cross(&v)) + c*(2.0*mu_r)*v + mu_d*gamma*w;

        return ImplicitDualQuaternion::new(rotation, translation);
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
    /// Implicit dual quaternion multiplication
    pub fn multiply(&self, other: &ImplicitDualQuaternion) -> ImplicitDualQuaternion {
        let mut out_rot = self.rotation * &other.rotation;
        let mut out_translation = self.rotation * &other.translation + &self.translation;
        return ImplicitDualQuaternion::new(out_rot, out_translation);
    }
    /// Returns same result as multiply, but checks if any component of the implicit dual quaternion
    /// is identity first for efficiency.
    pub fn multiply_shortcircuit(&self, other: &ImplicitDualQuaternion) -> ImplicitDualQuaternion {
        if self.is_identity { return other.clone(); }
        if other.is_identity { return self.clone(); }

        let mut out_rot = self.rotation.clone();
        if self.rot_is_identity { out_rot = other.rotation().clone(); }
        else if other.rot_is_identity {
            // do nothing
        }
        else {
            out_rot *= &other.rotation;
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
    /// Transforms the given point by the implicit dual quaternion
    pub fn multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return self.rotation * point + self.translation;
    }
    /// Transforms the given point by the implicit dual quaternion.  Tries to make the computation
    /// more efficient by checking if any component of the transform is an identity.
    pub fn multiply_by_point_shortcircuit(&self, point: &Vector3<f64>) -> Vector3<f64> {
        if self.is_identity { return point.clone(); }
        if self.rot_is_identity { return point + self.translation; }
        if self.translation_is_zeros { return self.rotation * point; }
        return self.rotation * point + self.translation;
    }
    /// Inverse multiplies by the given point.  inverse multiplication is useful for placing the
    /// given point in the transform's local coordinate system.
    pub fn inverse_multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return self.rotation.inverse() * (point - &self.translation);
    }
    /// Inverse multiplies by the given point.  inverse multiplication is useful for placing the
    /// given point in the transform's local coordinate system.  Tries to do this computation more
    /// efficiently by checking if any component of the transform is an identity.
    pub fn inverse_multiply_by_point_shortcircuit(&self, point: &Vector3<f64>) -> Vector3<f64> {
        if self.is_identity { return point.clone(); }
        if self.rot_is_identity { return point + self.translation; }
        if self.translation_is_zeros { return self.rotation.inverse() * point }
        return self.rotation.inverse() * (point - &self.translation);
    }
    /// The inverse transform such that T * T^-1 = I.
    pub fn inverse(&self) -> ImplicitDualQuaternion {
        let mut new_quat = self.rotation.inverse();
        let mut new_translation = &new_quat * -self.translation.clone();
        return ImplicitDualQuaternion::new(new_quat, new_translation);
    }
    /// The displacement transform such that T_self * T_disp = T_other.
    pub fn displacement(&self, other: &ImplicitDualQuaternion) -> ImplicitDualQuaternion {
        return self.inverse().multiply_shortcircuit(&other);
    }
    /// The natural logarithm of the implicit dual quaternion.  For details on this transform, see
    /// the IJRR paper Efficient Forward, Differential, and Inverse Kinematics using Dual Quaternions
    /// by Neil Dantam
    pub fn ln(&self) -> Vector6<f64> {
        let h_v = Vector3::new( self.rotation.i,self.rotation.j, self.rotation.k  );
        let s: f64 = h_v.norm();
        let c = self.rotation.w;
        let phi = s.atan2(c);
        let mut a = 0.0;
        if s > 0.0 { a = phi / s; }
        let rot_vec_diff = a * &h_v;

        let mut mu_r = 0.0;
        let mut mu_d = 0.0;

        if s < 0.00000000000001 {
            mu_r = 1.0 - (phi.powi(2) / 3.0) - (phi.powi(4) / 45.0);
        } else {
            mu_r = (c * phi) / s;
        }

        if phi < 0.000000000001 {
            mu_d = (1.0 / 3.0) + (phi.powi(2) / 45.0) + ((2.0 * phi.powi(4)) / 945.0);
        } else {
            mu_d = (1.0 - mu_r) / (phi.powi(2));
        }

        let tmp = (&self.translation / 2.0);
        let mut translation_diff = mu_d * ( &tmp.dot(&rot_vec_diff) ) * &rot_vec_diff + mu_r * &tmp + &tmp.cross(&rot_vec_diff);

        let mut out_vec = Vector6::new(rot_vec_diff[0], rot_vec_diff[1], rot_vec_diff[2], translation_diff[0], translation_diff[1], translation_diff[2]);

        out_vec
    }
    /// The norm of the natural logarithm
    pub fn ln_l2_magnitude(&self) -> f64 {
        return self.ln().norm();
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// Converts the SE(3) pose to other supported pose types.
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