#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use serde::{Serialize, Deserialize};
use nalgebra::{UnitQuaternion, Rotation3, Vector3, Unit, Matrix3};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_traits::ToAndFromRonString;

/// An enum used to represent a rotation or orientation.  The enum affords easy conversion between
/// rotation types and functions over singular or pairs of rotations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimaRotation {
    RotationMatrix{data: Rotation3<f64>, rotation_type: OptimaRotationType },
    UnitQuaternion{data: UnitQuaternion<f64>, rotation_type: OptimaRotationType }
}
impl OptimaRotation {
    pub fn new_rotation_matrix(data: Rotation3<f64>) -> OptimaRotation {
        OptimaRotation::RotationMatrix { data, rotation_type: OptimaRotationType::RotationMatrix }
    }
    pub fn new_unit_quaternion(data: UnitQuaternion<f64>) -> OptimaRotation {
        OptimaRotation::UnitQuaternion { data, rotation_type: OptimaRotationType::UnitQuaternion }
    }
    pub fn new_rotation_matrix_from_euler_angles(rx: f64, ry: f64, rz: f64) -> OptimaRotation {
        let data = Rotation3::from_euler_angles(rx, ry, rz);
        return Self::new_rotation_matrix(data);
    }
    pub fn new_unit_quaternion_from_euler_angles(rx: f64, ry: f64, rz: f64) -> OptimaRotation {
        let q = UnitQuaternion::from_euler_angles(rx, ry, rz);
        return Self::new_unit_quaternion(q);
    }
    pub fn new_rotation_matrix_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64) -> OptimaRotation {
        let data = Rotation3::from_axis_angle(axis, angle);
        return Self::new_rotation_matrix(data);
    }
    pub fn new_unit_quaternion_from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64) -> OptimaRotation {
        let data = UnitQuaternion::from_axis_angle(axis, angle);
        return Self::new_unit_quaternion(data);
    }
    pub fn new_rotation_matrix_from_lookat(lookat_direction: Vector3<f64>, lookat_axis: LookatAxis) -> OptimaRotation {
        let mut mat = Matrix3::identity();

        let lookat_column = match lookat_axis {
            LookatAxis::X => {0}
            LookatAxis::Y => {1}
            LookatAxis::Z => {2}
            LookatAxis::NegX => {0}
            LookatAxis::NegY => {1}
            LookatAxis::NegZ => {2}
        };
        let multiplier = match lookat_axis {
            LookatAxis::X => {1.0}
            LookatAxis::Y => {1.0}
            LookatAxis::Z => {1.0}
            LookatAxis::NegX => {-1.0}
            LookatAxis::NegY => {-1.0}
            LookatAxis::NegZ => {-1.0}
        };
        let column_a = (lookat_column + 1) % 3;
        let column_b = (lookat_column + 2) % 3;

        let l = multiplier * lookat_direction.normalize();
        mat[(0, lookat_column)] = l[0];
        mat[(1, lookat_column)] = l[1];
        mat[(2, lookat_column)] = l[2];

        let mut u = Vector3::new(0.,0.,1.);
        let dot_test = l.dot(&u);
        if dot_test == 1.0 || dot_test == -1.0 {
            u = Vector3::new(0.,1.,0.);
        }

        let c_a = l.cross(&u).normalize();
        mat[(0, column_a)] = c_a[0];
        mat[(1, column_a)] = c_a[1];
        mat[(2, column_a)] = c_a[2];

        let c_b = l.cross(&c_a).normalize();
        mat[(0, column_b)] = c_b[0];
        mat[(1, column_b)] = c_b[1];
        mat[(2, column_b)] = c_b[2];

        let data = Rotation3::from_matrix(&mat);
        return Self::new_rotation_matrix(data);
    }
    pub fn new_quaternion_from_lookat(lookat_direction: Vector3<f64>, lookat_axis: LookatAxis) -> OptimaRotation {
        let r = Self::new_rotation_matrix_from_lookat(lookat_direction, lookat_axis);
        return r.convert(&OptimaRotationType::UnitQuaternion);
    }
    /// Creates new rotation by exponentating the logarithm vector (the vector returned by ln()
    /// function).
    pub fn new_from_exp(ln_vec: &Vector3<f64>, rotation_type: &OptimaRotationType) -> Self {
        return match rotation_type {
            OptimaRotationType::RotationMatrix => {
                let data = Rotation3::new(ln_vec.clone());
                Self::new_rotation_matrix(data)
            }
            OptimaRotationType::UnitQuaternion => {
                let data = UnitQuaternion::new(ln_vec.clone());
                Self::new_unit_quaternion(data)
            }
        }
    }
    /// Converts the rotation to another provided rotation type.
    pub fn convert(&self, target_type: &OptimaRotationType) -> OptimaRotation {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                match target_type {
                    OptimaRotationType::RotationMatrix => {
                        self.clone()
                    }
                    OptimaRotationType::UnitQuaternion => {
                        let data = UnitQuaternion::from_rotation_matrix(data);
                        Self::new_unit_quaternion(data)
                    }
                }
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                match target_type {
                    OptimaRotationType::RotationMatrix => {
                        let data: Rotation3<f64> = data.to_rotation_matrix();
                        Self::new_rotation_matrix(data)
                    }
                    OptimaRotationType::UnitQuaternion => {
                        self.clone()
                    }
                }
            }
        }
    }
    /// Inverse rotation such that R * R^-1 = I
    pub fn inverse(&self) -> OptimaRotation {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                let new_data = data.inverse();
                Self::new_rotation_matrix(new_data)
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                let new_data = data.inverse();
                Self::new_unit_quaternion(new_data)
            }
        }
    }
    /// The angle that is encoded by the given rotation
    pub fn angle(&self) -> f64 {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => { data.angle() }
            OptimaRotation::UnitQuaternion { data, .. } => { data.angle() }
        }
    }
    /// Natural logarithm of the rotation.  This can be thought of as the rotation axis that is
    /// scaled by the length of the angle of rotation.
    pub fn ln(&self) -> Vector3<f64> {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                let scaled_axis = data.scaled_axis();
                scaled_axis
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                let out_vec: Vector3<f64> = data.ln().vector().into();
                out_vec
            }
        }
    }
    /// Rotation multiplication.
    pub fn multiply(&self, other: &OptimaRotation, conversion_if_necessary: bool) -> Result<OptimaRotation, OptimaError> {
        if self.get_rotation_type() != other.get_rotation_type() {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_rotation_type());
                self.multiply(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!()))
            }
        }

        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaRotation::RotationMatrix { data, .. } => {
                        let new_data = data0 * data;
                        Ok(Self::new_rotation_matrix(new_data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!())) }
                }
            }
            OptimaRotation::UnitQuaternion { data, rotation_type: _ } => {
                let data0 = data;
                match other {
                    OptimaRotation::UnitQuaternion { data, .. } => {
                        let new_data = data0 * data;
                        Ok(Self::new_unit_quaternion(new_data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!())) }
                }
            }
        }
    }
    /// Rotation multiplication by a point.
    pub fn multiply_by_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                data * point
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                data * point
            }
        }
    }
    /// Returns true if the rotation is identity.
    pub fn is_identity(&self) -> bool {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                if data.angle() == 0.0 { true } else { false }
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                if data.angle() == 0.0 { true } else { false }
            }
        }
    }
    /// The displacement between two rotations such that R_self * R_displacement = R_other
    pub fn displacement(&self, other: &OptimaRotation, conversion_if_necessary: bool) -> Result<OptimaRotation, OptimaError> {
        if self.get_rotation_type() != other.get_rotation_type() {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_rotation_type());
                self.displacement(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!()))
            }
        }

        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaRotation::RotationMatrix { data, .. } => {
                        let new_data = data0.inverse() * data;
                        Ok(Self::new_rotation_matrix(new_data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!())) }
                }
            }
            OptimaRotation::UnitQuaternion { data, rotation_type: _ } => {
                let data0 = data;
                match other {
                    OptimaRotation::UnitQuaternion { data, .. } => {
                        let new_data = data0.inverse() * data;
                        Ok(Self::new_unit_quaternion(new_data))
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!())) }
                }
            }
        }
    }
    /// The angle between two rotations.
    pub fn angle_between(&self, other: &OptimaRotation, conversion_if_necessary: bool) -> Result<f64, OptimaError> {
        if self.get_rotation_type() != other.get_rotation_type() {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_rotation_type());
                self.angle_between(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!()))
            }
        }

        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaRotation::RotationMatrix { data, .. } => {
                        let angle_between = data0.angle_to(data);
                        Ok(angle_between)
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!())) }
                }
            }
            OptimaRotation::UnitQuaternion { data, rotation_type: _ } => {
                let data0 = data;
                match other {
                    OptimaRotation::UnitQuaternion { data, .. } => {
                        let angle_between = data0.angle_to(data);
                        Ok(angle_between)
                    }
                    _ => { Err(OptimaError::new_generic_error_str("incompatible rotation types in multiply.", file!(), line!())) }
                }
            }
        }
    }
    /// Returns the 3x3 rotation matrix encoded by the rotation object.  Returns error if the
    /// underlying representation is not a RotationMatrix.
    pub fn unwrap_rotation_matrix(&self) -> Result<&Rotation3<f64>, OptimaError> {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                Ok(data)
            }
            OptimaRotation::UnitQuaternion { .. } => {
                Err(OptimaError::new_generic_error_str("tried to unwrap unit quaternion as rotation matrix.", file!(), line!()))
            }
        }
    }
    /// Returns the Unit Quaternion encoded by the rotation object.  Returns error if the
    /// underlying representation is not a UnitQuaternion.
    pub fn unwrap_unit_quaternion(&self) -> Result<&UnitQuaternion<f64>, OptimaError> {
        return match self {
            OptimaRotation::RotationMatrix { .. } => {
                Err(OptimaError::new_generic_error_str("tried to unwrap rotation matrix as unit quaternion.", file!(), line!()))
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                Ok(data)
            }
        }
    }
    /// Returns the euler angle representation of the rotation.
    pub fn to_euler_angles(&self) -> Vector3<f64> {
        let euler_angles = match self {
            OptimaRotation::RotationMatrix { data, .. } => { data.euler_angles() }
            OptimaRotation::UnitQuaternion { data, .. } => { data.euler_angles() }
        };
        let euler_angles_vec = Vector3::new(euler_angles.0, euler_angles.1, euler_angles.2);
        return euler_angles_vec;
    }
    /// To axis angle representation of a rotation.
    pub fn to_axis_angle(&self) -> (Vector3<f64>, f64) {
        let axis_angle = match self {
            OptimaRotation::RotationMatrix { data, .. } => { data.axis_angle() }
            OptimaRotation::UnitQuaternion { data, .. } => { data.axis_angle() }
        };
        match axis_angle {
            None => {
                (Vector3::new(0.,0.,0.), 0.0)
            }
            Some(axis_angle) => {
                (Vector3::new(axis_angle.0[0], axis_angle.0[1], axis_angle.0[2]), axis_angle.1)
            }
        }
    }
    /// Spherical linear interpolation.
    pub fn slerp(&self, other: &OptimaRotation, t: f64, conversion_if_necessary: bool) -> Result<OptimaRotation, OptimaError> {
        if self.get_rotation_type() != other.get_rotation_type() {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_rotation_type());
                self.slerp(&new_operand, t, conversion_if_necessary)
            } else {
                Err(OptimaError::new_generic_error_str("incompatible rotation types in interpolate.", file!(), line!()))
            }
        }

        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                let data0 = data;
                match other {
                    OptimaRotation::RotationMatrix { data, .. } => {
                        Ok(Self::new_rotation_matrix(data0.slerp(data, t)))
                    }
                    OptimaRotation::UnitQuaternion { .. } => {
                        Err(OptimaError::new_generic_error_str("incompatible rotation types in interpolate.", file!(), line!()))
                    }
                }
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                let data0 = data;
                match other {
                    OptimaRotation::RotationMatrix { .. } => {
                        Err(OptimaError::new_generic_error_str("incompatible rotation types in interpolate.", file!(), line!()))
                    }
                    OptimaRotation::UnitQuaternion { data, .. } => {
                        Ok(Self::new_unit_quaternion(data0.slerp(data, t)))
                    }
                }
            }
        }
    }
    fn get_rotation_type(&self) -> &OptimaRotationType {
        return match &self {
            OptimaRotation::RotationMatrix { data: _, rotation_type } => { rotation_type }
            OptimaRotation::UnitQuaternion { data: _, rotation_type } => { rotation_type }
        }
    }
    /// Converts to vector representation.
    ///
    /// If quaternion: [[q_i, q_j, q_k, q_w]]
    ///
    /// If rotation matrix: [[r_00, r_01, r_02], [r_10, r_11, r_12], [r_20, r_21, r_22]]
    pub fn to_vec_representation(&self) -> Vec<Vec<f64>> {
        let mut out_vec = vec![];
        match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                for i in 0..3 {
                    let mut tmp_vec = vec![];
                    for j in 0..3 {
                        tmp_vec.push(data[(i,j)]);
                    }
                    out_vec.push(tmp_vec);
                }
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                out_vec.push(vec![]);
                out_vec[0].push(data.i);
                out_vec[0].push(data.j);
                out_vec[0].push(data.k);
                out_vec[0].push(data.w);
            }
        }
        out_vec
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum OptimaRotationType {
    RotationMatrix,
    UnitQuaternion
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum LookatAxis {
    X,Y,Z,NegX,NegY,NegZ
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct OptimaRotationPy {
    rotation: OptimaRotation
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl OptimaRotationPy {
    #[staticmethod]
    pub fn new_rotation_matrix_from_lookat_py(lookat: Vec<f64>, lookat_axis: &str) -> Self {
        let v = Vector3::new(lookat[0], lookat[1], lookat[2]);
        let rotation = OptimaRotation::new_rotation_matrix_from_lookat(v, LookatAxis::from_ron_string(lookat_axis).expect("error"));
        return Self {
            rotation
        }
    }
    #[staticmethod]
    pub fn new_rotation_matrix_from_euler_angles_py(rx: f64, ry: f64, rz: f64) -> Self {
        let rotation = OptimaRotation::new_rotation_matrix_from_euler_angles(rx, ry, rz);
        return Self {
            rotation
        }
    }
    pub fn to_euler_angles_py(&self) -> Vec<f64> {
        let mut out_vec = vec![];

        let e = self.rotation.to_euler_angles();
        out_vec.push(e[0]);
        out_vec.push(e[1]);
        out_vec.push(e[2]);

        out_vec
    }
    pub fn to_axis_angle_py(&self) -> (Vec<f64>, f64) {
        let mut out_vec = vec![];

        let e = self.rotation.to_axis_angle();
        out_vec.push(e.0[0]);
        out_vec.push(e.0[1]);
        out_vec.push(e.0[2]);

        (out_vec, e.1)
    }
    pub fn print_summary_py(&self) {
        println!("{:?}", self);
    }
}
impl OptimaRotationPy {
    pub fn rotation(&self) -> &OptimaRotation { &self.rotation }
}