use serde::{Serialize, Deserialize};
use nalgebra::{UnitQuaternion, Rotation3, Vector3};
use crate::utils::utils_errors::OptimaError;

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
    pub fn angle(&self) -> f64 {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => { data.angle() }
            OptimaRotation::UnitQuaternion { data, .. } => { data.angle() }
        }
    }
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
    pub fn multiply(&self, other: &OptimaRotation, conversion_if_necessary: bool) -> Result<OptimaRotation, OptimaError> {
        if self.get_rotation_type() != other.get_rotation_type() {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_rotation_type());
                self.multiply(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply."))
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
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply.")) }
                }
            }
            OptimaRotation::UnitQuaternion { data, rotation_type } => {
                let data0 = data;
                match other {
                    OptimaRotation::UnitQuaternion { data, .. } => {
                        let new_data = data0 * data;
                        Ok(Self::new_unit_quaternion(new_data))
                    }
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply.")) }
                }
            }
        }
    }
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
    pub fn displacement(&self, other: &OptimaRotation, conversion_if_necessary: bool) -> Result<OptimaRotation, OptimaError> {
        if self.get_rotation_type() != other.get_rotation_type() {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_rotation_type());
                self.displacement(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply."))
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
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply.")) }
                }
            }
            OptimaRotation::UnitQuaternion { data, rotation_type } => {
                let data0 = data;
                match other {
                    OptimaRotation::UnitQuaternion { data, .. } => {
                        let new_data = data0.inverse() * data;
                        Ok(Self::new_unit_quaternion(new_data))
                    }
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply.")) }
                }
            }
        }
    }
    pub fn angle_between(&self, other: &OptimaRotation, conversion_if_necessary: bool) -> Result<f64, OptimaError> {
        if self.get_rotation_type() != other.get_rotation_type() {
            return if conversion_if_necessary {
                let new_operand = other.convert(self.get_rotation_type());
                self.angle_between(&new_operand, conversion_if_necessary)
            } else {
                Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply."))
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
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply.")) }
                }
            }
            OptimaRotation::UnitQuaternion { data, rotation_type } => {
                let data0 = data;
                match other {
                    OptimaRotation::UnitQuaternion { data, .. } => {
                        let angle_between = data0.angle_to(data);
                        Ok(angle_between)
                    }
                    _ => { Err(OptimaError::new_string_descriptor_error("incompatible rotation types in multiply.")) }
                }
            }
        }
    }
    pub fn unwrap_rotation_matrix(&self) -> Result<&Rotation3<f64>, OptimaError> {
        return match self {
            OptimaRotation::RotationMatrix { data, .. } => {
                Ok(data)
            }
            OptimaRotation::UnitQuaternion { .. } => {
                Err(OptimaError::new_string_descriptor_error("tried to unwrap unit quaternion as rotation matrix."))
            }
        }
    }
    pub fn unwrap_unit_quaternion(&self) -> Result<&UnitQuaternion<f64>, OptimaError> {
        return match self {
            OptimaRotation::RotationMatrix { .. } => {
                Err(OptimaError::new_string_descriptor_error("tried to unwrap rotation matrix as unit quaternion."))
            }
            OptimaRotation::UnitQuaternion { data, .. } => {
                Ok(data)
            }
        }
    }
    fn get_rotation_type(&self) -> &OptimaRotationType {
        return match &self {
            OptimaRotation::RotationMatrix { data, rotation_type } => { rotation_type }
            OptimaRotation::UnitQuaternion { data, rotation_type } => { rotation_type }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum OptimaRotationType {
    RotationMatrix,
    UnitQuaternion
}