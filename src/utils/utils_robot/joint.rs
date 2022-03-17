use nalgebra::{Vector3, Unit};
use serde::{Serialize, Deserialize};
use crate::utils::utils_robot::urdf_joint::URDFJoint;

#[derive(Clone, Serialize, Deserialize)]
pub struct Joint {
    name: String,
    urdf_joint: URDFJoint,
    joint_idx: usize,
    preceding_link_idx: Option<usize>,
    child_link_idx: Option<usize>,
    // origin_offset: ImplicitDualQuaternion,
    has_origin_offset: bool,
    dof_rotation_axes: Vec<Vector3<f64>>,
    dof_translation_axes: Vec<Vector3<f64>>,
    dof_rotation_axes_as_units: Vec<Unit<Vector3<f64>>>,
    num_dofs: usize,
    active: bool
}