use nalgebra::{Vector3, Unit};
use serde::{Serialize, Deserialize};
use crate::utils::utils_robot::urdf_joint::{JointTypeWrapper, URDFJoint};
use crate::utils::utils_se3::implicit_dual_quaternion::ImplicitDualQuaternion;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

/// A Joint holds all necessary information about a robot joint (specified by a robot URDF file)
/// in order to do kinematic and dynamic computations on a robot model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Joint {
    name: String,
    active: bool,
    joint_idx: usize,
    preceding_link_idx: Option<usize>,
    child_link_idx: Option<usize>,
    origin_offset_idq: OptimaSE3Pose,
    origin_offset_hm: OptimaSE3Pose,
    origin_offset_uqt: OptimaSE3Pose,
    origin_offset_rmt: OptimaSE3Pose,
    has_origin_offset: bool,
    dof_rotation_axes: Vec<Vector3<f64>>,
    dof_translation_axes: Vec<Vector3<f64>>,
    dof_rotation_axes_as_units: Vec<Unit<Vector3<f64>>>,
    num_dofs: usize,
    urdf_joint: URDFJoint
}
impl Joint {
    pub fn new(urdf_joint: URDFJoint, joint_idx: usize) -> Self {
        let name = urdf_joint.name().to_string();

        let rpy = urdf_joint.origin_rpy();
        let xyz = urdf_joint.origin_xyz();

        let mut out_self = Self {
            name,
            active: true,
            joint_idx,
            preceding_link_idx: None,
            child_link_idx: None,
            origin_offset_idq: OptimaSE3Pose::new_implicit_dual_quaternion_from_euler_angles(rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2]),
            origin_offset_hm: OptimaSE3Pose::new_homogeneous_matrix_from_euler_angles(rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2]),
            origin_offset_uqt: OptimaSE3Pose::new_unit_quaternion_and_translation_from_euler_angles(rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2]),
            origin_offset_rmt: OptimaSE3Pose::new_rotation_matrix_and_translation_from_euler_angles(rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2]),
            has_origin_offset: rpy.norm() != 0.0 || xyz.norm() != 0.0,
            dof_rotation_axes: vec![],
            dof_translation_axes: vec![],
            dof_rotation_axes_as_units: vec![],
            num_dofs: 0,
            urdf_joint
        };
        out_self.set_dof_axes();

        out_self
    }
    pub fn get_origin_offset(&self, pose_type: &OptimaSE3PoseType) -> &OptimaSE3Pose {
        return match pose_type {
            OptimaSE3PoseType::ImplicitDualQuaternion => { &self.origin_offset_idq }
            OptimaSE3PoseType::HomogeneousMatrix => { &self.origin_offset_hm }
            OptimaSE3PoseType::UnitQuaternionAndTranslation => { &self.origin_offset_uqt }
            OptimaSE3PoseType::RotationMatrixAndTranslation => { &self.origin_offset_rmt }
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn active(&self) -> bool {
        self.active
    }
    pub fn joint_idx(&self) -> usize {
        self.joint_idx
    }
    pub fn preceding_link_idx(&self) -> Option<usize> {
        self.preceding_link_idx
    }
    pub fn child_link_idx(&self) -> Option<usize> {
        self.child_link_idx
    }
    pub fn has_origin_offset(&self) -> bool {
        self.has_origin_offset
    }
    pub fn dof_rotation_axes(&self) -> &Vec<Vector3<f64>> {
        &self.dof_rotation_axes
    }
    pub fn dof_translation_axes(&self) -> &Vec<Vector3<f64>> {
        &self.dof_translation_axes
    }
    pub fn dof_rotation_axes_as_units(&self) -> &Vec<Unit<Vector3<f64>>> {
        &self.dof_rotation_axes_as_units
    }
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }
    pub fn urdf_joint(&self) -> &URDFJoint {
        &self.urdf_joint
    }
    fn set_dof_axes(&mut self) {
        let joint_type = self.urdf_joint.joint_type();
        let axis = self.urdf_joint.axis().clone();

        match joint_type {
            JointTypeWrapper::Revolute => {
                self.dof_rotation_axes.push(axis);
            }
            JointTypeWrapper::Continuous => {
                self.dof_rotation_axes.push(axis);
            }
            JointTypeWrapper::Prismatic => {
                self.dof_translation_axes.push(axis);
            }
            JointTypeWrapper::Fixed => {
                /* Do Nothing */
            }
            JointTypeWrapper::Floating => {
                self.dof_translation_axes.push( Vector3::new(1., 0., 0.) );
                self.dof_translation_axes.push( Vector3::new(0., 1., 0.) );
                self.dof_translation_axes.push( Vector3::new(0., 0., 1.) );

                self.dof_rotation_axes.push( Vector3::new(1., 0., 0.) );
                self.dof_rotation_axes.push( Vector3::new(0., 1., 0.) );
                self.dof_rotation_axes.push( Vector3::new(0., 0., 1.) );
            }
            JointTypeWrapper::Planar => {
                /*
                let mut v = axis.clone();

                let v1 = get_orthogonal_vector(&v)?;
                let v2 = v.cross(&v1);

                self.dof_translation_axes.push( Vector3::new(v1[0], v1[1], v1[2]) );
                self.dof_translation_axes.push( Vector3::new(v2[0], v2[1], v2[2]) );
                */
                todo!()
            }
            JointTypeWrapper::Spherical => {
                todo!()
            }
        }

        self.num_dofs = self.dof_rotation_axes.len() + self.dof_translation_axes.len();
    }
    pub fn set_preceding_link_idx(&mut self, preceding_link_idx: Option<usize>) {
        self.preceding_link_idx = preceding_link_idx;
    }
    pub fn set_child_link_idx(&mut self, child_link_idx: Option<usize>) {
        self.child_link_idx = child_link_idx;
    }
}