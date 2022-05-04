#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use nalgebra::{Vector3, Unit};
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::ContiguousChainMobilityMode;
use crate::utils::utils_console::{optima_print, optima_print_new_line, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_robot::urdf_joint::{JointTypeWrapper, URDFJoint};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3PoseAll, OptimaSE3Pose, OptimaSE3PoseType};
use crate::utils::utils_traits::ToAndFromRonString;

/// A Joint holds all necessary information about a robot joint (specified by a robot URDF file)
/// in order to do kinematic and dynamic computations on a robot model.
/// Each joint can contain multiple JointAxis objects.  A JointAxis encodes a possible degree of freedom
/// in a robot model.  A single joint axis can  characterize either a rotation around the axis or a
/// translation along a given axis.  A Joint can contain multiple joint axes, meaning that a single
/// "joint" may have more than one degree of freedom (e.g., in the case of a floating joint,
/// it will have 6 DOFs).
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct Joint {
    name: String,
    present: bool,
    joint_idx: usize,
    preceding_link_idx: Option<usize>,
    child_link_idx: Option<usize>,
    origin_offset_pose: OptimaSE3PoseAll,
    has_origin_offset: bool,
    joint_axes: Vec<JointAxis>,
    is_chain_base_connector_joint: bool,
    is_joint_with_all_standard_axes: bool,
    urdf_joint: URDFJoint
}
impl Joint {
    /// Returns a joint corresponding to the given URDFJoint.  This will be automatically called
    /// by the RobotModelModule.
    pub fn new(urdf_joint: URDFJoint, joint_idx: usize) -> Self {
        let name = urdf_joint.name().to_string();

        let rpy = urdf_joint.origin_rpy();
        let xyz = urdf_joint.origin_xyz();

        let mut out_self = Self {
            name,
            present: true,
            joint_idx,
            preceding_link_idx: None,
            child_link_idx: None,
            origin_offset_pose: OptimaSE3PoseAll::new(&OptimaSE3Pose::new_implicit_dual_quaternion_from_euler_angles(rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2])),
            has_origin_offset: rpy.norm() != 0.0 || xyz.norm() != 0.0,
            joint_axes: vec![],
            is_chain_base_connector_joint: false,
            is_joint_with_all_standard_axes: false,
            urdf_joint
        };
        out_self.set_dof_axes(joint_idx);
        out_self.set_is_joint_with_all_standard_axes();

        out_self
    }
    /// Returns a joint that can serve as a connector between a mobile base and the rest of the robot's
    /// kinematic chain.  This will be automatically used by the RobotConfigurationModule, so it will
    /// almost never need to be called by the end user.
    pub fn new_base_of_chain_connector_joint(mobile_base_mode: &ContiguousChainMobilityMode, joint_idx: usize, newly_created_link_idx: usize, child_link_idx: usize) -> Self {
        let mut joint_axes = vec![];

        match mobile_base_mode {
            ContiguousChainMobilityMode::Static => {}
            ContiguousChainMobilityMode::Floating { x_bounds, y_bounds, z_bounds, xr_bounds, yr_bounds, zr_bounds } => {
                joint_axes.push(JointAxis::new(joint_idx, 0, Vector3::new(1.,0.,0.), JointAxisPrimitiveType::Translation, *x_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 1, Vector3::new(0.,1.,0.), JointAxisPrimitiveType::Translation, *y_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 2, Vector3::new(0.,0.,1.), JointAxisPrimitiveType::Translation, *z_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 3, Vector3::new(1.,0.,0.), JointAxisPrimitiveType::Rotation, *xr_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 4, Vector3::new(0.,1.,0.), JointAxisPrimitiveType::Rotation, *yr_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 5, Vector3::new(0.,0.,1.), JointAxisPrimitiveType::Rotation, *zr_bounds));
            }
            ContiguousChainMobilityMode::PlanarTranslation { x_bounds, y_bounds } => {
                joint_axes.push(JointAxis::new(joint_idx, 0, Vector3::new(1.,0.,0.), JointAxisPrimitiveType::Translation, *x_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 1, Vector3::new(0.,1.,0.), JointAxisPrimitiveType::Translation, *y_bounds));
            }
            ContiguousChainMobilityMode::PlanarRotation { zr_bounds } => {
                joint_axes.push(JointAxis::new(joint_idx, 0, Vector3::new(0.,0.,1.), JointAxisPrimitiveType::Rotation, *zr_bounds));
            }
            ContiguousChainMobilityMode::PlanarTranslationAndRotation { x_bounds, y_bounds, zr_bounds } => {
                joint_axes.push(JointAxis::new(joint_idx, 0, Vector3::new(1.,0.,0.), JointAxisPrimitiveType::Translation, *x_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 1, Vector3::new(0.,1.,0.), JointAxisPrimitiveType::Translation, *y_bounds));
                joint_axes.push(JointAxis::new(joint_idx, 2, Vector3::new(0.,0.,1.), JointAxisPrimitiveType::Rotation, *zr_bounds));
            }
        }

        let name = format!("base_of_chain_connector_joint_with_child_link_{}", newly_created_link_idx);

        Self {
            name,
            present: true,
            joint_idx,
            preceding_link_idx: Some(newly_created_link_idx),
            child_link_idx: Some(child_link_idx),
            origin_offset_pose: OptimaSE3PoseAll::new_identity(),
            has_origin_offset: false,
            joint_axes,
            is_chain_base_connector_joint: true,
            is_joint_with_all_standard_axes: true,
            urdf_joint: URDFJoint::new_empty()
        }
    }
    pub fn get_origin_offset(&self, pose_type: &OptimaSE3PoseType) -> &OptimaSE3Pose {
        return self.origin_offset_pose.get_pose_by_type(pose_type);
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn active(&self) -> bool {
        self.present
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
    pub fn num_dofs(&self) -> usize {
        let mut num_dofs = 0;
        for ja in self.joint_axes() {
            if !ja.is_fixed() {
                num_dofs += 1;
            }
        }
        return num_dofs;
    }
    pub fn num_axes(&self) -> usize {
        return self.joint_axes.len();
    }
    pub fn urdf_joint(&self) -> &URDFJoint {
        &self.urdf_joint
    }
    pub fn origin_offset_pose(&self) -> &OptimaSE3PoseAll {
        &self.origin_offset_pose
    }
    pub fn joint_axes(&self) -> &Vec<JointAxis> {
        &self.joint_axes
    }
    pub fn set_preceding_link_idx(&mut self, preceding_link_idx: Option<usize>) {
        self.preceding_link_idx = preceding_link_idx;
    }
    pub fn set_child_link_idx(&mut self, child_link_idx: Option<usize>) {
        self.child_link_idx = child_link_idx;
    }
    pub fn print_summary(&self) {
        optima_print(&format!(">> Joint index: "), PrintMode::Print, PrintColor::Blue, true);
        optima_print(&format!(" {} ", self.joint_idx), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("  Joint name: "), PrintMode::Print, PrintColor::Blue, true);
        optima_print(&format!(" {} ", self.name), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("  Num dofs: "), PrintMode::Print, PrintColor::Blue, true);
        optima_print(&format!(" {} ", self.num_dofs()), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("  Present: "), PrintMode::Print, PrintColor::Blue, true);
        let c = if self.present { PrintColor::Green } else { PrintColor::Red };
        optima_print(&format!(" {} ", self.present), PrintMode::Print, c, false);
        if self.num_axes() > 0 {
            optima_print_new_line();
        }
        for (i, a) in self.joint_axes().iter().enumerate() {
            optima_print(&format!("      -- Joint sub idx {}: ", i), PrintMode::Print, PrintColor::Cyan, false);
            optima_print(&format!(" {:?} about axis {:?}, ", a.axis_primitive_type, a.axis), PrintMode::Print, PrintColor::None, false);

            match a.is_fixed() {
                true => {
                    optima_print(&format!("Fixed at value {}", a.fixed_value.unwrap()), PrintMode::Print, PrintColor::None, false);
                }
                false => {
                    optima_print("Not fixed.", PrintMode::Print, PrintColor::None, false);
                }
            };
            if self.joint_axes.len() > 1 && i < self.joint_axes.len()-1 {
                optima_print_new_line();
            }
        }
    }
    pub fn set_fixed_joint_sub_dof(&mut self, joint_sub_idx: usize, fixed_value: Option<f64>) -> Result<(), OptimaError> {
        if joint_sub_idx >= self.joint_axes.len() {
            return Err(OptimaError::new_idx_out_of_bound_error(joint_sub_idx, self.joint_axes.len(), file!(), line!()));
        }

        self.joint_axes[joint_sub_idx].fixed_value = fixed_value;
        Ok(())
    }
    pub fn set_present(&mut self, present: bool) {
        self.present = present;
    }
    pub fn present(&self) -> bool {
        self.present
    }
    pub fn is_chain_base_connector_joint(&self) -> bool {
        self.is_chain_base_connector_joint
    }
    pub fn is_joint_with_all_standard_axes(&self) -> bool {
        self.is_joint_with_all_standard_axes
    }
    fn set_dof_axes(&mut self, joint_idx: usize) {
        let joint_type = self.urdf_joint.joint_type();
        let lower_bound = self.urdf_joint.limits_lower();
        let upper_bound = self.urdf_joint.limits_upper();
        let axis = self.urdf_joint.axis().clone();

        match joint_type {
            JointTypeWrapper::Revolute => {
                self.joint_axes.push(JointAxis::new(joint_idx, 0, axis, JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));
            }
            JointTypeWrapper::Continuous => {
                self.joint_axes.push(JointAxis::new(joint_idx, 0, axis, JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));
            }
            JointTypeWrapper::Prismatic => {
                self.joint_axes.push(JointAxis::new(joint_idx, 0, axis, JointAxisPrimitiveType::Translation, (lower_bound, upper_bound)));
            }
            JointTypeWrapper::Fixed => {
                /* Do Nothing */
            }
            JointTypeWrapper::Floating => {
                self.joint_axes.push(JointAxis::new(joint_idx, 0, Vector3::new(1.,0.,0.), JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));
                self.joint_axes.push(JointAxis::new(joint_idx, 1, Vector3::new(0.,1.,0.), JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));
                self.joint_axes.push(JointAxis::new(joint_idx, 2, Vector3::new(0.,0.,1.), JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));

                self.joint_axes.push(JointAxis::new(joint_idx, 3, Vector3::new(1.,0.,0.), JointAxisPrimitiveType::Translation, (lower_bound, upper_bound)));
                self.joint_axes.push(JointAxis::new(joint_idx, 4, Vector3::new(0.,1.,0.), JointAxisPrimitiveType::Translation, (lower_bound, upper_bound)));
                self.joint_axes.push(JointAxis::new(joint_idx, 5, Vector3::new(0.,0.,1.), JointAxisPrimitiveType::Translation, (lower_bound, upper_bound)));
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
                self.joint_axes.push(JointAxis::new(joint_idx, 0, Vector3::new(1.,0.,0.), JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));
                self.joint_axes.push(JointAxis::new(joint_idx, 1, Vector3::new(0.,1.,0.), JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));
                self.joint_axes.push(JointAxis::new(joint_idx, 2, Vector3::new(0.,0.,1.), JointAxisPrimitiveType::Rotation, (lower_bound, upper_bound)));
            }
        }
    }
    fn set_is_joint_with_all_standard_axes(&mut self) {
        let mut out_val = true;
        for a in &self.joint_axes {
            let axis = &a.axis;
            if !(axis == &Vector3::new(1.,0.,0.) || axis == &Vector3::new(0.,1.,0.) || axis == &Vector3::new(0.,0.,1.)) {
                out_val = false;
            }
        }
        self.is_joint_with_all_standard_axes = out_val;
    }
}

/// Methods supported by python.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl Joint {
    pub fn name_py(&self) -> String {
        self.name.clone()
    }
    pub fn present_py(&self) -> bool {
        self.present
    }
    pub fn joint_idx_py(&self) -> usize { self.joint_idx }
    pub fn preceding_link_idx_py(&self) -> Option<usize> {
        self.preceding_link_idx
    }
    pub fn child_link_idx_py(&self) -> Option<usize> {
        self.child_link_idx
    }
    pub fn joint_axes_py(&self) -> Vec<JointAxis> {
        self.joint_axes.clone()
    }
}

/// Methods supported by WASM.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Joint {

}

/// A JointAxis encodes a single degree of freedom in a robot model.  A single joint axis can
/// characterize either a rotation around the axis or a translation along a given axis.
/// A Joint can contain multiple joint axes, meaning that a single "joint" may have more than one
/// degree of freedom (e.g., in the case of a floating joint, it will have 6 DOFs).
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]

pub struct JointAxis {
    joint_idx: usize,
    joint_sub_dof_idx: usize,
    fixed_value: Option<f64>,
    axis_as_unit: Unit<Vector3<f64>>,
    axis: Vector3<f64>,
    axis_primitive_type: JointAxisPrimitiveType,
    bounds: (f64, f64)
}
impl JointAxis {
    pub fn new(joint_idx: usize, joint_sub_dof_idx: usize, axis: Vector3<f64>, axis_primitive_type: JointAxisPrimitiveType, bounds: (f64, f64)) -> Self {
        Self {
            joint_idx,
            joint_sub_dof_idx,
            fixed_value: None,
            axis_as_unit: Unit::new_normalize(axis.clone()),
            axis,
            axis_primitive_type,
            bounds
        }
    }
    pub fn is_fixed(&self) -> bool {
        self.fixed_value.is_some()
    }
    pub fn joint_idx(&self) -> usize {
        self.joint_idx
    }
    pub fn joint_sub_dof_idx(&self) -> usize {
        self.joint_sub_dof_idx
    }
    pub fn fixed_value(&self) -> Option<f64> {
        self.fixed_value
    }
    pub fn axis_as_unit(&self) -> Unit<Vector3<f64>> {
        self.axis_as_unit
    }
    pub fn axis(&self) -> Vector3<f64> {
        self.axis
    }
    pub fn axis_primitive_type(&self) -> &JointAxisPrimitiveType {
        &self.axis_primitive_type
    }
    pub fn bounds(&self) -> (f64, f64) {
        self.bounds
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl JointAxis {
    pub fn joint_idx_py(&self) -> usize {
        self.joint_idx
    }
    pub fn joint_sub_dof_idx_py(&self) -> usize {
        self.joint_sub_dof_idx
    }
    pub fn fixed_value_py(&self) -> Option<f64> {
        self.fixed_value
    }
    pub fn axis_py(&self) -> Vec<f64> {
        let a = &self.axis;
        return vec![a[0], a[1], a[2]];
    }
    pub fn axis_primitive_type_py(&self) -> String {
        self.axis_primitive_type.to_ron_string()
    }
    pub fn bounds_py(&self) -> (f64, f64) {
        self.bounds
    }
}

/// Specifies the transform type for a JointAxis Object.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum JointAxisPrimitiveType {
    Rotation,
    Translation
}