use urdf_rs::*;
use nalgebra::{Vector3};
use serde::{Serialize, Deserialize};

/// This struct holds all information provided by a URDF file on a Joint when parsed by urdf_rs.
#[derive(Clone, Serialize, Deserialize)]
pub struct URDFJoint {
    name: String,
    joint_type: JointTypeWrapper,
    origin_xyz: Vector3<f64>,
    origin_rpy: Vector3<f64>,
    parent_link: String,
    child_link: String,
    axis: Vector3<f64>,
    includes_limits: bool,
    limits_lower: f64,
    limits_upper: f64,
    limits_effort: f64,
    limits_velocity: f64,
    dynamics_damping: Option<f64>,
    dynamics_friction: Option<f64>,
    mimic_joint: Option<String>,
    mimic_multiplier: Option<f64>,
    mimic_offset: Option<f64>,
    safety_soft_lower_limit: Option<f64>,
    safety_soft_upper_limit: Option<f64>,
    safety_k_position: Option<f64>,
    safety_k_velocity: Option<f64>
}
impl URDFJoint {
    pub fn new_from_urdf_joint(joint: &Joint) -> Self {
        Self {
            name: joint.name.clone(),
            joint_type: JointTypeWrapper::from_joint_type(&joint.joint_type),
            origin_xyz: Vector3::new(joint.origin.xyz[0], joint.origin.xyz[1], joint.origin.xyz[2]),
            origin_rpy: Vector3::new(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2]),
            parent_link: joint.parent.link.clone(),
            child_link: joint.child.link.clone(),
            axis: Vector3::new(joint.axis.xyz[0], joint.axis.xyz[1], joint.axis.xyz[2]),
            includes_limits: joint.limit.lower != 0.0 && joint.limit.upper != 0.0,
            limits_lower: joint.limit.lower,
            limits_upper: joint.limit.upper,
            limits_effort: joint.limit.effort,
            limits_velocity: joint.limit.velocity,
            dynamics_damping: if let Some(d) = &joint.dynamics { Some(d.damping) } else { None },
            dynamics_friction: if let Some(d) = &joint.dynamics { Some(d.friction) } else { None },
            mimic_joint: if let Some(m) = &joint.mimic { Some(m.joint.clone()) } else { None },
            mimic_multiplier: if let Some(m) = &joint.mimic { m.multiplier.clone() } else { None },
            mimic_offset: if let Some(m) = &joint.mimic { m.offset.clone() } else { None },
            safety_soft_lower_limit: if let Some(s) = &joint.safety_controller { Some(s.soft_lower_limit) } else { None },
            safety_soft_upper_limit: if let Some(s) = &joint.safety_controller { Some(s.soft_upper_limit) } else { None },
            safety_k_position: if let Some(s) = &joint.safety_controller { Some(s.k_position) } else { None },
            safety_k_velocity: if let Some(s) = &joint.safety_controller { Some(s.k_velocity) } else { None }
        }
    }
    pub fn new_empty() -> Self {
        Self {
            name: "".to_string(),
            joint_type: JointTypeWrapper::Revolute,
            origin_xyz: Default::default(),
            origin_rpy: Default::default(),
            parent_link: "".to_string(),
            child_link: "".to_string(),
            axis: Default::default(),
            includes_limits: false,
            limits_lower: 0.0,
            limits_upper: 0.0,
            limits_effort: 0.0,
            limits_velocity: 0.0,
            dynamics_damping: None,
            dynamics_friction: None,
            mimic_joint: None,
            mimic_multiplier: None,
            mimic_offset: None,
            safety_soft_lower_limit: None,
            safety_soft_upper_limit: None,
            safety_k_position: None,
            safety_k_velocity: None
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn joint_type(&self) -> &JointTypeWrapper {
        &self.joint_type
    }
    pub fn origin_xyz(&self) -> Vector3<f64> {
        self.origin_xyz
    }
    pub fn origin_rpy(&self) -> Vector3<f64> {
        self.origin_rpy
    }
    pub fn parent_link(&self) -> &str {
        &self.parent_link
    }
    pub fn child_link(&self) -> &str {
        &self.child_link
    }
    pub fn axis(&self) -> Vector3<f64> {
        self.axis
    }
    pub fn includes_limits(&self) -> bool {
        self.includes_limits
    }
    pub fn limits_lower(&self) -> f64 {
        self.limits_lower
    }
    pub fn limits_upper(&self) -> f64 {
        self.limits_upper
    }
    pub fn limits_effort(&self) -> f64 {
        self.limits_effort
    }
    pub fn limits_velocity(&self) -> f64 {
        self.limits_velocity
    }
    pub fn dynamics_damping(&self) -> Option<f64> {
        self.dynamics_damping
    }
    pub fn dynamics_friction(&self) -> Option<f64> {
        self.dynamics_friction
    }
    pub fn mimic_joint(&self) -> &Option<String> {
        &self.mimic_joint
    }
    pub fn mimic_multiplier(&self) -> Option<f64> {
        self.mimic_multiplier
    }
    pub fn mimic_offset(&self) -> Option<f64> {
        self.mimic_offset
    }
    pub fn safety_soft_lower_limit(&self) -> Option<f64> {
        self.safety_soft_lower_limit
    }
    pub fn safety_soft_upper_limit(&self) -> Option<f64> {
        self.safety_soft_upper_limit
    }
    pub fn safety_k_position(&self) -> Option<f64> {
        self.safety_k_position
    }
    pub fn safety_k_velocity(&self) -> Option<f64> {
        self.safety_k_velocity
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JointTypeWrapper {
    Revolute,
    Continuous,
    Prismatic,
    Fixed,
    Floating,
    Planar,
    Spherical
}
impl JointTypeWrapper {
    pub fn from_joint_type(j: &JointType) -> Self {
        match j {
            JointType::Revolute => { Self::Revolute }
            JointType::Continuous => { Self::Continuous }
            JointType::Prismatic => { Self::Prismatic }
            JointType::Fixed => { Self::Fixed }
            JointType::Floating => { Self::Floating }
            JointType::Planar => { Self::Planar }
            JointType::Spherical => { Self::Spherical }
        }
    }
}