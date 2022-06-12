use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use crate::utils::utils_generic_data_structures::{EnumBinarySearchSignatureContainer, EnumMapToSignature, EnumSignatureContainer};
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;
use crate::utils::utils_se3::optima_rotation::OptimaRotation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotSetLinkSpecification {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaSE3Pose },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: Vector3<f64> },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaRotation }
}
impl EnumMapToSignature<RobotSetLinkSpecificationSignature> for RobotSetLinkSpecification {
    fn map_to_signature(&self) -> RobotSetLinkSpecificationSignature {
        return match self {
            RobotSetLinkSpecification::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal: _} => {
                RobotSetLinkSpecificationSignature {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotSetLinkSpecification::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal: _ } => {
                RobotSetLinkSpecificationSignature {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotSetLinkSpecification::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal: _ } => {
                RobotSetLinkSpecificationSignature {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct RobotSetLinkSpecificationSignature {
    robot_idx_in_set: usize,
    link_idx_in_robot: usize
}
impl RobotSetLinkSpecificationSignature {
    pub fn new(robot_idx_in_set: usize, link_idx_in_robot: usize) -> Self {
        Self {
            robot_idx_in_set,
            link_idx_in_robot
        }
    }
    pub fn robot_idx_in_set(&self) -> usize {
        self.robot_idx_in_set
    }
    pub fn link_idx_in_robot(&self) -> usize {
        self.link_idx_in_robot
    }
}

pub struct RobotLinkSpecificationCollection {
    c: EnumBinarySearchSignatureContainer<RobotSetLinkSpecification, RobotSetLinkSpecificationSignature>
}
impl RobotLinkSpecificationCollection {
    pub fn new() -> Self {
        Self {
            c: EnumBinarySearchSignatureContainer::new()
        }
    }
    pub fn insert_or_replace(&mut self, r: RobotSetLinkSpecification) {
        self.c.insert_or_replace_object(r);
    }
    pub fn robot_set_link_specification_ref(&self, signature: &RobotSetLinkSpecificationSignature) -> Option<&RobotSetLinkSpecification> {
        self.c.object_ref(signature)
    }
    pub fn robot_set_link_specification_mut_ref(&mut self, signature: &RobotSetLinkSpecificationSignature) -> Option<&mut RobotSetLinkSpecification> {
        self.c.object_mut_ref(signature)
    }
    pub fn robot_set_link_specification_refs(&self) -> &Vec<RobotSetLinkSpecification> {
        &self.c.object_refs()
    }
    pub fn print_summary(&self) {
        for s  in self.robot_set_link_specification_refs() {
            println!("{:?}", s);
        }
    }
}