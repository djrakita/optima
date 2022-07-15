use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use crate::utils::utils_generic_data_structures::{EnumBinarySearchTypeContainer, EnumMapToType, EnumTypeContainer};
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;
use crate::utils::utils_se3::optima_rotation::OptimaRotation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotSetLinkTransformSpecification {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaSE3Pose, weight: Option<f64> },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: Vector3<f64>, weight: Option<f64> },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaRotation, weight: Option<f64> }
}
impl EnumMapToType<RobotSetLinkTransformSpecificationType> for RobotSetLinkTransformSpecification {
    fn map_to_type(&self) -> RobotSetLinkTransformSpecificationType {
        return match self {
            RobotSetLinkTransformSpecification::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotSetLinkTransformSpecificationType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotSetLinkTransformSpecification::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotSetLinkTransformSpecificationType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotSetLinkTransformSpecification::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotSetLinkTransformSpecificationType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct RobotSetLinkTransformSpecificationType {
    robot_idx_in_set: usize,
    link_idx_in_robot: usize
}
impl RobotSetLinkTransformSpecificationType {
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

pub struct RobotLinkTransformSpecificationCollection {
    c: EnumBinarySearchTypeContainer<RobotSetLinkTransformSpecification, RobotSetLinkTransformSpecificationType>
}
impl RobotLinkTransformSpecificationCollection {
    pub fn new() -> Self {
        Self {
            c: EnumBinarySearchTypeContainer::new()
        }
    }
    pub fn insert_or_replace(&mut self, r: RobotSetLinkTransformSpecification) {
        self.c.insert_or_replace_object(r);
    }
    pub fn robot_set_link_specification_ref(&self, signature: &RobotSetLinkTransformSpecificationType) -> Option<&RobotSetLinkTransformSpecification> {
        self.c.object_ref(signature)
    }
    pub fn robot_set_link_specification_mut_ref(&mut self, signature: &RobotSetLinkTransformSpecificationType) -> Option<&mut RobotSetLinkTransformSpecification> {
        self.c.object_mut_ref(signature)
    }
    pub fn robot_set_link_specification_refs(&self) -> &Vec<RobotSetLinkTransformSpecification> {
        &self.c.object_refs()
    }
    pub fn print_summary(&self) {
        for s  in self.robot_set_link_specification_refs() {
            println!("{:?}", s);
        }
    }
}