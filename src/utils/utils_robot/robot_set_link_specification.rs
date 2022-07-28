use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use crate::robot_set_modules::robot_set::RobotSet;
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::robot_set_modules::robot_set_kinematics_module::RobotSetFKResult;
use crate::utils::utils_generic_data_structures::{EnumBinarySearchTypeContainer, EnumMapToType, EnumTypeContainer};
use crate::utils::utils_robot::robot_generic_structures::GenericRobotJointState;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
use crate::utils::utils_se3::optima_rotation::OptimaRotation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotSetLinkTransformGoal {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaSE3Pose, weight: Option<f64> },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: Vector3<f64>, weight: Option<f64> },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaRotation, weight: Option<f64> }
}
impl EnumMapToType<RobotSetLinkTransformGoalType> for RobotSetLinkTransformGoal {
    fn map_to_type(&self) -> RobotSetLinkTransformGoalType {
        return match self {
            RobotSetLinkTransformGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotSetLinkTransformGoalType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotSetLinkTransformGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotSetLinkTransformGoalType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotSetLinkTransformGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotSetLinkTransformGoalType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
        }
    }
}
impl RobotSetLinkTransformGoal {
    pub fn compute_error_report(&self, robot_set_fk_res: &RobotSetFKResult) -> RobotSetLinkTransformGoalErrorReport {
        return match self {
            RobotSetLinkTransformGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, .. } => {
                let pose = robot_set_fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                let disp = pose.displacement(goal, true).expect("error");
                let euler_angles_and_translation = disp.to_euler_angles_and_translation();
                let e = euler_angles_and_translation.0;
                let t = euler_angles_and_translation.1;
                RobotSetLinkTransformGoalErrorReport::LinkSE3PoseGoal {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot,
                    rx_error: e[0],
                    ry_error: e[1],
                    rz_error: e[2],
                    x_error: t[0],
                    y_error: t[1],
                    z_error: t[2]
                }
            }
            RobotSetLinkTransformGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, .. } => {
                let pose = robot_set_fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                let disp = goal - pose.translation();
                RobotSetLinkTransformGoalErrorReport::LinkPositionGoal {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot,
                    x_error: disp[0].abs(),
                    y_error: disp[1].abs(),
                    z_error: disp[2].abs()
                }
            }
            RobotSetLinkTransformGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, .. } => {
                let pose = robot_set_fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                let disp = pose.rotation().displacement(goal, true).expect("error");
                let e = disp.to_euler_angles();
                RobotSetLinkTransformGoalErrorReport::LinkRotationGoal {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot,
                    rx_error: e[0],
                    ry_error: e[1],
                    rz_error: e[2],
                }
            }
        }
    }
    pub fn is_error_allowable(&self, robot_set_fk_res: &RobotSetFKResult, allowable_error: &RobotSetLinkTransformAllowableError) -> bool {
        let error_report = self.compute_error_report(robot_set_fk_res);
        match error_report {
            RobotSetLinkTransformGoalErrorReport::LinkSE3PoseGoal {  rx_error, ry_error, rz_error, x_error, y_error, z_error, .. } => {
                if rx_error.abs() > allowable_error.rx { return false; }
                if ry_error.abs() > allowable_error.ry { return false; }
                if rz_error.abs() > allowable_error.rz { return false; }
                if x_error.abs() > allowable_error.x { return false; }
                if y_error.abs() > allowable_error.y { return false; }
                if z_error.abs() > allowable_error.z { return false; }
            }
            RobotSetLinkTransformGoalErrorReport::LinkPositionGoal {  x_error, y_error, z_error, .. } => {
                if x_error.abs() > allowable_error.x { return false; }
                if y_error.abs() > allowable_error.y { return false; }
                if z_error.abs() > allowable_error.z { return false; }
            }
            RobotSetLinkTransformGoalErrorReport::LinkRotationGoal { rx_error, ry_error, rz_error, .. } => {
                if rx_error.abs() > allowable_error.rx { return false; }
                if ry_error.abs() > allowable_error.ry { return false; }
                if rz_error.abs() > allowable_error.rz { return false; }
            }
        }
        return true;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct RobotSetLinkTransformGoalType {
    robot_idx_in_set: usize,
    link_idx_in_robot: usize
}
impl RobotSetLinkTransformGoalType {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotSetLinkTransformGoalSignature {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize }
}

#[derive(Clone)]
pub enum RobotSetLinkTransformSpecification<'a> {
    Absolute { goal: RobotSetLinkTransformGoal },
    RelativeWRTGivenState { state: Box<&'a dyn GenericRobotJointState>, relative_goal: RobotSetLinkTransformGoal }
}
impl <'a> RobotSetLinkTransformSpecification<'a> {
    pub fn recover_goal(&self, robot_set: &RobotSet) -> RobotSetLinkTransformGoal {
        return match self {
            RobotSetLinkTransformSpecification::Absolute { goal } => { goal.clone() }
            RobotSetLinkTransformSpecification::RelativeWRTGivenState { state, relative_goal } => {
                let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(state.joint_state().clone()).expect("error");
                let fk_res = robot_set.robot_set_kinematics_module().compute_fk(&robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
                let pose = match relative_goal {
                    RobotSetLinkTransformGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, .. } => { fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot) }
                    RobotSetLinkTransformGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, .. } => { fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot) }
                    RobotSetLinkTransformGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, .. } => { fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot) }
                };

                match relative_goal {
                    RobotSetLinkTransformGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                        let out_goal = pose.multiply_separate_rotation_and_translation(goal, true).expect("error");
                        RobotSetLinkTransformGoal::LinkSE3PoseGoal {
                            robot_idx_in_set: *robot_idx_in_set,
                            link_idx_in_robot: *link_idx_in_robot,
                            goal: out_goal,
                            weight: *weight
                        }
                    }
                    RobotSetLinkTransformGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                        let out_goal = pose.translation() + goal;
                        RobotSetLinkTransformGoal::LinkPositionGoal {
                            robot_idx_in_set: *robot_idx_in_set,
                            link_idx_in_robot: *link_idx_in_robot,
                            goal: out_goal,
                            weight: *weight
                        }
                    }
                    RobotSetLinkTransformGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                        let out_goal = goal.multiply(&pose.rotation(), true).expect("error");
                        RobotSetLinkTransformGoal::LinkRotationGoal {
                            robot_idx_in_set: *robot_idx_in_set,
                            link_idx_in_robot: *link_idx_in_robot,
                            goal: out_goal,
                            weight: *weight
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct RobotSetLinkTransformSpecificationCollection<'a> {
    robot_set_link_transform_specifications: Vec<RobotSetLinkTransformSpecification<'a>>
}
impl <'a> RobotSetLinkTransformSpecificationCollection <'a> {
    pub fn new() -> Self {
        Self {
            robot_set_link_transform_specifications: vec![]
        }
    }
    pub fn add(&mut self, r: RobotSetLinkTransformSpecification<'a>) {
        self.robot_set_link_transform_specifications.push(r);
    }
    pub fn robot_set_link_transform_specifications(&self) -> &Vec<RobotSetLinkTransformSpecification<'a>> {
        &self.robot_set_link_transform_specifications
    }
}

#[derive(Clone)]
pub struct RobotSetLinkTransformSpecificationAndError<'a> {
    spec: RobotSetLinkTransformSpecification<'a>,
    allowable_error: RobotSetLinkTransformAllowableError
}
impl <'a> RobotSetLinkTransformSpecificationAndError<'a> {
    pub fn new(spec: RobotSetLinkTransformSpecification<'a>, allowable_error: Option<RobotSetLinkTransformAllowableError>) -> Self {
        Self {
            spec,
            allowable_error: match allowable_error {
                None => { RobotSetLinkTransformAllowableError::default() }
                Some(a) => { a }
            }
        }
    }
    pub fn spec(&self) -> &RobotSetLinkTransformSpecification<'a> {
        &self.spec
    }
    pub fn allowable_error(&self) -> &RobotSetLinkTransformAllowableError {
        &self.allowable_error
    }
}

#[derive(Clone)]
pub struct RobotSetLinkTransformSpecificationAndErrorCollection<'a> {
    robot_set_link_transform_specification_and_errors: Vec<RobotSetLinkTransformSpecificationAndError<'a>>
}
impl <'a> RobotSetLinkTransformSpecificationAndErrorCollection<'a> {
    pub fn new_empty() -> Self {
        Self {
            robot_set_link_transform_specification_and_errors: vec![]
        }
    }
    pub fn add(&mut self, r: RobotSetLinkTransformSpecificationAndError<'a>) {
        self.robot_set_link_transform_specification_and_errors.push(r);
    }
    pub fn robot_set_link_transform_specification_and_errors(&self) -> &Vec<RobotSetLinkTransformSpecificationAndError<'a>> {
        &self.robot_set_link_transform_specification_and_errors
    }
}

#[derive(Clone)]
pub struct RobotLinkTransformGoalCollection {
    c: EnumBinarySearchTypeContainer<RobotSetLinkTransformGoal, RobotSetLinkTransformGoalType>
}
impl RobotLinkTransformGoalCollection {
    pub fn new() -> Self {
        Self {
            c: EnumBinarySearchTypeContainer::new()
        }
    }
    pub fn insert_or_replace(&mut self, r: RobotSetLinkTransformGoal) {
        self.c.insert_or_replace_object(r);
    }
    pub fn remove(&mut self, signature: &RobotSetLinkTransformGoalType) {
        self.c.remove_object(signature);
    }
    pub fn remove_all(&mut self) {
        self.c.remove_all_objects();
    }
    pub fn robot_set_link_specification_ref(&self, signature: &RobotSetLinkTransformGoalType) -> Option<&RobotSetLinkTransformGoal> {
        self.c.object_ref(signature)
    }
    pub fn robot_set_link_specification_mut_ref(&mut self, signature: &RobotSetLinkTransformGoalType) -> Option<&mut RobotSetLinkTransformGoal> {
        self.c.object_mut_ref(signature)
    }
    pub fn robot_set_link_specification_refs(&self) -> &Vec<RobotSetLinkTransformGoal> {
        &self.c.object_refs()
    }
    pub fn all_robot_set_link_transform_goal_refs(&self) -> &Vec<RobotSetLinkTransformGoal> {
        self.c.object_refs()
    }
    pub fn print_summary(&self) {
        for s  in self.robot_set_link_specification_refs() {
            println!("{:?}", s);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotSetLinkTransformGoalErrorReport {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, rx_error: f64, ry_error: f64, rz_error: f64, x_error: f64, y_error: f64, z_error: f64 },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, x_error: f64, y_error: f64, z_error: f64 },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, rx_error: f64, ry_error: f64, rz_error: f64 }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetLinkTransformGoalErrorReportCollection {
    robot_set_link_transform_goal_error_reports: Vec<RobotSetLinkTransformGoalErrorReport>
}
impl RobotSetLinkTransformGoalErrorReportCollection {
    pub fn new() -> Self {
        Self {
            robot_set_link_transform_goal_error_reports: vec![]
        }
    }
    pub fn add(&mut self, r: RobotSetLinkTransformGoalErrorReport) {
        self.robot_set_link_transform_goal_error_reports.push(r);
    }
    pub fn robot_set_link_transform_goal_error_reports(&self) -> &Vec<RobotSetLinkTransformGoalErrorReport> {
        &self.robot_set_link_transform_goal_error_reports
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetLinkTransformAllowableError {
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64
}
impl RobotSetLinkTransformAllowableError {
    pub fn new(rx: Option<f64>, ry: Option<f64>, rz: Option<f64>, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Self {
        Self {
            rx: match rx {
                None => { Default::default() }
                Some(a) => {a}
            },
            ry: match ry {
                None => { Default::default() }
                Some(a) => {a}
            },
            rz: match rz {
                None => { Default::default() }
                Some(a) => {a}
            },
            x: match x {
                None => { Default::default() }
                Some(a) => {a}
            },
            y: match y {
                None => { Default::default() }
                Some(a) => {a}
            },
            z: match z {
                None => { Default::default() }
                Some(a) => {a}
            }
        }
    }
    pub fn new_uniform_value(val: f64) -> Self {
        Self {
            rx: val,
            ry: val,
            rz: val,
            x: val,
            y: val,
            z: val
        }
    }
    pub fn new_rotation_and_translation_values(rotation_val: Option<f64>, translation_val: Option<f64>) -> Self {
        Self {
            rx: match rotation_val {
                None => { Default::default() }
                Some(a) => {a}
            },
            ry: match rotation_val {
                None => { Default::default() }
                Some(a) => {a}
            },
            rz: match rotation_val {
                None => { Default::default() }
                Some(a) => {a}
            },
            x: match translation_val {
                None => { Default::default() }
                Some(a) => {a}
            },
            y: match translation_val {
                None => { Default::default() }
                Some(a) => {a}
            },
            z: match translation_val {
                None => { Default::default() }
                Some(a) => {a}
            }
        }
    }
}
impl Default for RobotSetLinkTransformAllowableError {
    fn default() -> Self {
        Self {
            rx: 0.0001,
            ry: 0.0001,
            rz: 0.0001,
            x: 0.0001,
            y: 0.0001,
            z: 0.0001
        }
    }
}