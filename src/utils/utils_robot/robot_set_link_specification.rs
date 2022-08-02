#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use nalgebra::{DVector, Vector3};
use serde::{Serialize, Deserialize};
use crate::robot_set_modules::robot_set::RobotSet;
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::robot_set_modules::robot_set_kinematics_module::RobotSetFKResult;
use crate::utils::utils_generic_data_structures::{EnumBinarySearchTypeContainer, EnumMapToType, EnumTypeContainer};
use crate::utils::utils_robot::robot_generic_structures::GenericRobotJointState;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PosePy, OptimaSE3PoseType};
use crate::utils::utils_se3::optima_rotation::{OptimaRotation, OptimaRotationPy};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotLinkTFGoal {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaSE3Pose, weight: Option<f64> },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: Vector3<f64>, weight: Option<f64> },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, goal: OptimaRotation, weight: Option<f64> }
}
impl EnumMapToType<RobotLinkTFGoalType> for RobotLinkTFGoal {
    fn map_to_type(&self) -> RobotLinkTFGoalType {
        return match self {
            RobotLinkTFGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotLinkTFGoalType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotLinkTFGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotLinkTFGoalType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
            RobotLinkTFGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, ..} => {
                RobotLinkTFGoalType {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot
                }
            }
        }
    }
}
impl RobotLinkTFGoal {
    pub fn compute_error_report(&self, robot_set_fk_res: &RobotSetFKResult) -> RobotLinkTFGoalErrorReport {
        return match self {
            RobotLinkTFGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, .. } => {
                let pose = robot_set_fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                let disp = pose.displacement(goal, true).expect("error");
                let euler_angles_and_translation = disp.to_euler_angles_and_translation();
                let e = euler_angles_and_translation.0;
                let t = euler_angles_and_translation.1;
                RobotLinkTFGoalErrorReport::LinkSE3PoseGoal {
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
            RobotLinkTFGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, .. } => {
                let pose = robot_set_fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                let disp = goal - pose.translation();
                RobotLinkTFGoalErrorReport::LinkPositionGoal {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot,
                    x_error: disp[0].abs(),
                    y_error: disp[1].abs(),
                    z_error: disp[2].abs()
                }
            }
            RobotLinkTFGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, .. } => {
                let pose = robot_set_fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                let disp = pose.rotation().displacement(goal, true).expect("error");
                let e = disp.to_euler_angles();
                RobotLinkTFGoalErrorReport::LinkRotationGoal {
                    robot_idx_in_set: *robot_idx_in_set,
                    link_idx_in_robot: *link_idx_in_robot,
                    rx_error: e[0],
                    ry_error: e[1],
                    rz_error: e[2],
                }
            }
        }
    }
    pub fn is_error_allowable(&self, robot_set_fk_res: &RobotSetFKResult, allowable_error: &RobotLinkTFAllowableError) -> bool {
        let error_report = self.compute_error_report(robot_set_fk_res);
        match error_report {
            RobotLinkTFGoalErrorReport::LinkSE3PoseGoal {  rx_error, ry_error, rz_error, x_error, y_error, z_error, .. } => {
                if rx_error.abs() > allowable_error.rx { return false; }
                if ry_error.abs() > allowable_error.ry { return false; }
                if rz_error.abs() > allowable_error.rz { return false; }
                if x_error.abs() > allowable_error.x { return false; }
                if y_error.abs() > allowable_error.y { return false; }
                if z_error.abs() > allowable_error.z { return false; }
            }
            RobotLinkTFGoalErrorReport::LinkPositionGoal {  x_error, y_error, z_error, .. } => {
                if x_error.abs() > allowable_error.x { return false; }
                if y_error.abs() > allowable_error.y { return false; }
                if z_error.abs() > allowable_error.z { return false; }
            }
            RobotLinkTFGoalErrorReport::LinkRotationGoal { rx_error, ry_error, rz_error, .. } => {
                if rx_error.abs() > allowable_error.rx { return false; }
                if ry_error.abs() > allowable_error.ry { return false; }
                if rz_error.abs() > allowable_error.rz { return false; }
            }
        }
        return true;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
#[pyclass]
pub struct RobotLinkTFGoalPy {
    robot_idx_in_set: usize,
    link_idx_in_robot: usize,
    se3_pose_goal: Option<OptimaSE3PosePy>,
    position_goal: Option<Vec<f64>>,
    rotation_goal: Option<OptimaRotationPy>,
    weight: Option<f64>
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotLinkTFGoalPy {
    #[staticmethod]
    pub fn new_se3_pose_goal(robot_idx_in_set: usize, link_idx_in_robot: usize, se3_pose_goal: OptimaSE3PosePy, weight: Option<f64>) -> Self {
        Self {
            robot_idx_in_set,
            link_idx_in_robot,
            se3_pose_goal: Some(se3_pose_goal),
            position_goal: None,
            rotation_goal: None,
            weight
        }
    }
    #[staticmethod]
    pub fn new_position_goal(robot_idx_in_set: usize, link_idx_in_robot: usize, position_goal: Vec<f64>, weight: Option<f64>) -> Self {
        Self {
            robot_idx_in_set,
            link_idx_in_robot,
            se3_pose_goal: None,
            position_goal: Some(position_goal),
            rotation_goal: None,
            weight
        }
    }
    #[staticmethod]
    pub fn new_rotation_goal(robot_idx_in_set: usize, link_idx_in_robot: usize, rotation_goal: OptimaRotationPy, weight: Option<f64>) -> Self {
        Self {
            robot_idx_in_set,
            link_idx_in_robot,
            se3_pose_goal: None,
            position_goal: None,
            rotation_goal: Some(rotation_goal),
            weight
        }
    }
}
impl RobotLinkTFGoalPy {
    pub fn convert_to_robot_set_link_transform_goal(&self) -> RobotLinkTFGoal {
        if let Some(se3_pose_goal) = &self.se3_pose_goal {
            let se3_pose_goal = se3_pose_goal.pose().clone();
            return RobotLinkTFGoal::LinkSE3PoseGoal {
                robot_idx_in_set: self.robot_idx_in_set,
                link_idx_in_robot: self.link_idx_in_robot,
                goal: se3_pose_goal,
                weight: self.weight
            }
        }

        if let Some(position_goal) = &self.position_goal {
            let position_goal = Vector3::new(position_goal[0], position_goal[1], position_goal[2]);
            return RobotLinkTFGoal::LinkPositionGoal {
                robot_idx_in_set: self.robot_idx_in_set,
                link_idx_in_robot: self.link_idx_in_robot,
                goal: position_goal,
                weight: self.weight
            }
        }

        if let Some(rotation_goal) = &self.rotation_goal {
            let rotation_goal = rotation_goal.rotation().clone();
            return RobotLinkTFGoal::LinkRotationGoal {
                robot_idx_in_set: self.robot_idx_in_set,
                link_idx_in_robot: self.link_idx_in_robot,
                goal: rotation_goal,
                weight: self.weight
            }
        }

        unreachable!()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct RobotLinkTFGoalType {
    robot_idx_in_set: usize,
    link_idx_in_robot: usize
}
impl RobotLinkTFGoalType {
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
pub enum RobotLinkTFGoalSignature {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize }
}

#[derive(Clone)]
pub struct RobotLinkTFGoalCollection {
    c: EnumBinarySearchTypeContainer<RobotLinkTFGoal, RobotLinkTFGoalType>
}
impl RobotLinkTFGoalCollection {
    pub fn new() -> Self {
        Self {
            c: EnumBinarySearchTypeContainer::new()
        }
    }
    pub fn insert_or_replace(&mut self, r: RobotLinkTFGoal) {
        self.c.insert_or_replace_object(r);
    }
    pub fn remove(&mut self, signature: &RobotLinkTFGoalType) {
        self.c.remove_object(signature);
    }
    pub fn remove_all(&mut self) {
        self.c.remove_all_objects();
    }
    pub fn robot_set_link_specification_ref(&self, signature: &RobotLinkTFGoalType) -> Option<&RobotLinkTFGoal> {
        self.c.object_ref(signature)
    }
    pub fn robot_set_link_specification_mut_ref(&mut self, signature: &RobotLinkTFGoalType) -> Option<&mut RobotLinkTFGoal> {
        self.c.object_mut_ref(signature)
    }
    pub fn robot_set_link_specification_refs(&self) -> &Vec<RobotLinkTFGoal> {
        &self.c.object_refs()
    }
    pub fn all_robot_set_link_transform_goal_refs(&self) -> &Vec<RobotLinkTFGoal> {
        self.c.object_refs()
    }
    pub fn print_summary(&self) {
        for s  in self.robot_set_link_specification_refs() {
            println!("{:?}", s);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub enum RobotLinkTFSpec<'a> {
    Absolute { goal: RobotLinkTFGoal },
    RelativeWRTGivenState { state: Box<&'a dyn GenericRobotJointState>, relative_goal: RobotLinkTFGoal }
}
impl <'a> RobotLinkTFSpec<'a> {
    pub fn recover_goal(&self, robot_set: &RobotSet) -> RobotLinkTFGoal {
        return match self {
            RobotLinkTFSpec::Absolute { goal } => { goal.clone() }
            RobotLinkTFSpec::RelativeWRTGivenState { state, relative_goal } => {
                let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(state.joint_state().clone()).expect("error");
                let fk_res = robot_set.robot_set_kinematics_module().compute_fk(&robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
                let pose = match relative_goal {
                    RobotLinkTFGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, .. } => { fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot) }
                    RobotLinkTFGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, .. } => { fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot) }
                    RobotLinkTFGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, .. } => { fk_res.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot) }
                };

                match relative_goal {
                    RobotLinkTFGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                        let out_goal = pose.multiply_separate_rotation_and_translation(goal, true).expect("error");
                        RobotLinkTFGoal::LinkSE3PoseGoal {
                            robot_idx_in_set: *robot_idx_in_set,
                            link_idx_in_robot: *link_idx_in_robot,
                            goal: out_goal,
                            weight: *weight
                        }
                    }
                    RobotLinkTFGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                        let out_goal = pose.translation() + goal;
                        RobotLinkTFGoal::LinkPositionGoal {
                            robot_idx_in_set: *robot_idx_in_set,
                            link_idx_in_robot: *link_idx_in_robot,
                            goal: out_goal,
                            weight: *weight
                        }
                    }
                    RobotLinkTFGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                        let out_goal = goal.multiply(&pose.rotation(), true).expect("error");
                        RobotLinkTFGoal::LinkRotationGoal {
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
#[pyclass]
pub struct RobotLinkTFSpecPy {
    goal: RobotLinkTFGoalPy,
    relative_state: Option<DVector<f64>>
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotLinkTFSpecPy {
    #[staticmethod]
    pub fn new_absolute(goal: RobotLinkTFGoalPy) -> Self {
        Self {
            goal,
            relative_state: None
        }
    }
    #[staticmethod]
    pub fn new_relative_wrt_given_state(state: Vec<f64>, goal: RobotLinkTFGoalPy) -> Self {
        Self {
            goal,
            relative_state: Some(DVector::from_vec(state))
        }
    }
}
impl RobotLinkTFSpecPy {
    pub fn convert_to_robot_set_link_transform_specification(&self) -> RobotLinkTFSpec {
        return if let Some(state) = &self.relative_state {
            RobotLinkTFSpec::RelativeWRTGivenState { state: Box::new(state), relative_goal: self.goal.convert_to_robot_set_link_transform_goal() }
        } else {
            RobotLinkTFSpec::Absolute { goal: self.goal.convert_to_robot_set_link_transform_goal() }
        }
    }
}

#[derive(Clone)]
pub struct RobotLinkTFSpecCollection<'a> {
    robot_set_link_transform_specifications: Vec<RobotLinkTFSpec<'a>>
}
impl <'a> RobotLinkTFSpecCollection<'a> {
    pub fn new() -> Self {
        Self {
            robot_set_link_transform_specifications: vec![]
        }
    }
    pub fn add(&mut self, r: RobotLinkTFSpec<'a>) {
        self.robot_set_link_transform_specifications.push(r);
    }
    pub fn robot_set_link_transform_specifications(&self) -> &Vec<RobotLinkTFSpec<'a>> {
        &self.robot_set_link_transform_specifications
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
#[pyclass]
pub struct RobotLinkTFSpecCollectionPy {
    robot_set_link_transform_specifications: Vec<RobotLinkTFSpecPy>
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotLinkTFSpecCollectionPy {
    #[staticmethod]
    pub fn new() -> Self {
        Self {
            robot_set_link_transform_specifications: vec![]
        }
    }
    pub fn add(&mut self, r: RobotLinkTFSpecPy) {
        self.robot_set_link_transform_specifications.push(r);
    }
}
impl RobotLinkTFSpecCollectionPy {
    pub fn convert_to_robot_set_link_transform_specification_collection(&self) -> RobotLinkTFSpecCollection {
        let mut out = RobotLinkTFSpecCollection::new();
        for r in &self.robot_set_link_transform_specifications {
            out.add(r.convert_to_robot_set_link_transform_specification());
        }
        out
    }
}

#[derive(Clone)]
pub struct RobotLinkTFSpecAndAllowableError<'a> {
    spec: RobotLinkTFSpec<'a>,
    allowable_error: RobotLinkTFAllowableError
}
impl <'a> RobotLinkTFSpecAndAllowableError<'a> {
    pub fn new(spec: RobotLinkTFSpec<'a>, allowable_error: Option<RobotLinkTFAllowableError>) -> Self {
        Self {
            spec,
            allowable_error: match allowable_error {
                None => { RobotLinkTFAllowableError::default() }
                Some(a) => { a }
            }
        }
    }
    pub fn spec(&self) -> &RobotLinkTFSpec<'a> {
        &self.spec
    }
    pub fn allowable_error(&self) -> &RobotLinkTFAllowableError {
        &self.allowable_error
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
#[pyclass]
pub struct RobotLinkTFSpecAndAllowableErrorPy {
    spec: RobotLinkTFSpecPy,
    allowable_error: RobotLinkTFAllowableErrorPy
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotLinkTFSpecAndAllowableErrorPy {
    #[staticmethod]
    pub fn new(spec: RobotLinkTFSpecPy, allowable_error: RobotLinkTFAllowableErrorPy) -> Self {
        Self {
            spec,
            allowable_error
        }
    }
}
impl RobotLinkTFSpecAndAllowableErrorPy {
    pub fn convert_to_robot_set_link_transform_specification_and_error(&self) -> RobotLinkTFSpecAndAllowableError {
        RobotLinkTFSpecAndAllowableError {
            spec: self.spec.convert_to_robot_set_link_transform_specification(),
            allowable_error: self.allowable_error.robot_set_link_transform_allowable_error.clone()
        }
    }
}

#[derive(Clone)]
pub struct RobotLinkTFSpecAndAllowableErrorCollection<'a> {
    robot_set_link_transform_specification_and_errors: Vec<RobotLinkTFSpecAndAllowableError<'a>>
}
impl <'a> RobotLinkTFSpecAndAllowableErrorCollection<'a> {
    pub fn new_empty() -> Self {
        Self {
            robot_set_link_transform_specification_and_errors: vec![]
        }
    }
    pub fn add(&mut self, r: RobotLinkTFSpecAndAllowableError<'a>) {
        self.robot_set_link_transform_specification_and_errors.push(r);
    }
    pub fn robot_set_link_transform_specification_and_errors(&self) -> &Vec<RobotLinkTFSpecAndAllowableError<'a>> {
        &self.robot_set_link_transform_specification_and_errors
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
#[pyclass]
pub struct RobotLinkTFSpecAndAllowableErrorCollectionPy {
    robot_set_link_transform_specification_and_errors: Vec<RobotLinkTFSpecAndAllowableErrorPy>
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotLinkTFSpecAndAllowableErrorCollectionPy {
    #[staticmethod]
    pub fn new() -> Self {
        Self {
            robot_set_link_transform_specification_and_errors: vec![]
        }
    }
    pub fn add(&mut self, r: RobotLinkTFSpecAndAllowableErrorPy) {
        self.robot_set_link_transform_specification_and_errors.push(r);
    }
}
impl RobotLinkTFSpecAndAllowableErrorCollectionPy {
    pub fn convert_to_robot_set_link_transform_specification_and_error_collection(&self) -> RobotLinkTFSpecAndAllowableErrorCollection {
        let mut out = RobotLinkTFSpecAndAllowableErrorCollection::new_empty();
        for r in &self.robot_set_link_transform_specification_and_errors {
            out.add(r.convert_to_robot_set_link_transform_specification_and_error());
        }
        out
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotLinkTFGoalErrorReport {
    LinkSE3PoseGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, rx_error: f64, ry_error: f64, rz_error: f64, x_error: f64, y_error: f64, z_error: f64 },
    LinkPositionGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, x_error: f64, y_error: f64, z_error: f64 },
    LinkRotationGoal { robot_idx_in_set: usize, link_idx_in_robot: usize, rx_error: f64, ry_error: f64, rz_error: f64 }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotLinkTFGoalErrorReportCollection {
    robot_set_link_transform_goal_error_reports: Vec<RobotLinkTFGoalErrorReport>
}
impl RobotLinkTFGoalErrorReportCollection {
    pub fn new() -> Self {
        Self {
            robot_set_link_transform_goal_error_reports: vec![]
        }
    }
    pub fn add(&mut self, r: RobotLinkTFGoalErrorReport) {
        self.robot_set_link_transform_goal_error_reports.push(r);
    }
    pub fn robot_set_link_transform_goal_error_reports(&self) -> &Vec<RobotLinkTFGoalErrorReport> {
        &self.robot_set_link_transform_goal_error_reports
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotLinkTFAllowableError {
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64
}
impl RobotLinkTFAllowableError {
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
impl Default for RobotLinkTFAllowableError {
    fn default() -> Self {
        Self {
            rx: 0.001,
            ry: 0.001,
            rz: 0.001,
            x: 0.001,
            y: 0.001,
            z: 0.001
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
#[pyclass]
pub struct RobotLinkTFAllowableErrorPy {
    robot_set_link_transform_allowable_error: RobotLinkTFAllowableError
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotLinkTFAllowableErrorPy {
    #[staticmethod]
    pub fn new(rx: Option<f64>, ry: Option<f64>, rz: Option<f64>, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Self {
        Self {
            robot_set_link_transform_allowable_error: RobotLinkTFAllowableError::new(rx, ry, rz, x, y, z)
        }
    }
    #[staticmethod]
    pub fn new_uniform_value(val: f64) -> Self {
        Self {
            robot_set_link_transform_allowable_error: RobotLinkTFAllowableError::new_uniform_value(val)
        }
    }
    #[staticmethod]
    pub fn new_rotation_and_translation_values(rotation_val: Option<f64>, translation_val: Option<f64>) -> Self {
        Self {
            robot_set_link_transform_allowable_error: RobotLinkTFAllowableError::new_rotation_and_translation_values(rotation_val, translation_val)
        }
    }
}

