use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_fk_module::{RobotFKModule, RobotFKResult};
use crate::robot_modules::robot_joint_state_module::RobotJointStateModule;
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;

pub struct RobotSetFKModule {
    robot_fk_modules: Vec<RobotFKModule>
}
impl RobotSetFKModule {
    pub fn new(robot_set_configuration_module: &RobotSetConfigurationModule) -> Self {
        let mut robot_fk_modules = vec![];
        for r in robot_set_configuration_module.robot_configuration_modules() {
            let robot_joint_state_module = RobotJointStateModule::new(r.clone());
            let robot_fk_module = RobotFKModule::new(r.clone(), robot_joint_state_module);
            robot_fk_modules.push(robot_fk_module);
        }
        Self {
            robot_fk_modules
        }
    }
    pub fn new_from_set_name(set_name: &str) -> Result<Self, OptimaError> {
        let robot_set_configuration_module = RobotSetConfigurationModule::new_from_set_name(set_name)?;
        return Ok(Self::new(&robot_set_configuration_module));
    }
    pub fn compute_fk(&self, set_joint_state: &RobotSetJointState, t: &OptimaSE3PoseType) -> Result<RobotSetFKResult, OptimaError> {
        let mut out_vec = vec![];

        for (i, joint_state) in set_joint_state.robot_joint_states().iter().enumerate() {
            let fk_res = self.robot_fk_modules[i].compute_fk(joint_state, t)?;
            out_vec.push(fk_res);
        }

        Ok(RobotSetFKResult {
            robot_fk_results: out_vec
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetFKResult {
    robot_fk_results: Vec<RobotFKResult>
}
impl RobotSetFKResult {
    pub fn robot_fk_results(&self) -> &Vec<RobotFKResult> {
        &self.robot_fk_results
    }
    pub fn robot_fk_result(&self, robot_idx_in_set: usize) -> Result<&RobotFKResult, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(robot_idx_in_set, self.robot_fk_results.len(), file!(), line!())?;

        return Ok(&self.robot_fk_results[robot_idx_in_set]);
    }
    pub fn print_summary(&self) {
        for (i, robot_fk_result) in self.robot_fk_results.iter().enumerate() {
            optima_print(&format!("Robot {} ---> ", i), PrintMode::Println, PrintColor::Cyan, true);
            robot_fk_result.print_summary();
        }
    }
}