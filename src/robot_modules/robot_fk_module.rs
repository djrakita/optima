use nalgebra::DVector;
use crate::robot_modules::robot_configuration_generator_module::RobotConfigurationGeneratorModule;
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_state_module::RobotStateModule;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseAll, OptimaSE3PoseType};

#[derive(Clone, Debug)]
pub struct RobotFKModule {
    robot_configuration_module: RobotConfigurationModule,
    robot_state_module: RobotStateModule
}
impl RobotFKModule {
    pub fn new(robot_configuration_module: RobotConfigurationModule, robot_state_module: RobotStateModule) -> Self {
        Self {
            robot_configuration_module,
            robot_state_module
        }
    }
    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationGeneratorModule::new(robot_name)?.generate_configuration(configuration_name)?;
        let robot_state_module = RobotStateModule::new(robot_configuration_module.clone());
        return Ok(Self::new(robot_configuration_module, robot_state_module));
    }
    pub fn compute_fk_full_state(&self, full_state: &DVector<f64>) -> Result<(), OptimaError> {
        todo!()
    }
    pub fn compute_fk_dof_state(&self, dof_state: &DVector<f64>) -> Result<(), OptimaError> {
        let full_state = self.robot_state_module.convert_dof_state_to_full_state(dof_state)?;
        return self.compute_fk_full_state(&full_state);
    }
    fn compute_fk_on_single_link(&self, full_state: &DVector<f64>, link_idx: usize, t: &OptimaSE3PoseType, out_vec: &mut Vec<Option<OptimaSE3Pose>>) -> Result<(), OptimaError> {
        let link = self.robot_configuration_module.robot_model_module().get_link_by_idx(link_idx)?;
        let preceding_link_option = link.preceding_link_idx();
        if preceding_link_option.is_none() {
            out_vec[link_idx] = Some(OptimaSE3Pose::new_from_euler_angles(0.,0.,0.,0.,0.,0., t));
            return Ok(());
        }

        let preceding_link = preceding_link_option.unwrap();

        let preceding_joint_option = link.preceding_joint_idx();
        if preceding_joint_option.is_none() {
            out_vec[link_idx] = out_vec[preceding_link].clone();
        }

        Ok(())
    }
}