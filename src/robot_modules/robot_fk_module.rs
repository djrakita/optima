use nalgebra::DVector;
use crate::robot_modules::robot_configuration_generator_module::RobotConfigurationGeneratorModule;
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_state_module::RobotStateModule;
use crate::utils::utils_console::{optima_print, optima_print_new_line, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_robot::joint::JointAxisPrimitiveType;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

#[derive(Clone, Debug)]
pub struct RobotFKModule {
    robot_configuration_module: RobotConfigurationModule,
    robot_state_module: RobotStateModule,
    starter_out_vec: Vec<Option<OptimaSE3Pose>>
}
impl RobotFKModule {
    pub fn new(robot_configuration_module: RobotConfigurationModule, robot_state_module: RobotStateModule) -> Self {
        let mut starter_out_vec = vec![];
        let l = robot_configuration_module.robot_model_module().links().len();
        for _ in 0..l { starter_out_vec.push(None); }

        Self {
            robot_configuration_module,
            robot_state_module,
            starter_out_vec
        }
    }
    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationGeneratorModule::new(robot_name)?.generate_configuration(configuration_name)?;
        let robot_state_module = RobotStateModule::new(robot_configuration_module.clone());
        return Ok(Self::new(robot_configuration_module, robot_state_module));
    }
    pub fn compute_fk_full_state(&self, full_state: &DVector<f64>, t: &OptimaSE3PoseType) -> Result<Vec<Option<OptimaSE3Pose>>, OptimaError> {
        if full_state.len() != self.robot_state_module.num_axes() {
            return Err(OptimaError::new_robot_state_vec_wrong_size_error("compute_fk_full_state", full_state.len(), self.robot_state_module.num_axes(), file!(), line!()))
        }

        let mut out_vec = self.starter_out_vec.clone();
        let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();

        for link_tree_traversal_layer in link_tree_traversal_layers {
            for link_idx in link_tree_traversal_layer {
                self.compute_fk_on_single_link(full_state, *link_idx, t, &mut out_vec)?;
            }
        }

        return Ok(out_vec)
    }
    pub fn compute_fk_dof_state(&self, dof_state: &DVector<f64>, t: &OptimaSE3PoseType) -> Result<Vec<Option<OptimaSE3Pose>>, OptimaError> {
        let full_state = self.robot_state_module.convert_dof_state_to_full_state(dof_state)?;
        return self.compute_fk_full_state(&full_state, t);
    }
    pub fn print_results(&self, result_vec: &Vec<Option<OptimaSE3Pose>>) {
        let links = self.robot_configuration_module.robot_model_module().links();
        for (i, link) in links.iter().enumerate() {
            optima_print(&format!("Link {} ({}) ---> ", i, link.name()), PrintMode::Print, PrintColor::Blue, true);
            optima_print(&format!("{:?}", result_vec[i]), PrintMode::Print, PrintColor::None, false);
            optima_print_new_line();
        }
    }
    fn compute_fk_on_single_link(&self, full_state: &DVector<f64>, link_idx: usize, t: &OptimaSE3PoseType, out_vec: &mut Vec<Option<OptimaSE3Pose>>) -> Result<(), OptimaError> {
        let link = self.robot_configuration_module.robot_model_module().get_link_by_idx(link_idx)?;
        let preceding_link_option = link.preceding_link_idx();
        if preceding_link_option.is_none() {
            out_vec[link_idx] = Some(OptimaSE3Pose::new_from_euler_angles(0.,0.,0.,0.,0.,0., t));
            return Ok(());
        }

        let preceding_link_idx = preceding_link_option.unwrap();

        let preceding_joint_option = link.preceding_joint_idx();
        if preceding_joint_option.is_none() {
            out_vec[link_idx] = out_vec[preceding_link_idx].clone();
            return Ok(());
        }

        let preceding_joint_idx = preceding_joint_option.unwrap();
        let preceding_joint = &self.robot_configuration_module.robot_model_module().joints()[preceding_joint_idx];

        let full_state_idxs = self.robot_state_module.map_joint_idx_to_full_state_idxs(preceding_joint_idx)?;

        let mut out_pose = out_vec[preceding_link_idx].clone().expect("error");

        let offset_pose_all = preceding_joint.origin_offset_pose();
        out_pose = out_pose.multiply(offset_pose_all.get_pose_by_type(t), false)?;

        let joint_axes = preceding_joint.joint_axes();

        for (i, full_state_idx) in full_state_idxs.iter().enumerate() {
            let joint_axis = &joint_axes[i];
            let joint_value = full_state[*full_state_idx];

            let axis_pose = match joint_axis.axis_primitive_type() {
                JointAxisPrimitiveType::Rotation => {
                    let axis = &joint_axis.axis_as_unit();
                    OptimaSE3Pose::new_from_axis_angle(axis, joint_value, 0.,0.,0., t)
                }
                JointAxisPrimitiveType::Translation => {
                    let axis = joint_value * &joint_axis.axis();
                    OptimaSE3Pose::new_from_euler_angles(0.,0.,0., axis[0], axis[1], axis[2], t)
                }
            };

            out_pose = out_pose.multiply(&axis_pose, false)?;
        }

        out_vec[link_idx] = Some(out_pose);

        Ok(())
    }
}