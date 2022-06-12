use crate::robot_set_modules::robot_set::RobotSet;

pub trait GetRobotSet {
    fn get_robot_set(&self) -> &RobotSet;
}




pub mod robot_set_configuration_module;
pub mod robot_set_joint_state_module;
pub mod robot_set_kinematics_module;
pub mod robot_set_mesh_file_manager_module;
pub mod robot_set_geometric_shape_module;
pub mod robot_set;