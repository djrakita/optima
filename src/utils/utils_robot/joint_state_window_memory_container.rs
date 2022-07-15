/*
use crate::robot_modules::robot_joint_state_module::RobotJointState;
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::utils::utils_generic_data_structures::WindowMemoryContainer;

#[derive(Clone, Debug)]
pub struct RobotJointStateWindowMemoryContainer {
    pub c: WindowMemoryContainer<RobotJointState>
}
impl RobotJointStateWindowMemoryContainer {
    pub fn new(window_size: usize, init_state: RobotJointState) -> Self {
        Self {
            c: WindowMemoryContainer::new(window_size, init_state)
        }
    }
}

#[derive(Clone, Debug)]
pub struct RobotSetJointStateWindowMemoryContainer {
    pub c: WindowMemoryContainer<RobotSetJointState>
}
impl RobotSetJointStateWindowMemoryContainer {
    pub fn new(window_size: usize, init_state: RobotSetJointState) -> Self {
        Self {
            c: WindowMemoryContainer::new(window_size, init_state)
        }
    }
}
*/
