use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetWrapper<T> {
    object: T,
    robot_idx_in_robot_set: usize
}
impl <T> RobotSetWrapper<T> {
    pub fn new(object: T, robot_idx_in_robot_set: usize) -> Self {
        Self {
            object,
            robot_idx_in_robot_set
        }
    }
    pub fn object(&self) -> &T {
        &self.object
    }
    pub fn object_mut(&mut self) -> &mut T {
        &mut self.object
    }
    pub fn robot_idx_in_robot_set(&self) -> usize {
        self.robot_idx_in_robot_set
    }
}