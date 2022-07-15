use std::fmt::Debug;
use nalgebra::DVector;
use crate::utils::utils_generic_data_structures::WindowMemoryContainer;

pub trait GenericRobotJointState: Debug + GenericRobotJointStateClone {
    fn joint_state(&self) -> &DVector<f64>;
}

pub trait GenericRobotJointStateClone {
    fn clone_box(&self) -> Box<dyn GenericRobotJointState>;
}
impl<T> GenericRobotJointStateClone for T where T: 'static + GenericRobotJointState + Clone {
    fn clone_box(&self) -> Box<dyn GenericRobotJointState> {
        Box::new(self.clone())
    }
}
impl Clone for Box<dyn GenericRobotJointState> {
    fn clone(&self) -> Box<dyn GenericRobotJointState> {
        self.clone_box()
    }
}

#[derive(Clone, Debug)]
pub struct TimedGenericRobotJointState {
    joint_state: Box<dyn GenericRobotJointState>,
    time: f64
}
impl TimedGenericRobotJointState {
    pub fn new<T: GenericRobotJointState + 'static>(joint_state: T, time: f64) -> Self {
        Self {
            joint_state: Box::new(joint_state),
            time
        }
    }
    pub fn joint_state(&self) -> &DVector<f64> {
        self.joint_state.joint_state()
    }
    pub fn time(&self) -> f64 {
        self.time
    }
}

#[derive(Clone, Debug)]
pub struct TimedGenericRobotJointStateWindowMemoryContainer {
    pub c: WindowMemoryContainer<TimedGenericRobotJointState>
}
impl TimedGenericRobotJointStateWindowMemoryContainer {
    pub fn new(window_size: usize, init_state: TimedGenericRobotJointState) -> Self {
        Self {
            c: WindowMemoryContainer::new(window_size, init_state)
        }
    }
}

/*
#[derive(Clone, Debug)]
pub struct GenericRobotJointStateWithOptTimeWindowMemoryContainerObject {
    pub c: WindowMemoryContainer<dyn GenericRobotJointStateWithOptTime>
}
impl GenericRobotJointStateWithOptTimeWindowMemoryContainerObject {
    pub fn new(window_size: usize, init_state: GenericRobotJointStateWithOptTimeObject<dyn GenericRobotJointStateWithOptTime>) -> Self {
        Self {
            c: WindowMemoryContainer::new(window_size, init_state)
        }
    }
}
*/