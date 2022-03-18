use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use urdf_rs::read_file;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::RobotFolderUtils;
use crate::utils::utils_robot::joint::Joint;
use crate::utils::utils_robot::link::Link;
use crate::utils::utils_robot::urdf_joint::URDFJoint;
use crate::utils::utils_robot::urdf_link::URDFLink;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotModelModule {
    robot_name: String,
    links: Vec<Link>,
    joints: Vec<Joint>,
    world_link_idx: usize,
    robot_base_link_idx: usize,
    link_tree_traversal_layers: Vec<Vec<usize>>,
    link_tree_max_depth: usize,
    preceding_actuated_joint_idxs: Vec<Option<usize>>,
    link_name_to_idx_hashmap: HashMap<String, usize>,
    joint_name_to_idx_hashmap: HashMap<String, usize>
}
impl RobotModelModule {
    pub fn new(robot_name: &str) -> Result<Self, OptimaError> {
        let mut joints = vec![];
        let mut links = vec![];

        let mut urdf_robot_joints = vec![];
        let mut urdf_robot_links = vec![];

        let mut link_name_to_idx_hashmap = HashMap::new();
        let mut joint_name_to_idx_hashmap = HashMap::new();

        let path_to_urdf = RobotFolderUtils::get_path_to_urdf_file(robot_name)?;
        let urdf_robot_res = urdf_rs::read_file(path_to_urdf);
        match &urdf_robot_res {
            Ok(urdf_robot) => {
                for (i, j) in urdf_robot.joints.iter().enumerate() {
                    joint_name_to_idx_hashmap.insert(j.name.clone(), i);
                    joints.push(Joint::new(URDFJoint::new_from_urdf_joint(j), i));
                    urdf_robot_joints.push(j);
                }
                for (i, l) in urdf_robot.links.iter().enumerate() {
                    link_name_to_idx_hashmap.insert(l.name.clone(), i);
                    links.push(Link::new(URDFLink::new_from_urdf_link(l), i));
                    urdf_robot_links.push(l);
                }
            }
            Err(_) => {
                return Err(OptimaError::new_string_descriptor_error("Error when parsing urdf."))
            }
        }

        let mut out_self = Self {
            robot_name: robot_name.to_string(),
            links,
            joints,
            world_link_idx: 0,
            robot_base_link_idx: 0,
            link_tree_traversal_layers: vec![],
            link_tree_max_depth: 0,
            preceding_actuated_joint_idxs: vec![],
            link_name_to_idx_hashmap,
            joint_name_to_idx_hashmap
        };

        out_self.assign_all_link_connections_manual();
        out_self.assign_all_joint_connections_manual();
        out_self.set_world_link_idx_manual();
        out_self.set_link_tree_traversal_info();

        Ok(out_self)
    }
    fn assign_all_link_connections_manual(&mut self) {
        let l1 = self.links.len();
        let l2 = self.joints.len();

        for i in 0..l1 {
            for j in 0..l2 {
                if self.links[i].name() == self.joints[j].urdf_joint().child_link() {
                    let link_idx = self.get_link_idx_from_name( &self.joints[j].urdf_joint().parent_link().to_string() );
                    let joint_idx = self.get_joint_idx_from_name( &self.joints[j].name().to_string() );
                    self.links[i].set_preceding_link_idx( link_idx );
                    self.links[i].set_preceding_joint_idx( joint_idx );
                }

                if self.links[i].name() == self.joints[j].urdf_joint().parent_link() {
                    let link_idx = self.get_link_idx_from_name( &self.joints[j].urdf_joint().child_link().to_string() );
                    if link_idx.is_some() { self.links[i].add_child_link_idx(link_idx.unwrap()); }
                    let joint_idx = self.get_joint_idx_from_name( &self.joints[j].name().to_string());
                    if joint_idx.is_some() { self.links[i].add_child_joint_idx(joint_idx.unwrap()); }
                }
            }
        }
    }
    fn assign_all_joint_connections_manual(&mut self) {
        let l = self.joints.len();

        for i in 0..l {
            let link_idx = self.get_link_idx_from_name(  &self.joints[i].urdf_joint().parent_link().to_string()  );
            self.joints[i].set_preceding_link_idx(link_idx);
            let link_idx = self.get_link_idx_from_name(  &self.joints[i].urdf_joint().child_link().to_string()  );
            self.joints[i].set_child_link_idx(link_idx);
        }
    }
    fn set_world_link_idx_manual(&mut self) {
        let l = self.links.len();
        for i in 0..l {
            if self.links[i].preceding_link_idx().is_none() {
                self.world_link_idx = i;
                self.robot_base_link_idx = i;
                return;
            }
        }
    }
    pub fn set_link_tree_traversal_info(&mut self) {
        self.link_tree_traversal_layers = Vec::new();
        self.link_tree_traversal_layers.push( vec![ self.world_link_idx ] );

        let num_links = self.links.len();
        let mut curr_layer = 1 as usize;
        loop {
            let mut change_on_this_loop = false;
            for i in 0..num_links {
                if self.links[i].preceding_link_idx().is_some() && self.links[i].active() {
                    if self.link_tree_traversal_layers[curr_layer - 1].contains(&self.links[i].preceding_link_idx().unwrap()) {
                        if self.link_tree_traversal_layers.len() == curr_layer { self.link_tree_traversal_layers.push( Vec::new() ); }

                        self.link_tree_traversal_layers[curr_layer].push( i );
                        change_on_this_loop = true;
                    }
                }
            }

            if change_on_this_loop {
                curr_layer += 1;
            } else {
                self.link_tree_max_depth = self.link_tree_traversal_layers.len();
                return;
            }
        }
    }

    fn get_link_idx_from_name(&self, link_name: &String) -> Option<usize> {
        let res = self.link_name_to_idx_hashmap.get(link_name);
        match res {
            None => { return None }
            Some(u) => { return Some(*u) }
        }
    }
    fn get_joint_idx_from_name(&self, joint_name: &String) -> Option<usize> {
        let res = self.joint_name_to_idx_hashmap.get(joint_name);
        match res {
            None => { return None }
            Some(u) => { return Some(*u) }
        }
    }
}