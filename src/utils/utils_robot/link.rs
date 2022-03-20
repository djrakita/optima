use serde::{Serialize, Deserialize};
use crate::utils::utils_robot::urdf_link::URDFLink;

#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// A Link holds all necessary information about a robot link (specified by a robot URDF file)
/// in order to do kinematic and dynamic computations on a robot model.
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct Link {
    name: String,
    active: bool,
    link_idx: usize,
    preceding_link_idx: Option<usize>,
    children_link_idxs: Vec<usize>,
    preceding_joint_idx: Option<usize>,
    children_joint_idxs: Vec<usize>,
    is_mobile_base_link: bool,
    urdf_link: URDFLink,
}
impl Link {
    pub fn new(urdf_link: URDFLink, link_idx: usize) -> Self {
        Self {
            name: urdf_link.name().to_string(),
            active: true,
            link_idx,
            preceding_link_idx: None,
            children_link_idxs: vec![],
            preceding_joint_idx: None,
            children_joint_idxs: vec![],
            is_mobile_base_link: false,
            urdf_link
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn active(&self) -> bool {
        self.active
    }
    pub fn link_idx(&self) -> usize {
        self.link_idx
    }
    pub fn preceding_link_idx(&self) -> Option<usize> {
        self.preceding_link_idx
    }
    pub fn children_link_idxs(&self) -> &Vec<usize> {
        &self.children_link_idxs
    }
    pub fn preceding_joint_idx(&self) -> Option<usize> {
        self.preceding_joint_idx
    }
    pub fn children_joint_idxs(&self) -> &Vec<usize> {
        &self.children_joint_idxs
    }
    pub fn is_mobile_base_link(&self) -> bool {
        self.is_mobile_base_link
    }
    pub fn urdf_link(&self) -> &URDFLink {
        &self.urdf_link
    }
    pub fn set_is_mobile_base_link(&mut self, is_mobile_base_link: bool) {
        self.is_mobile_base_link = is_mobile_base_link;
    }
    pub fn set_preceding_link_idx(&mut self, preceding_link_idx: Option<usize>) {
        self.preceding_link_idx = preceding_link_idx;
    }
    pub fn set_children_link_idxs(&mut self, children_link_idxs: Vec<usize>) {
        self.children_link_idxs = children_link_idxs;
    }
    pub fn set_preceding_joint_idx(&mut self, preceding_joint_idx: Option<usize>) {
        self.preceding_joint_idx = preceding_joint_idx;
    }
    pub fn set_children_joint_idxs(&mut self, children_joint_idxs: Vec<usize>) {
        self.children_joint_idxs = children_joint_idxs;
    }
    pub fn add_child_joint_idx(&mut self, idx: usize) {
        self.children_joint_idxs.push(idx);
    }
    pub fn add_child_link_idx(&mut self, idx: usize) {
        self.children_link_idxs.push(idx);
    }
}

/// Methods supported by python.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl Link {

}

/// Methods supported by WASM.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Link {

}