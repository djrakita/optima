#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_robot::urdf_link::URDFLink;

/// A Link holds all necessary information about a robot link (specified by a robot URDF file)
/// in order to do kinematic and dynamic computations on a robot model.
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct Link {
    name: String,
    present: bool,
    link_idx: usize,
    preceding_link_idx: Option<usize>,
    children_link_idxs: Vec<usize>,
    preceding_joint_idx: Option<usize>,
    is_chain_base_link: bool,
    urdf_link: URDFLink,
}
impl Link {
    pub fn new(urdf_link: URDFLink, link_idx: usize) -> Self {
        Self {
            name: urdf_link.name().to_string(),
            present: true,
            link_idx,
            preceding_link_idx: None,
            children_link_idxs: vec![],
            preceding_joint_idx: None,
            is_chain_base_link: false,
            urdf_link
        }
    }
    /// Returns a link that can serve as a base link.  This will be automatically used by the
    /// `RobotConfigurationModule`, so it will almost never need to be called by the end user.
    /// Here, the `child_link_idx` will be the link that the chain connects to.  For example,
    /// if the child_link_idx is the world_link_idx of the `RobotModelModule`, this will essentially
    /// create a mobile base link for the whole robot model.
    pub fn new_base_of_chain_link(link_idx: usize, child_link_idx: usize, newly_created_joint_idx: usize, world_link_idx: usize) -> Self {
        let name = format!("base_of_chain_link_with_child_link_{}", child_link_idx);
        Self {
            name,
            present: true,
            link_idx,
            preceding_link_idx: if child_link_idx == world_link_idx { None } else { Some(world_link_idx) } ,
            children_link_idxs: vec![child_link_idx],
            preceding_joint_idx: Some(newly_created_joint_idx),
            is_chain_base_link: true,
            urdf_link: URDFLink::new_empty()
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn present(&self) -> bool {
        self.present
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
    pub fn is_chain_base_link(&self) -> bool {
        self.is_chain_base_link
    }
    pub fn urdf_link(&self) -> &URDFLink {
        &self.urdf_link
    }
    pub fn set_is_mobile_base_link(&mut self, is_mobile_base_link: bool) {
        self.is_chain_base_link = is_mobile_base_link;
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
    pub fn add_child_link_idx(&mut self, idx: usize) {
        self.children_link_idxs.push(idx);
    }
    pub fn print_summary(&self) {
        optima_print(&format!("  Link index: "), PrintMode::Print, PrintColor::Blue, true);
        optima_print(&format!(" {} ", self.link_idx), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("  Link name: "), PrintMode::Print, PrintColor::Blue, true);
        optima_print(&format!(" {} ", self.name), PrintMode::Print, PrintColor::None, false);
        optima_print(&format!("  Present: "), PrintMode::Print, PrintColor::Blue, true);
        let color = match self.present {
            true => { PrintColor::Green }
            false => { PrintColor::Red }
        };
        optima_print(&format!(" {} ", self.present), PrintMode::Print, color, false);
    }
    pub fn set_present(&mut self, present: bool) {
        self.present = present;
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