use std::vec;
#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_console::{ConsoleInputUtils, optima_print, PrintColor, PrintMode};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseAll, OptimaSE3PosePy};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaAssetLocation, OptimaStemCellPath};
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_traits::SaveAndLoadable;

/// A `RobotConfigurationModule` is a description of a robot model one abstraction layer above the
/// `RobotModelModule`.  A robot configuration affords extra specificity and functionality over a robot
/// model.  For example, a robot configuration can include a mobile base, a base offset, removed links,
/// as well as fixed joint values that will not be considered degrees of freedom.
///
/// A robot configuration consists of two components:
/// - A `RobotConfigurationInfo` object that describes key features of the particular configuration.
/// - The given robot's model that has been modified based on a given `RobotConfigurationInfo`.
///
/// In many cases, the `RobotConfigurationInfo` will reflect a default base model configuration, meaning
/// its respective configuration will be the base robot model given directly by the robot's URDF.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotConfigurationModule {
    robot_configuration_info: RobotConfigurationInfo,
    robot_model_module: RobotModelModule,
    base_robot_model_module: RobotModelModule
}
impl RobotConfigurationModule {
    pub fn new_from_names(robot_names: RobotNames) -> Result<Self, OptimaError> {
        return match robot_names.configuration_name() {
            None => { Self::new_base_model(robot_names.robot_name()) }
            Some(configuration_name) => {
                let mut path = OptimaStemCellPath::new_asset_path()?;
                path.append_file_location(&OptimaAssetLocation::RobotConfigurations { robot_name: robot_names.robot_name().to_string() });
                path.append(&(configuration_name.to_string() + ".JSON"));

                if !path.exists() {
                    return Err(OptimaError::new_generic_error_str(&format!("Robot {} does not have configuration {} at path {:?}.", robot_names.robot_name(), configuration_name, path), file!(), line!()))
                }

                return Self::load_from_path(&path);
            }
        }
    }
    pub fn new_from_robot_name_and_info(robot_name: &str, robot_configuration_info: RobotConfigurationInfo) -> Result<Self, OptimaError> {
        let base_model_module = RobotModelModule::new(robot_name)?;

        let mut out_self = Self {
            robot_configuration_info,
            robot_model_module: base_model_module.clone(),
            base_robot_model_module: base_model_module
        };

        out_self.update()?;

        return Ok(out_self);
    }
    fn new_base_model(robot_name: &str) -> Result<Self, OptimaError> {
        let robot_model_module = RobotModelModule::new(robot_name)?;
        Ok(Self {
            robot_configuration_info: Default::default(),
            robot_model_module: robot_model_module.clone(),
            base_robot_model_module: robot_model_module
        })
    }
    fn update(&mut self) -> Result<(), OptimaError> {
        let mut robot_model_module = self.base_robot_model_module.clone();
        let robot_configuration_info = &self.robot_configuration_info;

        let contiguous_chain_infos = &robot_configuration_info.contiguous_chain_infos;
        if contiguous_chain_infos.len() > 0 {
            // Set all links as not present.
            let num_links = robot_model_module.links().len();
            for link_idx in 0..num_links {
                robot_model_module.set_link_as_not_present(link_idx).expect("error");
            }
        }

        let mut names_to_remove = vec![];

        let mut link_idxs_that_are_already_a_part_of_chains = vec![];
        for contiguous_chain_info in contiguous_chain_infos {
            let link_idxs_to_possibly_add = match contiguous_chain_info.end_link_idx {
                None => {
                    self.robot_model_module.get_all_downstream_links(contiguous_chain_info.start_link_idx)?
                }
                Some(end_link_idx) => {
                    let chain = self.robot_model_module.get_link_chain(contiguous_chain_info.start_link_idx, end_link_idx)?;
                    if chain.is_none() { return Err(OptimaError::new_generic_error_str(&format!("Link chain does not exist between link {} and {}.", contiguous_chain_info.start_link_idx, end_link_idx), file!(), line!())) }
                    let chain = chain.unwrap().clone();
                    chain
                }
            };

            let mut clash = false;
            for link_idx_to_possibly_add in &link_idxs_to_possibly_add {
                for link_idx_that_is_already_a_part_of_chains in &link_idxs_that_are_already_a_part_of_chains {
                    if link_idx_to_possibly_add == link_idx_that_is_already_a_part_of_chains {
                        clash = true;
                        break;
                    }
                }
            }

            if !clash {
                for link_idx in &link_idxs_to_possibly_add {
                    link_idxs_that_are_already_a_part_of_chains.push(*link_idx);
                    robot_model_module.set_link_as_present(*link_idx)?;
                }
                robot_model_module.add_contiguous_chain_link_and_joint(&contiguous_chain_info.mobility_mode, contiguous_chain_info.start_link_idx);
            } else {
                let print_string = format!("WARNING: Could not add contiguous chain {:?} because it conflicts with an already added chain.", contiguous_chain_info);
                optima_print(&print_string, PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
                names_to_remove.push(contiguous_chain_info.chain_name.clone());
            }
        }

        let dead_end_link_idxs = &robot_configuration_info.dead_end_link_idxs;
        for d in dead_end_link_idxs {
            let all_downstream_links = robot_model_module.get_all_downstream_links(*d)?;
            for dl in &all_downstream_links {
                robot_model_module.set_link_as_not_present(*dl)?;
            }
        }

        let fixed = &robot_configuration_info.fixed_joint_infos;
        for f in fixed {
            robot_model_module.set_fixed_joint_sub_dof(f.joint_idx, f.joint_sub_idx, Some(f.fixed_joint_value))?;
        }

        for name_to_remove in &names_to_remove { self.remove_contiguous_chain(name_to_remove); }

        robot_model_module.set_link_tree_traversal_info();

        self.robot_model_module = robot_model_module;

        Ok(())
    }
    /// Returns a reference to the `RobotConfigurationInfo` that was used to change the configuration's
    /// underlying model module.
    pub fn robot_configuration_info(&self) -> &RobotConfigurationInfo {
        &self.robot_configuration_info
    }
    /// Returns a reference to the robot model module that reflects the configuration's `RobotConfigurationInfo`.
    pub fn robot_model_module(&self) -> &RobotModelModule {
        &self.robot_model_module
    }
    pub fn set_contiguous_chain(&mut self, chain_name: &str, start_link_idx: usize, end_link_idx: Option<usize>, mobility_mode: ContiguousChainMobilityMode) -> Result<(), OptimaError> {
        for c in &self.robot_configuration_info.contiguous_chain_infos {
            if &c.chain_name == chain_name {
                let print_string = format!("WARNING: Could not add contiguous chain {:?} because its name conflicts with an already added chain.", chain_name);
                optima_print(&print_string, PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
                return Ok(());
            }
        }

        self.robot_configuration_info.contiguous_chain_infos.push(ContiguousChainInfo {
            chain_name: chain_name.to_string(),
            start_link_idx,
            end_link_idx,
            mobility_mode
        });

        return self.update();
    }
    pub fn remove_contiguous_chain(&mut self, chain_name: &str) {
        let l = self.robot_configuration_info.contiguous_chain_infos.len();
        for i in 0..l {
            if &self.robot_configuration_info.contiguous_chain_infos[i].chain_name == chain_name {
                self.robot_configuration_info.contiguous_chain_infos.remove(i);
                return;
            }
        }
    }
    /// Note that setting a mobile base is done using `set_contiguous_chain`
    pub fn set_mobile_base(&mut self, mobility_mode: ContiguousChainMobilityMode) -> Result<(), OptimaError> {
        let world_idx = self.robot_model_module.world_link_idx();
        // let children_link_idxs = self.robot_model_module.links()[world_idx].children_link_idxs();
        // assert!(children_link_idxs.len() > 0);
        // self.set_contiguous_chain("mobile_base", children_link_idxs[0], None, mobility_mode)
        self.set_contiguous_chain("mobile_base", world_idx, None, mobility_mode)
    }
    /// Sets the given link as a "dead end" link.  A dead end link is a link such that it and all
    /// links that occur as successors in the kinematic chain will be inactive (essentially, removed)
    /// from the robot model.
    pub fn set_dead_end_link(&mut self, link_idx: usize) -> Result<(), OptimaError> {
        self.robot_configuration_info.dead_end_link_idxs.push(link_idx);
        return self.update();
    }
    /// Removes the given link as a dead end link.
    pub fn remove_dead_end_link(&mut self, link_idx: usize) -> Result<(), OptimaError> {
        self.robot_configuration_info.dead_end_link_idxs =
            self.robot_configuration_info.dead_end_link_idxs
            .iter().filter_map(|s| if *s == link_idx { None } else { Some(*s) } ).collect();
        return self.update();
    }
    /// Fixes the given joint to the given value.  Thus, this joint will not be a degree of freedom
    /// in the current configuration.
    pub fn set_fixed_joint(&mut self, joint_idx: usize, joint_sub_idx: usize, fixed_joint_value: f64) -> Result<(), OptimaError> {
        self.robot_configuration_info.fixed_joint_infos.push(FixedJointInfo {
            joint_idx,
            joint_sub_idx,
            fixed_joint_value
        });
        return self.update();
    }
    /// Removes the given joint as a fixed joint.  Thus, this joint will become a degree of freedom.
    pub fn remove_fixed_joint(&mut self, joint_idx: usize, joint_sub_idx: usize) -> Result<(), OptimaError> {
        self.robot_configuration_info.fixed_joint_infos =
            self.robot_configuration_info.fixed_joint_infos
                .iter().filter_map(|s| if s.joint_idx == joint_idx && s.joint_sub_idx == joint_sub_idx { None } else { Some(s.clone()) } ).collect();

        return self.update();
    }
    /// sets the base offset of the robot configuration.
    pub fn set_base_offset(&mut self, p: &OptimaSE3Pose) -> Result<(), OptimaError> {
        self.robot_configuration_info.base_offset = OptimaSE3PoseAll::new(p);
        return self.update();
    }
    pub fn print_contiguous_chains(&self) {
        for c in &self.robot_configuration_info.contiguous_chain_infos {
            println!("{:?}", c);
        }
    }
    /*
    /// Saves the `RobotConfigurationModule` to its robot's `RobotConfigurationGeneratorModule`.
    /// The configuration will be saved to a json file such that the `RobotConfigurationGeneratorModule`
    /// will be able to load this configuration in the future.
    pub fn save(&mut self, configuration_name: &str) -> Result<(), OptimaError> {
        self.set_configuration_name(configuration_name);
        let mut r = RobotConfigurationGeneratorModule::new(self.robot_model_module.robot_name())?;
        r.save_robot_configuration_module(&self.robot_configuration_info)?;
        Ok(())
    }
    */
    pub fn save(&self, configuration_name: &str) -> Result<(), OptimaError> {
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotConfigurations { robot_name: self.robot_model_module.robot_name().to_string() });
        path.append(&(configuration_name.to_string() + ".JSON"));

        if path.exists() {
            let response = ConsoleInputUtils::get_console_input_string(&format!("Configuration with name {} already exists.  Overwrite?  (y or n)", configuration_name), PrintColor::Cyan)?;
            if response == "y" {
                self.save_to_path(&path)?;
            } else {
                return Ok(());
            }
        } else {
            self.save_to_path(&path)?;
        }

        Ok(())
    }
    /// Prints summary of underlying robot model module.
    pub fn print_summary(&self) {
        self.robot_model_module.print_summary();
    }
    pub fn robot_name(&self) -> &str {
        return self.robot_model_module.robot_name()
    }
}
impl SaveAndLoadable for RobotConfigurationModule {
    type SaveType = (String, RobotConfigurationInfo);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_model_module.robot_name().to_string(), self.robot_configuration_info.clone())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        return RobotConfigurationModule::new_from_robot_name_and_info(&load.0, load.1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug))]
pub struct RobotConfigurationModulePy {
    pub robot_configuration_module: RobotConfigurationModule,
    #[cfg(not(target_arch = "wasm32"))]
    #[pyo3(get)]
    pub robot_model_module_py: Py<RobotModelModule>
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotConfigurationModulePy {
    #[cfg(not(target_arch = "wasm32"))]
    #[new]
    pub fn new(robot_name: &str, py: Python) -> Self {
        let robot_configuration_module = RobotConfigurationModule::new_base_model(robot_name).expect("error");
        let robot_model_module_py = Py::new(py, robot_configuration_module.robot_model_module.clone()).expect("error");
        Self {
            robot_configuration_module,
            robot_model_module_py
        }
    }

    #[staticmethod]
    pub fn new_from_configuration_module(robot_configuration_module: RobotConfigurationModule, py: Python) -> Self {
        let robot_model_module_py = Py::new(py, robot_configuration_module.robot_model_module.clone()).expect("error");
        Self {
            robot_configuration_module,
            robot_model_module_py
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn copy_robot_model_module_to_py(&mut self, py: Python) {
        self.robot_model_module_py = Py::new(py, self.robot_configuration_module.robot_model_module.clone()).expect("error");
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn print_summary(&self) {
        self.robot_configuration_module.print_summary();
    }

    /// Sets the given link as a "dead end" link.  A dead end link is a link such that it and all
    /// links that occur as successors in the kinematic chain will be inactive (essentially, removed)
    /// from the robot model.
    pub fn set_dead_end_link(&mut self, link_idx: usize, py: Python) {
        self.robot_configuration_module.set_dead_end_link(link_idx).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

    /// Removes the given link as a dead end link.
    pub fn remove_dead_end_link(&mut self, link_idx: usize, py: Python) {
        self.robot_configuration_module.remove_dead_end_link(link_idx).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

    /// Fixes the given joint to the given value.  Thus, this joint will not be a degree of freedom
    /// in the current configuration.
    pub fn set_fixed_joint(&mut self, joint_idx: usize, joint_sub_idx: usize, fixed_joint_value: f64, py: Python) {
        self.robot_configuration_module.set_fixed_joint(joint_idx, joint_sub_idx, fixed_joint_value).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

    /// Removes the given joint as a fixed joint.  Thus, this joint will become a degree of freedom.
    pub fn remove_fixed_joint(&mut self, joint_idx: usize, joint_sub_idx: usize, py: Python) {
        self.robot_configuration_module.remove_fixed_joint(joint_idx, joint_sub_idx).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

    /*
    pub fn set_mobile_base_mode(&mut self, mobile_base_mode: MobileBaseInfo, py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(mobile_base_mode).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    */

    /*
    /// Sets the mobile base mode of the robot configuration.
    pub fn set_mobile_base_mode(&mut self, mobile_base_mode: MobileBaseInfo, py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(mobile_base_mode).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

    /// sets the base offset of the robot configuration.
    pub fn set_base_offset(&mut self, p: &OptimaSE3Pose) {
        self.robot_configuration_info.base_offset = OptimaSE3PoseAll::new(p);
        return self.update();
    }
    */

    /*
    pub fn set_static_mobile_base_mode_py(&mut self, py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(BaseOfChainMobilityMode::Static).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    pub fn set_floating_mobile_base_mode_py(&mut self, x_bounds: (f64, f64), y_bounds: (f64, f64), z_bounds: (f64, f64), xr_bounds: (f64, f64), yr_bounds: (f64, f64), zr_bounds: (f64, f64), py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(BaseOfChainMobilityMode::Floating {
            x_bounds,
            y_bounds,
            z_bounds,
            xr_bounds,
            yr_bounds,
            zr_bounds
        }).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    pub fn set_planar_translation_mobile_base_mode_py(&mut self, x_bounds: (f64, f64), y_bounds: (f64, f64), py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(BaseOfChainMobilityMode::PlanarTranslation { x_bounds, y_bounds }).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    pub fn set_planar_rotation_mobile_base_mode_py(&mut self, zr_bounds: (f64, f64), py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(BaseOfChainMobilityMode::PlanarRotation { zr_bounds }).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    pub fn set_planar_translation_and_rotation_mobile_base_mode_py(&mut self, x_bounds: (f64, f64), y_bounds: (f64, f64), zr_bounds: (f64, f64), py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(BaseOfChainMobilityMode::PlanarTranslationAndRotation { x_bounds, y_bounds, zr_bounds }).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    */

    pub fn set_base_offset_py(&mut self, pose: &OptimaSE3PosePy, py: Python) {
        self.robot_configuration_module.set_base_offset(pose.pose()).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

    /// sets the base offset of the robot configuration.
    pub fn set_base_offset_euler_angles(&mut self, rx: f64, ry: f64, rz: f64, x: f64, y: f64, z: f64, py: Python) {
        self.robot_configuration_module.set_base_offset(&OptimaSE3Pose::new_unit_quaternion_and_translation_from_euler_angles(rx, ry, rz, x, y, z)).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

    /// Saves the RobotConfigurationModule to its robot's RobotConfigurationGeneratorModule.
    /// The configuration will be saved to a json file such that the RobotConfigurationGeneratorModule
    /// will be able to load this configuration in the future.
    pub fn save(&mut self, configuration_name: &str) {
        self.robot_configuration_module.save(configuration_name).expect("error");
    }
}

/// Methods supported by WASM.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotConfigurationModule {
    pub fn print_summary_wasm(&self) {
        self.print_summary();
    }
}

/// Stores information about the robot configuration that will be used to modify the configuration's
/// underlying `RobotModelModule`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotConfigurationInfo {
    contiguous_chain_infos: Vec<ContiguousChainInfo>,
    dead_end_link_idxs: Vec<usize>,
    fixed_joint_infos: Vec<FixedJointInfo>,
    base_offset: OptimaSE3PoseAll
}
impl Default for RobotConfigurationInfo {
    /// By default, we will just have the robot's given base model directly from the robot's URDF.
    fn default() -> Self {
        Self {
            contiguous_chain_infos: vec![],
            dead_end_link_idxs: vec![],
            fixed_joint_infos: vec![],
            base_offset: OptimaSE3PoseAll::new_identity()
        }
    }
}
impl RobotConfigurationInfo {
    pub fn dead_end_link_idxs(&self) -> &Vec<usize> {
        &self.dead_end_link_idxs
    }
    pub fn inactive_joint_infos(&self) -> &Vec<FixedJointInfo> {
        &self.fixed_joint_infos
    }
    pub fn base_offset(&self) -> &OptimaSE3PoseAll {
        &self.base_offset
    }
    pub fn contiguous_chain_infos(&self) -> &Vec<ContiguousChainInfo> {
        &self.contiguous_chain_infos
    }
}

/// An object that describes a fixed joint.  The joint_sub_idx refers to the index of a joint's
/// joint_axes list of `JointAxis` objects.  The fixed_joint_value will be the floating point value
/// that the given joint axis will be locked to.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixedJointInfo {
    pub joint_idx: usize,
    pub joint_sub_idx: usize,
    pub fixed_joint_value: f64
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContiguousChainInfo {
    chain_name: String,
    start_link_idx: usize,
    end_link_idx: Option<usize>,
    mobility_mode: ContiguousChainMobilityMode
}
impl ContiguousChainInfo {
    pub fn chain_name(&self) -> &str {
        &self.chain_name
    }
    pub fn start_link_idx(&self) -> usize {
        self.start_link_idx
    }
    pub fn end_link_idx(&self) -> Option<usize> {
        self.end_link_idx
    }
    pub fn mobility_mode(&self) -> &ContiguousChainMobilityMode {
        &self.mobility_mode
    }
}

/// Enum that characterizes the mobility of a robot's base.  Enum variants take as inputs boundary values
/// along any relevant dimensions.
/// Note that this enum does not implicitly handle something like differential drive constraints,
/// this would have to be handled separately.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ContiguousChainMobilityMode {
    /// The robot's base is immobile.
    Static,
    /// The robot's base is floating and can move along any spatial dimension (including rotation).
    /// This mobile base type is useful for humanoid robots where the robot's hips can move freely
    /// through space.
    Floating { x_bounds: (f64, f64), y_bounds: (f64, f64), z_bounds: (f64, f64), xr_bounds: (f64, f64), yr_bounds: (f64, f64), zr_bounds: (f64, f64) },
    /// The robot's base can translate in space along the x and y axis (assuming z is up).
    PlanarTranslation { x_bounds: (f64, f64), y_bounds: (f64, f64) },
    /// The robot's base can rotate in space along the z axis.
    PlanarRotation { zr_bounds: (f64, f64) },
    /// The robot's base can translate in space along the x and y axis and rotate along the z axis.
    PlanarTranslationAndRotation { x_bounds: (f64, f64), y_bounds: (f64, f64), zr_bounds: (f64, f64) }
}
impl ContiguousChainMobilityMode {
    pub fn new_default(t: &ContiguousChainMobilityModeType) -> Self {
        let default_translation = 5.0;
        let default_rotation = 2.0*std::f64::consts::PI;
        return match t {
            ContiguousChainMobilityModeType::Static => {
                ContiguousChainMobilityMode::Static
            }
            ContiguousChainMobilityModeType::Floating => {
                ContiguousChainMobilityMode::Floating {
                    x_bounds: (-default_translation, default_translation),
                    y_bounds: (-default_translation, default_translation),
                    z_bounds: (-default_translation, default_translation),
                    xr_bounds: (-default_rotation, default_rotation),
                    yr_bounds: (-default_rotation, default_rotation),
                    zr_bounds: (-default_rotation, default_rotation)
                }
            }
            ContiguousChainMobilityModeType::PlanarTranslation => {
                ContiguousChainMobilityMode::PlanarTranslation {
                    x_bounds: (-default_translation, default_translation),
                    y_bounds: (-default_translation, default_translation)
                }
            }
            ContiguousChainMobilityModeType::PlanarRotation => {
                ContiguousChainMobilityMode::PlanarRotation {
                    zr_bounds: (-default_rotation, default_rotation)
                }
            }
            ContiguousChainMobilityModeType::PlanarTranslationAndRotation => {
                ContiguousChainMobilityMode::PlanarTranslationAndRotation {
                    x_bounds: (-default_translation, default_translation),
                    y_bounds: (-default_translation, default_translation),
                    zr_bounds: (-default_rotation, default_rotation)
                }
            }
        }
    }
    pub fn get_bounds(&self) -> Vec<(f64, f64)> {
        match self {
            ContiguousChainMobilityMode::Static => {
                vec![]
            }
            ContiguousChainMobilityMode::Floating { x_bounds, y_bounds, z_bounds, xr_bounds, yr_bounds, zr_bounds } => {
                vec![x_bounds.clone(), y_bounds.clone(), z_bounds.clone(), xr_bounds.clone(), yr_bounds.clone(), zr_bounds.clone()]
            }
            ContiguousChainMobilityMode::PlanarTranslation { x_bounds, y_bounds } => {
                vec![ x_bounds.clone(), y_bounds.clone() ]
            }
            ContiguousChainMobilityMode::PlanarRotation { zr_bounds } => {
                vec![ zr_bounds.clone() ]
            }
            ContiguousChainMobilityMode::PlanarTranslationAndRotation { x_bounds, y_bounds, zr_bounds } => {
                vec![ x_bounds.clone(), y_bounds.clone(), zr_bounds.clone() ]
            }
        }
    }
}

/// An Enum that describes the robot base mode without any of the underlying data (e.g., bounds).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContiguousChainMobilityModeType {
    Static,
    Floating,
    PlanarTranslation,
    PlanarRotation,
    PlanarTranslationAndRotation
}

