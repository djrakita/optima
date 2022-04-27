#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_console::{ConsoleInputUtils, PrintColor};
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

        robot_model_module.add_mobile_base_link_and_joint(&robot_configuration_info.mobile_base_mode);

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
    /// Sets the mobile base mode of the robot configuration.
    pub fn set_mobile_base_mode(&mut self, mobile_base_mode: MobileBaseInfo) -> Result<(), OptimaError> {
        self.robot_configuration_info.mobile_base_mode = mobile_base_mode;
        return self.update();
    }
    /// sets the base offset of the robot configuration.
    pub fn set_base_offset(&mut self, p: &OptimaSE3Pose) -> Result<(), OptimaError> {
        self.robot_configuration_info.base_offset = OptimaSE3PoseAll::new(p);
        return self.update();
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
                path.save_object_to_file_as_json(&self.robot_configuration_info)?;
            } else {
                return Ok(());
            }
        } else {
            path.save_object_to_file_as_json(&self.robot_configuration_info)?;
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

    pub fn set_static_mobile_base_mode_py(&mut self, py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(MobileBaseInfo::Static).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    pub fn set_floating_mobile_base_mode_py(&mut self, x_bounds: (f64, f64), y_bounds: (f64, f64), z_bounds: (f64, f64), xr_bounds: (f64, f64), yr_bounds: (f64, f64), zr_bounds: (f64, f64), py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(MobileBaseInfo::Floating {
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
        self.robot_configuration_module.set_mobile_base_mode(MobileBaseInfo::PlanarTranslation { x_bounds, y_bounds }).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    pub fn set_planar_rotation_mobile_base_mode_py(&mut self, zr_bounds: (f64, f64), py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(MobileBaseInfo::PlanarRotation { zr_bounds }).expect("error");
        self.copy_robot_model_module_to_py(py);
    }
    pub fn set_planar_translation_and_rotation_mobile_base_mode_py(&mut self, x_bounds: (f64, f64), y_bounds: (f64, f64), zr_bounds: (f64, f64), py: Python) {
        self.robot_configuration_module.set_mobile_base_mode(MobileBaseInfo::PlanarTranslationAndRotation { x_bounds, y_bounds, zr_bounds }).expect("error");
        self.copy_robot_model_module_to_py(py);
    }

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
    dead_end_link_idxs: Vec<usize>,
    fixed_joint_infos: Vec<FixedJointInfo>,
    base_offset: OptimaSE3PoseAll,
    mobile_base_mode: MobileBaseInfo
}
impl Default for RobotConfigurationInfo {
    /// By default, we will just have the robot's given base model directly from the robot's URDF.
    fn default() -> Self {
        Self {
            dead_end_link_idxs: vec![],
            fixed_joint_infos: vec![],
            base_offset: OptimaSE3PoseAll::new_identity(),
            mobile_base_mode: MobileBaseInfo::Static
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
    pub fn mobile_base_mode(&self) -> &MobileBaseInfo {
        &self.mobile_base_mode
    }
}

/// An Enum used to identify a given configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotConfigurationIdentifier {
    BaseModel,
    NamedConfiguration(String)
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

/// Enum that characterizes the mobility of a robot's base.  Enum variants take as inputs boundary values
/// along any relevant dimensions.
/// Note that this enum does not implicitly handle something like differential drive constraints,
/// this would have to be handled separately.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MobileBaseInfo {
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
impl MobileBaseInfo {
    pub fn new_default(t: &MobileBaseType) -> Self {
        let default_translation = 5.0;
        let default_rotation = 2.0*std::f64::consts::PI;
        return match t {
            MobileBaseType::Static => {
                MobileBaseInfo::Static
            }
            MobileBaseType::Floating => {
                MobileBaseInfo::Floating {
                    x_bounds: (-default_translation, default_translation),
                    y_bounds: (-default_translation, default_translation),
                    z_bounds: (-default_translation, default_translation),
                    xr_bounds: (-default_rotation, default_rotation),
                    yr_bounds: (-default_rotation, default_rotation),
                    zr_bounds: (-default_rotation, default_rotation)
                }
            }
            MobileBaseType::PlanarTranslation => {
                MobileBaseInfo::PlanarTranslation {
                    x_bounds: (-default_translation, default_translation),
                    y_bounds: (-default_translation, default_translation)
                }
            }
            MobileBaseType::PlanarRotation => {
                MobileBaseInfo::PlanarRotation {
                    zr_bounds: (-default_rotation, default_rotation)
                }
            }
            MobileBaseType::PlanarTranslationAndRotation => {
                MobileBaseInfo::PlanarTranslationAndRotation {
                    x_bounds: (-default_translation, default_translation),
                    y_bounds: (-default_translation, default_translation),
                    zr_bounds: (-default_rotation, default_rotation)
                }
            }
        }
    }
    pub fn get_bounds(&self) -> Vec<(f64, f64)> {
        match self {
            MobileBaseInfo::Static => {
                vec![]
            }
            MobileBaseInfo::Floating { x_bounds, y_bounds, z_bounds, xr_bounds, yr_bounds, zr_bounds } => {
                vec![x_bounds.clone(), y_bounds.clone(), z_bounds.clone(), xr_bounds.clone(), yr_bounds.clone(), zr_bounds.clone()]
            }
            MobileBaseInfo::PlanarTranslation { x_bounds, y_bounds } => {
                vec![ x_bounds.clone(), y_bounds.clone() ]
            }
            MobileBaseInfo::PlanarRotation { zr_bounds } => {
                vec![ zr_bounds.clone() ]
            }
            MobileBaseInfo::PlanarTranslationAndRotation { x_bounds, y_bounds, zr_bounds } => {
                vec![ x_bounds.clone(), y_bounds.clone(), zr_bounds.clone() ]
            }
        }
    }
}

/// An Enum that describes the robot base mode without any of the underlying data (e.g., bounds).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MobileBaseType {
    Static,
    Floating,
    PlanarTranslation,
    PlanarRotation,
    PlanarTranslationAndRotation
}

