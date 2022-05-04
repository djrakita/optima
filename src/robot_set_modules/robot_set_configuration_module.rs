#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::{RobotConfigurationModule, RobotConfigurationModulePy};
use crate::utils::utils_console::{ConsoleInputUtils, optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaStemCellPath};
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_traits::SaveAndLoadable;

/// The robot set analogue of the `RobotConfigurationModule`.  Multiple robot configurations
/// can be added and configured in the set of robots here, then can be saved to a file for loading
/// at a later time.  This is a relatively simple wrapper around multiple `RobotConfigurationModule`
/// objects that will be used to initialize many other robot set modules.
///
/// # Example
/// ```
/// use optima::robot_modules::robot_configuration_module::{ContiguousChainMobilityMode, RobotConfigurationModule};
/// use optima::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
/// use optima::utils::utils_robot::robot_module_utils::RobotNames;
/// use optima::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
///
/// let mut r = RobotSetConfigurationModule::new_empty();
///
/// r.add_robot_configuration_from_names(RobotNames::new_base("ur5"))?;
/// let mut sawyer_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("sawyer"))?;
/// sawyer_configuration.set_mobile_base_mode(ContiguousChainMobilityMode::PlanarTranslation {x_bounds: (-2.0, 2.0),y_bounds: (-2.0, 2.0)})?;
/// sawyer_configuration.set_base_offset(&OptimaSE3Pose::new_from_euler_angles(0.,0.,0.,1.0,0.,0., &OptimaSE3PoseType::ImplicitDualQuaternion))?;
/// r.add_robot_configuration(sawyer_configuration)?;
///
/// r.save_robot_set_configuration_module("test_set")?;
/// ```
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotSetConfigurationModule {
    robot_configuration_modules: Vec<RobotConfigurationModule>
}
impl RobotSetConfigurationModule {
    pub fn new_empty() -> Self {
        Self {
            robot_configuration_modules: vec![]
        }
    }
    pub fn new_from_set_name(set_name: &str) -> Result<Self, OptimaError> {
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotSet { set_name: set_name.to_string() });
        OptimaError::new_check_for_stem_cell_path_does_not_exist(&path, file!(), line!())?;
        path.append(&("robot_set_configuration_module.JSON"));
        OptimaError::new_check_for_stem_cell_path_does_not_exist(&path, file!(), line!())?;
        return Self::load_from_path(&path);
    }
    pub fn add_robot_configuration_from_names(&mut self, robot_names: RobotNames) -> Result<(), OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_names)?;
        return self.add_robot_configuration(robot_configuration_module);
    }
    pub fn add_robot_configuration(&mut self, robot_configuration: RobotConfigurationModule) -> Result<(), OptimaError> {
        self.robot_configuration_modules.push(robot_configuration);
        Ok(())
    }
    pub fn robot_configuration_modules(&self) -> &Vec<RobotConfigurationModule> {
        &self.robot_configuration_modules
    }
    pub fn robot_configuration_module(&self, idx: usize) -> Result<&RobotConfigurationModule, OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(idx, self.robot_configuration_modules.len(), file!(), line!())?;
        Ok(&self.robot_configuration_modules[idx])
    }
    /// Robot set configurations are saved to the optima_assets/optima_robot_sets directory.
    pub fn save_robot_set_configuration_module(&self, set_name: &str) -> Result<(), OptimaError> {
        if self.robot_configuration_modules.len() <= 1 {
            optima_print("WARNING: Cannot save RobotSetConfigurationModule with <= 1 robot.", PrintMode::Println, PrintColor::Yellow, true);
            return Ok(());
        }

        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotSet { set_name: set_name.to_string() });
        if path.exists() {
            let response = ConsoleInputUtils::get_console_input_string(&format!("Robot set with name {} already exists.  Overwrite?  (y or n)", set_name), PrintColor::Cyan)?;
            if response == "y" {
                path.delete_all_items_in_directory()?;
                path.append(&("robot_set_configuration_module.JSON"));
                self.save_to_path(&path)?;
            } else {
                return Ok(());
            }
        } else {
            path.append(&("robot_set_configuration_module.JSON"));
            self.save_to_path(&path)?;
        }

        Ok(())
    }
}
impl SaveAndLoadable for RobotSetConfigurationModule {
    type SaveType = Vec<String>;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        self.robot_configuration_modules.get_save_serialization_object()
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load = Vec::load_from_json_string(&json_str)?;

        Ok(Self {
            robot_configuration_modules: load
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotSetConfigurationModule {
    #[staticmethod]
    pub fn new_empty_py() -> Self {
        Self::new_empty()
    }
    #[staticmethod]
    pub fn new_from_set_name_py(set_name: &str) -> Self { Self::new_from_set_name(set_name).expect("error") }
    pub fn add_robot_configuration_from_names_py(&mut self, robot_name: &str, configuration_name: Option<&str>) {
        let robot_names = RobotNames::new(robot_name, configuration_name);
        self.add_robot_configuration_from_names(robot_names).expect("error");
    }
    pub fn add_robot_configuration_py(&mut self, robot_configuration: RobotConfigurationModulePy) {
        self.robot_configuration_modules.push(robot_configuration.robot_configuration_module);
    }
    pub fn save_robot_set_configuration_module_py(&self, set_name: &str) {
        self.save_robot_set_configuration_module(set_name).expect("error");
    }
    pub fn num_robot_configurations(&self) -> usize {
        return self.robot_configuration_modules.len();
    }
    pub fn robot_configuration_modules_py(&self, py: Python) -> Vec<RobotConfigurationModulePy> {
        let mut out_vec = vec![];

        for c in &self.robot_configuration_modules {
            out_vec.push(RobotConfigurationModulePy::new_from_configuration_module(c.clone(), py));
        }

        out_vec
    }
}


