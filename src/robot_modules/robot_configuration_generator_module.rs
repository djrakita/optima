#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::robot_modules::robot_configuration_module::RobotConfigurationModulePy;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::io;
use std::io::BufRead;
use serde::{Deserialize, Serialize};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::robot_modules::robot_configuration_module::{RobotConfigurationIdentifier, RobotConfigurationInfo, RobotConfigurationModule};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_robot::robot_module_utils::RobotModuleSaveAndLoad;
use crate::utils::utils_console_output::{optima_print, optima_print_new_line, PrintColor, PrintMode};
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaStemCellPath, RobotModuleJsonType};

/// The `RobotConfigurationGeneratorModule` is used to generate `RobotConfigurationModule`s.  This generator
/// object can be easily stored on the end user's computer in a JSON file, meaning configurations
/// can be saved by name and re-generated at a later time.  It is recommended to always use the
/// `RobotConfigurationGeneratorModule` to generate `RobotConfigurationModule`s, even when just
/// generating a default base model configuration.
///
/// # Example
/// ```
/// use optima::robot_modules::robot_configuration_generator_module::RobotConfigurationGeneratorModule;
/// use optima::robot_modules::robot_configuration_module::MobileBaseInfo;
///
/// // initialize a generator for the ur5 robot
/// let generator = RobotConfigurationGeneratorModule::new("ur5").expect("error");
/// // generator an initial configuration (just the base model configuration)
/// let mut initial_configuration = generator.generate_base_configuration().expect("error");
/// // change the initial configuration to have a planar rotation mobile base
/// initial_configuration.set_mobile_base_mode(MobileBaseInfo::PlanarRotation {zr_bounds: (-3.14, 3.14)}).expect("error");
/// // save the modified configuration to the ur5's RobotConfigurationGeneratorModule.
/// initial_configuration.save("test_configuration");
///
/// // (At a future session).  Load in the previously saved configuration.
/// let generator = RobotConfigurationGeneratorModule::new("ur5").expect("error");
/// let loaded_configuration = generator.generate_named_configuration("test_configuration").expect("error");
/// ```
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotConfigurationGeneratorModule {
    robot_configuration_infos: Vec<RobotConfigurationInfo>,
    base_robot_model_module: RobotModelModule
}
impl RobotConfigurationGeneratorModule {
    /// Initializes the RobotConfigurationGeneratorModule corresponding to the given robot.

    pub fn new(robot_name: &str) -> Result<Self, OptimaError> {
        let mut p = OptimaStemCellPath::new_asset_path()?;
        p.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ConfigurationGeneratorModule });
        if p.exists() {
            let loaded_self = p.load_object_from_json_file();
            return loaded_self;
        } else {
            let robot_model_module = RobotModelModule::new(robot_name)?;
            let out_self = Self {
                robot_configuration_infos: vec![],
                base_robot_model_module: robot_model_module
            };
            out_self.save_to_json_file()?;
            Ok(out_self)
        }
    }


    /// A function that is automatically called by RobotConfigurationModule.
    /// Should not need to be called by end user.
    pub fn save_robot_configuration_module(&mut self, robot_configuration_info: &RobotConfigurationInfo) -> Result<(), OptimaError> {
        match robot_configuration_info.configuration_identifier() {
            RobotConfigurationIdentifier::NamedConfiguration(name) => {
                let idx_option = self.get_idx(name);
                match &idx_option {
                    None => {
                        self.robot_configuration_infos.push(robot_configuration_info.clone());
                        self.save_to_json_file()?;
                    }
                    Some(idx) => {
                        optima_print(&format!("Robot configuration with name {:?} already exits.  Overwrite?  (y or n)", name), PrintMode::Println, PrintColor::Blue, true);
                        let stdin = io::stdin();
                        let line = stdin.lock().lines().next().unwrap().unwrap();
                        if &line == "y" {
                            optima_print("Configuration saved.", PrintMode::Println, PrintColor::Blue, false);
                            self.robot_configuration_infos[*idx] = robot_configuration_info.clone();
                            self.save_to_json_file()?;
                        } else {
                            optima_print("Configuration not saved.", PrintMode::Println, PrintColor::Yellow, true);
                        }

                        self.save_to_json_file()?;
                    }
                }
            }
            _ => {  }
        }
        Ok(())
    }

    /// Deletes a named robot configuration module from the generator.  Will post a confirmation
    /// message in the console to verify decision.
    pub fn delete_named_robot_configuration_module(&mut self, name: &str) -> Result<(), OptimaError> {
        let idx_option = self.get_idx(name);
        match &idx_option {
            None => {
                optima_print(&format!("Configuration with name {:?} does not exist.  Nothing deleted.", name), PrintMode::Println, PrintColor::Yellow, true);
            }
            Some(idx) => {
                optima_print(&format!("Ready to delete configuration {:?}.  Confirm?  (y or n)", name), PrintMode::Println, PrintColor::Blue, true);
                let stdin = io::stdin();
                let line = stdin.lock().lines().next().unwrap().unwrap();
                if &line == "y" {
                    self.robot_configuration_infos.remove(*idx);
                    self.save_to_json_file()?;
                    optima_print("Configuration removed.", PrintMode::Println, PrintColor::Blue, true);
                } else {
                    optima_print(&format!("Did not delete configuration."), PrintMode::Println, PrintColor::Yellow, true);
                }
            }
        }

        Ok(())
    }

    fn generate_configuration_module(&self, identifier: RobotConfigurationIdentifier) -> Result<RobotConfigurationModule, OptimaError> {
        return match &identifier {
            RobotConfigurationIdentifier::BaseModel => {
                RobotConfigurationModule::new_base_model_from_absolute_paths(self.base_robot_model_module.robot_name())
            }
            RobotConfigurationIdentifier::NamedConfiguration(name) => {
                // let info_option = self.robot_configuration_infos.get(name);
                let idx_option = self.get_idx(name);
                match &idx_option {
                    None => {
                        Err(OptimaError::new_generic_error_str(&format!("configuration with name {:?} does not exist.", name)))
                    }
                    Some(idx) => {
                        let info = &self.robot_configuration_infos[*idx];
                        RobotConfigurationModule::new_from_base_model_module_and_info(self.base_robot_model_module.clone(), info.clone())
                    }
                }
            }
        }
    }

    /// Generates a named configuration.
    pub fn generate_named_configuration(&self, name: &str) -> Result<RobotConfigurationModule, OptimaError> {
        return self.generate_configuration_module(RobotConfigurationIdentifier::NamedConfiguration(name.to_string()));
    }

    /// Generates the given robot's base configuration (straight from the URDF).
    pub fn generate_base_configuration(&self) -> Result<RobotConfigurationModule, OptimaError> {
        return RobotConfigurationModule::new_from_base_model_module_and_info(self.base_robot_model_module.clone(), RobotConfigurationInfo::default());
    }

    /// Prints the names of all saved configurations in this generator.
    pub fn print_saved_configurations(&self) {
        optima_print("Saved Configurations ---> ", PrintMode::Println, PrintColor::Blue, true);
        for r in &self.robot_configuration_infos {
            optima_print(&format!("   {:?}", r.configuration_identifier()), PrintMode::Println, PrintColor::None, false);
        }
        optima_print_new_line();
    }

    fn get_idx(&self, name: &str) -> Option<usize> {
        for (i, c) in self.robot_configuration_infos.iter().enumerate() {
            match c.configuration_identifier() {
                RobotConfigurationIdentifier::NamedConfiguration(n) => {
                    if n == name { return Some(i) }
                }
                _ => { }
            }
        }
        return None;
    }
}
impl RobotModuleSaveAndLoad for RobotConfigurationGeneratorModule {
    fn get_robot_name(&self) -> &str {
        self.base_robot_model_module.robot_name()
    }
    fn get_robot_module_json_type(&self) -> RobotModuleJsonType {
        RobotModuleJsonType::ConfigurationGeneratorModule
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotConfigurationGeneratorModule {
    #[new]
    pub fn new_py(robot_name: &str) -> Self {
        Self::new(robot_name).expect("error")
    }

    pub fn generate_named_configuration_py(&self, name: &str, py: Python) -> PyResult<RobotConfigurationModulePy> {
        let robot_configuration_module = self.generate_named_configuration(name).expect("error");
        return Ok(RobotConfigurationModulePy::new_from_configuration_module(robot_configuration_module, py));
    }

    pub fn generate_base_configuration_py(&self, py: Python) -> PyResult<RobotConfigurationModulePy> {
        let robot_configuration_module = self.generate_base_configuration().expect("error");
        return Ok(RobotConfigurationModulePy::new_from_configuration_module(robot_configuration_module, py));
    }

    pub fn print_saved_configurations_py(&self) {
        self.print_saved_configurations();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RobotConfigurationModulePy {
    #[cfg(not(target_arch = "wasm32"))]
    fn new_from_configuration_module(robot_configuration_module: RobotConfigurationModule, py: Python) -> Self {
        let robot_model_module_py = Py::new(py, robot_configuration_module.robot_model_module().clone()).expect("error");
        Self {
            robot_configuration_module,
            robot_model_module_py
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotConfigurationGeneratorModule {

    #[wasm_bindgen(constructor)]
    pub fn new_wasm(json_string: &str) -> Self {
        Self::new_load_from_json_string(json_string).expect("error")
    }

    /// Generates a named configuration.
    pub fn generate_named_configuration_wasm(&self, name: &str) -> RobotConfigurationModule {
        let out = self.generate_named_configuration(name).expect("error");
        return out;
    }

    /// Generates the given robot's base configuration (straight from the URDF).
    pub fn generate_base_configuration_wasm(&self) -> RobotConfigurationModule {
        return self.generate_base_configuration().expect("error");
    }
}

