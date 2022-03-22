#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::collections::HashMap;
use std::io;
use std::io::BufRead;
use serde::{Deserialize, Serialize};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::robot_modules::robot_configuration_module::{RobotConfigurationIdentifier, RobotConfigurationInfo, RobotConfigurationModule};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::{FileUtils, RobotDirUtils, RobotModuleJsonType};
use crate::utils::utils_robot::robot_module_utils::RobotModuleSaveAndLoad;
use crate::utils::utils_console_output::{optima_print, optima_print_new_line, PrintColor, PrintMode};

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotConfigurationGeneratorModule {
    robot_configuration_infos: HashMap<String, RobotConfigurationInfo>,
    base_robot_model_module: RobotModelModule
}
impl RobotConfigurationGeneratorModule {
    pub fn new(robot_name: &str) -> Result<Self, OptimaError> {
        let path = RobotDirUtils::get_path_to_robot_module_json(robot_name, RobotModuleJsonType::ConfigurationGeneratorModule)?;
        if path.exists() {
            let loaded_self = FileUtils::load_object_from_json_file::<Self>(&path)?;
            return Ok(loaded_self);
        } else {
            let robot_model_module = RobotModelModule::new(robot_name)?;
            let out_self = Self {
                robot_configuration_infos: Default::default(),
                base_robot_model_module: robot_model_module
            };
            out_self.save_to_json_file()?;
            Ok(out_self)
        }
    }

    pub fn save_robot_configuration_module(&mut self, robot_configuration_info: &RobotConfigurationInfo) -> Result<(), OptimaError> {
        match robot_configuration_info.configuration_identifier() {
            RobotConfigurationIdentifier::NamedConfiguration(name) => {
                let robot_configuration_info_option = self.robot_configuration_infos.get(name);
                match robot_configuration_info_option {
                    None => {
                        self.robot_configuration_infos.insert(name.clone(), robot_configuration_info.clone());

                        self.save_to_json_file()?;
                    }
                    Some(_) => {
                        optima_print(&format!("Robot configuration with name {:?} already exits.  Overwrite?  (y or n)", name), PrintMode::Println, PrintColor::Blue, true);
                        let stdin = io::stdin();
                        let line = stdin.lock().lines().next().unwrap().unwrap();
                        if &line == "y" {
                            optima_print("Configuration saved.", PrintMode::Println, PrintColor::Blue, false);
                            self.robot_configuration_infos.insert(name.clone(), robot_configuration_info.clone());
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

    pub fn delete_named_robot_configuration_module(&mut self, name: &str) -> Result<(), OptimaError> {
        let res = self.robot_configuration_infos.remove(name);
        if res.is_some() {
            optima_print(&format!("Ready to delete configuration {:?}.  Confirm?  (y or n)", name), PrintMode::Println, PrintColor::Blue, true);
            let stdin = io::stdin();
            let line = stdin.lock().lines().next().unwrap().unwrap();
            if &line == "y" {
                self.save_to_json_file()?;
            } else {
                optima_print(&format!("Did not delete configuration."), PrintMode::Println, PrintColor::Yellow, true);
            }
        } else {
            optima_print(&format!("Configuration with name {:?} does not exist.  Nothing deleted.", name), PrintMode::Println, PrintColor::Yellow, true);
        }
        Ok(())
    }

    pub fn generate_configuration_module(&self, identifier: RobotConfigurationIdentifier) -> Result<RobotConfigurationModule, OptimaError> {
        return match &identifier {
            RobotConfigurationIdentifier::BaseModel => {
                RobotConfigurationModule::new_base_model(self.base_robot_model_module.robot_name())
            }
            RobotConfigurationIdentifier::NamedConfiguration(name) => {
                let info_option = self.robot_configuration_infos.get(name);
                match info_option {
                    None => {
                        Err(OptimaError::new_generic_error_str(&format!("configuration with name {:?} does not exist.", name)))
                    }
                    Some(info) => {
                        RobotConfigurationModule::new_from_base_model_module_and_info(self.base_robot_model_module.clone(), info.clone())
                    }
                }
            }
        }
    }

    pub fn generate_named_configuration(&self, name: &str) -> Result<RobotConfigurationModule, OptimaError> {
        return self.generate_configuration_module(RobotConfigurationIdentifier::NamedConfiguration(name.to_string()));
    }

    pub fn generate_base_configuration(&self) -> Result<RobotConfigurationModule, OptimaError> {
        return self.generate_configuration_module(RobotConfigurationIdentifier::BaseModel);
    }
    
    pub fn print_saved_configurations(&self) {
        optima_print("Saved Configurations ---> ", PrintMode::Println, PrintColor::Blue, true);
        for (k, _) in &self.robot_configuration_infos {
            optima_print(&format!("   {}", k), PrintMode::Println, PrintColor::None, false);
        }
        optima_print_new_line();
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

}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotConfigurationGeneratorModule {

}

