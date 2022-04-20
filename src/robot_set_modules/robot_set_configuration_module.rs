use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::utils::utils_console::{ConsoleInputUtils, optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaStemCellPath};
use crate::utils::utils_robot::robot_module_utils::RobotNames;

/// The robot set analogue of the `RobotConfigurationModule`.  Multiple robot configurations
/// can be added and configured in the set of robots here, then can be saved to disk for loading
/// at a later time.  This is a relatively simple wrapper around multiple `RobotConfigurationModule`
/// objects that will be used to initialize many other robot set modules.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
        return path.load_object_from_json_file();
    }
    pub fn add_robot_configuration_from_names(&mut self, robot_names: RobotNames) -> Result<(), OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_names)?;
        self.robot_configuration_modules.push(robot_configuration_module);
        Ok(())
    }
    pub fn robot_configuration_modules(&self) -> &Vec<RobotConfigurationModule> {
        &self.robot_configuration_modules
    }
    pub fn robot_configuration_modules_mut(&mut self) -> &mut Vec<RobotConfigurationModule> {
        &mut self.robot_configuration_modules
    }
    pub fn robot_configuration_module(&self, idx: usize) -> Result<&RobotConfigurationModule, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(idx, self.robot_configuration_modules.len(), file!(), line!())?;
        Ok(&self.robot_configuration_modules[idx])
    }
    pub fn robot_configuration_module_mut(&mut self, idx: usize) -> Result<&mut RobotConfigurationModule, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(idx, self.robot_configuration_modules.len(), file!(), line!())?;
        Ok(&mut self.robot_configuration_modules[idx])
    }
    pub fn save(&self, set_name: &str) -> Result<(), OptimaError> {
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
                path.save_object_to_file_as_json(&self)?;
            } else {
                return Ok(());
            }
        } else {
            path.append(&("robot_set_configuration_module.JSON"));
            path.save_object_to_file_as_json(&self)?;
        }

        Ok(())
    }
}