use serde::{Serialize};
use serde::de::DeserializeOwned;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::{FileUtils, RobotDirUtils, RobotModuleJsonType};

/// Convenience struct that groups together utility functions for robot modules.
pub struct RobotModuleUtils;
impl RobotModuleUtils {
    fn save_to_json_file_generic<T: Serialize>(save_obj: &T, robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<(), OptimaError> {
        let p = RobotDirUtils::get_path_to_robot_module_json(robot_name, robot_module_json_type)?;
        FileUtils::save_object_to_file_as_json(save_obj, &p)?;
        Ok(())
    }

    fn new_load_from_json_file_generic<T: DeserializeOwned>(robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<T, OptimaError> {
        let p = RobotDirUtils::get_path_to_robot_module_json(robot_name, robot_module_json_type)?;
        let json_string = FileUtils::read_file_contents_to_string(&p)?;
        return Self::new_load_from_json_string_generic(&json_string);
    }

    pub fn new_load_from_json_string_generic<T: DeserializeOwned>(json_string: &str) -> Result<T, OptimaError> {
        FileUtils::load_object_from_json_string::<T>(json_string)
    }
}

/// Trait that can be implemented by robot modules to allow for easy serializing and deserializing
/// to and from json files.  The only functions that must be implemented for this trait are
/// get_robot_name() and get_robot_module_json_type().
pub trait RobotModuleSaveAndLoad where Self: Serialize + DeserializeOwned {
    fn get_robot_name(&self) -> &str;
    fn get_robot_module_json_type(&self) -> RobotModuleJsonType;
    fn save_to_json_file(&self) -> Result<(), OptimaError> where Self: Sized {
        return RobotModuleUtils::save_to_json_file_generic(self, self.get_robot_name(), self.get_robot_module_json_type());
    }
    fn new_from_json_file(robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<Self, OptimaError>  {
        return RobotModuleUtils::new_load_from_json_file_generic::<Self>(robot_name, robot_module_json_type);
    }
    fn new_load_from_json_string(json_string: &str) -> Result<Self, OptimaError> {
        return RobotModuleUtils::new_load_from_json_string_generic::<Self>(json_string);
    }
}