use serde::{Serialize};
use serde::de::DeserializeOwned;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaStemCellPath, RobotModuleJsonType};

/// Convenience struct that groups together utility functions for robot modules.
pub struct RobotModuleUtils;
impl RobotModuleUtils {
    pub fn save_to_json_file_generic<T: Serialize>(save_obj: &T, robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<(), OptimaError> {
        let mut o = OptimaStemCellPath::new_asset_path()?;
        o.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: robot_module_json_type });
        return o.save_object_to_file_as_json(save_obj);
    }
    pub fn load_from_json_file_generic<T: DeserializeOwned>(robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<T, OptimaError> {
        let mut o = OptimaStemCellPath::new_asset_path()?;
        o.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: robot_module_json_type });
        return o.load_object_from_json_file();
    }
}

/// Trait that can be implemented by robot modules to allow for easy serializing and deserializing
/// to and from json files.  The only functions that must be implemented for this trait are
/// get_robot_name() and get_robot_module_json_type().
pub trait RobotModuleSaveAndLoad where Self: Serialize + DeserializeOwned {
    fn get_robot_name(&self) -> &str;
    fn save_to_json_file(&self, robot_module_json_type: RobotModuleJsonType) -> Result<(), OptimaError> where Self: Sized {
        return RobotModuleUtils::save_to_json_file_generic(self, self.get_robot_name(), robot_module_json_type);
    }
    fn load_from_json_file(robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<Self, OptimaError>  {
        return RobotModuleUtils::load_from_json_file_generic::<Self>(robot_name, robot_module_json_type);
    }
}

/// Used to initialize robot modules.
#[derive(Clone, Debug)]
pub struct RobotNames<'a> {
    robot_name: &'a str,
    configuration_name: Option<&'a str>
}
impl <'a> RobotNames<'a> {
    pub fn new_base(robot_name: &'a str) -> Self {
        Self {
            robot_name,
            configuration_name: None
        }
    }
    pub fn new(robot_name: &'a str, configuration_name: Option<&'a str>) -> Self {
        Self {
            robot_name,
            configuration_name
        }
    }
    pub fn robot_name(&self) -> &'a str {
        self.robot_name
    }
    pub fn configuration_name(&self) -> Option<&'a str> {
        self.configuration_name
    }
}