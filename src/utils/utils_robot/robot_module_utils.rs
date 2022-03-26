use serde::{Serialize};
use serde::de::DeserializeOwned;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaAssetLocation, OptimaStemCellPath, RobotModuleJsonType};

/// Convenience struct that groups together utility functions for robot modules.
pub struct RobotModuleUtils;
impl RobotModuleUtils {
    fn save_to_json_file_generic<T: Serialize>(save_obj: &T, robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<(), OptimaError> {
        let mut o = OptimaStemCellPath::new_asset_path()?;
        o.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: robot_module_json_type });
        return o.save_object_to_file_as_json(save_obj);
    }

    fn new_load_from_json_file_generic<T: DeserializeOwned>(robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<T, OptimaError> {
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
    fn get_robot_module_json_type(&self) -> RobotModuleJsonType;
    fn save_to_json_file(&self) -> Result<(), OptimaError> where Self: Sized {
        return RobotModuleUtils::save_to_json_file_generic(self, self.get_robot_name(), self.get_robot_module_json_type());
    }
    fn new_from_json_file(robot_name: &str, robot_module_json_type: RobotModuleJsonType) -> Result<Self, OptimaError>  {
        return RobotModuleUtils::new_load_from_json_file_generic::<Self>(robot_name, robot_module_json_type);
    }
    fn new_load_from_json_string(json_string: &str) -> Result<Self, OptimaError> {
        load_object_from_json_string::<Self>(json_string)
    }
}