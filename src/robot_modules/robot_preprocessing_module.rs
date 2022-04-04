#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::utils::utils_console::{ConsoleInputUtils, optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_generator_module::RobotConfigurationGeneratorModule;
use crate::robot_modules::robot_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaPathMatchingPattern, OptimaPathMatchingStopCondition, OptimaStemCellPath, RobotModuleJsonType};
use crate::utils::utils_robot::robot_module_utils::RobotModuleSaveAndLoad;

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotPreprocessingModule {
    pub replace_robot_model_module_json: bool,
    pub replace_configuration_generator_module_json: bool,
    pub replace_robot_link_convex_shapes: bool,
    pub replace_robot_link_convex_shape_subcomponents: bool
}
impl RobotPreprocessingModule {
    pub fn preprocess_robot_from_console_input(robot_name: &str) -> Result<(), OptimaError> {
        optima_print("Welcome to the Optima Robot Preprocessing Module.", PrintMode::Println, PrintColor::Blue, true);
        let line = ConsoleInputUtils::get_console_input_string("Use all default options?  (y or n)", PrintColor::Cyan)?;
        let default_options = if &line == "y" { true } else { false };
        if default_options { return Self::default().preprocess_robot(robot_name); }

        let line = ConsoleInputUtils::get_console_input_string("Replace robot model module?  (y or n)", PrintColor::Cyan)?;
        let replace_robot_model_module_json = if &line == "y" { true } else { false };
        let line = ConsoleInputUtils::get_console_input_string("Replace robot configuration generator module?  (y or n)", PrintColor::Cyan)?;
        let replace_configuration_generator_module_json = if &line == "y" { true } else { false };
        let line = ConsoleInputUtils::get_console_input_string("Replace robot link convex shapes?  (y or n)", PrintColor::Cyan)?;
        let replace_robot_link_convex_shapes = if &line == "y" { true } else { false };
        let line = ConsoleInputUtils::get_console_input_string("Replace robot link convex subcomponents?  (y or n)", PrintColor::Cyan)?;
        let replace_robot_link_convex_shape_subcomponents = if &line == "y" { true } else { false };

        return Self {
            replace_robot_model_module_json,
            replace_configuration_generator_module_json,
            replace_robot_link_convex_shapes,
            replace_robot_link_convex_shape_subcomponents
        }.preprocess_robot(robot_name);
    }

    pub fn preprocess_robot(&self, robot_name: &str) -> Result<(), OptimaError> {
        if cfg!(feature = "only_use_embedded_assets") {
            return Err(OptimaError::new_unsupported_operation_error("preprocess_robot", "Cannot preprocess robot using only_use_embedded_assets feature.", file!(), line!()));
        }

        self.preprocess_robot_model_module_json(robot_name)?;
        self.preprocess_robot_configuration_generator_module_json(robot_name)?;
        self.copy_link_meshes_to_assets_folder(robot_name)?;
        self.preprocess_robot_link_meshes(robot_name)?;
        self.preprocess_robot_link_convex_shapes(robot_name)?;
        self.preprocess_robot_link_convex_shape_subcomponents(robot_name)?;

        Ok(())
    }

    fn preprocess_robot_model_module_json(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut file_path = OptimaStemCellPath::new_asset_path()?;
        file_path.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ModelModule });
        if !file_path.exists() || self.replace_robot_model_module_json {
            optima_print("Preprocessing robot model module...", PrintMode::Println, PrintColor::Cyan, false);
            file_path.delete_file()?;

            let robot_model_module = RobotModelModule::new(robot_name)?;
            robot_model_module.save_to_json_file()?;

            optima_print("Successfully preprocessed robot model module!", PrintMode::Println, PrintColor::Green, false);
        }
        Ok(())
    }

    fn preprocess_robot_configuration_generator_module_json(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut file_path = OptimaStemCellPath::new_asset_path()?;
        file_path.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ConfigurationGeneratorModule });
        if !file_path.exists() || self.replace_configuration_generator_module_json {
            optima_print("Preprocessing robot configuration generator module...", PrintMode::Println, PrintColor::Cyan, false);
            file_path.delete_file()?;

            let robot_configuration_generator_module = RobotConfigurationGeneratorModule::new(robot_name)?;
            robot_configuration_generator_module.save_to_json_file()?;

            optima_print("Successfully preprocessed robot configuration generator module!", PrintMode::Println, PrintColor::Green, false);
        }
        Ok(())
    }

    fn copy_link_meshes_to_assets_folder(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut file_path = OptimaStemCellPath::new_asset_path()?;
        file_path.append_file_location(&OptimaAssetLocation::RobotInputMeshes { robot_name: robot_name.to_string() });
        let file_names = file_path.get_all_items_in_directory(false, false);
        if !file_path.exists() || file_names.len() == 0 {
            optima_print("Copying link meshes to assets folder...", PrintMode::Println, PrintColor::Cyan, false);
            let robot_model_module = RobotModelModule::new(robot_name)?;
            let mesh_file_manager_module = RobotMeshFileManagerModule::new(&robot_model_module)?;

            mesh_file_manager_module.find_and_copy_visual_meshes_to_assets()?;

            optima_print("Successfully copied link meshes to assets folder!", PrintMode::Println, PrintColor::Green, false);
        }
        Ok(())
    }

    fn preprocess_robot_link_meshes(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut directory_path = OptimaStemCellPath::new_asset_path()?;
        directory_path.append_file_location(&OptimaAssetLocation::RobotMeshes { robot_name: robot_name.to_string() });
        // let files_in_directory = directory_path.get_all_items_in_directory(false, false);
        optima_print("Preprocessing robot link meshes...", PrintMode::Println, PrintColor::Cyan, false);
        directory_path.delete_all_items_in_directory()?;

        let mut base_meshes_directory_path = OptimaStemCellPath::new_asset_path()?;
        base_meshes_directory_path.append_file_location(&OptimaAssetLocation::RobotInputMeshes {robot_name: robot_name.to_string()});

        let robot_model_module = RobotModelModule::new(robot_name)?;
        let links = robot_model_module.links();

        for (i, link) in links.iter().enumerate() {
            let has_visual_mesh = link.urdf_link().visual_mesh_filename().is_some();
            if has_visual_mesh {
                let res = base_meshes_directory_path.walk_directory_and_match(OptimaPathMatchingPattern::PathComponentsWithoutExtension(vec![format!("{}", i)]), OptimaPathMatchingStopCondition::First);
                let optima_path = res[0].clone();
                let trimesh = optima_path.load_file_to_trimesh_engine()?;
                optima_print(&format!("   > computed mesh for link {:?}", i), PrintMode::Println, PrintColor::None, false);

                let mut directory_path_copy = directory_path.clone();
                directory_path_copy.append(&format!("{}.stl", i));
                directory_path_copy.save_trimesh_engine_to_stl(&trimesh)?;
            }
        }
        optima_print("Successfully preprocessed robot link meshes!", PrintMode::Println, PrintColor::Green, false);
        Ok(())
    }

    fn preprocess_robot_link_convex_shapes(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut directory_path = OptimaStemCellPath::new_asset_path()?;
        directory_path.append_file_location(&OptimaAssetLocation::RobotConvexShapes { robot_name: robot_name.to_string() });
        let files_in_directory = directory_path.get_all_items_in_directory(false, false);
        if !directory_path.exists() || files_in_directory.len() == 0 || self.replace_robot_link_convex_shapes {
            optima_print("Preprocessing robot link convex shapes...", PrintMode::Println, PrintColor::Cyan, false);
            directory_path.delete_all_items_in_directory()?;

            let mut base_meshes_directory_path = OptimaStemCellPath::new_asset_path()?;
            base_meshes_directory_path.append_file_location(&OptimaAssetLocation::RobotInputMeshes {robot_name: robot_name.to_string()});

            let robot_model_module = RobotModelModule::new(robot_name)?;
            let links = robot_model_module.links();

            for (i, link) in links.iter().enumerate() {
                let has_visual_mesh = link.urdf_link().visual_mesh_filename().is_some();
                if has_visual_mesh {
                    let res = base_meshes_directory_path.walk_directory_and_match(OptimaPathMatchingPattern::PathComponentsWithoutExtension(vec![format!("{}", i)]), OptimaPathMatchingStopCondition::First);
                    let optima_path = res[0].clone();
                    let trimesh = optima_path.load_file_to_trimesh_engine()?;
                    let convex_hull = trimesh.compute_convex_hull();
                    optima_print(&format!("   > computed convex hull for link {:?}", i), PrintMode::Println, PrintColor::None, false);

                    let mut directory_path_copy = directory_path.clone();
                    directory_path_copy.append(&format!("{}.stl", i));
                    directory_path_copy.save_trimesh_engine_to_stl(&convex_hull)?;
                }
            }
            optima_print("Successfully preprocessed robot link convex shapes!", PrintMode::Println, PrintColor::Green, false);
        }
        Ok(())
    }

    fn preprocess_robot_link_convex_shape_subcomponents(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut directory_path = OptimaStemCellPath::new_asset_path()?;
        directory_path.append_file_location(&OptimaAssetLocation::RobotConvexSubcomponents { robot_name: robot_name.to_string() });
        let files_in_directory = directory_path.get_all_items_in_directory(false, false);
        if !directory_path.exists() || files_in_directory.len() == 0 || self.replace_robot_link_convex_shapes {
            optima_print("Preprocessing robot link convex shape subcomponents...", PrintMode::Println, PrintColor::Cyan, false);
            directory_path.delete_all_items_in_directory()?;

            let mut base_meshes_directory_path = OptimaStemCellPath::new_asset_path()?;
            base_meshes_directory_path.append_file_location(&OptimaAssetLocation::RobotInputMeshes {robot_name: robot_name.to_string()});

            let robot_model_module = RobotModelModule::new(robot_name)?;
            let links = robot_model_module.links();

            for (i, link) in links.iter().enumerate() {
                let has_visual_mesh = link.urdf_link().visual_mesh_filename().is_some();
                if has_visual_mesh {
                    let res = base_meshes_directory_path.walk_directory_and_match(OptimaPathMatchingPattern::PathComponentsWithoutExtension(vec![format!("{}", i)]), OptimaPathMatchingStopCondition::First);
                    let optima_path = res[0].clone();
                    let trimesh = optima_path.load_file_to_trimesh_engine()?;

                    let convex_components = trimesh.compute_convex_decomposition();
                    optima_print(&format!("   > computed {:?} convex subcomponents for link {:?}", convex_components.len(), i), PrintMode::Println, PrintColor::None, false);
                    for (j, c) in convex_components.iter().enumerate() {
                        let mut directory_path_copy = directory_path.clone();
                        directory_path_copy.append(&format!("{}_{}.stl", i, j));
                        directory_path_copy.save_trimesh_engine_to_stl(&c)?;
                    }
                }
            }
            optima_print("Successfully preprocessed robot link convex shape subcomponents!", PrintMode::Println, PrintColor::Green, false);
        }
        Ok(())
    }
}
impl Default for RobotPreprocessingModule {
    fn default() -> Self {
        Self {
            replace_robot_model_module_json: true,
            replace_configuration_generator_module_json: false,
            replace_robot_link_convex_shapes: true,
            replace_robot_link_convex_shape_subcomponents: true
        }
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotPreprocessingModule {
    #[staticmethod]
    pub fn preprocess_robot_from_console_input_py(robot_name: &str) {
        Self::preprocess_robot_from_console_input(robot_name).expect("error");
    }

    pub fn preprocess_robot_py(&self, robot_name: &str) {
        self.preprocess_robot(robot_name).expect("error");
    }
}


