#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::utils::utils_console::{ConsoleInputUtils, get_default_progress_bar, optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::robot_modules::robot_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::robot_modules::robot_geometric_shape_module::RobotGeometricShapeModule;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaPathMatchingPattern, OptimaPathMatchingStopCondition, OptimaStemCellPath, RobotModuleJsonType};
use crate::utils::utils_robot::robot_module_utils::RobotModuleSaveAndLoad;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotPreprocessingModule {
    pub replace_robot_model_module_json: bool,
    pub replace_robot_link_convex_shapes: bool,
    pub replace_robot_link_convex_shape_subcomponents: bool
}
impl RobotPreprocessingModule {
    pub fn preprocess_all_robots_from_console_input() -> Result<(), OptimaError> {
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::Robots);
        let all_robot_strings = path.get_all_directories_in_directory();

        optima_print("Welcome to the Optima Robot Preprocessing Module.", PrintMode::Println, PrintColor::Green, true);
        let line = ConsoleInputUtils::get_console_input_string("Use all default options?  (y or n)", PrintColor::Blue)?;
        let default_options = if &line == "y" { true } else { false };
        if default_options {
            for robot_name in &all_robot_strings {
                optima_print(&format!(" Preprocessing robot {:?}", robot_name), PrintMode::Println, PrintColor::Green, true);
                let res = RobotPreprocessingModule::default().preprocess_robot(robot_name);
                if res.is_err() {
                    optima_print(&format!(" Could not successfully preprocess robot {:?}.  Encountered error {:?}", robot_name, res), PrintMode::Println, PrintColor::Red, true);
                }
            }
            return Ok(());
        }

        let line = ConsoleInputUtils::get_console_input_string("Replace robot model module?  (y or n)", PrintColor::Blue)?;
        let replace_robot_model_module_json = if &line == "y" { true } else { false };
        let line = ConsoleInputUtils::get_console_input_string("Replace robot link convex shapes?  (y or n)", PrintColor::Blue)?;
        let replace_robot_link_convex_shapes = if &line == "y" { true } else { false };
        let line = ConsoleInputUtils::get_console_input_string("Replace robot link convex subcomponents?  (y or n)", PrintColor::Blue)?;
        let replace_robot_link_convex_shape_subcomponents = if &line == "y" { true } else { false };

        for robot_name in &all_robot_strings {
            optima_print(&format!("Preprocessing robot {:?}", robot_name), PrintMode::Println, PrintColor::Blue, true);
            let res = RobotPreprocessingModule {
                replace_robot_model_module_json,
                replace_robot_link_convex_shapes,
                replace_robot_link_convex_shape_subcomponents
            }.preprocess_robot(robot_name);
            if res.is_err() {
                optima_print(&format!("Could not successfully preprocess robot {:?}.  Encountered error {:?}", robot_name, res), PrintMode::Println, PrintColor::Red, true);
            }
        }

        Ok(())
    }
    pub fn preprocess_robot_from_console_input(robot_name: &str) -> Result<(), OptimaError> {
        optima_print("Welcome to the Optima Robot Preprocessing Module.", PrintMode::Println, PrintColor::Blue, true);
        let line = ConsoleInputUtils::get_console_input_string("Use all default options?  (y or n)", PrintColor::Blue)?;
        let default_options = if &line == "y" { true } else { false };
        if default_options { return Self::default().preprocess_robot(robot_name); }

        let line = ConsoleInputUtils::get_console_input_string("Replace robot model module?  (y or n)", PrintColor::Blue)?;
        let replace_robot_model_module_json = if &line == "y" { true } else { false };
        let line = ConsoleInputUtils::get_console_input_string("Replace robot link convex shapes?  (y or n)", PrintColor::Blue)?;
        let replace_robot_link_convex_shapes = if &line == "y" { true } else { false };
        let line = ConsoleInputUtils::get_console_input_string("Replace robot link convex subcomponents?  (y or n)", PrintColor::Blue)?;
        let replace_robot_link_convex_shape_subcomponents = if &line == "y" { true } else { false };

        return Self {
            replace_robot_model_module_json,
            replace_robot_link_convex_shapes,
            replace_robot_link_convex_shape_subcomponents
        }.preprocess_robot(robot_name);
    }
    pub fn preprocess_robot(&self, robot_name: &str) -> Result<(), OptimaError> {
        if cfg!(feature = "only_use_embedded_assets") {
            return Err(OptimaError::new_unsupported_operation_error("preprocess_robot", "Cannot preprocess robot using only_use_embedded_assets feature.", file!(), line!()));
        }

        self.preprocess_robot_model_module_json(robot_name)?;
        self.copy_link_meshes_to_assets_folder(robot_name)?;
        self.preprocess_robot_link_meshes(robot_name)?;
        self.preprocess_robot_link_convex_shapes(robot_name)?;
        self.preprocess_robot_link_convex_shape_subcomponents(robot_name)?;
        self.preprocess_robot_shape_geometry_module(robot_name)?;

        println!();
        optima_print(&format!("Successfully preprocessed robot {}!", robot_name), PrintMode::Println, PrintColor::Green, true);
        Ok(())
    }
    fn preprocess_robot_model_module_json(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut file_path = OptimaStemCellPath::new_asset_path()?;
        file_path.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ModelModule });
        if !file_path.exists() || self.replace_robot_model_module_json {
            optima_print("Preprocessing robot model module...", PrintMode::Println, PrintColor::Blue, true);
            file_path.delete_file()?;

            let robot_model_module = RobotModelModule::new(robot_name)?;
            robot_model_module.save_to_json_file(RobotModuleJsonType::ModelModule)?;

            optima_print("Successfully preprocessed robot model module.", PrintMode::Println, PrintColor::Blue, true);
        }
        Ok(())
    }
    fn copy_link_meshes_to_assets_folder(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut file_path = OptimaStemCellPath::new_asset_path()?;
        file_path.append_file_location(&OptimaAssetLocation::RobotInputMeshes { robot_name: robot_name.to_string() });
        let file_names = file_path.get_all_items_in_directory(false, false);
        if !file_path.exists() || file_names.len() == 0 {
            let mesh_file_manager_module = RobotMeshFileManagerModule::new_from_name(robot_name)?;
            mesh_file_manager_module.find_and_copy_visual_meshes_to_assets()?;
        }
        Ok(())
    }
    fn preprocess_robot_link_meshes(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut directory_path = OptimaStemCellPath::new_asset_path()?;
        directory_path.append_file_location(&OptimaAssetLocation::RobotMeshes { robot_name: robot_name.to_string() });
        optima_print("Preprocessing robot link meshes...", PrintMode::Println, PrintColor::Blue, true);
        directory_path.delete_all_items_in_directory()?;

        let mut base_meshes_directory_path = OptimaStemCellPath::new_asset_path()?;
        base_meshes_directory_path.append_file_location(&OptimaAssetLocation::RobotInputMeshes {robot_name: robot_name.to_string()});

        let robot_model_module = RobotModelModule::new(robot_name)?;
        let links = robot_model_module.links();

        let mut pb = get_default_progress_bar(links.len());
        pb.show_counter = true;

        for (i, link) in links.iter().enumerate() {
            let has_visual_mesh = link.urdf_link().visual_mesh_filename().is_some();
            if has_visual_mesh {
                let res = base_meshes_directory_path.walk_directory_and_match(OptimaPathMatchingPattern::PathComponentsWithoutExtension(vec![format!("{}", i)]), OptimaPathMatchingStopCondition::First);
                if res.len() == 0 {
                    return Err(OptimaError::new_generic_error_str(&format!("Path for link {:?} does not exist in {:?}.", i, base_meshes_directory_path), file!(), line!()));
                }
                let optima_path = res[0].clone();
                let mut trimesh = optima_path.load_file_to_trimesh_engine()?;

                let visual_origin_rpy = link.urdf_link().visual_origin_rpy();
                let visual_origin_xyz = link.urdf_link().visual_origin_xyz();
                if let Some(r) = visual_origin_rpy  {
                    if let Some(t) = visual_origin_xyz {
                        let pose = OptimaSE3Pose::new_from_euler_angles(r[0], r[1], r[2], t[0], t[1], t[2], &OptimaSE3PoseType::ImplicitDualQuaternion);
                        trimesh.transform_vertices(&pose);
                    }
                }

                let mut directory_path_copy = directory_path.clone();
                directory_path_copy.append(&format!("{}.stl", i));
                directory_path_copy.save_trimesh_engine_to_stl(&trimesh)?;
            }
            pb.set(i as u64 + 1);
        }

        println!();

        Ok(())
    }
    fn preprocess_robot_link_convex_shapes(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut directory_path = OptimaStemCellPath::new_asset_path()?;
        directory_path.append_file_location(&OptimaAssetLocation::RobotConvexShapes { robot_name: robot_name.to_string() });
        let files_in_directory = directory_path.get_all_items_in_directory(false, false);
        if !directory_path.exists() || files_in_directory.len() == 0 || self.replace_robot_link_convex_shapes {
            optima_print("Preprocessing robot link convex shapes...", PrintMode::Println, PrintColor::Blue, true);
            directory_path.delete_all_items_in_directory()?;

            let mut base_meshes_directory_path = OptimaStemCellPath::new_asset_path()?;
            base_meshes_directory_path.append_file_location(&OptimaAssetLocation::RobotMeshes {robot_name: robot_name.to_string()});

            let robot_model_module = RobotModelModule::new(robot_name)?;
            let links = robot_model_module.links();

            let mut pb = get_default_progress_bar(links.len());
            pb.show_counter = true;

            for (i, link) in links.iter().enumerate() {
                let has_visual_mesh = link.urdf_link().visual_mesh_filename().is_some();
                if has_visual_mesh {
                    let res = base_meshes_directory_path.walk_directory_and_match(OptimaPathMatchingPattern::PathComponentsWithoutExtension(vec![format!("{}", i)]), OptimaPathMatchingStopCondition::First);
                    let optima_path = res[0].clone();
                    let trimesh = optima_path.load_file_to_trimesh_engine()?;
                    let convex_hull = trimesh.compute_convex_hull();

                    let mut directory_path_copy = directory_path.clone();
                    directory_path_copy.append(&format!("{}.stl", i));
                    directory_path_copy.save_trimesh_engine_to_stl(&convex_hull)?;
                }
                pb.set(i as u64  + 1);
            }

            println!();
        }
        Ok(())
    }
    fn preprocess_robot_link_convex_shape_subcomponents(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut directory_path = OptimaStemCellPath::new_asset_path()?;
        directory_path.append_file_location(&OptimaAssetLocation::RobotConvexSubcomponents { robot_name: robot_name.to_string() });
        let files_in_directory = directory_path.get_all_items_in_directory(false, false);
        if !directory_path.exists() || files_in_directory.len() == 0 || self.replace_robot_link_convex_shapes {
            optima_print("Preprocessing robot link convex shape subcomponents...", PrintMode::Println, PrintColor::Blue, true);
            directory_path.delete_all_items_in_directory()?;

            let mut base_meshes_directory_path = OptimaStemCellPath::new_asset_path()?;
            base_meshes_directory_path.append_file_location(&OptimaAssetLocation::RobotMeshes {robot_name: robot_name.to_string()});

            let robot_model_module = RobotModelModule::new(robot_name)?;
            let links = robot_model_module.links();

            let mut pb = get_default_progress_bar(links.len());
            pb.show_counter = true;

            let mut messages = vec![];
            for (i, link) in links.iter().enumerate() {
                let has_visual_mesh = link.urdf_link().visual_mesh_filename().is_some();
                if has_visual_mesh {
                    let res = base_meshes_directory_path.walk_directory_and_match(OptimaPathMatchingPattern::PathComponentsWithoutExtension(vec![format!("{}", i)]), OptimaPathMatchingStopCondition::First);
                    let optima_path = res[0].clone();
                    let trimesh = optima_path.load_file_to_trimesh_engine()?;

                    let convex_components = trimesh.compute_convex_decomposition();
                    messages.push(format!("{:?} convex subcomponents for link {:?}: {}. ", convex_components.len(), link.link_idx(), link.name()));
                    for (j, c) in convex_components.iter().enumerate() {
                        let mut directory_path_copy = directory_path.clone();
                        directory_path_copy.append(&format!("{}_{}.stl", i, j));
                        directory_path_copy.save_trimesh_engine_to_stl(&c)?;
                    }
                }
                pb.set(i as u64 + 1);
            }

            println!();
            for m in messages { println!("{}", m); }

        }
        Ok(())
    }
    fn preprocess_robot_shape_geometry_module(&self, robot_name: &str) -> Result<(), OptimaError> {
        let mut directory_path = OptimaStemCellPath::new_asset_path()?;
        directory_path.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ShapeGeometryModule });

        let mut directory_path_permanent = OptimaStemCellPath::new_asset_path()?;
        directory_path_permanent.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ShapeGeometryModule });

        if !directory_path.exists() || !directory_path_permanent.exists() || self.replace_robot_link_convex_shapes || self.replace_robot_link_convex_shape_subcomponents {
            optima_print("Preprocessing robot shape geometry module...", PrintMode::Println, PrintColor::Blue, true);
            let robot_shape_geometry_module = RobotGeometricShapeModule::new_from_names(robot_name, None, true)?;
            robot_shape_geometry_module.save_to_json_file(RobotModuleJsonType::ShapeGeometryModule)?;
            robot_shape_geometry_module.save_to_json_file(RobotModuleJsonType::ShapeGeometryModulePermanent)?;
        }
        Ok(())
    }
}
impl Default for RobotPreprocessingModule {
    fn default() -> Self {
        Self {
            replace_robot_model_module_json: true,
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
