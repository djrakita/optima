use std::path::{Component, Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::{AssetDirLocation, AssetDirUtils, AssetFileMode, FileUtils, RobotModuleJsonType};
use crate::utils::utils_robot::link::Link;
use crate::utils::utils_robot::robot_module_utils::RobotModuleSaveAndLoad;

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFileManagerModule {
    robot_name: String,
    links: Vec<Link>
}
impl RobotFileManagerModule {
    pub fn new(robot_model_module: &RobotModelModule) -> Result<Self, OptimaError> {
        Ok(Self {
            robot_name: robot_model_module.robot_name().to_string(),
            links: robot_model_module.links().clone()
        })
    }

    pub fn new_from_robot_model_module_json_file(robot_name: &str) -> Result<Self, OptimaError> {
        let robot_model_module = RobotModelModule::new_from_json_file(robot_name, RobotModuleJsonType::ModelModule)?;
        return Self::new(&robot_model_module);
    }

    pub fn new_from_robot_model_module_json_string(json_str: &str) -> Result<Self, OptimaError> {
        let robot_model_module = RobotModelModule::new_load_from_json_string(json_str)?;
        return Self::new(&robot_model_module);
    }

    pub fn get_all_link_mesh_path_stubs(&self, link_mesh_type: &LinkMeshType, with_file_extension: bool) -> Vec<Option<PathBuf>> {
        let mut out_vec = vec![];
        for link in &self.links {
            let ros_filestring_option = match link_mesh_type {
                LinkMeshType::Visual => { link.urdf_link().visual_mesh_filename().clone() }
                LinkMeshType::Collision => { link.urdf_link().collision_mesh_filename().clone() }
            };

            match &ros_filestring_option {
                None => { out_vec.push(None); }
                Some(s) => {
                    let path_from_string = Path::new(s);
                    let components = path_from_string.components();
                    let component_vec: Vec<Component> = components.collect();
                    if component_vec.len() <= 1 { out_vec.push(None); }
                    else {
                        let last_component = component_vec[component_vec.len() - 1];
                        let second_last_component = component_vec[component_vec.len() - 2];

                        let last_component_path_option = match last_component {
                            Component::Normal(s) => { Some(Path::new(s).to_path_buf()) }
                            _ => { None }
                        };

                        let second_last_component_path_option = match second_last_component {
                            Component::Normal(s) => { Some(Path::new(s).to_path_buf()) }
                            _ => { None }
                        };

                        if last_component_path_option.is_none() || second_last_component_path_option.is_none() {
                            out_vec.push(None); continue;
                        } else {
                            let last_component_path = last_component_path_option.unwrap();
                            let second_last_component_path = second_last_component_path_option.unwrap();

                            let mut combined_path = second_last_component_path.join(last_component_path);
                            if !with_file_extension {
                                combined_path.set_extension("");
                            }
                            out_vec.push(Some(combined_path));
                        }
                    }
                }
            }
        }
        out_vec
    }

    pub fn get_all_link_mesh_filenames(&self, link_mesh_type: &LinkMeshType, with_file_extension: bool) -> Vec<Option<PathBuf>> {
        let mut out_vec = vec![];

        let stubs = self.get_all_link_mesh_path_stubs(link_mesh_type, with_file_extension);
        for stub_option in &stubs {
            match stub_option {
                None => {
                    out_vec.push(None);
                }
                Some(stub) => {
                    out_vec.push(Some(FileUtils::get_filename(stub, with_file_extension).expect("error")));
                }
            }
        }

        out_vec
    }

    pub fn get_all_link_mesh_directories(&self, link_mesh_type: &LinkMeshType) -> Vec<PathBuf> {
        let mut out_vec = vec![];

        let stubs = self.get_all_link_mesh_path_stubs(link_mesh_type, true);
        for stub_option in &stubs {
            if let Some(stub) = stub_option {
                let parent = stub.parent().unwrap().to_path_buf();
                if !out_vec.contains(&parent) {
                    out_vec.push(parent);
                }
            }
        }

        out_vec
    }

    pub fn get_all_link_mesh_paths(&self, link_mesh_type: &LinkMeshType, asset_file_mode: AssetFileMode, with_file_extension: bool) -> Vec<Option<PathBuf>> {
        let mut out_vec = vec![];
        let path = AssetDirUtils::get_path_to_location(AssetDirLocation::RobotMeshes { robot_name: self.robot_name.clone() }, asset_file_mode).expect("error");
        let stubs = self.get_all_link_mesh_path_stubs(link_mesh_type, with_file_extension);

        for stub_option in &stubs {
            match stub_option {
                None => { out_vec.push(None); }
                Some(stub) => { out_vec.push(Some(path.join(stub))) }
            }
        }

        out_vec
    }

    /*
    pub fn get_all_mesh_file_directories(&self) -> Vec<PathBuf> {
        for l in &self.links {
            let aa = l.urdf_link().visual_mesh_filename();
            if let Some(a) = aa {
                let p = Path::new(a);
                // println!("{:?}", p);
                let c: Vec<Component> = p.components().collect();
                match c[c.len()-1] {
                    Component::Normal(pp) => {
                        let path = Path::new(pp);
                        println!("{:?}", path);
                    }
                    _ => { }
                }
                // println!("{:?}", c[c.len()-1]);
            }
        }
        vec![]
    }
    */
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LinkMeshType {
    Visual,
    Collision
}