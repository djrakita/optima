#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_console_output::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaPath, OptimaPathMatchingPattern, OptimaPathMatchingStopCondition};
use crate::utils::utils_robot::link::Link;
use crate::utils::utils_robot::robot_module_utils::RobotModuleSaveAndLoad;

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotMeshFileManagerModule {
    robot_name: String,
    links: Vec<Link>
}
impl RobotMeshFileManagerModule {
    pub fn new(robot_model_module: &RobotModelModule) -> Result<Self, OptimaError> {
        Ok(Self {
            robot_name: robot_model_module.robot_name().to_string(),
            links: robot_model_module.links().clone()
        })
    }

    pub fn new_from_robot_model_module_json_file(robot_name: &str) -> Result<Self, OptimaError> {
        let robot_model_module = RobotModelModule::new(robot_name)?;
        return Self::new(&robot_model_module);
    }

    pub fn new_from_robot_model_module_json_string(json_str: &str) -> Result<Self, OptimaError> {
        let robot_model_module = RobotModelModule::new_load_from_json_string(json_str)?;
        return Self::new(&robot_model_module);
    }

    fn get_urdf_link_mesh_path_split_vecs(&self, link_mesh_type: &LinkMeshType) -> Vec<Option<Vec<String>>> {
        let mut out_vec = vec![];

        let links = &self.links;
        for link in links {
            let path_string_string = match link_mesh_type {
                LinkMeshType::Visual => { link.urdf_link().visual_mesh_filename().clone() }
                LinkMeshType::Collision => { link.urdf_link().collision_mesh_filename().clone() }
            };

            match path_string_string {
                None => { out_vec.push(None); }
                Some(path_string) => {
                    let path_string_split: Vec<&str> = path_string.split("/").collect();
                    let mut v = vec![];
                    for path_string in path_string_split {
                        v.push(path_string.to_string());
                    }
                    out_vec.push(Some(v));
                }
            }
        }

        out_vec
    }

    fn get_final_n_subcomponents_from_urdf_link_mesh_path_split_vecs(&self, link_mesh_type: &LinkMeshType, n: usize) -> Vec<Option<Vec<String>>> {
        let mut out_vec = vec![];

        let split = self.get_urdf_link_mesh_path_split_vecs(link_mesh_type);
        for s in &split {
            match s {
                None => { out_vec.push(None); }
                Some(ss) => {
                    let mut v = vec![];
                    let n_local = n.min(ss.len());
                    for i in 0..n_local {
                        v.push(ss[ss.len() - n_local + i].clone());
                    }
                    out_vec.push(Some(v));
                }
            }
        }

        out_vec
    }

    pub fn get_optima_paths_to_urdf_link_meshes(&self, link_mesh_type: &LinkMeshType) -> Result<Vec<Option<OptimaPath>>, OptimaError> {
        let mut out_vec = vec![];

        let mut directory_string_vecs = vec![];
        let mut directory_idxs = vec![];
        let subcomponents = self.get_final_n_subcomponents_from_urdf_link_mesh_path_split_vecs(link_mesh_type, 3);
        for s in &subcomponents {
            if let Some(ss) = s {
                let check_vec = vec![ ss[0].clone(), ss[1].clone() ];
                if !directory_string_vecs.contains(&check_vec) {
                    directory_idxs.push(Some(directory_string_vecs.len()));
                    directory_string_vecs.push(check_vec.clone());
                } else {
                    for (i, d) in directory_string_vecs.iter().enumerate() {
                        if d == &check_vec {
                            directory_idxs.push(Some(i));
                        }
                    }
                }
            } else {
                directory_idxs.push(None);
            }
        }

        let mut directory_optima_paths = vec![];
        optima_print("Finding mesh file directories.  This may take a while...", PrintMode::Println, PrintColor::Cyan, true);
        for d in &directory_string_vecs {
            let p = OptimaPath::new_home_path()?;
            let res = p.walk_directory_and_match(OptimaPathMatchingPattern::PathComponents(d.clone()), OptimaPathMatchingStopCondition::First);
            if res.len() > 0 { directory_optima_paths.push(res[0].clone()); }
            else {
                return Err(OptimaError::new_generic_error_str(&format!("Could not find directory corresponding to path components {:?}.", d), file!(), line!()));
            }
        }

        let subcomponents = self.get_final_n_subcomponents_from_urdf_link_mesh_path_split_vecs(link_mesh_type, 1);
        for (i, directory_optima_path_idx_option) in directory_idxs.iter().enumerate() {
            match directory_optima_path_idx_option {
                None => { out_vec.push(None); }
                Some(idx) => {
                    let ss = subcomponents[i].as_ref().unwrap();
                    let mut out_path_clone = directory_optima_paths[*idx].clone();
                    out_path_clone.append(&ss[0]);
                    out_vec.push(Some(out_path_clone));
                }
            }
        }

        Ok(out_vec)
    }

    #[allow(unused_must_use)]
    pub fn find_and_copy_visual_meshes_to_assets(&self) -> Result<(), OptimaError> {
        let destination = OptimaPath::new_asset_path_from_json_file()?;
        let paths = self.get_optima_paths_to_urdf_link_meshes(&LinkMeshType::Visual)?;
        for (i, path) in paths.iter().enumerate() {
            if let Some(p) = path {
                let extension = p.extension().unwrap();
                let new_filename = format!("{}.{}", i, extension);
                let mut destination_clone = destination.clone();
                destination_clone.append_file_location(&OptimaAssetLocation::RobotMeshes { robot_name: self.robot_name.clone() });
                destination_clone.append(&new_filename);
                p.copy_file_to_destination(&destination_clone);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LinkMeshType {
    Visual,
    Collision
}