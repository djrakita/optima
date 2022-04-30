#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_geometric_shape_module::RobotLinkShapeRepresentation;
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::utils_console::get_default_progress_bar;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaAssetLocation, OptimaPath, OptimaPathMatchingPattern, OptimaPathMatchingStopCondition, OptimaStemCellPath};
use crate::utils::utils_robot::link::Link;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeSignature};
use crate::utils::utils_traits::SaveAndLoadable;

/// The `RobotMeshFileManagerModule` has numerous utility functions relating to mesh files.
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotMeshFileManagerModule {
    robot_name: String,
    links: Vec<Link>
}
impl RobotMeshFileManagerModule {
    pub fn new_from_name(robot_name: &str) -> Result<Self, OptimaError> {
        let robot_model_module = RobotModelModule::new(robot_name)?;
        return Self::new(&robot_model_module);
    }
    pub fn new(robot_model_module: &RobotModelModule) -> Result<Self, OptimaError> {
        Ok(Self {
            robot_name: robot_model_module.robot_name().to_string(),
            links: robot_model_module.links().clone()
        })
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
    fn find_optima_paths_to_urdf_link_meshes(&self, link_mesh_type: &LinkMeshType) -> Result<Vec<Option<OptimaPath>>, OptimaError> {
        let mut out_vec = vec![];

        let mut directory_string_vecs = vec![];
        let mut directory_idxs = vec![];
        let subcomponents = self.get_final_n_subcomponents_from_urdf_link_mesh_path_split_vecs(link_mesh_type, 4);
        for s in &subcomponents {
            if let Some(ss) = s {
                let check_vec = vec![ ss[0].clone(), ss[1].clone(), ss[2].clone() ];
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
    /// Attempts to find mesh files on the user's computer based on the file paths specified in the robot URDF.
    /// The mesh files can be anywhere on the computer, but these files will be easiest to find in
    /// a major directory like the desktop.  If the files are found, they are copied to the local
    /// optima_assets directory.  If the files cannot be found, this function will return an error.
    #[allow(unused_must_use)]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn find_and_copy_visual_meshes_to_assets(&self) -> Result<(), OptimaError> {
        optima_print(&format!("Finding and copying visual meshes to assets folder..."), PrintMode::Println, PrintColor::Blue, true);
        let destination = OptimaPath::new_asset_physical_path_from_json_file()?;
        let paths = self.find_optima_paths_to_urdf_link_meshes(&LinkMeshType::Visual)?;
        let num_paths = paths.len();
        let mut pb = get_default_progress_bar(num_paths);

        for (i, path) in paths.iter().enumerate() {
            if let Some(p) = path {
                let extension = p.extension().unwrap();
                let new_filename = format!("{}.{}", i, extension);
                let mut destination_clone = destination.clone();
                destination_clone.append_file_location(&OptimaAssetLocation::RobotInputMeshes { robot_name: self.robot_name.clone() });
                destination_clone.append(&new_filename);
                p.copy_file_to_destination(&destination_clone)?;
            }
            pb.set(i as u64);
        }
        println!();
        Ok(())
    }
    /// Returns the paths to visual meshes.  The vector here has an entry for each robot link in the
    /// robot model.  If a given link does not have a visual component, the entry will be None.
    /// Files are either drawn from the robot's mesh folder as stls or the robot's glb_mesh directory as glbs.
    /// Files in the glb_mesh directory are prioritized, if present, but they are optional.
    pub fn get_paths_to_visual_meshes(&self) -> Result<Vec<Option<OptimaStemCellPath>>, OptimaError> {
        let paths_to_meshes = self.get_paths_to_meshes()?;
        let paths_to_glb_meshes = self.get_paths_to_glb_meshes()?;

        let mut out_vec = vec![];

        let l = paths_to_meshes.len();
        for i in 0..l {
            if paths_to_glb_meshes[i].is_some() { out_vec.push(paths_to_glb_meshes[i].clone()); }
            else { out_vec.push(paths_to_meshes[i].clone()); }
        }

        Ok(out_vec)
    }
    /// Returns the paths to convex shape stls.  The vector here has an entry for each robot link in the
    /// robot model.  If a given link does not have a visual component, the entry will be None.
    pub fn get_paths_to_convex_shape_meshes(&self) -> Result<Vec<Option<OptimaStemCellPath>>, OptimaError> {
        let mut out_vec = vec![];

        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotConvexShapes { robot_name: self.robot_name.clone() });
        for (i, link) in self.links.iter().enumerate() {
            if link.urdf_link().visual_mesh_filename().is_some() {
                let mut path_copy = path.clone();
                path_copy.append(&format!("{}.stl", i));
                if path_copy.exists() {
                    out_vec.push(Some(path_copy));
                } else {
                    out_vec.push(None);
                }
            } else {
                out_vec.push(None);
            }
        }

        Ok(out_vec)
    }
    /// Returns the paths to convex shape subcomponent stls.  The vector here has a vector entry for
    /// each robot link in the robot model.
    pub fn get_paths_to_convex_shape_subcomponent_meshes(&self) -> Result<Vec<Vec<OptimaStemCellPath>>, OptimaError> {
        let mut out_vec = vec![];
        let num_links = self.links.len();
        for _ in 0..num_links { out_vec.push(vec![]); }

        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotConvexSubcomponents { robot_name: self.robot_name.clone() });

        let all_files = path.get_all_items_in_directory(false, false);
        for filename in &all_files {
            let split: Vec<&str> = filename.split("_").collect();
            if split.len() <= 1 { continue; }
            let num_as_string = split[0];
            let link_idx = num_as_string.parse::<usize>().expect(&format!("Could not parse {} as usize", num_as_string));
            let mut path_copy = path.clone();
            path_copy.append(filename);
            out_vec[link_idx].push(path_copy);
        }

        Ok(out_vec)
    }
    pub fn get_paths_to_meshes(&self) -> Result<Vec<Option<OptimaStemCellPath>>, OptimaError> {
        let mut out_vec = vec![];

        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotMeshes { robot_name: self.robot_name.clone() });
        for (i, link) in self.links.iter().enumerate() {
            if link.urdf_link().visual_mesh_filename().is_some() {
                let mut path_copy = path.clone();
                path_copy.append(&format!("{}.stl", i));
                if path_copy.exists() {
                    out_vec.push(Some(path_copy));
                } else {
                    out_vec.push(None);
                }
            } else {
                out_vec.push(None);
            }
        }

        Ok(out_vec)
    }
    fn get_paths_to_glb_meshes(&self) -> Result<Vec<Option<OptimaStemCellPath>>, OptimaError> {
        let mut out_vec = vec![];

        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotGLBMeshes { robot_name: self.robot_name.clone() });
        for (i, link) in self.links.iter().enumerate() {
            if link.urdf_link().visual_mesh_filename().is_some() {
                let mut path_copy = path.clone();
                path_copy.append(&format!("{}.glb", i));
                if path_copy.exists() {
                    out_vec.push(Some(path_copy));
                } else {
                    out_vec.push(None);
                }
            } else {
                out_vec.push(None);
            }
        }


        Ok(out_vec)
    }
    pub fn get_geometric_shapes(&self, shape_representation: &RobotLinkShapeRepresentation) -> Result<Vec<Option<GeometricShape>>, OptimaError> {
        let mut out_vec = vec![];

        match shape_representation {
            RobotLinkShapeRepresentation::Cubes => {
                let paths = self.get_paths_to_meshes()?;
                for (link_idx, path) in paths.iter().enumerate() {
                    match path {
                        None => { out_vec.push(None); }
                        Some(path) => {
                            let base_shape = GeometricShape::new_triangle_mesh(path, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: 0 });
                            let cube_shape = base_shape.to_best_fit_cube();
                            out_vec.push(Some(cube_shape));
                        }
                    }
                }
            }
            RobotLinkShapeRepresentation::ConvexShapes => {
                let paths = self.get_paths_to_convex_shape_meshes()?;
                for (link_idx, path) in paths.iter().enumerate() {
                    match path {
                        None => { out_vec.push(None); }
                        Some(path) => {
                            let base_shape = GeometricShape::new_convex_shape(path, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: 0 });
                            out_vec.push(Some(base_shape));
                        }
                    }
                }
            }
            RobotLinkShapeRepresentation::SphereSubcomponents => {
                let paths = self.get_paths_to_convex_shape_subcomponent_meshes()?;
                for (link_idx, v) in paths.iter().enumerate() {
                    if v.len() == 0 { out_vec.push(None); }
                    for (shape_idx_in_link, path) in v.iter().enumerate() {
                        let base_shape = GeometricShape::new_convex_shape(path, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link });
                        let sphere_shape = base_shape.to_best_fit_sphere();
                        out_vec.push(Some(sphere_shape));
                    }
                }
            }
            RobotLinkShapeRepresentation::CubeSubcomponents => {
                let paths = self.get_paths_to_convex_shape_subcomponent_meshes()?;
                for (link_idx, v) in paths.iter().enumerate() {
                    if v.len() == 0 { out_vec.push(None); }
                    for (shape_idx_in_link, path) in v.iter().enumerate() {
                        let base_shape = GeometricShape::new_convex_shape(path, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link });
                        let cube_shape = base_shape.to_best_fit_cube();
                        out_vec.push(Some(cube_shape));
                    }
                }
            }
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents => {
                let paths = self.get_paths_to_convex_shape_subcomponent_meshes()?;
                for (link_idx, v) in paths.iter().enumerate() {
                    if v.len() == 0 { out_vec.push(None); }
                    for (shape_idx_in_link, path) in v.iter().enumerate() {
                        let base_shape = GeometricShape::new_convex_shape(path, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link });
                        out_vec.push(Some(base_shape));
                    }
                }
            }
            RobotLinkShapeRepresentation::TriangleMeshes => {
                let paths = self.get_paths_to_convex_shape_meshes()?;
                for (link_idx, path) in paths.iter().enumerate() {
                    match path {
                        None => { out_vec.push(None); }
                        Some(path) => {
                            let base_shape = GeometricShape::new_triangle_mesh(path, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: 0 });
                            out_vec.push(Some(base_shape));
                        }
                    }
                }
            }
        }

        Ok(out_vec)
    }
    pub fn robot_name(&self) -> &str {
        &self.robot_name
    }
}
impl SaveAndLoadable for RobotMeshFileManagerModule {
    type SaveType = String;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        self.robot_name.clone()
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let robot_name: Self::SaveType = load_object_from_json_string(json_str)?;
        return RobotMeshFileManagerModule::new_from_name(&robot_name);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotMeshFileManagerModule {
    #[new]
    pub fn new_from_name_py(robot_name: &str) -> Self {
        return Self::new_from_name(robot_name).expect("error");
    }

    pub fn get_paths_to_meshes_as_strings(&self) -> Vec<Option<String>> {
        let mut out_vec = vec![];

        let res = self.get_paths_to_meshes().expect("error");
        for optima_path_option in &res {
            match optima_path_option {
                None => { out_vec.push(None); }
                Some(optima_path) => { out_vec.push(Some(optima_path.to_string())); }
            }
        }

        out_vec
    }

    pub fn get_paths_to_visual_meshes_as_strings(&self) -> Vec<Option<String>> {
        let mut out_vec = vec![];

        let res = self.get_paths_to_visual_meshes().expect("error");
        for optima_path_option in &res {
            match optima_path_option {
                None => { out_vec.push(None); }
                Some(optima_path) => { out_vec.push(Some(optima_path.to_string())); }
            }
        }

        out_vec
    }

    pub fn get_paths_to_convex_shape_meshes_as_strings(&self) -> Vec<Option<String>> {
        let mut out_vec = vec![];

        let res = self.get_paths_to_convex_shape_meshes().expect("error");
        for optima_path_option in &res {
            match optima_path_option {
                None => { out_vec.push(None); }
                Some(optima_path) => { out_vec.push(Some(optima_path.to_string())); }
            }
        }

        out_vec
    }

    pub fn get_paths_to_convex_shape_subcomponent_meshes_as_strings(&self) -> Vec<Vec<String>> {
        let mut out_vec = vec![];

        let res = self.get_paths_to_convex_shape_subcomponent_meshes().expect("error");
        for optima_path_vec in &res {
            let idx = out_vec.len();
            out_vec.push(vec![]);
            for optima_paths in optima_path_vec {
                out_vec[idx].push(optima_paths.to_string());
            }
        }

        out_vec
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LinkMeshType {
    Visual,
    Collision
}

