use vfs::*;
use rust_embed::RustEmbed;
use std::{fs, vec};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{PathBuf};
use serde::de::DeserializeOwned;
use serde::{Serialize, Deserialize};
use urdf_rs::Robot;
use walkdir::WalkDir;
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;

/// An `OptimaStemCellPath` has the same functionality as an `OptimaPath`, but it
/// will try to automatically select whether it should use a physical or virtual file path based on
/// your target (rust executable, web-assembly, python module, etc).  When in doubt, use this over
/// an `OptimaPath`.
///
/// # Example
///```
/// use optima::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaStemCellPath};
///
/// let mut p = OptimaStemCellPath::new_asset_path().expect("error");
/// p.append_file_location(&OptimaAssetLocation::RobotMeshes {robot_name: "ur5".to_string()});
/// p.append("0.dae");
///
///```
#[derive(Clone, Debug)]
pub struct OptimaStemCellPath {
    optima_file_paths: Vec<OptimaPath>
}
impl OptimaStemCellPath {
    pub fn new_asset_path() -> Result<Self, OptimaError> {
        let mut optima_file_paths = vec![];

        if cfg!(target_arch = "wasm32") || cfg!(feature = "only_use_embedded_assets") {
            let p_res = OptimaPath::new_asset_virtual_path();
            if let Ok(p) = p_res { optima_file_paths.push(p); }
        } else if cfg!(feature = "do_not_embed_assets") {
            let p_res = OptimaPath::new_asset_physical_path_from_json_file();
            if let Ok(p) = p_res { optima_file_paths.push(p); }
        } else {
            let p_res1 = OptimaPath::new_asset_physical_path_from_json_file();
            if let Ok(p) = p_res1 { optima_file_paths.push(p); }
            let p_res2 = OptimaPath::new_asset_virtual_path();
            if let Ok(p) = p_res2 { optima_file_paths.push(p); }
        }

        if optima_file_paths.len() == 0 {
            return Err(OptimaError::new_generic_error_str("OptimaStemCellPath has zero valid paths.", file!(), line!()))
        }

        Ok(Self {
            optima_file_paths
        })
    }
    pub fn new_asset_path_from_string_components(components: &Vec<String>) -> Result<Self, OptimaError> {
        let mut out_path = Self::new_asset_path()?;
        for s in components { out_path.append(s); }
        Ok(out_path)
    }
    pub fn append(&mut self, s: &str) {
        for p in &mut self.optima_file_paths { p.append(s); }
    }
    pub fn append_vec(&mut self, v: &Vec<String>) {
        for p in &mut self.optima_file_paths {
            p.append_vec(v);
        }
    }
    pub fn append_file_location(&mut self, location: &OptimaAssetLocation) {
        for p in &mut self.optima_file_paths {
            p.append_file_location(location);
        }
    }
    pub fn read_file_contents_to_string(&self) -> Result<String, OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::read_file_contents_to_string, "read_file_contents_to_string")
    }
    pub fn write_string_to_file(&self, s: &String) -> Result<(), OptimaError> {
        self.try_function_on_all_optima_file_paths_with_one_param(OptimaPath::write_string_to_file, s, "write_string_to_file")
    }
    pub fn exists(&self) -> bool {
        return self.optima_file_paths[0].exists();
    }
    pub fn get_file_for_writing(&self) -> Result<File, OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::get_file_for_writing, "get_file_for_writing")
    }
    pub fn to_string(&self) -> String {
        return self.optima_file_paths[0].to_string();
    }
    pub fn filename(&self) -> Option<String> {
        return self.optima_file_paths[0].filename();
    }
    pub fn filename_without_extension(&self) -> Option<String> {
        return self.optima_file_paths[0].filename_without_extension();
    }
    pub fn extension(&self) -> Option<String> {
        return self.optima_file_paths[0].extension();
    }
    pub fn set_extension(&mut self, extension: &str) {
        for p in &mut self.optima_file_paths {
            p.set_extension(extension);
        }
    }
    pub fn split_path_into_string_components(&self) -> Vec<String> {
        return self.optima_file_paths[0].split_path_into_string_components();
    }
    pub fn split_path_into_string_components_back_to_assets_dir(&self) -> Result<Vec<String>, OptimaError> {
        return self.optima_file_paths[0].split_path_into_string_components_back_to_asset_dir();
    }
    pub fn delete_file(&self) -> Result<(), OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::delete_file, "delete_file")
    }
    pub fn delete_all_items_in_directory(&self) -> Result<(), OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::delete_all_items_in_directory, "delete_all_items_in_directory")
    }
    pub fn copy_file_to_destination(&self, destination: &OptimaPath) -> Result<(), OptimaError> {
        self.try_function_on_all_optima_file_paths_with_one_param(OptimaPath::copy_file_to_destination, destination, "copy_file_to_destination")
    }
    pub fn verify_extension(&self, extensions: &Vec<&str>) -> Result<(), OptimaError> {
        self.try_function_on_all_optima_file_paths_with_one_param(OptimaPath::verify_extension, extensions, "verify_extension")
    }
    pub fn get_all_items_in_directory(&self, include_directories: bool, include_hidden_files: bool) -> Vec<String> {
        for p in &self.optima_file_paths {
            let items = p.get_all_items_in_directory(include_directories, include_hidden_files);
            if items.len() > 0 { return items; }
        }
        return vec![];
    }
    pub fn get_all_directories_in_directory(&self) -> Vec<String> {
        for p in &self.optima_file_paths {
            let items = p.get_all_directories_in_directory();
            if items.len() > 0 { return items; }
        }
        return vec![];
    }
    pub fn save_object_to_file_as_json<T: Serialize>(&self, object: &T) -> Result<(), OptimaError> {
        self.try_function_on_all_optima_file_paths_with_one_param(OptimaPath::save_object_to_file_as_json, object, "save_object_to_file_as_json")
    }
    pub fn load_object_from_json_file<T: DeserializeOwned>(&self) -> Result<T, OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::load_object_from_json_file, "load_object_from_json_file")
    }
    pub fn walk_directory_and_match(&self, pattern: OptimaPathMatchingPattern, stop_condition: OptimaPathMatchingStopCondition) -> Vec<OptimaPath> {
        for p in &self.optima_file_paths {
            let res = p.walk_directory_and_match(pattern.clone(), stop_condition.clone());
            if res.len() > 0 { return res; }
        }
        return vec![];
    }
    pub fn load_urdf(&self) -> Result<Robot, OptimaError> {
        return self.try_function_on_all_optima_file_paths(OptimaPath::load_urdf, "load_urdf");
    }
    pub fn try_function_on_all_optima_file_paths<T>(&self, f: fn(&OptimaPath) -> Result<T, OptimaError>, function_name: &str) -> Result<T, OptimaError> {
        for p in &self.optima_file_paths {
            let res = f(p);
            if res.is_ok() { return res; }
        }
        return Err(OptimaError::new_generic_error_str(&format!("No valid optima_path in function {:?}", function_name), file!(), line!()));
    }
    pub fn try_function_on_all_optima_file_paths_with_one_param<T, P>(&self, f: fn(&OptimaPath, &P) -> Result<T, OptimaError>, param: &P, function_name: &str) -> Result<T, OptimaError> {
        for p in &self.optima_file_paths {
            let res = f(p, param);
            if res.is_ok() { return res; }
        }
        return Err(OptimaError::new_generic_error_str(&format!("No valid optima_path in function {:?}", function_name), file!(), line!()));
    }
    pub fn optima_file_paths(&self) -> &Vec<OptimaPath> {
        &self.optima_file_paths
    }
}

/// An `OptimaPath` is an object used to load files and write to files in the Optima library.
/// The object is designed to be cross-platform and flexible such that it can either write to or
/// read from the physical harddrive on the user's computer or read from a virtual file system that
/// is embedded into the library binary itself.  When in doubt, use the `OptimaStemCellPath` instead of
/// this object as it has the same underlying functionality, but it will try to automatically select
/// whether it should use a physical or virtual file path based on your target (rust executable, web-assembly,
/// python module, etc).
///
/// # Example
/// ```
/// use optima::utils::utils_files::optima_path::OptimaAssetLocation;
/// use optima::utils::utils_files::optima_path::OptimaPath;
///
/// // Initializes an OptimaPath linking to the embedded virtual path.
/// let mut p = OptimaPath::new_asset_virtual_path().expect("error");
/// p.append_file_location(&OptimaAssetLocation::RobotMeshes {robot_name: "ur5".to_string()});
/// p.append("0.dae");
/// let d = p.load_dae().expect("error");
/// ```
/// Note that the virtual file system (VfsPath) variant does not support any
/// writing operations; if tried, an `UnsupportedOperationError` will be returned.
#[derive(Clone, Debug)]
pub enum OptimaPath {
    Path(PathBuf),
    VfsPath(VfsPath)
}
impl OptimaPath {
    pub fn new_home_path() -> Result<Self, OptimaError> {
        if cfg!(target_arch = "wasm32") {
            return Err(OptimaError::new_unsupported_operation_error("new_home_path",
            "Not supported by wasm32.", file!(), line!()));
        }
        Ok(Self::Path(dirs::home_dir().unwrap().to_path_buf()))
    }
    pub fn new_asset_physical_path_from_json_file() -> Result<Self, OptimaError> {
        if cfg!(target_arch = "wasm32") {
            return Err(OptimaError::new_unsupported_operation_error("new_asset_path_from_json_file",
            "Not supported by wasm32.", file!(), line!()));
        }

        let mut check_path = Self::new_home_path()?;
        check_path.append(".optima_asset_path.JSON");
        if check_path.exists() {
            let path_to_assets_dir_res = check_path.load_object_from_json_file::<PathToAssetsDir>();
            match path_to_assets_dir_res {
                Ok(path_to_asset_dir) => {
                    return Ok(Self::Path(path_to_asset_dir.path_to_assets_dir));
                }
                Err(_) => {
                    let found = Self::auto_create_optima_asset_path_json_file();
                    if !found { return Err(OptimaError::new_generic_error_str("optima_asset folder not found on computer.", file!(), line!())); }
                    else { return Self::new_asset_physical_path_from_json_file(); }
                }
            }
        } else {
            let mut check_path = Self::new_home_path()?;
            check_path.append(".optima_asset_path.lock");
            if check_path.exists() {
                return Err(OptimaError::new_generic_error_str("optima_assets folder not found on computer.  This was indicated by the .optima_asset_path.lock file.", file!(), line!()))
            }
            let found = Self::auto_create_optima_asset_path_json_file();
            if !found { return Err(OptimaError::new_generic_error_str("optima_asset folder not found on computer.", file!(), line!())); }
            else { return Self::new_asset_physical_path_from_json_file(); }
        }
    }
    pub fn new_asset_virtual_path() -> Result<Self, OptimaError> {
        /*
        let root_path = VfsPath::new(PhysicalFS::new(env::current_dir()
            .expect("error")
            .join("..")
            .join("optima_assets")));
        */
        let e: EmbeddedFS<AssetEmbed> = EmbeddedFS::new();
        let root_path = VfsPath::new(e);
        return Ok(Self::VfsPath(root_path));
    }
    pub fn new_asset_physical_path_from_string_components(components: &Vec<String>) -> Result<Self, OptimaError> {
        if cfg!(target_arch = "wasm32") {
            return Err(OptimaError::new_unsupported_operation_error("new_asset_physical_path_from_string_components",
            "Not supported by wasm32.", file!(), line!()));
        }

        let mut path = Self::new_asset_physical_path_from_json_file()?;
        for s in components { path.append(s); }

        return Ok(path);
    }
    pub fn new_asset_virtual_path_from_string_components(components: &Vec<String>) -> Result<Self, OptimaError> {
        let mut path = Self::new_asset_virtual_path()?;
        for s in components { path.append(s); }

        return Ok(path);
    }
    pub fn append(&mut self, s: &str) {
        if s == "" { return; }
        match self {
            OptimaPath::Path(p) => { p.push(s); }
            OptimaPath::VfsPath(p) => { *p = p.join(s).expect("error"); }
        }
    }
    pub fn append_vec(&mut self, v: &Vec<String>) {
        for s in v {
            self.append(s);
        }
    }
    pub fn append_file_location(&mut self, location: &OptimaAssetLocation) {
        let v = location.get_path_wrt_asset_folder();
        match self {
            OptimaPath::Path(p) => {
                for s in v { p.push(s); }
            }
            OptimaPath::VfsPath(p) => {
                for s in v { *p = p.join(s).expect("error"); }
            }
        }
    }
    pub fn read_file_contents_to_string(&self) -> Result<String, OptimaError> {
        return match self {
            OptimaPath::Path(p) => {
                let mut file_res = File::open(p);
                match &mut file_res {
                    Ok(f) => {
                        let mut contents = String::new();
                        let res = f.read_to_string(&mut contents);
                        if res.is_err() {
                            return Err(OptimaError::new_generic_error_str(&format!("Could not read file contents to string for path {:?}", self), file!(), line!()));
                        }
                        Ok(contents)
                    }
                    Err(e) => {
                        Err(OptimaError::new_generic_error_str(e.to_string().as_str(), file!(), line!()))
                    }
                }
            }
            OptimaPath::VfsPath(p) => {
                let mut content = String::new();

                let mut seek_and_read_res = p.open_file();
                match &mut seek_and_read_res {
                    Ok(seek_and_read) => {
                        seek_and_read.read_to_string(&mut content).expect("error");
                        Ok(content)
                    }
                    Err(e) => {
                        Err(OptimaError::new_generic_error_str(&format!("Could not read file.  Error is {:?}.", e.to_string()), file!(), line!()))
                    }
                }
            }
        }
    }
    pub fn write_string_to_file(&self, s: &String) -> Result<(), OptimaError> {
        return match self {
            OptimaPath::Path(p) => {
                let parent_option = p.parent();
                match parent_option {
                    None => { return Err(OptimaError::new_generic_error_str("Could not get parent of path in save_object_to_file_as_json.", file!(), line!())) }
                    Some(parent) => {
                        fs::create_dir_all(parent).expect("error");
                    }
                }

                if p.exists() { fs::remove_file(p).expect("error"); }

                let mut file_res = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(p);

                match &mut file_res {
                    Ok(f) => {
                        f.write(s.as_bytes()).expect("error");
                        Ok(())
                    }
                    Err(e) => {
                        Err(OptimaError::new_generic_error_str(e.to_string().as_str(), file!(), line!()))
                    }
                }
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("save_object_to_file_as_json()",
                                                                 "Writing is not supported by VfsPath.  \
                                                                    Try using a Path variant instead.", file!(), line!()))
            }
        }
    }
    pub fn exists(&self) -> bool {
        return match self {
            OptimaPath::Path(p) => { p.exists() }
            OptimaPath::VfsPath(p) => { p.exists().expect("error") }
        }
    }
    #[allow(unused_must_use)]
    pub fn get_file_for_writing(&self) -> Result<File, OptimaError> {
        return match self {
            OptimaPath::Path(p) => {
                let prefix = p.parent().unwrap();
                std::fs::create_dir_all(prefix).unwrap();
                if p.exists() { std::fs::remove_file(p); }
                let file = OpenOptions::new().write(true).create_new(true).open(p).unwrap();
                Ok(file)
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("get_file_for_writing", "Cannot get file for writing from VfsPath.", file!(), line!()))
            }
        }
    }
    pub fn to_string(&self) -> String {
        return match self {
            OptimaPath::Path(p) => {
                p.to_str().unwrap().to_string()
            }
            OptimaPath::VfsPath(p) => {
                p.as_str().to_string()
            }
        }
    }
    pub fn filename(&self) -> Option<String> {
        return match self {
            OptimaPath::Path(p) => {
                let f = p.file_name();
                match f {
                    None => { None }
                    Some(ff) => { Some(ff.to_str().unwrap().to_string()) }
                }
            }
            OptimaPath::VfsPath(p) => {
                let f = p.filename();
                Some(f)
            }
        }
    }
    pub fn filename_without_extension(&self) -> Option<String> {
        let f = self.filename();
        return match f {
            None => { None }
            Some(ff) => {
                let split: Vec<&str> = ff.split(".").collect();
                Some(split[0].to_string())
            }
        }
    }
    pub fn extension(&self) -> Option<String> {
        return match self {
            OptimaPath::Path(p) => {
                let ext_option = p.extension();
                match ext_option {
                    None => { None }
                    Some(ext) => { Some(ext.to_str().unwrap().to_string()) }
                }
            }
            OptimaPath::VfsPath(p) => {
                let filename = p.filename();
                let split: Vec<&str> = filename.split(".").collect();
                if split.len() <= 1 { None } else { Some(split[split.len() - 1].to_string()) }
            }
        }
    }
    pub fn set_extension(&mut self, extension: &str) {
        match self {
            OptimaPath::Path(p) => {
                p.set_extension(extension);
            }
            OptimaPath::VfsPath(p) => {
                let parent_path_option = p.parent();
                if let Some(parent_path) = &parent_path_option {
                    let filename = p.filename();
                    let split: Vec<&str> = filename.split(".").collect();
                    let mut new_filename = split[0].to_string();
                    if extension != "" {
                        new_filename += format!(".{}", extension).as_str();
                    }
                    *p = parent_path.join(new_filename.as_str()).expect("error");
                }
            }
        }
    }
    pub fn save_object_to_file_as_json<T: Serialize>(&self, object: &T) -> Result<(), OptimaError> {
        return match self {
            OptimaPath::Path(p) => {
                let parent_option = p.parent();
                match parent_option {
                    None => { return Err(OptimaError::new_generic_error_str("Could not get parent of path in save_object_to_file_as_json.", file!(), line!())) }
                    Some(parent) => {
                        fs::create_dir_all(parent).expect("error");
                    }
                }

                if p.exists() { fs::remove_file(p).expect("error"); }

                let mut file_res = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(p);

                match &mut file_res {
                    Ok(f) => {
                        serde_json::to_writer(f, object).expect("error");
                        Ok(())
                    }
                    Err(e) => {
                        Err(OptimaError::new_generic_error_str(e.to_string().as_str(), file!(), line!()))
                    }
                }
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("save_object_to_file_as_json()",
                                                                 "Writing is not supported by VfsPath.  \
                                                                    Try using a Path variant instead.", file!(), line!()))
            }
        }
    }
    pub fn load_object_from_json_file<T: DeserializeOwned>(&self) -> Result<T, OptimaError> {
        OptimaError::new_check_for_path_does_not_exist(self, file!(), line!())?;
        let contents = self.read_file_contents_to_string();
        return match &contents {
            Ok(s) => {
                load_object_from_json_string(s)
            }
            Err(e) => {
                Err(e.clone())
            }
        }
    }
    pub fn walk_directory_and_match(&self, pattern: OptimaPathMatchingPattern, stop_condition: OptimaPathMatchingStopCondition) -> Vec<OptimaPath> {
        let mut out_vec = vec![];

        match self {
            OptimaPath::Path(p) => {
                for entry_res in WalkDir::new(p) {
                    if let Ok(entry) = entry_res {
                        let entry_path = entry.path().to_path_buf();
                        let mut optima_path = Self::Path(entry_path);
                        let matched = Self::directory_walk_standard_entry(&mut optima_path, &mut out_vec, &pattern);
                        if matched {
                            match stop_condition {
                                OptimaPathMatchingStopCondition::First => {
                                    return out_vec;
                                }
                                OptimaPathMatchingStopCondition::All => {}
                            }
                        }
                    }
                }
            }
            OptimaPath::VfsPath(p) => {
                let it_res = p.walk_dir();
                for it in it_res {
                    for entry_res in it {
                        if let Ok(entry) = entry_res {
                            let mut optima_path = Self::VfsPath(entry.clone());
                            let matched = Self::directory_walk_standard_entry(&mut optima_path, &mut out_vec, &pattern);
                            if matched {
                            match stop_condition {
                                OptimaPathMatchingStopCondition::First => {
                                    return out_vec;
                                }
                                OptimaPathMatchingStopCondition::All => {}
                            }
                        }
                        }
                    }
                }
            }
        }

        out_vec
    }
    pub fn split_path_into_string_components(&self) -> Vec<String> {
        let mut out_vec = vec![];

        match self {
            OptimaPath::Path(p) => {
                let mut par = p.clone();
                loop {
                    let filename_option = par.file_name();
                    if let Some(filename) = filename_option {
                        out_vec.insert(0, filename.to_str().unwrap().to_string());
                    }
                    let par_option = par.parent();
                    if let Some(par_some) = par_option {
                        par = par_some.to_path_buf();
                    } else {
                        return out_vec;
                    }
                }
            }
            OptimaPath::VfsPath(p) => {
                let mut par = p.clone();
                loop {
                    let filename = par.filename();
                    out_vec.insert(0, filename);
                    let par_option = par.parent();
                    if let Some(par_some) = par_option {
                        par = par_some.clone();
                    } else {
                        return out_vec;
                    }
                }
            }
        };
    }
    pub fn split_path_into_string_components_back_to_asset_dir(&self) -> Result<Vec<String>, OptimaError> {
        return match self {
            OptimaPath::Path(_) => {
                let string_components = self.split_path_into_string_components();
                let mut optima_assets_idx: Option<usize> = None;
                for (i, s) in string_components.iter().enumerate() {
                    if s == "optima_assets" {
                        optima_assets_idx = Some(i);
                        break;
                    }
                }

                if optima_assets_idx.is_none() {
                    return Err(OptimaError::new_generic_error_str(&format!("optima_assets was not found on the given path {:?}", self), file!(), line!()));
                }

                let optima_assets_idx = optima_assets_idx.unwrap();
                let num_string_components = string_components.len();

                if optima_assets_idx == num_string_components - 1 { return Ok(vec![]); }

                let mut out_vec = vec![];
                for i in optima_assets_idx + 1..num_string_components {
                    out_vec.push(string_components[i].clone());
                }

                Ok(out_vec)
            }
            OptimaPath::VfsPath(_) => {
                Ok(self.split_path_into_string_components())
            }
        }
    }
    #[allow(unused_must_use)]
    pub fn delete_file(&self) -> Result<(), OptimaError> {
        return match self {
            OptimaPath::Path(p) => {
                fs::remove_file(p);
                Ok(())
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("delete_file", "VfsPath does not support deleting files.", file!(), line!()))
            }
        }
    }
    #[allow(unused_must_use)]
    pub fn delete_all_items_in_directory(&self) -> Result<(), OptimaError> {
        return match self {
            OptimaPath::Path(p) => {
                fs::remove_dir_all(p);
                fs::create_dir(p);
                Ok(())
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("delete_all_files_in_directory", "VfsPath does not support deleting files.", file!(), line!()))
            }
        }
    }
    #[allow(unused_must_use)]
    pub fn copy_file_to_destination(&self, destination: &OptimaPath) -> Result<(), OptimaError> {
        if !self.exists() {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to copy file {:?} but it does not exist!", self), file!(), line!()));
        }
        return match self {
            OptimaPath::Path(p) => {
                match destination {
                    OptimaPath::Path(p2) => {
                        if !p2.exists() {
                            let par = p2.parent().unwrap();
                            fs::create_dir_all(par);
                        }
                        fs::copy(p, p2);

                        Ok(())
                    }
                    OptimaPath::VfsPath(_) => {
                        Err(OptimaError::new_unsupported_operation_error("copy_file_to_destination", "VfsPath does not support copying files.", file!(), line!()))
                    }
                }
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("copy_file_to_destination", "VfsPath does not support copying files.", file!(), line!()))
            }
        }
    }
    pub fn verify_extension(&self, extensions: &Vec<&str>) -> Result<(), OptimaError> {
        let ext_option = self.extension();
        match ext_option {
            None => {
                return Err(OptimaError::new_generic_error_str(&format!("Path {:?} does not have one of the following extensions: {:?} ", self, extensions), file!(), line!()));
            }
            Some(ext) => {
                for e in extensions {
                    if e == &ext {
                        return Ok(());
                    }
                }
            }
        }
        return Err(OptimaError::new_generic_error_str(&format!("Path {:?} does not have one of the following extensions: {:?} ", self, extensions), file!(), line!()));
    }
    pub fn get_all_items_in_directory(&self, include_directories: bool, include_hidden_files: bool) -> Vec<String> {
        let mut out_vec = vec![];

        match self {
            OptimaPath::Path(p) => {
                let res = p.read_dir();
                if let Ok(read_dir) = res {
                    for dir_entry_res in read_dir {
                        if let Ok(dir_entry) = dir_entry_res {
                            let filename = dir_entry.file_name();
                            if include_directories || dir_entry.path().is_file() {
                                let f = filename.to_str().unwrap().to_string();
                                if !(f.chars().nth(0).unwrap().to_string() == ".") || include_hidden_files {
                                    out_vec.push(filename.to_str().unwrap().to_string());
                                }
                            }
                        }
                    }
                }
            }
            OptimaPath::VfsPath(p) => {
                let res = p.read_dir();
                if let Ok(read_dir) = res {
                    for i in read_dir {
                        if include_directories || i.is_file().unwrap() {
                            let f = i.filename();
                            if !(f.chars().nth(0).unwrap().to_string() == ".") || include_hidden_files {
                                out_vec.push(i.filename());
                            }
                        }
                    }
                }
            }
        }

        out_vec
    }
    pub fn get_all_directories_in_directory(&self) -> Vec<String> {
        let mut out_vec = vec![];

        match self {
            OptimaPath::Path(p) => {
                let res = p.read_dir();
                if let Ok(read_dir) = res {
                    for dir_entry_res in read_dir {
                        if let Ok(dir_entry) = dir_entry_res {
                            if dir_entry.path().is_dir() {
                                let filename = dir_entry.file_name();
                                let filename_string = filename.to_str().unwrap().to_string();
                                out_vec.push(filename_string);
                            }
                        }
                    }
                }
            }
            OptimaPath::VfsPath(p) => {
                let res = p.read_dir();
                if let Ok(read_dir) = res {
                    for i in read_dir {
                        if i.is_dir().unwrap() {
                            out_vec.push(i.filename());
                        }
                    }
                }
            }
        }

        out_vec
    }
    pub fn load_urdf(&self) -> Result<Robot, OptimaError> {
        let s = self.read_file_contents_to_string()?;
        let robot_res = urdf_rs::read_from_string(&s);
        match robot_res {
            Ok(r) => { Ok(r) }
            Err(_) => { Err(OptimaError::new_generic_error_str(&format!("Robot could not be loaded from path {:?}", self), file!(), line!())) }
        }
    }
    fn directory_walk_standard_entry(optima_path: &mut OptimaPath,
                                     out_vec: &mut Vec<OptimaPath>,
                                     pattern: &OptimaPathMatchingPattern) -> bool {
        let mut matched = false;
        match pattern {
            OptimaPathMatchingPattern::FileOrDirName(s) => {
                let filename_option = optima_path.filename();
                if let Some(filename) = filename_option {
                    if &filename == s {
                        out_vec.push(optima_path.clone());
                        matched = true;
                    }
                }
            }
            OptimaPathMatchingPattern::Extension(s) => {
                let extension_option = optima_path.extension();
                if let Some(extension) = extension_option {
                    if &extension == s {
                        out_vec.push(optima_path.clone());
                        matched = true;
                    }
                }
            }
            OptimaPathMatchingPattern::FilenameWithoutExtension(s) => {
                let filename_option = optima_path.filename_without_extension();
                if let Some(filename) = filename_option {
                    if &filename == s {
                        out_vec.push(optima_path.clone());
                        matched = true;
                    }
                }
            }
            OptimaPathMatchingPattern::PathComponents(v) => {
                let split = optima_path.split_path_into_string_components();
                let v_len = v.len();
                let split_len = split.len();
                if split_len < v_len { return false; }

                let mut local_match = true;
                for i in 0..v_len {
                    if &v[i] != &split[split_len - v_len + i] {
                        local_match = false;
                        break;
                    }
                }

                if local_match {
                    out_vec.push(optima_path.clone());
                    matched = true;
                }
            }
            OptimaPathMatchingPattern::PathComponentsWithoutExtension(v) => {
                let mut optima_path_copy = optima_path.clone();
                optima_path_copy.set_extension("");
                let split = optima_path_copy.split_path_into_string_components();
                let v_len = v.len();
                let split_len = split.len();
                if split_len < v_len { return false; }

                let mut local_match = true;
                for i in 0..v_len {
                    if &v[i] != &split[split_len - v_len + i] {
                        local_match = false;
                        break;
                    }
                }

                if local_match {
                    out_vec.push(optima_path.clone());
                    matched = true;
                }
            }
        }
        return matched;
    }
    #[allow(unused_must_use)]
    fn auto_create_optima_asset_path_json_file() -> bool {
        optima_print("Searching for Optima assets folder...", PrintMode::Println, PrintColor::Cyan, true, 0, None, vec![]);
        let mut home_dir = Self::new_home_path().expect("error");
        let walk_vec = home_dir.walk_directory_and_match(OptimaPathMatchingPattern::FileOrDirName("optima_assets".to_string()), OptimaPathMatchingStopCondition::First);
        return if walk_vec.is_empty() {
            optima_print("WARNING: optima_assets folder not found on your computer.", PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
            let mut lock_path = Self::new_home_path().expect("error");
            lock_path.append(".optima_asset_path.lock");
            lock_path.write_string_to_file(&"".to_string());
            optima_print(&format!("Adding a lock file here: {:?}", lock_path), PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
            optima_print(&format!("If you would like to use a local optima_assets directory on your computer, please delete the lock file once the assets directory is on your computer"), PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
            false
        } else {
            let found_path = walk_vec[0].clone();
            match &found_path {
                OptimaPath::Path(p) => {
                    optima_print(&format!("Optima assets folder found at {:?}", p), PrintMode::Println, PrintColor::Green, true, 0, None, vec![]);
                    home_dir.append(".optima_asset_path.JSON");
                    optima_print(&format!("Saved found path at {:?}", home_dir), PrintMode::Println, PrintColor::Green, true, 0, None, vec![]);
                    let path_to_assets_dir = PathToAssetsDir { path_to_assets_dir: p.clone() };
                    home_dir.save_object_to_file_as_json(&path_to_assets_dir).expect("error");
                    true
                }
                _ => { false }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum OptimaPathType {
    Path,
    Vfs
}

/// Loads an object that implements the `Deserialize` trait from a deserialized json string.
pub fn load_object_from_json_string<T: DeserializeOwned>(json_str: &str) -> Result<T, OptimaError> {
    let o_res = serde_json::from_str(json_str);
    return match o_res {
        Ok(o) => {
            // optima_print(json_str, PrintMode::Println, PrintColor::Green, false);
            Ok(o)
        }
        Err(_) => {
            optima_print(json_str, PrintMode::Println, PrintColor::Red, false, 0, None, vec![]);
            Err(OptimaError::new_generic_error_str("load_object_from_json_string() failed.  The given json_string is incompatible with the requested type.", file!(), line!()))
        }
    }
}

#[derive(RustEmbed, Debug)]
#[folder = "../optima_assets"]
#[exclude = "*/.DS_Store"]
#[cfg_attr(feature = "exclude_all_robot_asset_embedding", exclude = "optima_robots/*")]
#[cfg_attr(feature = "exclude_robot_visual_meshes_embedding", exclude = "*/meshes/*")]
#[cfg_attr(feature = "exclude_robot_visual_meshes_embedding", exclude = "*/input_meshes/*")]
#[cfg_attr(feature = "ur5", include = "*/ur5/*")]
#[cfg_attr(feature = "sawyer", include = "*/sawyer/*")]
#[cfg_attr(feature = "fetch", include = "*/fetch/*")]
#[cfg_attr(feature = "hubo", include = "*/hubo/*")]
struct AssetEmbed;

/// Asset folder location.  Will be used to easily access paths to these locations with respect to
/// the asset folder.
#[derive(Clone, Debug)]
pub enum OptimaAssetLocation {
    RobotSets,
    RobotSet { set_name: String },
    Robots,
    Robot { robot_name: String },
    RobotConfigurations { robot_name: String },
    RobotInputMeshes { robot_name: String },
    RobotMeshes { robot_name: String  },
    RobotGLBMeshes { robot_name: String  },
    RobotPreprocessedData { robot_name: String },
    RobotModuleJsons { robot_name: String },
    RobotModuleJson { robot_name: String, t: RobotModuleJsonType },
    RobotConvexShapes { robot_name: String },
    RobotConvexSubcomponents { robot_name: String },
    Scenes,
    SceneMeshFiles,
    SceneMeshFile { name: String },
    SceneMeshFilePreprocessing { name: String },
    SceneMeshFileConvexShape { name: String },
    SceneMeshFileConvexShapeSubcomponents { name: String },
    FileIO
}
impl OptimaAssetLocation {
    pub fn get_path_wrt_asset_folder(&self) -> Vec<String> {
        return match self {
            OptimaAssetLocation::RobotSets => {
                vec!["optima_robot_sets".to_string()]
            }
            OptimaAssetLocation::RobotSet { set_name } => {
                let mut v = Self::RobotSets.get_path_wrt_asset_folder();
                v.push(set_name.clone());
                v
            }
            OptimaAssetLocation::Robots => {
                vec!["optima_robots".to_string()]
            }
            OptimaAssetLocation::Robot { robot_name } => {
                let mut v = Self::Robots.get_path_wrt_asset_folder();
                v.push(robot_name.clone());
                v
            }
            OptimaAssetLocation::RobotConfigurations { robot_name } => {
                let mut v = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("configurations".to_string());
                v
            }
            OptimaAssetLocation::RobotInputMeshes { robot_name } => {
                let mut v = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("input_meshes".to_string());
                v
            }
            OptimaAssetLocation::RobotMeshes { robot_name } => {
                let mut v = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("meshes".to_string());
                v
            }
            OptimaAssetLocation::RobotGLBMeshes { robot_name } => {
                let mut v = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("glb_meshes".to_string());
                v
            }
            OptimaAssetLocation::RobotPreprocessedData { robot_name } => {
                let mut v = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("preprocessed_data".to_string());
                v
            }
            OptimaAssetLocation::RobotModuleJsons { robot_name } => {
                let mut v = Self::RobotPreprocessedData { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("robot_module_jsons".to_string());
                v
            }
            OptimaAssetLocation::RobotModuleJson { robot_name, t } => {
                let mut v = Self::RobotModuleJsons { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push(t.filename().to_string());
                v
            }
            OptimaAssetLocation::RobotConvexShapes { robot_name } => {
                let mut v = Self::RobotPreprocessedData { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("convex_shapes".to_string());
                v
            }
            OptimaAssetLocation::RobotConvexSubcomponents { robot_name } => {
                let mut v = Self::RobotPreprocessedData { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("convex_shape_subcomponents".to_string());
                v
            }
            OptimaAssetLocation::Scenes => {
                vec!["optima_scenes".to_string()]
            }
            OptimaAssetLocation::SceneMeshFiles => {
                let mut v = Self::Scenes.get_path_wrt_asset_folder();
                v.push("mesh_files".to_string());
                v
            }
            OptimaAssetLocation::SceneMeshFile { name } => {
                let mut v = Self::SceneMeshFiles.get_path_wrt_asset_folder();
                v.push(name.clone());
                v
            }
            OptimaAssetLocation::SceneMeshFilePreprocessing { name } => {
                let mut v = Self::SceneMeshFile { name: name.clone() }.get_path_wrt_asset_folder();
                v.push("preprocessing".to_string());
                v
            }
            OptimaAssetLocation::SceneMeshFileConvexShape { name } => {
                let mut v = Self::SceneMeshFilePreprocessing { name: name.clone() }.get_path_wrt_asset_folder();
                v.push("convex_shape".to_string());
                v
            }
            OptimaAssetLocation::SceneMeshFileConvexShapeSubcomponents { name } => {
                let mut v = Self::SceneMeshFilePreprocessing { name: name.clone() }.get_path_wrt_asset_folder();
                v.push("convex_shape_subcomponents".to_string());
                v
            }
            OptimaAssetLocation::FileIO => {
                vec!["fileIO".to_string()]
            }
        }
    }
}

/// Convenience class that will be used for path_to_assets_dir.JSON file.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PathToAssetsDir {
    pub path_to_assets_dir: PathBuf
}

/// An Enum used to specify a particular patten that should be matched during a directory walk.
#[derive(Debug, Clone)]
pub enum OptimaPathMatchingPattern {
    FileOrDirName(String),
    Extension(String),
    FilenameWithoutExtension(String),
    PathComponents(Vec<String>),
    PathComponentsWithoutExtension(Vec<String>)
}

/// An Enum used to specify when a particular directory walk should stop based on when a pattern is
/// matched.  For example, `First` will stop the directory walk when the given pattern is matched
/// for the first time, while `All` will continue recursively searching through the whole directory
/// until all potential matches are found.
#[derive(Debug, Clone)]
pub enum OptimaPathMatchingStopCondition {
    First,
    All
}

/// Specifies a particular robot module json type.  This enum provides a unified and convenient way
/// to handle paths to particular module json files.
#[derive(Clone, Debug)]
pub enum RobotModuleJsonType {
    ModelModule,
    ShapeGeometryModule,
    ShapeGeometryModulePermanent
}
impl RobotModuleJsonType {
    pub fn filename(&self) -> &str {
        match self {
            RobotModuleJsonType::ModelModule => { "robot_model_module.JSON" }
            RobotModuleJsonType::ShapeGeometryModule => { "robot_shape_geometry_module.JSON" }
            RobotModuleJsonType::ShapeGeometryModulePermanent => { "robot_shape_geometry_module_permanent.JSON" }
        }
    }
}