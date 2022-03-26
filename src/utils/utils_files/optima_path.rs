use vfs::*;
use rust_embed::RustEmbed;
use std::{fs};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{PathBuf};
use serde::de::DeserializeOwned;
use serde::{Serialize, Deserialize};
use urdf_rs::Robot;
use walkdir::WalkDir;
use crate::utils::utils_console_output::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;

#[derive(Clone, Debug)]
pub struct OptimaStemCellPath {
    optima_file_paths: Vec<OptimaPath>
}
impl OptimaStemCellPath {
    pub fn new_asset_path() -> Result<Self, OptimaError> {
        let mut optima_file_paths = vec![];

        if cfg!(target = "wasm32") || cfg!(feature = "only_use_embedded_assets") {
            let p_res = OptimaPath::new_asset_virtual_path();
            if let Ok(p) = p_res { optima_file_paths.push(p); }
        } else if cfg!(feature = "do_not_embed_assets") {
            let p_res = OptimaPath::new_asset_path_from_json_file();
            if let Ok(p) = p_res { optima_file_paths.push(p); }
        } else {
            let p_res1 = OptimaPath::new_asset_path_from_json_file();
            if let Ok(p) = p_res1 { optima_file_paths.push(p); }
            let p_res2 = OptimaPath::new_asset_virtual_path();
            if let Ok(p) = p_res2 { optima_file_paths.push(p); }
        }

        if optima_file_paths.len() == 0 {
            return Err(OptimaError::new_generic_error_str("OptimaStemCellPath has zero valid paths."))
        }

        Ok(Self {
            optima_file_paths
        })
    }
    pub fn print_number_of_paths(&self) {
        println!("{:?}", self.optima_file_paths.len());
    }
    pub fn append(&mut self, s: &str) {
        for p in &mut self.optima_file_paths {
            p.append(s);
        }
    }
    pub fn append_file_location(&mut self, location: &OptimaAssetLocation) {
        for p in &mut self.optima_file_paths {
            p.append_file_location(location);
        }
    }
    pub fn read_file_contents_to_string(&self) -> Result<String, OptimaError> {
        for p in &self.optima_file_paths {
            let res = p.read_file_contents_to_string();
            if res.is_ok() { return res; }
        }
        return Err(OptimaError::new_generic_error_str("No valid optima_path in function read_file_contents_to_string()"));
    }
    pub fn write_string_to_file(&self, s: &String) -> Result<(), OptimaError> {
        for p in &self.optima_file_paths {
            let res = p.write_string_to_file(s);
            if res.is_ok() { return res; }
        }
        return Err(OptimaError::new_generic_error_str("No valid optima_path in function write_string_to_file()"));
    }
    pub fn exists(&self) -> bool {
        return self.optima_file_paths[0].exists();
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
    pub fn save_object_to_file_as_json<T: Serialize>(&self, object: &T) -> Result<(), OptimaError> {
        for p in &self.optima_file_paths {
            let res = p.save_object_to_file_as_json(object);
            if res.is_ok() { return res; }
        }
        return Err(OptimaError::new_generic_error_str("No valid optima_path in function save_object_to_file_as_json()"));
    }
    pub fn load_object_from_json_file<T: DeserializeOwned>(&self) -> Result<T, OptimaError> {
        for p in &self.optima_file_paths {
            let res = p.load_object_from_json_file();
            if res.is_ok() { return res; }
        }
        return Err(OptimaError::new_generic_error_str("No valid optima_path in function load_object_from_json_file()"));
    }
    pub fn walk_directory_and_match(&self, pattern: OptimaPathMatchingPattern, stop_condition: OptimaPathMatchingStopCondition) -> Vec<OptimaPath> {
        for p in &self.optima_file_paths {
            let res = p.walk_directory_and_match(pattern.clone(), stop_condition.clone());
            if res.len() > 0 { return res; }
        }
        return vec![];
    }
    pub fn load_urdf(&self) -> Result<Robot, OptimaError> {
        for p in &self.optima_file_paths {
            let res = p.load_urdf();
            if res.is_ok() { return res; }
        }
        return Err(OptimaError::new_generic_error_str("No valid optima_path in function load_urdf()"));
    }
}

/*
#[derive(Clone, Debug)]
pub struct OptimaFilePath {
    root_path: VfsPath,
    path: VfsPath
}
impl OptimaFilePath {
    pub fn new_asset_dir_from_env() -> Self {
        let root_path = VfsPath::new(PhysicalFS::new(env::current_dir()
                .expect("error")
                .join("..")
                .join("optima_assets")));
        Self {
            root_path: root_path.clone(),
            path: root_path
        }
    }
    pub fn new_asset_dir_from_json_file() -> Self {
        todo!()
    }
    pub fn append(&mut self, s: &str) {
        self.path = self.path.join(s).expect("error");
    }
    pub fn append_file_location(&mut self, location: &OptimaFilePathLocation) {
        let v = location.get_path_wrt_asset_folder();
        for s in v {
            self.path = self.path.join(s).expect("error");
        }
    }
    pub fn revert_to_root_path(&mut self) {
        self.path = self.root_path.clone();
    }
    pub fn path(&self) -> &VfsPath {
        &self.path
    }
    pub fn read_file_contents_to_string(&self) -> Result<String, OptimaError> {
        let mut content = String::new();

        let mut seek_and_read_res = self.path.open_file();
        match &mut seek_and_read_res {
            Ok(seek_and_read) => {
                seek_and_read.read_to_string(&mut content).expect("error");
                return Ok(content);
            }
            Err(e) => {
                return Err(OptimaError::new_generic_error_str(&format!("Could not read file.  Error is {:?}.", e.to_string())))
            }
        }
    }
    pub fn read_file_contents_to_bytes(&self) -> Result<Vec<u8>, OptimaError> {
        let mut content = vec![];

        let mut seek_and_read_res = self.path.open_file();
        match &mut seek_and_read_res {
            Ok(seek_and_read) => {
                seek_and_read.read(&mut content).expect("error");
                return Ok(content);
            }
            Err(e) => {
                return Err(OptimaError::new_generic_error_str(&format!("Could not read file.  Error is {:?}.", e.to_string())))
            }
        }
    }
    pub fn get_extension(&self) -> Option<String> {
        let filename = self.path.filename();
        let split: Vec<&str> = filename.split(".").collect();
        if split.len() <= 1 { return None; }
        else { return Some(split[split.len()-1].to_string()) }
    }
    pub fn set_extension(&mut self, extension: &str) {
        let parent_path_option = self.path.parent();
        if let Some(parent_path) = &parent_path_option {
            let filename = self.path.filename();
            let split: Vec<&str> = filename.split(".").collect();
            let mut new_filename = split[0].to_string();
            if extension != "" {
                new_filename += format!(".{}", extension).as_str();
            }
            self.path = parent_path.join(new_filename.as_str()).expect("error");
        }
    }
    pub fn save_object_to_file_as_json<T: Serialize>(&self, object: &T) -> Result<(), OptimaError> {
        let mut write_res = self.path.create_file();
        match &mut write_res {
            Ok(write) => {
                let object_string = serde_json::to_string(object).expect("error");
                write.write(object_string.as_bytes()).expect("error");
            }
            Err(e) => {
                return Err(OptimaError::new_generic_error_str(&format!("Could not save object to file.  Error is {:?}.", e.to_string())))
            }
        }

        Ok(())
    }
    pub fn parse_stl(&self) -> Result<(), OptimaError> {
        // let b = self.read_file_contents_to_string()?;

        let mut seek_and_read_res = self.path.open_file();
        match &mut seek_and_read_res {
            Ok(seek_and_read) => {
                let r = stl_io::read_stl(seek_and_read);
                println!("{:?}", r);
            }
            Err(e) => {
                return Err(OptimaError::new_generic_error_str(&format!("Could not read file.  Error is {:?}.", e.to_string())))
            }
        }

        Ok(())
    }
}

*/
#[derive(Clone, Debug)]
pub enum OptimaPath {
    Path(PathBuf),
    VfsPath(VfsPath)
}
impl OptimaPath {
    pub fn new_home_path() -> Result<Self, OptimaError> {
        if cfg!(target_os = "wasm32") {
            return Err(OptimaError::new_unsupported_operation_error("new_home_path",
            "Not supported by wasm32."));
        }
        Ok(Self::Path(dirs::home_dir().unwrap().to_path_buf()))
    }

    pub fn new_asset_path_from_json_file() -> Result<Self, OptimaError> {
        if cfg!(target_os = "wasm32") {
            return Err(OptimaError::new_unsupported_operation_error("new_asset_path_from_json_file",
            "Not supported by wasm32."));
        }

        let mut check_path = Self::new_home_path()?;
        check_path.append(".optima_asset_path.json");
        if check_path.exists() {
            let path_to_assets_dir_res = check_path.load_object_from_json_file::<PathToAssetsDir>();
            match path_to_assets_dir_res {
                Ok(path_to_asset_dir) => {
                    return Ok(Self::Path(path_to_asset_dir.path_to_assets_dir));
                }
                Err(_) => {
                    let found = Self::auto_create_optima_asset_path_json_file();
                    if !found { return Err(OptimaError::new_generic_error_str("optima_asset folder not found on computer.")); }
                    else { return Self::new_asset_path_from_json_file(); }
                }
            }
        } else {
            let mut check_path = Self::new_home_path()?;
            check_path.append(".optima_asset_path.lock");
            if check_path.exists() {
                return Err(OptimaError::new_generic_error_str("optima_assets folder not found on computer.  This was indicated by the .optima_asset_path.lock file."))
            }
            let found = Self::auto_create_optima_asset_path_json_file();
            if !found { return Err(OptimaError::new_generic_error_str("optima_asset folder not found on computer.")); }
            else { return Self::new_asset_path_from_json_file(); }
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

    pub fn append(&mut self, s: &str) {
        match self {
            OptimaPath::Path(p) => { p.push(s); }
            OptimaPath::VfsPath(p) => { *p = p.join(s).expect("error"); }
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
                        f.read_to_string(&mut contents).expect("error");
                        Ok(contents)
                    }
                    Err(e) => {
                        Err(OptimaError::new_generic_error_str(e.to_string().as_str()))
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
                        Err(OptimaError::new_generic_error_str(&format!("Could not read file.  Error is {:?}.", e.to_string())))
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
                    None => { return Err(OptimaError::new_generic_error_str("Could not get parent of path in save_object_to_file_as_json.")) }
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
                        Err(OptimaError::new_generic_error_str(e.to_string().as_str()))
                    }
                }
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("save_object_to_file_as_json()",
                                                                 "Writing is not supported by VfsPath.  \
                                                                    Try using a Path variant instead."))
            }
        }
    }

    pub fn exists(&self) -> bool {
        return match self {
            OptimaPath::Path(p) => { p.exists() }
            OptimaPath::VfsPath(p) => { p.exists().expect("error") }
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
                    None => { return Err(OptimaError::new_generic_error_str("Could not get parent of path in save_object_to_file_as_json.")) }
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
                        Err(OptimaError::new_generic_error_str(e.to_string().as_str()))
                    }
                }
            }
            OptimaPath::VfsPath(_) => {
                Err(OptimaError::new_unsupported_operation_error("save_object_to_file_as_json()",
                                                                 "Writing is not supported by VfsPath.  \
                                                                    Try using a Path variant instead."))
            }
        }
    }

    pub fn load_object_from_json_file<T: DeserializeOwned>(&self) -> Result<T, OptimaError> {
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
                        let optima_path = Self::Path(entry_path);
                        let matched = Self::directory_walk_standard_entry(&optima_path, &mut out_vec, &pattern);
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
                            let optima_path = Self::VfsPath(entry.clone());
                            let matched = Self::directory_walk_standard_entry(&optima_path, &mut out_vec, &pattern);
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

    pub fn load_urdf(&self) -> Result<Robot, OptimaError> {
        let s = self.read_file_contents_to_string()?;
        let robot_res = urdf_rs::read_from_string(&s);
        match robot_res {
            Ok(r) => { Ok(r) }
            Err(_) => { Err(OptimaError::new_generic_error_str(&format!("Robot could not be loaded from path {:?}", self))) }
        }
    }

    fn directory_walk_standard_entry(optima_path: &OptimaPath,
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
        }
        return matched;
    }

    #[allow(unused_must_use)]
    fn auto_create_optima_asset_path_json_file() -> bool {
        optima_print("Searching for Optima assets folder...", PrintMode::Println, PrintColor::Cyan, true);
        let mut home_dir = Self::new_home_path().expect("error");
        let walk_vec = home_dir.walk_directory_and_match(OptimaPathMatchingPattern::FileOrDirName("optima_assets".to_string()), OptimaPathMatchingStopCondition::First);
        return if walk_vec.is_empty() {
            optima_print("WARNING: optima_assets folder not found on your computer.", PrintMode::Println, PrintColor::Yellow, true);
            let mut lock_path = Self::new_home_path().expect("error");
            lock_path.append(".optima_asset_path.lock");
            lock_path.write_string_to_file(&"".to_string());
            optima_print(&format!("Adding a lock file here: {:?}", lock_path), PrintMode::Println, PrintColor::Yellow, true);
            optima_print(&format!("If you would like to use a local optima_assets directory on your computer, please delete the lock file once the assets directory is on your computer"), PrintMode::Println, PrintColor::Yellow, true);
            false
        } else {
            let found_path = walk_vec[0].clone();
            match &found_path {
                OptimaPath::Path(p) => {
                    optima_print(&format!("Optima assets folder found at {:?}", p), PrintMode::Println, PrintColor::Green, true);
                    home_dir.append(".optima_asset_path.json");
                    optima_print(&format!("Saved found path at {:?}", home_dir), PrintMode::Println, PrintColor::Green, true);
                    let path_to_assets_dir = PathToAssetsDir { path_to_assets_dir: p.clone() };
                    home_dir.save_object_to_file_as_json(&path_to_assets_dir).expect("error");
                    true
                }
                _ => { false }
            }
        }
    }
}

pub fn load_object_from_json_string<T: DeserializeOwned>(json_str: &str) -> Result<T, OptimaError> {
    let o_res = serde_json::from_str(json_str);
    return match o_res {
        Ok(o) => {
            // optima_print(json_str, PrintMode::Println, PrintColor::Green, false);
            Ok(o)
        }
        Err(_) => {
            optima_print(json_str, PrintMode::Println, PrintColor::Red, false);
            Err(OptimaError::new_generic_error_str("load_object_from_json_string() failed.  The given json_string is incompatible with the requested type."))
        }
    }
}

#[derive(RustEmbed, Debug)]
#[folder = "../optima_assets"]
#[exclude = "*/.DS_Store"]
#[cfg_attr(feature = "exclude_all_robot_asset_embedding", exclude = "optima_robots/*")]
#[cfg_attr(feature = "exclude_robot_visual_meshes_embedding", exclude = "meshes/*")]
#[cfg_attr(feature = "ur5", include = "*/ur5/*")]
#[cfg_attr(feature = "sawyer", include = "*/sawyer/*")]
#[cfg_attr(feature = "fetch", include = "*/fetch/*")]
#[cfg_attr(feature = "hubo", include = "*/hubo/*")]
struct AssetEmbed;

/// Asset folder location.  Will be used to easily access paths to these locations with respect to
/// the asset folder.
#[derive(Clone, Debug)]
pub enum OptimaAssetLocation {
    Robots,
    Robot { robot_name: String },
    RobotMeshes { robot_name: String  },
    RobotPreprocessedData { robot_name: String },
    RobotModuleJsons { robot_name: String },
    RobotModuleJson { robot_name: String, t: RobotModuleJsonType },
    RobotConvexShapes { robot_name: String },
    RobotConvexSubcomponents { robot_name: String },
    Environments,
    FileIO
}
impl OptimaAssetLocation {
    pub fn get_path_wrt_asset_folder(&self) -> Vec<String> {
        return match self {
            OptimaAssetLocation::Robots => {
                vec!["optima_robots".to_string()]
            }
            OptimaAssetLocation::Robot { robot_name } => {
                let mut v = Self::Robots.get_path_wrt_asset_folder();
                v.push(robot_name.clone());
                v
            }
            OptimaAssetLocation::RobotMeshes { robot_name } => {
                let mut v = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                v.push("meshes".to_string());
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
            OptimaAssetLocation::Environments => {
                vec!["environments".to_string()]
            }
            OptimaAssetLocation::FileIO => {
                vec!["fileIO".to_string()]
            }
        }
    }
}

/// Convenience class that will be used for path_to_assets_dir.json file.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PathToAssetsDir {
    pub path_to_assets_dir: PathBuf
}

#[derive(Debug, Clone)]
pub enum OptimaPathMatchingPattern {
    FileOrDirName(String),
    Extension(String),
    FilenameWithoutExtension(String)
}

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
    ConfigurationGeneratorModule
}
impl RobotModuleJsonType {
    pub fn filename(&self) -> &str {
        match self {
            RobotModuleJsonType::ModelModule => { "robot_model_module.json" }
            RobotModuleJsonType::ConfigurationGeneratorModule => { "configuration_generator_module.json" }
        }
    }
}