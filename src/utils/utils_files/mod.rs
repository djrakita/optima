use std::{env, fs};
use std::fs::{File, read_dir, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use crate::utils::utils_errors::OptimaError;

/// Convenience struct that holds many class functions related to file utils.
pub struct FileUtils;
impl FileUtils {
    /// Returns file path to the location from which the program is being executed.
    pub fn get_path_to_src() -> PathBuf {
        let path_buf = env::current_dir().expect("error");
        return path_buf;
    }
    /// Reads contents of file and outputs it to a string.
    pub fn read_file_contents_to_string(p: &PathBuf) -> Result<String, OptimaError> {
        let mut file_res = File::open(p);
        return match &mut file_res {
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
    /// Returns file extension of path as string.
    pub fn get_file_extension_string(p: &PathBuf) -> Option<String> {
        let e = p.extension();
        return match e {
            None => { None }
            Some(o) => { Some(o.to_str().expect("error").to_string()) }
        }
    }
    /// Returns the paths of all files within a directory.
    pub fn get_all_files_in_directory(p: &PathBuf) -> Result<Vec<PathBuf>, OptimaError> {
        let mut out: Vec<PathBuf> = Vec::new();
        let it_res = read_dir(p.clone());
        match it_res {
            Ok(it) => {
                for i in it {
                    let path = i.expect("error").path();
                    out.push(path);
                }
            }
            Err(_) => {
                return Err(OptimaError::new_generic_error_string(format!("filepath {:?} does not exist.", p)));
            }
        }
        Ok(out)
    }
    /// Saves given object to a file as a JSON string.  The object must be serializable using serde json.
    pub fn save_object_to_file_as_json<T: Serialize>(object: &T, p: &PathBuf) -> Result<(), OptimaError> {
        let parent_option = p.parent();
        match parent_option {
            None => { return Err(OptimaError::new_generic_error_str("Could not get parent of path in save_object_to_file_as_json.")) }
            Some(parent) => {
                fs::create_dir_all(parent).expect("error");
            }
        }

        let mut file_res = OpenOptions::new()
            .write(true)
            .create(true)
            .open(p);
        return match &mut file_res {
            Ok(f) => {
                serde_json::to_writer(f, object).expect("error");
                Ok(())
            }
            Err(e) => {
                Err(OptimaError::new_generic_error_str(e.to_string().as_str()))
            }
        }
    }
    /// Reads object that was serialized by serde JSON from a file.
    /// ## Example
    /// ```
    /// use std::path::Path;
    /// use nalgebra::Vector3;
    /// use optima::utils::utils_files::FileUtils;
    ///
    /// let res = FileUtils::load_object_from_json_file::<Vector3<f64>>(&Path::new("data.json").to_path_buf());
    /// ```
    pub fn load_object_from_json_file<T: DeserializeOwned>(p: &PathBuf) -> Result<T, OptimaError> {
        let contents = Self::read_file_contents_to_string(p);
        return match &contents {
            Ok(s) => {
                Self::load_object_from_json_string(s)
            }
            Err(e) => {
                Err(e.clone())
            }
        }
    }

    pub fn load_object_from_json_string<T: DeserializeOwned>(json_str: &str) -> Result<T, OptimaError> {
        let o_res = serde_json::from_str(json_str);
        return match o_res {
            Ok(o) => { Ok(o) }
            Err(_) => {
                Err(OptimaError::new_generic_error_str("load_object_from_json_string() failed.  The given json_string is incompatible with the requested type."))
            }
        }
    }
}

/// Convenience struct that holds many class functions related to the assets folder utils.
pub struct AssetDirUtils;
impl AssetDirUtils {

    #[cfg(target_arch = "wasm32")]
    pub fn get_path_to_assets_dir() -> Result<PathBuf, OptimaError> {
        return Err(OptimaError::new_generic_error_str("tried to call get_path_to_assets_dir() from wasm.\
            This is not a supported operation from wasm.  If you need to access files from wasm, \
            you should do something like fetch commands from javascript and provide the file contents \
            to Rust functions."));
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Returns file path to the Optima toolbox assets directory.
    /// This is read in from a file, path_to_optima_toolbox_assets.json, which is stored in the folder
    /// that the program is being executed from.
    /// If this file is not present, this function will automatically write a file for the user.
    /// If this file contains inaccurate information, this function will return an error.
    pub fn get_path_to_assets_dir() -> Result<PathBuf, OptimaError> {
        return Self::get_path_to_assets_dir_via_json_file();
    }

    fn get_path_to_assets_dir_via_json_file() -> Result<PathBuf, OptimaError> {
        let mut path_to_assets_dir_file = FileUtils::get_path_to_src();
        path_to_assets_dir_file.push("path_to_optima_toolbox_assets.json");
        let path_exists = path_to_assets_dir_file.exists();

        return match path_exists {
            true => {
                let path_to_assets_dir_res = FileUtils::load_object_from_json_file::<PathToAssetsDir>(&path_to_assets_dir_file);
                match &path_to_assets_dir_res {
                    Ok(p) => {
                        let path_buffer = match p.path_type {
                            PathType::Absolute => {
                                p.path_to_assets_dir.clone()
                            }
                            PathType::Relative => {
                                let mut path = FileUtils::get_path_to_src();
                                path.push(p.path_to_assets_dir.clone());
                                path
                            }
                        };

                        // let path_buffer = p.path_to_assets_dir.clone();
                        let path_exists = path_buffer.exists();
                        match path_exists {
                            true => {
                                Ok(path_buffer)
                            }
                            false => {
                                let console_strings = vec![
                                    format!("The path specified in path_to_optima_toolbox_assets.json file at {:?} is incorrect.", FileUtils::get_path_to_src()),
                                    format!("Please correct this path and re-run the application.")
                                ];

                                for s in &console_strings {
                                    // print_termion_string(s.as_str(),
                                    //                     PrintMode::Println,
                                    //                     color::Red,
                                    //                     true);
                                    println!("{}", s);
                                }
                                Err(OptimaError::new_generic_error_str("The path specified in path_to_optima_toolbox_assets.json is incorrect."))
                            }
                        }
                    }
                    Err(e) => {
                        let console_strings = vec![
                            format!("The path specified in path_to_optima_toolbox_assets.json file at {:?} is not a valid path.", FileUtils::get_path_to_src()),
                            format!("Please correct this path and re-run the application.")
                        ];

                        for s in &console_strings {
                            // print_termion_string(s.as_str(),
                            //                      PrintMode::Println,
                            //                      color::Red,
                            //                      true);
                            println!("{}", s);
                        }
                        Err(e.clone())
                    }
                }
            }
            false => {
                let console_strings = vec![
                    format!("I noticed that there is not a path_to_optima_toolbox_assets.json file at {:?}", FileUtils::get_path_to_src()),
                    format!("{}", "I will save a file there."),
                    format!("{}", "Please open the file at this location and fill in the absolute path.  Once this path is specified, please run the program again.")
                ];

                for s in &console_strings {
                    // print_termion_string(s.as_str(),
                    //                      PrintMode::Println,
                    //                      color::Cyan,
                    //                      true);
                    println!("{}", s);
                }

                let pp = PathToAssetsDir::default();
                FileUtils::save_object_to_file_as_json(&pp, &path_to_assets_dir_file)?;
                Err(OptimaError::new_generic_error_str("path_to_optima_toolbox_assets.json file did not exist yet."))
            }
        }
    }

    /// Returns file path to the given location in the Optima toolbox assets directory
    pub fn get_path_to_asset_dir_location(l: AssetDirLocation) -> Result<PathBuf, OptimaError> {
        let mut p = Self::get_path_to_assets_dir()?;
        let a = l.get_path_wrt_asset_folder();
        p = p.join(a);
        return Ok(p);
    }
}

/// Asset folder location.  Will be used to easily access paths to these locations with respect to
/// the asset folder.
#[derive(Clone, Debug)]
pub enum AssetDirLocation {
    Robots,
    Robot { robot_name: String },
    RobotMeshes { robot_name: String  },
    RobotPreprocessedData { robot_name: String },
    RobotModuleJsons { robot_name: String },
    RobotConvexShapes { robot_name: String },
    RobotConvexSubcomponents { robot_name: String },
    Environments,
    FileIO
}
impl AssetDirLocation {
    pub fn get_path_wrt_asset_folder(&self) -> PathBuf {
        return match self {
            AssetDirLocation::Robots => {
                Path::new("optima_robots").to_path_buf()
            }
            AssetDirLocation::Robot { robot_name } => {
                let mut out_path = Self::Robots.get_path_wrt_asset_folder();
                out_path = out_path.join(robot_name.as_str());
                out_path
            }
            AssetDirLocation::RobotMeshes { robot_name } => {
                let mut out_path = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                out_path = out_path.join("meshes");
                out_path
            }
            AssetDirLocation::RobotPreprocessedData { robot_name } => {
                let mut out_path = Self::Robot { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                out_path = out_path.join("preprocessed_data");
                out_path
            }
            AssetDirLocation::RobotModuleJsons { robot_name } => {
                let mut out_path = Self::RobotPreprocessedData { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                out_path = out_path.join("robot_module_jsons");
                out_path
            }
            AssetDirLocation::RobotConvexShapes { robot_name } => {
                let mut out_path = Self::RobotPreprocessedData { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                out_path = out_path.join("convex_shapes");
                out_path
            }
            AssetDirLocation::RobotConvexSubcomponents { robot_name } => {
                let mut out_path = Self::RobotPreprocessedData { robot_name: robot_name.clone() }.get_path_wrt_asset_folder();
                out_path = out_path.join("convex_shape_subcomponents");
                out_path
            }
            AssetDirLocation::Environments => {
                Path::new("environments").to_path_buf()
            }
            AssetDirLocation::FileIO => {
                Path::new("fileIO").to_path_buf()
            }
        }
    }
}

/// Convenience class that will be used for path_to_assets_dir.json file.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PathToAssetsDir {
    path_type: PathType,
    path_to_assets_dir: PathBuf
}
impl Default for PathToAssetsDir {
    fn default() -> Self {
        // let mut path = FileUtils::get_path_to_src();
        let mut path = PathBuf::new();
        path.push("..");
        path.push("optima_assets");
        Self {
            path_type: PathType::Relative,
            path_to_assets_dir: path
        }
    }
}

/// Convenience enum that will be used for path_to_assets_dir.json file.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
enum PathType {
    Absolute,
    Relative
}

/// Convenience struct that holds many class functions related to the robot folder within assets.
pub struct RobotDirUtils;
impl RobotDirUtils {
    pub fn get_path_to_urdf_file(robot_name: &str) -> Result<PathBuf, OptimaError> {
        let path = AssetDirUtils::get_path_to_asset_dir_location(AssetDirLocation::Robot { robot_name: robot_name.to_string() })?;
        let all_files = FileUtils::get_all_files_in_directory(&path)?;
        for f in &all_files {
            let ext_option = f.extension();
            if let Some(ext) = ext_option {
                if ext == "urdf" || ext == "URDF" {
                    return Ok(f.clone());
                }
            }
        }
        return Err(OptimaError::new_generic_error_str(format!("Robot directory for robot {:?} does not contain a urdf.", robot_name).as_str()))
    }
    pub fn get_path_to_robot_module_json(robot_name: &str, robot_module_json: RobotModuleJsonType) -> Result<PathBuf, OptimaError> {
        let mut p = AssetDirUtils::get_path_to_asset_dir_location(AssetDirLocation::RobotPreprocessedData { robot_name: robot_name.to_string() })?;
        p.push("robot_module_jsons");
        p.push(robot_module_json.filename());
        return Ok(p);
    }
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