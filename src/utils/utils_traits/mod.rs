use serde::de::DeserializeOwned;
use serde::{Serialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaAssetLocation, OptimaStemCellPath};

pub trait SaveAndLoadable {
    type SaveType: Serialize + DeserializeOwned;

    fn get_save_serialization_object(&self) -> Self::SaveType;
    fn get_serialization_string(&self) -> String {
        serde_json::to_string(&self.get_save_serialization_object()).expect("error")
    }
    fn save_to_path(&self, path: &OptimaStemCellPath) -> Result<(), OptimaError> {
        path.save_object_to_file_as_json(&self.get_save_serialization_object())
    }
    fn load_from_path(path: &OptimaStemCellPath) -> Result<Self, OptimaError> where Self: Sized {
        let s = path.read_file_contents_to_string()?;
        return Self::load_from_json_string(&s);
    }
    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized;
}
impl <T> SaveAndLoadable for Vec<T> where T: SaveAndLoadable{
    type SaveType = Vec<String>;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        let mut out_vec = vec![];

        for s in self {
            out_vec.push(s.get_serialization_string());
        }

        out_vec
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;

        let mut out_vec = vec![];
        for s in &load {
            out_vec.push(T::load_from_json_string(s)?);
        }

        Ok(out_vec)
    }
}

pub trait AssetSaveAndLoadable: SaveAndLoadable {
    fn save_as_asset(&self, location: OptimaAssetLocation) -> Result<(), OptimaError> {
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&location);
        self.save_to_path(&path)
    }
    fn load_as_asset(location: OptimaAssetLocation) -> Result<Self, OptimaError> where Self: Sized {
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&location);
        Self::load_from_path(&path)
    }
}
impl <T> AssetSaveAndLoadable for T where T: SaveAndLoadable { }

pub trait ToAndFromRonString: Serialize + DeserializeOwned {
    fn to_ron_string(&self) -> String {
        ron::to_string(self).expect("error")
    }
    fn from_ron_string(ron_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Result<Self, _> = ron::from_str(ron_str);
        return if let Ok(load) = load { Ok(load) } else {
            Err(OptimaError::new_generic_error_str(&format!("Could not load ron string {:?} into correct type.", ron_str), file!(), line!()))
        }
    }
}
impl <T> ToAndFromRonString for T where T: Serialize + DeserializeOwned {  }

pub trait ToAndFromJsonString: Serialize + DeserializeOwned {
    fn to_json_string(&self) -> String {
        serde_json::to_string(self).expect("error")
    }
    fn from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Result<Self, _> = serde_json::from_str(json_str);
        return if let Ok(load) = load { Ok(load) } else {
            Err(OptimaError::new_generic_error_str(&format!("Could not load json string {:?} into correct type.", json_str), file!(), line!()))
        }
    }
}
impl <T> ToAndFromJsonString for T where T: Serialize + DeserializeOwned {  }

pub trait ToAndFromTomlString: Serialize + DeserializeOwned {
    fn to_toml_string(&self) -> String {
        toml::to_string(self).expect("error")
    }
    fn from_toml_string(toml_string: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Result<Self, _> = toml::from_str(toml_string);
        return if let Ok(load) = load { Ok(load) } else {
            Err(OptimaError::new_generic_error_str(&format!("Could not load toml string {:?} into correct type.", toml_string), file!(), line!()))
        }
    }
}
impl <T> ToAndFromTomlString for T where T: Serialize + DeserializeOwned {  }

