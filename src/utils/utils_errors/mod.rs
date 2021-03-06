// use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};

use crate::utils::utils_files::optima_path::{OptimaPath, OptimaStemCellPath};

/// A common error type returned by functions throughout the toolbox.
#[derive(Clone, Debug)]
pub enum OptimaError {
    GenericError(String),
    IdxOutOfBoundError(String),
    UnsupportedOperationError(String),
    RobotStateVecWrongSizeError(String),
    CannotBeNoneError(String),
    PathDoesNotExist(String),
    OptimaTensorFunctionInputError(String)
}
impl OptimaError {
    pub fn new_generic_error_str(s: &str, file: &str, line: u32) -> Self {
        let s = format!("ERROR: {} -- File: {}, Line: {}", s.to_string(), file, line);
        // optima_print(&s, PrintMode::Println, PrintColor::Red, true);
        return Self::GenericError(s);
    }
    pub fn new_idx_out_of_bound_error(given_idx: usize, length_of_array: usize, file: &str, line: u32) -> Self {
        let s = format!("ERROR: Index {:?} is too large for the array of length {:?} -- File: {}, Line: {}", given_idx, length_of_array, file, line);
        // optima_print(&s, PrintMode::Println, PrintColor::Red, true);
        return Self::IdxOutOfBoundError(s)
    }
    pub fn new_check_from_idx_out_of_bound_error(given_idx: usize, length_of_array: usize, file: &str, line: u32) -> Result<(), Self> {
        return if given_idx >= length_of_array {
            Err(Self::new_idx_out_of_bound_error(given_idx, length_of_array, file, line))
        } else {
            Ok(())
        }
    }
    pub fn new_check_for_idx_out_of_bound_error(given_idx: usize, length_of_array: usize, file: &str, line: u32) -> Result<(), Self> {
        return if given_idx >= length_of_array {
            Err(Self::new_idx_out_of_bound_error(given_idx, length_of_array, file, line))
        } else {
            Ok(())
        }
    }
    pub fn new_check_for_cannot_be_none_error<T>(data: &Option<T>, file: &str, line: u32) -> Result<(), Self> {
        return match data {
            None => { Err(Self::CannotBeNoneError(format!("Data cannot be none. -- File {:?}, line {:?}", file, line))) }
            Some(_) => { Ok(()) }
        }
    }
    pub fn new_check_for_path_does_not_exist(path: &OptimaPath, file: &str, line: u32) -> Result<(), Self> {
        return if path.exists() { Ok(()) } else {
            Err(Self::PathDoesNotExist(format!("path: {:?} -- file: {:?}, line: {:?}", path, file, line)))
        }
    }
    pub fn new_check_for_stem_cell_path_does_not_exist(path: &OptimaStemCellPath, file: &str, line: u32) -> Result<(), Self> {
        return if path.exists() { Ok(()) } else {
            Err(Self::PathDoesNotExist(format!("path: {:?} -- file: {:?}, line: {:?}", path, file, line)))
        }
    }
    pub fn new_unsupported_operation_error(function_name: &str, message: &str, file: &str, line: u32) -> Self {
        let s = format!("ERROR: Unsupported operation error in function {}.  {} -- File: {}, Line: {}", function_name, message, file, line);
        // optima_print(&s, PrintMode::Println, PrintColor::Red, true);
        return Self::UnsupportedOperationError(s);
    }
    pub fn new_robot_state_vec_wrong_size_error(function_name: &str, given_robot_state_vec_len: usize, correct_robot_state_vec_len: usize, file: &str, line: u32) -> Self {
        let s = format!("Wrong size of robot state vector in function {}.  It should be length {}, but is currently length {}. -- {}, {}", function_name, correct_robot_state_vec_len, given_robot_state_vec_len, file, line);
        return Self::RobotStateVecWrongSizeError(s);
    }
}