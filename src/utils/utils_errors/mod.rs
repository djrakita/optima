// use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};

/// A common error type returned by functions throughout the toolbox.
#[derive(Clone, Debug)]
pub enum OptimaError {
    GenericError(String),
    IdxOutOfBoundError(String),
    UnsupportedOperationError(String)
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
    pub fn new_unsupported_operation_error(function_name: &str, message: &str, file: &str, line: u32) -> Self {
        let s = format!("ERROR: Unsupported operation error in function {}.  {} -- File: {}, Line: {}", function_name, message, file, line);
        // optima_print(&s, PrintMode::Println, PrintColor::Red, true);
        return Self::UnsupportedOperationError(s);
    }
}