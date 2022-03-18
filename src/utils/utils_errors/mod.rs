
/// A common error type returned by functions throughout the toolbox.
#[derive(Clone, Debug)]
pub enum OptimaError {
    GenericError(String),
    IdxOutOfBoundError(String)
}
impl OptimaError {
    pub fn new_generic_error_str(s: &str) -> Self {
        Self::GenericError(s.to_string())
    }
    pub fn new_generic_error_string(s: String) -> Self {
        Self::GenericError(s)
    }
    pub fn new_idx_out_of_bound_error(given_idx: usize, length_of_array: usize, function_name: &str) -> Self {
        let mut s = format!("Index {:?} is too large for the array of length {:?} in function {}", given_idx, length_of_array, function_name);
        Self::IdxOutOfBoundError(s)
    }
}