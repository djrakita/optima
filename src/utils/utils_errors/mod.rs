
/// A common error type returned by functions throughout the toolbox.
#[derive(Clone, Debug)]
pub enum OptimaError {
    StringDescriptor(String)
}
impl OptimaError {
    pub fn new_string_descriptor_error(s: &str) -> Self {
        Self::StringDescriptor(s.to_string())
    }
}