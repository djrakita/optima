#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), derive(Clone, Debug, Serialize, Deserialize))]
pub struct JsMatrix {
    matrix: Vec<Vec<f64>>
}
impl JsMatrix {
    pub fn matrix(&self) -> &Vec<Vec<f64>> {
        &self.matrix
    }
}