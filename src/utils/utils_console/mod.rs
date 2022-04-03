use std::io;
use std::io::BufRead;
#[cfg(not(target_arch = "wasm32"))]
use termion::{style, color::Rgb, color};

/// Prints the given string with the given color.
///
/// ## Example
/// ```
/// use optima::utils::utils_console::{optima_print, PrintMode, PrintColor};
/// optima_print("test", PrintMode::Print, PrintColor::Blue, false);
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub fn optima_print(s: &str, mode: PrintMode, color: PrintColor, bolded: bool) {
    let mut string = "".to_string();
    if bolded { string += format!("{}", style::Bold).as_str() }
    if &color != &PrintColor::None {
        let c = color.get_color_triple();
        string += format!("{}", color::Fg(Rgb(c.0, c.1, c.2))).as_str();
    }
    string += s;
    string += format!("{}", style::Reset).as_str();
    match mode {
        PrintMode::Println => { println!("{}", string); }
        PrintMode::Print => { print!("{}", string); }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn optima_print_new_line() {
    optima_print("\n", PrintMode::Print, PrintColor::None, false);
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use crate::utils::utils_errors::OptimaError;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
#[allow(unused)]
pub fn optima_print(s: &str, mode: PrintMode, color: PrintColor, bolded: bool) {
    println!("{}", s);
    log(s);
}

#[cfg(target_arch = "wasm32")]
pub fn optima_print_new_line() {
    optima_print("\n", PrintMode::Print, PrintColor::None, false);
}

/// Enum that is used in print_termion_string function.
/// Println will cause a new line after each line, while Print will not.
#[derive(Clone, Debug)]
pub enum PrintMode {
    Println,
    Print
}

/// Defines color for an optima print command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrintColor {
    None,
    Blue,
    Green,
    Red,
    Yellow,
    Cyan,
    Magenta
}
#[cfg(not(target_arch = "wasm32"))]
impl PrintColor {
    pub fn get_color_triple(&self) -> (u8, u8, u8) {
        match self {
            PrintColor::None => { (0,0,0) }
            PrintColor::Blue => { return (0, 0, 255) }
            PrintColor::Green => { return (0, 255, 0) }
            PrintColor::Red => { return (255, 0, 0) }
            PrintColor::Yellow => { return (255, 255, 0) }
            PrintColor::Cyan => { return (0, 255, 255) }
            PrintColor::Magenta => { return (255, 0, 255) }
        }
    }
}

pub struct ConsoleInputUtils;
impl ConsoleInputUtils {
    pub fn get_console_input_string(prompt: &str, print_color: PrintColor) -> Result<String, OptimaError> {
        return if cfg!(target_arch = "wasm32") {
            Err(OptimaError::new_generic_error_str("wasm32 does not support console input.", file!(), line!()))
        } else {
            optima_print(prompt, PrintMode::Println, print_color, true);
            let stdin = io::stdin();
            let line = stdin.lock().lines().next().unwrap().unwrap();
            Ok(line)
        }
    }
}
