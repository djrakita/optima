use serde::{Serialize, Deserialize};
use std::{io, vec};
use std::io::{BufRead, Stdout};
#[cfg(not(target_arch = "wasm32"))]
use pbr::ProgressBar;
#[cfg(not(target_arch = "wasm32"))]
use colored::Colorize;

pub const NUM_SPACES_PER_TAB: usize = 10;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimaDebug {
    True { debug_level: usize, num_indentation: usize, num_indentation_history: Vec<usize>},
    False
}
impl OptimaDebug {
    pub fn new_true_default() -> Self {
        Self::True {
            debug_level: 2,
            num_indentation: 0,
            num_indentation_history: vec![]
        }
    }
    pub fn spawn_child(&self) -> Self {
        return match self {
            OptimaDebug::True { debug_level, num_indentation, num_indentation_history } => {
                let mut num_indentation_history = num_indentation_history.clone();
                num_indentation_history.push(*num_indentation);
                OptimaDebug::True { debug_level: *debug_level, num_indentation: *num_indentation + crate::utils::utils_console::NUM_SPACES_PER_TAB, num_indentation_history }
            }
            OptimaDebug::False => { OptimaDebug::False }
        }
    }
}

/// Prints the given string with the given color.
///
/// ## Example
/// ```
/// use std::vec;
/// use optima::utils::utils_console::{optima_print, PrintMode, PrintColor};
/// optima_print("test", PrintMode::Print, PrintColor::Blue, false, 0, None,vec![]);
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub fn optima_print(s: &str, mode: PrintMode, color: PrintColor, bolded: bool, num_indentation: usize, fill_character: Option<char>, leading_marks: Vec<(usize, &str)>) {
    let mut string = "".to_string();
    let fill_character = match fill_character {
        None => { ' ' }
        Some(c) => {c}
    };
    for _ in 0..num_indentation {
        string.insert(0, fill_character);
    }
    for (idx, c) in leading_marks {
        if idx < num_indentation {
            string.replace_range(idx..idx+1,c);
        }
    }
    /*
    if bolded { string += format!("{}", style::Bold).as_str() }
    if &color != &PrintColor::None {
        let c = color.get_color_triple();
        string += format!("{}", color::Fg(Rgb(c.0, c.1, c.2))).as_str();
    }
    */
    string += s;

    let mut colored_string = string.normal();
    if bolded { colored_string = colored_string.bold() };
    if &color != &PrintColor::None {
        let c = color.get_color_triple();
        colored_string = colored_string.truecolor(c.0, c.1, c.2);
    }
    // string += format!("{}", style::Reset).as_str();
    match mode {
        PrintMode::Println => { println!("{}", colored_string); }
        PrintMode::Print => { print!("{}", colored_string); }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn optima_print_new_line() {
    optima_print("\n", PrintMode::Print, PrintColor::None, false, 0, None, vec![]);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn optima_print_multi_entry(m: OptimaPrintMultiEntryCollection, num_indentation: usize, fill_character: Option<char>, leading_marks: Vec<(usize, &str)>) {
    for (i, entry) in m.entries.iter().enumerate() {
        let mode = PrintMode::Print;
        let num_indentation = if i == 0 { num_indentation } else { if entry.add_space_before {1} else {0} };
        let fill_character = if i == 0 { fill_character.clone() } else { None };
        let leading_marks = if i == 0 { leading_marks.clone() } else { vec![] };
        optima_print(&entry.line, mode, entry.color.clone(), entry.bolded, num_indentation, fill_character, leading_marks);
    }
    optima_print_new_line();
    // print!("\n");
}

#[derive(Clone)]
pub struct OptimaPrintMultiEntryCollection {
    entries: Vec<OptimaPrintMultiEntry>
}
impl OptimaPrintMultiEntryCollection {
    pub fn new_empty() -> Self {
        Self {
            entries: vec![]
        }
    }
    pub fn add(&mut self, entry: OptimaPrintMultiEntry) { self.entries.push(entry); }
}

#[derive(Clone)]
pub struct OptimaPrintMultiEntry {
    line: String,
    color: PrintColor,
    bolded: bool,
    add_space_before: bool
}
impl OptimaPrintMultiEntry {
    pub fn new_from_str(line: &str, color: PrintColor, bolded: bool, add_space_before: bool) -> Self {
        Self {
            line: line.to_string(),
            color,
            bolded,
            add_space_before
        }
    }
    pub fn new_from_string(line: String, color: PrintColor, bolded: bool, add_space_before: bool) -> Self {
        Self {
            line,
            color,
            bolded,
            add_space_before
        }
    }
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
pub fn optima_print(s: &str, mode: PrintMode, color: PrintColor, bolded: bool, num_indentation: usize, fill_character: Option<char>, leading_marks: Vec<(usize, &str)>) {
    println!("{}", s);
    log(s);
}

#[cfg(target_arch = "wasm32")]
pub fn optima_print_new_line() {
    optima_print("\n", PrintMode::Print, PrintColor::None, false, 0, None, vec![]);
}

#[cfg(target_arch = "wasm32")]
pub fn optima_print_multi_entry(m: OptimaPrintMultiEntryCollection, num_indentation: usize, fill_character: Option<char>) {
    optima_print("multicolor line print not supported in WASM.", PrintMode::Print, PrintColor::None, false, 0);
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
            PrintColor::Cyan => { return (0, 200, 200) }
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
            optima_print(prompt, PrintMode::Println, print_color, true, 0, None, vec![]);
            let stdin = io::stdin();
            let line = stdin.lock().lines().next().unwrap().unwrap();
            Ok(line)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn get_default_progress_bar(max_total_of_bar: usize) -> ProgressBar<Stdout> {
    let mut out_self = ProgressBar::new(max_total_of_bar as u64);
    out_self.show_counter = false;
    out_self.format(&get_progress_bar_format_string());
    out_self
}

#[cfg(not(target_arch = "wasm32"))]
fn get_progress_bar_format_string() -> String {
    // return "".to_string();
    // return "╢▌▌░╟".to_string();
    return "|#--|".to_string();
}
