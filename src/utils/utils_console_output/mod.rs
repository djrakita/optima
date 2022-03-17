use std::fmt::Display;
use termion::color::Color;
use termion::{style, color};

/// Prints the given string with the given color.
///
/// ## Example
/// ```
/// print_termion_string("test", PrintMode::Print, termion::color::Blue, false);
/// ```
pub fn print_termion_string<T: Color + Clone>(s: &str, mode: PrintMode, color: T, bolded: bool) {
    let mut string = "".to_string();
    if bolded { string += format!("{}", style::Bold).as_str() }
    string += format!("{}", color::Fg(color.clone())).as_str();
    string += s;
    string += format!("{}", style::Reset).as_str();
    match mode {
        PrintMode::Println => { println!("{}", string); }
        PrintMode::Print => { print!("{}", string); }
    }
}

/// Enum that is used in print_termion_string function.
/// Println will cause a new line after each line, while Print will not.
#[derive(Clone, Debug)]
pub enum PrintMode {
    Println,
    Print
}
