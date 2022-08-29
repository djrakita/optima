use std::collections::HashMap;
use bevy::input::Input;
use bevy::prelude::{KeyCode, Res, ResMut, Windows};
use bevy_egui::egui::{Align, Color32, Direction, Id, Layout, Ui, Visuals};
use bevy_egui::{egui, EguiContext};
use bevy_egui::egui::panel::{Side, TopBottomSide};
use strum::IntoEnumIterator;
use crate::optima_bevy::optima_bevy_utils::gui::GuiGlobalInfo;
use crate::robot_set_modules::robot_set::RobotSet;
use crate::utils::utils_enums::EnumUtils;
use crate::utils::utils_traits::ToAndFromRonString;

pub struct EguiActions;
impl EguiActions {
    pub fn action_is_cursor_over_egui_window(gui_global_info: &mut ResMut<GuiGlobalInfo>,
                                             ui: &mut Ui,
                                             windows: &Res<Windows>,
                                             standard_check: bool,
                                             top_buffer: f64,
                                             bottom_buffer: f64,
                                             right_buffer: f64,
                                             left_buffer: f64) {
        if !gui_global_info.is_hovering_over_gui {
            let res = if standard_check {
                ui.ui_contains_pointer()
            } else {
                EguiUtils::is_cursor_over_egui_window(ui, windows, top_buffer, bottom_buffer, right_buffer, left_buffer)
            };

            if res {
                gui_global_info.is_hovering_over_gui = true;
            }
        }
    }
    pub fn action_egui_container_generic<F: FnMut(&mut Ui)>(mut f: F,
                                                            egui_container_mode: &EguiContainerMode,
                                                            id_source: &str,
                                                            windows: &Res<Windows>,
                                                            egui_context: &mut ResMut<EguiContext>,
                                                            egui_window_state_container: &mut ResMut<EguiWindowStateContainer>,
                                                            gui_global_info: &mut ResMut<GuiGlobalInfo>) {
        match egui_container_mode {
            EguiContainerMode::LeftPanel { resizable, default_width } => {
                egui::SidePanel::new(Side::Left, id_source)
                    .resizable(*resizable)
                    .default_width(*default_width)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, false, 1000.0, 1000.0, 20.0, 1000.0);
                        f(ui);
                });
            }
            EguiContainerMode::RightPanel { resizable, default_width } => {
                egui::SidePanel::new(Side::Right, id_source)
                    .resizable(*resizable)
                    .default_width(*default_width)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, false, 1000.0, 1000.0, 1000.0, 20.0);
                        f(ui);
                });
            }
            EguiContainerMode::TopPanel { resizable } => {
                egui::TopBottomPanel::new(TopBottomSide::Top, id_source)
                    .resizable(*resizable)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, false, 1000.0, 20.0, 1000.0, 1000.0);
                        f(ui);
                });
            }
            EguiContainerMode::BottomPanel { resizable } => {
                egui::TopBottomPanel::new(TopBottomSide::Bottom, id_source)
                    .resizable(*resizable)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, false, 20.0, 1000.0, 1000.0, 1000.0);
                        f(ui);
                });
            }
            EguiContainerMode::FloatingWindow { title, resizable } => {
                egui::Window::new(title)
                    .id(Id::new(id_source))
                    .resizable(*resizable)
                    .open(&mut egui_window_state_container.get_window_state_mut_ref(id_source).open)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, false, 40.0, 40.0, 40.0, 40.0);
                        f(ui);
                    });
            }
        }
    }

    pub fn action_egui_selection_over_enum<T: IntoEnumIterator + ToAndFromRonString>(ui: &mut Ui,
                                                                                     name: &str,
                                                                                     selection_mode: EguiSelectionMode,
                                                                                     egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>,
                                                                                     keys: &Res<Input<KeyCode>>,
                                                                                     allow_multiple_selections: bool) {
        let selection_choices = EnumUtils::convert_all_variants_of_enum_into_ron_strings::<T>();
        Self::action_egui_selection_generic(ui, name, selection_mode, selection_choices, egui_selection_block_container, keys, allow_multiple_selections);
    }
    pub fn action_egui_selection_over_strings(ui: &mut Ui,
                                              name: &str,
                                              selection_mode: EguiSelectionMode,
                                              selection_choices_as_strings: Vec<String>,
                                              egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>,
                                              keys: &Res<Input<KeyCode>>,
                                              allow_multiple_selections: bool) {
        let mut ron_strings = vec![];
        for s in &selection_choices_as_strings { ron_strings.push(s.to_ron_string()); }

        Self::action_egui_selection_generic(ui, name, selection_mode, ron_strings, egui_selection_block_container, keys, allow_multiple_selections);
    }
    pub fn action_egui_selection_over_given_items<T: ToAndFromRonString>(ui: &mut Ui,
                                                                         name: &str,
                                                                         selection_mode: EguiSelectionMode,
                                                                         selection_choices: Vec<T>,
                                                                         egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>,
                                                                         keys: &Res<Input<KeyCode>>,
                                                                         allow_multiple_selections: bool) {
        let mut ron_strings = vec![];
        for s in &selection_choices { ron_strings.push(s.to_ron_string()); }

        Self::action_egui_selection_generic(ui, name, selection_mode, ron_strings, egui_selection_block_container, keys, allow_multiple_selections);
    }
    fn action_egui_selection_generic(ui: &mut Ui,
                                         name: &str,
                                         selection_mode: EguiSelectionMode,
                                         selection_choices_as_ron_strings: Vec<String>,
                                         egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>,
                                         keys: &Res<Input<KeyCode>>,
                                         allow_multiple_selections: bool) {
        let selection_block = egui_selection_block_container.get_selection_mut_ref(name);

        ui.group(|ui| {
            for selection_choice in &selection_choices_as_ron_strings {
                let mut currently_selected = selection_block.selections.contains(selection_choice);
                let mut currently_selected_copy = currently_selected.clone();
                let selection_code = match &selection_mode {
                    EguiSelectionMode::RadioButtons => {
                        if ui.radio(currently_selected_copy, selection_choice.clone()).clicked() {
                            if !currently_selected { 1 } else { -1 }
                        } else { 0 }
                    }
                    EguiSelectionMode::Checkboxes => {
                        if ui.checkbox(&mut currently_selected_copy, selection_choice.clone()).clicked() {
                            if !currently_selected { 1 } else { -1 }
                        } else { 0 }
                    }
                    EguiSelectionMode::SelectionText => {
                        if ui.selectable_label(currently_selected_copy, selection_choice.clone()).clicked() {
                            if !currently_selected { 1 } else { -1 }
                        } else { 0 }
                    }
                };

                if selection_code == -1 && (keys.pressed(KeyCode::RShift) || keys.pressed(KeyCode::LShift)) && allow_multiple_selections {
                    selection_block.remove_selection_by_string(selection_choice);
                }
                else if selection_code == -1 {
                    selection_block.flush_selections();
                    selection_block.insert_selection_by_string(selection_choice.clone());
                }
                else if selection_code == 1 && selection_block.selections.len() == 0 {
                    selection_block.insert_selection_by_string(selection_choice.clone());
                }
                else if selection_code == 1 && selection_block.selections.len() >= 1 && allow_multiple_selections && (keys.pressed(KeyCode::RShift) || keys.pressed(KeyCode::LShift)) {
                    selection_block.insert_selection_by_string(selection_choice.clone());
                }
                else if selection_code == 1 && selection_block.selections.len() >= 1 && allow_multiple_selections && !(keys.pressed(KeyCode::RShift) || keys.pressed(KeyCode::LShift)) {
                    selection_block.flush_selections();
                    selection_block.insert_selection_by_string(selection_choice.clone());
                }
                else if selection_code == 1 && selection_block.selections.len() >= 1 {
                    selection_block.flush_selections();
                    selection_block.insert_selection_by_string(selection_choice.clone());
                }
            }
        });
    }

    pub fn action_egui_selection_combobox_dropdown_over_enum<T: IntoEnumIterator + ToAndFromRonString>(ui: &mut Ui,
                                                                                                       name: &str,
                                                                                                       egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>) {
        let strings = EnumUtils::convert_all_variants_of_enum_into_ron_strings::<T>();
        Self::action_egui_selection_combobox_dropdown_generic(ui, name, strings, egui_selection_block_container);
    }
    pub fn action_egui_selection_combobox_dropdown_over_strings(ui: &mut Ui,
                                                                name: &str,
                                                                selection_choices_as_strings: Vec<String>,
                                                                egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>) {
        let mut ron_strings = vec![];
        for s in &selection_choices_as_strings { ron_strings.push(s.to_ron_string()); }

        Self::action_egui_selection_combobox_dropdown_generic(ui, name, ron_strings, egui_selection_block_container);
    }
    pub fn action_egui_selection_combobox_dropdown_over_given_items<T: ToAndFromRonString>(ui: &mut Ui,
                                                                                           name: &str,
                                                                                           selection_choices: Vec<T>,
                                                                                           egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>) {
        let mut ron_strings = vec![];
        for s in &selection_choices { ron_strings.push(s.to_ron_string()); }

        Self::action_egui_selection_combobox_dropdown_generic(ui, name, ron_strings, egui_selection_block_container);
    }
    fn action_egui_selection_combobox_dropdown_generic(ui: &mut Ui,
                                                       name: &str,
                                                       selection_choices_as_ron_strings: Vec<String>,
                                                       egui_selection_block_container: &mut ResMut<EguiSelectionBlockContainer>) {
        assert!(selection_choices_as_ron_strings.len() > 0);

        let selection_block = egui_selection_block_container.get_selection_mut_ref(name);
        let mut selected = if selection_block.selections.len() > 0 {
            selection_block.selections[0].clone()
        } else {
            selection_block.insert_selection_by_string(selection_choices_as_ron_strings[0].clone());
            selection_choices_as_ron_strings[0].clone()
        };

        egui::ComboBox::new(format!("{}_combobox", name), "")
            .selected_text(format!("{}", selected))
            .show_ui(ui, |ui| {
                for selection_choice in &selection_choices_as_ron_strings {
                    let mut ss = selection_choice.clone();
                    if ui.selectable_value(&mut ss, selected.clone(), selection_choice.as_str()).clicked() {
                        selection_block.flush_selections();
                        selection_block.insert_selection_by_string(selection_choice.clone());
                    }
                }
        });
    }
}

pub enum EguiContainerMode {
    LeftPanel { resizable: bool, default_width: f32 },
    RightPanel { resizable: bool, default_width: f32 },
    TopPanel { resizable: bool },
    BottomPanel { resizable: bool },
    FloatingWindow { title: String, resizable: bool }
}

pub enum EguiSelectionMode {
    RadioButtons,
    Checkboxes,
    SelectionText
}

pub struct EguiSystems;
impl EguiSystems {
    pub fn init_egui_system(mut egui_context: ResMut<EguiContext>) {
        let mut visuals = Visuals::dark();
        let bg = visuals.widgets.noninteractive.bg_fill;
        visuals.widgets.noninteractive.bg_fill = Color32::from_rgba_premultiplied(bg.r(), bg.g(), bg.b(), 150);
        egui_context.ctx_mut().set_visuals(visuals);
    }
}

pub struct EguiUtils;
impl EguiUtils {
    pub fn is_cursor_over_egui_window(ui: &mut Ui, windows: &Res<Windows>, top_buffer: f64, bottom_buffer: f64, right_buffer: f64, left_buffer: f64) -> bool {
        let rect = ui.available_rect_before_wrap();
        let primary_window = windows.get_primary();
        if let Some(primary_window) = primary_window {
            let cursor = primary_window.cursor_position();
            if let Some(cursor) = cursor {
                let x = cursor.x;
                let y = primary_window.height() - cursor.y;

                let right = rect.right();
                let left = rect.left();
                let top = rect.top();
                let bottom = rect.bottom();

                if x < right + right_buffer as f32 && x > left - left_buffer as f32 && y < bottom + bottom_buffer as f32 && y > top - top_buffer as f32 {
                    return true;
                }
            }
        }
        return false;
    }
    pub fn get_default_layout() -> Layout {
        egui::Layout::centered_and_justified(Direction::LeftToRight).with_cross_align(Align::Center).with_cross_justify(true)
    }
}

/*
#[derive(Clone, Debug)]
pub struct EguiEngine {
    pub egui_window_state_container: EguiWindowStateContainer,
    pub egui_selection_block_container: EguiSelectionBlockContainer
}
impl EguiEngine {
    pub fn new() -> Self {
        Self {
            egui_window_state_container: EguiWindowStateContainer::new(),
            egui_selection_block_container: EguiSelectionBlockContainer::new()
        }
    }
}
*/

#[derive(Clone, Debug)]
pub struct EguiWindowStateContainer {
    map: HashMap<String, EguiWindowState>
}
impl EguiWindowStateContainer {
    pub fn new() -> Self {
        Self {
            map: Default::default()
        }
    }
    pub fn get_window_state_mut_ref(&mut self, name: &str) -> &mut EguiWindowState {
        if !self.map.contains_key(name) { self.map.insert(name.to_string(), EguiWindowState::default()); }
        return self.map.get_mut(name).unwrap();
    }
}

#[derive(Clone, Debug)]
pub struct EguiWindowState {
    pub open: bool
}
impl Default for EguiWindowState {
    fn default() -> Self {
        Self {
            open: true
        }
    }
}

#[derive(Clone, Debug)]
pub struct EguiSelectionBlockContainer {
    map: HashMap<String, EguiSelectionBlock>,
}
impl EguiSelectionBlockContainer {
    pub fn new() -> Self {
        Self {
            map: Default::default(),
        }
    }
    pub fn get_selection_mut_ref(&mut self, name: &str) -> &mut EguiSelectionBlock {
        if !self.map.contains_key(name) { self.map.insert(name.to_string(), EguiSelectionBlock::default()); }
        return self.map.get_mut(name).unwrap();
    }
}

/// All selections are ron strings.  If you want to make sure that an Enum is compatible, make sure
/// it derives Serialize, Deserialize.
#[derive(Clone, Debug)]
pub struct EguiSelectionBlock {
    selections: Vec<String>
}
impl Default for EguiSelectionBlock {
    fn default() -> Self {
        Self {
            selections: vec![]
        }
    }
}
impl EguiSelectionBlock {
    pub fn remove_selection_by_string(&mut self, s: &String) {
        for (i, ss) in self.selections.iter().enumerate() {
            if ss == s { self.selections.remove(i); return; }
        }
    }
    pub fn set_selections_by_string(&mut self, selections: Vec<String>) {
        self.selections = selections;
    }
    pub fn insert_selection_by_string(&mut self, selection: String) {
        self.selections.push(selection);
    }
    pub fn set_selections<T: ToAndFromRonString>(&mut self, selections: Vec<T>) {
        self.selections = vec![];
        for s in &selections { self.selections.push(s.to_ron_string()); }
    }
    pub fn insert_selection<T: ToAndFromRonString>(&mut self, selection: T) {
        self.selections.push(selection.to_ron_string());
    }
    pub fn flush_selections(&mut self) { self.selections.clear(); }
    pub fn unwrap_selections<T: ToAndFromRonString>(&self) -> Vec<T> {
        let mut out = vec![];
        for s in &self.selections { out.push(T::from_ron_string(s).expect("error.  Must be wrong type.")); }
        out
    }
}

/*
pub struct EguiEnumSelection {
    pub selections: Vec<Box<dyn IntoEnumIterator>>
}
impl Default for EguiEnumSelection {
    fn default() -> Self {
        Self {
            selections: vec![]
        }
    }
}
*/

