use std::collections::HashMap;
use bevy::prelude::{Res, ResMut, Windows};
use bevy_egui::egui::{Color32, Id, Ui, Visuals};
use bevy_egui::{egui, EguiContext};
use bevy_egui::egui::panel::{Side, TopBottomSide};
use crate::optima_bevy::optima_bevy_utils::gui::GuiGlobalInfo;
use crate::robot_set_modules::robot_set::RobotSet;

pub struct EguiActions;
impl EguiActions {
    pub fn action_is_cursor_over_egui_window(gui_global_info: &mut ResMut<GuiGlobalInfo>,
                                             ui: &mut Ui,
                                             windows: &Res<Windows>,
                                             top_buffer: f64,
                                             bottom_buffer: f64,
                                             right_buffer: f64,
                                             left_buffer: f64) {
        if !gui_global_info.is_hovering_over_gui {
            let res = EguiUtils::is_cursor_over_egui_window(ui, windows, top_buffer, bottom_buffer, right_buffer, left_buffer);
            if res {
                gui_global_info.is_hovering_over_gui = true;
            }
        }
    }
    pub fn action_egui_generic<F: FnMut(&mut Ui)>(mut f: F,
                                               egui_container_mode: &EguiContainerMode,
                                               id_source: &str,
                                               windows: &Res<Windows>,
                                               egui_context: &mut ResMut<EguiContext>,
                                               egui_engine: &mut ResMut<EguiEngine>,
                                               gui_global_info: &mut ResMut<GuiGlobalInfo>) {
        match egui_container_mode {
            EguiContainerMode::LeftPanel { resizable } => {
                egui::SidePanel::new(Side::Left, id_source)
                    .resizable(*resizable)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, 1000.0, 1000.0, 50.0, 1000.0);
                        f(ui);
                });
            }
            EguiContainerMode::RightPanel { resizable } => {
                egui::SidePanel::new(Side::Right, id_source)
                    .resizable(*resizable)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, 1000.0, 1000.0, 1000.0, 30.0);
                        f(ui);
                });
            }
            EguiContainerMode::TopPanel { resizable } => {
                egui::TopBottomPanel::new(TopBottomSide::Top, id_source)
                    .resizable(*resizable)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, 1000.0, 60.0, 1000.0, 1000.0);
                        f(ui);
                });
            }
            EguiContainerMode::BottomPanel { resizable } => {
                egui::TopBottomPanel::new(TopBottomSide::Bottom, id_source)
                    .resizable(*resizable)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, 30.0, 1000.0, 1000.0, 1000.0);
                        f(ui);
                });
            }
            EguiContainerMode::FloatingWindow { title, resizable } => {
                egui::Window::new(title)
                    .id(Id::new(id_source))
                    .resizable(*resizable)
                    .open(&mut egui_engine.egui_window_state_container.get_window_state_mut_ref(id_source).open)
                    .show(egui_context.ctx_mut(), |ui| {
                        Self::action_is_cursor_over_egui_window(gui_global_info, ui, windows, 40.0, 40.0, 40.0, 40.0);
                        f(ui);
                    });
            }
        }
    }
}

pub enum EguiContainerMode {
    LeftPanel { resizable: bool },
    RightPanel { resizable: bool },
    TopPanel { resizable: bool },
    BottomPanel { resizable: bool },
    FloatingWindow { title: String, resizable: bool }
}

pub struct EguiSystems;
impl EguiSystems {
    pub fn init_egui_system(mut egui_context: ResMut<EguiContext>) {
        let mut visuals = Visuals::dark();
        let bg = visuals.widgets.noninteractive.bg_fill;
        visuals.widgets.noninteractive.bg_fill = Color32::from_rgba_premultiplied(bg.r(), bg.g(), bg.b(), 100);
        egui_context.ctx_mut().set_visuals(visuals);
    }
    pub fn system_egui_test(windows: Res<Windows>,
                            mut egui_context: ResMut<EguiContext>,
                            mut egui_engine: ResMut<EguiEngine>,
                            mut gui_global_info: ResMut<GuiGlobalInfo>,
                            robot_set: Res<RobotSet>) {

        let f = |ui: &mut Ui| {
            ui.add(egui::Slider::new(&mut 0.0, 0.0..=1.0));
            ui.group(|ui| {
                ui.visuals_mut().override_text_color = Some(Color32::from_rgb(255,0,0));
                ui.visuals_mut().dark_mode = true;
                ui.label("test");
            });
            ui.label("test");
        };
        let f2 = |ui: &mut Ui| {
            ui.add(egui::Slider::new(&mut 0.0, 0.0..=1.0));
            ui.group(|ui| {
                ui.visuals_mut().override_text_color = Some(Color32::from_rgb(255,0,0));
                ui.visuals_mut().dark_mode = true;
                ui.label("test");
            });
            ui.label("test");
        };

        EguiActions::action_egui_generic(f, &EguiContainerMode::RightPanel { resizable: false }, "test", &windows, &mut egui_context, &mut egui_engine, &mut gui_global_info);
        EguiActions::action_egui_generic(f2, &EguiContainerMode::LeftPanel { resizable: false }, "test2", &windows, &mut egui_context, &mut egui_engine, &mut gui_global_info);
        EguiActions::action_egui_generic(f2, &EguiContainerMode::TopPanel { resizable: false }, "test3", &windows, &mut egui_context, &mut egui_engine, &mut gui_global_info);
        EguiActions::action_egui_generic(f2, &EguiContainerMode::BottomPanel { resizable: false }, "test4", &windows, &mut egui_context, &mut egui_engine, &mut gui_global_info);
        EguiActions::action_egui_generic(f2, &EguiContainerMode::FloatingWindow { title: "Test".to_string(), resizable: false }, "test5", &windows, &mut egui_context, &mut egui_engine, &mut gui_global_info);

    }
}

pub struct EguiUtils;
impl EguiUtils {
    pub fn is_cursor_over_egui_window(ui: &mut Ui, windows: &Res<Windows>, top_buffer: f64, bottom_buffer: f64, right_buffer: f64, left_buffer: f64) -> bool {
        let rect = ui.min_rect();
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
}

#[derive(Clone, Debug)]
pub struct EguiEngine {
    pub egui_window_state_container: EguiWindowStateContainer
}
impl EguiEngine {
    pub fn new() -> Self {
        Self {
            egui_window_state_container: EguiWindowStateContainer::new()
        }
    }
}

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



