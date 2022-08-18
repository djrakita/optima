use bevy::prelude::ResMut;

pub struct GuiSystems;
impl GuiSystems {
    pub fn system_reset_cursor_over_gui_window(mut gui_global_info: ResMut<GuiGlobalInfo>) {
        gui_global_info.is_hovering_over_gui = false;
    }
}

#[derive(Clone, Debug)]
pub struct GuiGlobalInfo {
    pub is_hovering_over_gui: bool
}
impl GuiGlobalInfo {
    pub fn new() -> Self {
        Self {
            is_hovering_over_gui: false
        }
    }
}