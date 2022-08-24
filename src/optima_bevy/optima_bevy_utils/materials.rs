use bevy::asset::AssetServer;
use bevy::ecs::component::Component;
use bevy::prelude::{Assets, Color, Handle, Query, Res, ResMut, StandardMaterial};
use bevy::ecs::query::Changed;
use bevy::pbr::AlphaMode;
use crate::optima_bevy::optima_bevy_utils::generic_item::GenericItemSignature;

pub struct MaterialSystems;
impl MaterialSystems {
    pub fn update_optima_bevy_material_components_from_change_requests(mut material_change_request_container: ResMut<MaterialChangeRequestContainer>,
                                                                       mut query: Query<(&mut OptimaBevyMaterialComponent, &GenericItemSignature)>) {
        for (mut optima_bevy_material_component_, signature_) in &mut query {
            let optima_bevy_material_component: &mut OptimaBevyMaterialComponent = &mut optima_bevy_material_component_;
            let signature: &GenericItemSignature = &signature_;

            let binary_search_res = material_change_request_container.material_change_requests.binary_search_by(|x| x.signature.partial_cmp(&signature).unwrap());
            match binary_search_res {
                Ok(idx) => {
                    let material_change_request = &material_change_request_container.material_change_requests[idx];
                    optima_bevy_material_component.updated = true;
                    match &material_change_request.material_change_request_type {
                        MaterialChangeRequestType::Reset => {
                            optima_bevy_material_component.curr_material = optima_bevy_material_component.base_material.clone();
                            optima_bevy_material_component.material_auto_update_mode = MaterialAutoUpdateMode::None;
                        }
                        MaterialChangeRequestType::Change { material } => {
                            optima_bevy_material_component.curr_material = material.clone();
                            optima_bevy_material_component.material_auto_update_mode = MaterialAutoUpdateMode::None;
                        }
                        MaterialChangeRequestType::ChangeButResetInNFrames { material, n } => {
                            optima_bevy_material_component.curr_material = material.clone();
                            optima_bevy_material_component.material_auto_update_mode = MaterialAutoUpdateMode::ResetInNFrames { n: *n + 1 };
                        }
                    }
                }
                Err(_) => {
                    optima_bevy_material_component.updated = false;
                }
            }
        }
    }
    pub fn update_optima_bevy_material_components_from_auto_update(mut query: Query<(&mut OptimaBevyMaterialComponent)>) {
        for mut q in query.iter_mut() {
            match &mut q.material_auto_update_mode {
                MaterialAutoUpdateMode::ResetInNFrames { n } => {
                    *n -= 1;
                    if *n == 0 {
                        q.curr_material = q.base_material.clone();
                        q.material_auto_update_mode = MaterialAutoUpdateMode::None;
                        q.updated = true;
                    }
                }
                MaterialAutoUpdateMode::None => {}
            }
        }
    }
    pub fn update_materials(mut query: Query<(&Handle<StandardMaterial>, &mut OptimaBevyMaterialComponent)>,
                            mut materials: ResMut<Assets<StandardMaterial>>,
                            mut material_change_request_container: ResMut<MaterialChangeRequestContainer>) {
        for (standard_material_, mut optima_bevy_material_component_) in &mut query {
            let standard_material: &Handle<StandardMaterial> = &standard_material_;
            let optima_bevy_material_component: &mut OptimaBevyMaterialComponent = &mut optima_bevy_material_component_;

            if optima_bevy_material_component.updated {
                let m = materials.get_mut(standard_material).unwrap();
                *m = optima_bevy_material_component.curr_material.map_to_color().into();
            }
        }

        material_change_request_container.flush();
    }
}

#[derive(Debug, Clone)]
pub enum OptimaBevyMaterial {
    Color(Color)
}
impl OptimaBevyMaterial {
    pub fn map_to_type(&self) -> OptimaBevyMaterialType {
        match self {
            OptimaBevyMaterial::Color(_) => { OptimaBevyMaterialType::Color }
        }
    }
    pub fn map_to_color(&self) -> Color {
        match self {
            OptimaBevyMaterial::Color(c) => {c.clone()}
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum OptimaBevyMaterialType {
    Color
}

#[derive(Debug, Clone)]
pub enum MaterialAutoUpdateMode {
    None,
    ResetInNFrames { n: usize }
}

#[derive(Component, Debug, Clone)]
pub struct OptimaBevyMaterialComponent {
    curr_material: OptimaBevyMaterial,
    base_material: OptimaBevyMaterial,
    material_auto_update_mode: MaterialAutoUpdateMode,
    updated: bool
}
impl OptimaBevyMaterialComponent {
    pub fn new(material: OptimaBevyMaterial) -> Self {
        Self {
            curr_material: material.clone(),
            base_material: material,
            material_auto_update_mode: MaterialAutoUpdateMode::None,
            updated: false
        }
    }
    pub fn base_material(&self) -> &OptimaBevyMaterial {
        &self.base_material
    }
    pub fn curr_material(&self) -> &OptimaBevyMaterial {
        &self.curr_material
    }
}

#[derive(Debug, Clone)]
pub struct MaterialChangeRequest {
    signature: GenericItemSignature,
    importance: usize,
    material_change_request_type: MaterialChangeRequestType
}
impl MaterialChangeRequest {
    pub fn new(signature: GenericItemSignature, importance: usize, material_change_request_type: MaterialChangeRequestType) -> Self {
        Self {
            signature,
            importance,
            material_change_request_type
        }
    }
}

#[derive(Debug, Clone)]
pub enum MaterialChangeRequestType {
    Reset,
    Change {material: OptimaBevyMaterial},
    ChangeButResetInNFrames { material: OptimaBevyMaterial, n: usize }
}

pub struct MaterialChangeRequestContainer {
    material_change_requests: Vec<MaterialChangeRequest>
}
impl MaterialChangeRequestContainer {
    pub fn new() -> Self {
        Self {
            material_change_requests: vec![]
        }
    }
    pub fn add_request(&mut self, request: MaterialChangeRequest) {
        let binary_search_result = self.material_change_requests.binary_search_by(|x| x.signature.partial_cmp(&request.signature).unwrap());

        match binary_search_result {
            Ok(idx) => {
                let curr_importance = self.material_change_requests[idx].importance;
                let new_importance = request.importance;

                if new_importance > curr_importance { self.material_change_requests[idx] = request; }
            }
            Err(idx) => {
                self.material_change_requests.insert(idx, request);
            }
        }
    }
    pub fn flush(&mut self) { self.material_change_requests.clear(); }
}

