use std::path::PathBuf;
use std::sync::Mutex;
use bevy::asset::Assets;
use bevy::math::Vec3;
use bevy::pbr::{AlphaMode, PbrBundle};
use bevy::prelude::{AssetServer, Changed, Color, Commands, KeyCode, Query, Res, ResMut, StandardMaterial, Transform, Visibility};
use bevy::ecs::component::Component;
use bevy::input::Input;
use bevy::window::Windows;
use bevy_egui::{egui, EguiContext};
use bevy_egui::egui::{Color32, Ui};
use bevy_prototype_debug_lines::DebugLines;
use rayon::prelude::*;
use crate::optima_bevy::optima_bevy_utils::egui::{EguiActions, EguiContainerMode, EguiSelectionBlockContainer, EguiSelectionMode, EguiWindowStateContainer};
use crate::optima_bevy::optima_bevy_utils::engine::FrameCount;
use crate::optima_bevy::optima_bevy_utils::generic_item::{GenericItemSignature};
use crate::optima_bevy::optima_bevy_utils::gui::GuiGlobalInfo;
use crate::optima_bevy::optima_bevy_utils::materials::{MaterialChangeRequest, MaterialChangeRequestContainer, MaterialChangeRequestType, OptimaBevyMaterial, OptimaBevyMaterialComponent};
use crate::optima_bevy::optima_bevy_utils::transform::TransformUtils;
use crate::optima_tensor_function::OTFImmutVars;
use crate::robot_modules::robot::Robot;
use crate::robot_modules::robot_geometric_shape_module::{RobotGeometricShapeModule, RobotLinkShapeRepresentation, RobotShapeCollectionQuery};
use crate::robot_set_modules::GetRobotSet;
use crate::robot_set_modules::robot_set::RobotSet;
use crate::robot_set_modules::robot_set_joint_state_module::{RobotSetJointState, RobotSetJointStateType};
use crate::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaPathMatchingPattern, OptimaPathMatchingStopCondition, OptimaStemCellPath, path_buf_from_string_components};
use crate::utils::utils_robot::joint::JointAxisPrimitiveType;
use crate::utils::utils_robot::link::Link;
use crate::utils::utils_robot::robot_generic_structures::GenericRobotJointState;
use crate::utils::utils_se3::optima_rotation::OptimaRotationType;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShapeSignature, LogCondition, StopCondition};

pub struct RobotSceneActions;
impl RobotSceneActions {
    pub fn action_spawn_robot_set(commands: &mut Commands,
                                  asset_server: &Res<AssetServer>,
                                  materials: &mut ResMut<Assets<StandardMaterial>>,
                                  robot_set: &RobotSet,
                                  robot_set_joint_state: &RobotSetJointState,
                                  color: Color,
                                  wireframe: bool,
                                  robot_set_idx: usize,
                                  global_offset: Option<Vec3>) {
        let robot_set_configuration_module = robot_set.robot_set_configuration_module();
        let robot_configuration_modules = robot_set_configuration_module.robot_configuration_modules();
        let robot_set_mesh_file_manager = robot_set.robot_set_mesh_file_manager();
        let robot_mesh_file_managers = robot_set_mesh_file_manager.robot_mesh_file_manager_modules();

        let robot_set_fk_res = robot_set.robot_set_kinematics_module().compute_fk(robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");

        let global_offset_transform = match global_offset {
            None => { Vec3::new(0., 0., 0.) }
            Some(g) => { TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(g) }
        };

        for (robot_idx_in_set, robot_configuration_module) in robot_configuration_modules.iter().enumerate() {
            let robot_mesh_file_manager = robot_mesh_file_managers.get(robot_idx_in_set).unwrap();

            let links = robot_configuration_module.robot_model_module().links();
            let paths_to_meshes = robot_mesh_file_manager.get_paths_to_meshes().expect("error");

            for (link_idx_in_robot, link) in links.iter().enumerate() {
                if link.present() {
                    let path_to_mesh = paths_to_meshes.get(link_idx_in_robot).unwrap();
                    if let Some(path_to_mesh) = path_to_mesh {
                        let mut path_buf_back_to_optima_assets = PathBuf::new();
                        path_buf_back_to_optima_assets.push("..");
                        path_buf_back_to_optima_assets.push("..");
                        let string_components = path_to_mesh.split_path_into_string_components_back_to_given_dir("optima_toolbox").unwrap();
                        let path_buf_from_optima_toolbox = path_buf_from_string_components(&string_components);
                        let combined_path_buf = path_buf_back_to_optima_assets.join(path_buf_from_optima_toolbox);
                        let combined_path = combined_path_buf.to_str().unwrap().to_string();
                        let path = if wireframe {
                            combined_path + "#wireframe"
                        } else {
                            combined_path
                        };

                        let pose = robot_set_fk_res.get_pose_from_idxs(robot_idx_in_set, link_idx_in_robot);
                        let mut transform = TransformUtils::util_convert_se3_pose_to_y_up_bevy_transform(pose);

                        transform.translation += global_offset_transform;

                        let mat = StandardMaterial {
                            base_color: color,
                            alpha_mode: AlphaMode::Opaque,
                            ..Default::default()
                        };

                        commands.spawn_bundle(PbrBundle {
                            mesh: asset_server.load(&path),
                            material: materials.add(mat),
                            transform,
                            global_transform: Default::default(),
                            visibility: Visibility::visible(),
                            ..Default::default()
                        }).insert(RobotLinkSpawn {
                            robot_set_idx,
                            robot_idx_in_set,
                            link_idx_in_robot,
                            global_offset: global_offset_transform.clone(),
                        }).insert(GenericItemSignature::RobotLink {
                            robot_set_idx,
                            robot_idx_in_set,
                            link_idx_in_robot,
                        }).insert(OptimaBevyMaterialComponent::new(OptimaBevyMaterial::Color(color.clone())));
                    }
                }
            }
        }

        commands.spawn().insert(RobotSetJointStateBevyComponent {
            robot_set_idx,
            joint_state: robot_set_joint_state.clone(),
        });
    }
    pub fn action_spawn_robot_geometric_shape_scene(commands: &mut Commands,
                                                    asset_server: &Res<AssetServer>,
                                                    materials: &mut ResMut<Assets<StandardMaterial>>,
                                                    robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                                    robot_set_joint_state: &RobotSetJointState,
                                                    robot_color: Color,
                                                    environment_color: Color,
                                                    robot_wireframe: bool,
                                                    environment_wireframe: bool,
                                                    robot_geometric_shape_scene_idx: usize,
                                                    global_offset: Option<Vec3>) {
        Self::action_spawn_robot_set(commands, asset_server, materials, robot_geometric_shape_scene.get_robot_set(), robot_set_joint_state, robot_color, robot_wireframe, robot_geometric_shape_scene_idx, global_offset.clone());

        let poses = robot_geometric_shape_scene.recover_poses(Some(robot_set_joint_state), None, &RobotLinkShapeRepresentation::Cubes).expect("error");

        let global_offset_transform = match global_offset {
            None => { Vec3::new(0., 0., 0.) }
            Some(g) => { TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(g) }
        };

        let env_object_spawners = robot_geometric_shape_scene.env_obj_info_blocks();
        for (env_obj_idx, env_object_spawner) in env_object_spawners.iter().enumerate() {
            let asset_name = env_object_spawner.asset_name.to_string();
            let mut path = OptimaStemCellPath::new_asset_path().expect("error");
            path.append_file_location(&OptimaAssetLocation::SceneMeshFile { name: asset_name.clone() });

            let optima_paths = path.walk_directory_and_match(OptimaPathMatchingPattern::FilenameWithoutExtension(asset_name), OptimaPathMatchingStopCondition::All);
            assert!(optima_paths.len() > 0, "mesh file was not found.");
            let optima_path = optima_paths[optima_paths.len() - 1].clone();

            let mut path_buf_back_to_optima_assets = PathBuf::new();
            path_buf_back_to_optima_assets.push("..");
            path_buf_back_to_optima_assets.push("..");
            let string_components = optima_path.split_path_into_string_components_back_to_given_dir("optima_toolbox").unwrap();
            let path_buf_from_optima_toolbox = path_buf_from_string_components(&string_components);
            let combined_path_buf = path_buf_back_to_optima_assets.join(path_buf_from_optima_toolbox);
            let combined_path = combined_path_buf.to_str().unwrap().to_string();
            let path = if environment_wireframe {
                combined_path + "#wireframe"
            } else {
                combined_path
            };

            let shape_idxs = robot_geometric_shape_scene.get_shape_idxs_from_env_obj_idx(env_obj_idx, &RobotLinkShapeRepresentation::Cubes).expect("error");
            let shape_idx = shape_idxs[0];
            let pose = poses.poses()[shape_idx].clone().unwrap();
            let mut transform = TransformUtils::util_convert_se3_pose_to_y_up_bevy_transform(&pose);
            transform.scale = Vec3::new(env_object_spawner.scale[0] as f32, env_object_spawner.scale[1] as f32, env_object_spawner.scale[2] as f32);

            transform.translation += global_offset_transform;

            let mat = StandardMaterial {
                base_color: environment_color,
                alpha_mode: AlphaMode::Opaque,
                ..Default::default()
            };

            commands.spawn_bundle(PbrBundle {
                mesh: asset_server.load(&path),
                material: materials.add(mat),
                transform,
                ..Default::default()
            })
                .insert(EnvObjSpawn {
                    robot_geometric_shape_scene_idx,
                    env_obj_idx,
                    global_offset: global_offset_transform.clone(),
                })
                .insert(GenericItemSignature::EnvObj { robot_geometric_shape_scene_idx, env_obj_idx })
                .insert(OptimaBevyMaterialComponent::new(OptimaBevyMaterial::Color(environment_color.clone())));
        }
    }
    pub fn action_set_robot_set_joint_state(query: &mut Query<(&RobotLinkSpawn, &mut Transform)>,
                                            robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                            robot_set_joint_state: &RobotSetJointStateBevyComponent) {
        let fk_res = robot_geometric_shape_scene.robot_set().robot_set_kinematics_module().compute_fk(&robot_set_joint_state.joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");

        for (robot_link_spawn, mut transform) in query.iter_mut() {
            let robot_link_spawn_: &RobotLinkSpawn = &robot_link_spawn;

            if robot_link_spawn_.robot_set_idx == robot_set_joint_state.robot_set_idx {
                let pose = fk_res.get_pose_from_idxs(robot_link_spawn_.robot_idx_in_set, robot_link_spawn_.link_idx_in_robot);
                let mut new_transform = TransformUtils::util_convert_se3_pose_to_y_up_bevy_transform(&pose);
                new_transform.translation += robot_link_spawn_.global_offset;

                let transform_: &mut Transform = &mut transform;
                *transform_ = new_transform;
            }
        }
    }
    pub fn action_update_robot_set_joint_state_bevy_component(query: &mut Query<&mut RobotSetJointStateBevyComponent>,
                                                              robot_set_joint_state: &RobotSetJointState,
                                                              robot_set_idx: usize) {
        for mut r in query.iter_mut() {
            if r.robot_set_idx == robot_set_idx {
                r.joint_state = robot_set_joint_state.clone();
            }
        }
    }
    pub fn action_robot_set_joint_sliders_egui(ui: &mut Ui,
                                               robot_set: &RobotSet,
                                               query: &mut Query<&mut RobotSetJointStateBevyComponent>) {
        let joint_axes = robot_set.robot_set_joint_state_module().ordered_joint_axes();

        ui.heading("Robot Set Joint Sliders");

        ui.group(|ui| {
            egui::ScrollArea::vertical()
                .id_source("robot_set_joint_sliders_scroll_area")
                .show(ui, |ui| {
                    for (_, mut robot_set_joint_state_bevy_component) in query.iter_mut().enumerate() {
                        if ui.button("Reset State").clicked() {
                            let zeros_state = robot_set.robot_set_joint_state_module().spawn_zeros_robot_set_joint_state(RobotSetJointStateType::Full);
                            robot_set_joint_state_bevy_component.joint_state = zeros_state;
                        }

                        let id = format!("robot_set_joint_sliders_collapsing_header_robot_{}", robot_set_joint_state_bevy_component.robot_set_idx);
                        egui::CollapsingHeader::new(format!("Robot Set {}", robot_set_joint_state_bevy_component.robot_set_idx))
                            .id_source(id)
                            .default_open(true)
                            .show(ui, |ui| {
                                let robot_set_idx = robot_set_joint_state_bevy_component.robot_set_idx;
                                let id = format!("robot_set_joint_sliders_scroll_area_robot_{}", robot_set_idx);
                                egui::ScrollArea::vertical()
                                    .id_source(id)
                                    .show(ui, |ui| {
                                        let robot_set_joint_state = &mut robot_set_joint_state_bevy_component.joint_state;
                                        assert_eq!(robot_set_joint_state.joint_state().len(), joint_axes.len(), "check to make sure that the joint state is a Full joint state and not DOF.");
                                        for (joint_axis_idx, robot_set_joint_axis) in joint_axes.iter().enumerate() {
                                            ui.group(|ui| {
                                                ui.visuals_mut().override_text_color = Some(Color32::from_rgb(0, 0, 255));
                                                ui.label(format!("Joint Axis {}", joint_axis_idx));
                                                ui.visuals_mut().override_text_color = None;
                                                ui.label(format!(" -- Robot idx in set: {}", robot_set_joint_axis.robot_idx_in_set()));
                                                ui.label(format!(" -- Joint Name: {}, Joint idx: {}", robot_set_joint_axis.joint_axis().joint_name(), robot_set_joint_axis.joint_axis().joint_idx()));
                                                ui.label(format!(" -- Joint Sub-dof idx: {}", robot_set_joint_axis.joint_axis().joint_sub_dof_idx()));
                                                ui.label(format!(" -- Type: {:?}, Axis: {:?}", robot_set_joint_axis.joint_axis().axis_primitive_type(), robot_set_joint_axis.joint_axis().axis_as_unit()));
                                                match robot_set_joint_axis.joint_axis().fixed_value() {
                                                    None => {
                                                        ui.horizontal(|ui| {
                                                            if ui.button("+0.1").clicked() { robot_set_joint_state[joint_axis_idx] += 0.1; }
                                                            if ui.button("-0.1").clicked() { robot_set_joint_state[joint_axis_idx] -= 0.1; }
                                                            if ui.button("+0.01").clicked() { robot_set_joint_state[joint_axis_idx] += 0.01; }
                                                            if ui.button("-0.01").clicked() { robot_set_joint_state[joint_axis_idx] -= 0.01; }
                                                            if ui.button("Reset").clicked() { robot_set_joint_state[joint_axis_idx] = 0.0; }
                                                        });

                                                        ui.horizontal(|ui| {
                                                            let bounds = robot_set_joint_axis.joint_axis().bounds();
                                                            ui.add(egui::Slider::new(&mut robot_set_joint_state[joint_axis_idx], bounds.0..=bounds.1));
                                                        });
                                                    }
                                                    Some(fixed_value) => {
                                                        ui.visuals_mut().override_text_color = Some(Color32::from_rgb(100, 100, 100));
                                                        ui.label(format!("Fixed at value {}", fixed_value));
                                                    }
                                                }
                                            });
                                        }
                                    });
                            });
                    }
                });
        });
    }
    pub fn action_robot_set_link_info_egui(ui: &mut Ui,
                                           robot_set: &RobotSet,
                                           query: &Query<&RobotSetJointStateBevyComponent>,
                                           lines: &mut ResMut<DebugLines>,
                                           robot_link_info_vars: &mut ResMut<RobotLinkInfoVars>,
                                           material_change_request_container: &mut ResMut<MaterialChangeRequestContainer>) {
        let robot_configuration_modules = robot_set.robot_set_configuration_module().robot_configuration_modules();

        ui.heading("Robot Set Link Info");

        ui.horizontal(|ui| {
            if ui.button("Show all frames").clicked() {
                for rr in &mut robot_link_info_vars.link_frame_display { for r in rr { *r = true; } }
            }

            if ui.button("Show no frames").clicked() {
                for rr in &mut robot_link_info_vars.link_frame_display { for r in rr { *r = false; } }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Frame size: ");
            ui.add(egui::DragValue::new(&mut robot_link_info_vars.frame_display_size).clamp_range(0.0..=0.6).speed(0.01))
        });

        let mut f0 = |ui: &mut Ui, robot_set_idx: usize, robot_idx_in_set: usize, link_idx_in_robot: usize, link: &Link, link_pose: &OptimaSE3Pose| {
            let euler_angles_and_translation = link_pose.to_euler_angles_and_translation();
            let e = euler_angles_and_translation.0;
            let t = euler_angles_and_translation.1;
            let bind = link_pose.rotation().convert(&OptimaRotationType::UnitQuaternion);
            let quat = bind.unwrap_unit_quaternion().expect("error");
            ui.group(|ui| {
                ui.label(format!("Robot idx in set: {}", robot_idx_in_set));
                ui.label(format!("Link idx: {}", link_idx_in_robot));
                ui.label(format!("Link name: {}", link.name()));
                ui.label(format!("x: {:.2}, y: {:.2}, z: {:.2}", t[0], t[1], t[2]));
                ui.label(format!("rx: {:.2}, ry: {:.2}, rz: {:.2}", e[0], e[1], e[2]));
                ui.label(format!("qi: {:.2}, qj: {:.2}, qk: {:.2}, qw: {:.2}", quat.i, quat.j, quat.k, quat.w));
                // ui.label(format!("Quaternion: {:?}", quat));
                let display_frame = &mut robot_link_info_vars.link_frame_display[robot_idx_in_set][link_idx_in_robot];
                ui.horizontal(|ui| {
                    ui.checkbox(display_frame, "Show Frame");
                    if ui.radio(false, "Highlight").hovered() {
                        material_change_request_container.add_request(MaterialChangeRequest::new(GenericItemSignature::RobotLink {
                            robot_set_idx,
                            robot_idx_in_set,
                            link_idx_in_robot,
                        }, 0, MaterialChangeRequestType::ChangeButResetInNFrames {
                            material: OptimaBevyMaterial::Color(Color::rgb(0.4, 0.5, 0.1)),
                            n: 1,
                        }))
                    };
                });

                if *display_frame {
                    let bind = link_pose.rotation().convert(&OptimaRotationType::RotationMatrix);
                    let rot_mat = bind.unwrap_rotation_matrix().expect("error").matrix();

                    let size = robot_link_info_vars.frame_display_size;
                    let x_end = t + size * rot_mat.column(0);
                    let y_end = t + size * rot_mat.column(1);
                    let z_end = t + size * rot_mat.column(2);

                    let center_bevy = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(Vec3::new(t[0] as f32, t[1] as f32, t[2] as f32));
                    let x_end_bevy = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(Vec3::new(x_end[0] as f32, x_end[1] as f32, x_end[2] as f32));
                    let y_end_bevy = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(Vec3::new(y_end[0] as f32, y_end[1] as f32, y_end[2] as f32));
                    let z_end_bevy = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(Vec3::new(z_end[0] as f32, z_end[1] as f32, z_end[2] as f32));

                    lines.line_colored(center_bevy, x_end_bevy, 0.0, Color::rgb(1.0, 0.0, 0.0));
                    lines.line_colored(center_bevy, y_end_bevy, 0.0, Color::rgb(0.0, 1.0, 0.0));
                    lines.line_colored(center_bevy, z_end_bevy, 0.0, Color::rgb(0.0, 0.0, 1.0));
                }
            });
        };

        egui::ScrollArea::vertical()
            .id_source("robot_set_link_info_scroll_area")
            .show(ui, |ui| {
                for (_, robot_set_joint_state_bevy_component) in query.iter().enumerate() {
                    let robot_set_joint_state = &robot_set_joint_state_bevy_component.joint_state;
                    let robot_set_fk_res = robot_set.robot_set_kinematics_module().compute_fk(&robot_set_joint_state, &OptimaSE3PoseType::EulerAnglesAndTranslation).expect("error");
                    let robot_set_idx = robot_set_joint_state_bevy_component.robot_set_idx;

                    egui::CollapsingHeader::new(format!("Robot Set {}", robot_set_joint_state_bevy_component.robot_set_idx)).default_open(true).show(ui, |ui| {
                        for (robot_idx_in_set, robot_configuration_module) in robot_configuration_modules.iter().enumerate() {
                            let links = robot_configuration_module.robot_model_module().links();
                            for (link_idx_in_robot, link) in links.iter().enumerate() {
                                let link_pose = robot_set_fk_res.get_pose_option_from_idxs(robot_idx_in_set, link_idx_in_robot);
                                if let Some(pose) = link_pose {
                                    f0(ui, robot_set_idx, robot_idx_in_set, link_idx_in_robot, link, pose);
                                }
                            }
                        }
                    });
                }
            });
    }
    pub fn action_robot_self_collision_calibrator_egui(ui: &mut Ui,
                                                       robot: &Res<Robot>,
                                                       robot_geometric_shape_module: &mut ResMut<RobotGeometricShapeModule>,
                                                       query: &Query<&RobotSetJointStateBevyComponent>) {
        if query.iter().len() != 1 { return; }

        let robot_link_shape_representations = vec![
            RobotLinkShapeRepresentation::Cubes,
            RobotLinkShapeRepresentation::ConvexShapes,
            RobotLinkShapeRepresentation::SphereSubcomponents,
            RobotLinkShapeRepresentation::CubeSubcomponents,
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents,
        ];

        let res_vec = Mutex::new(vec![]);

        for robot_set_joint_state_bevy_component in query.iter() {
            let robot_set_joint_state = &robot_set_joint_state_bevy_component.joint_state;

            let robot_joint_state = robot.spawn_robot_joint_state(robot_set_joint_state.concatenated_state().clone()).expect("It looks like the robot_set used here is more than one robot.  Not legal in this function.");

            let input = RobotShapeCollectionQuery::Contact {
                robot_joint_state: &robot_joint_state,
                prediction: 0.2,
                inclusion_list: &None,
            };

            robot_link_shape_representations.par_iter().for_each(|x| {
                let res = robot_geometric_shape_module.shape_collection_query(&input, x.clone(), StopCondition::None, LogCondition::LogAll, true).expect("error");
                res_vec.lock().unwrap().push((x.clone(), res));
            });
        }

        ui.heading("Robot Self Collision Calibration");

        let res_vec = res_vec.lock().unwrap();

        for (robot_link_shape_representation, res) in res_vec.iter() {
            ui.horizontal(|ui| {
                ui.label(format!("{:?}: ", robot_link_shape_representation));
                if res.intersection_found() {
                    ui.visuals_mut().override_text_color = Some(Color32::from_rgb(255, 0, 0));
                    ui.label("In Collision!");
                } else {
                    ui.visuals_mut().override_text_color = Some(Color32::from_rgb(0, 255, 0));
                    ui.label("Not in Collision.");
                }
                ui.visuals_mut().override_text_color = None;
            });
        }
    }
}

pub struct RobotSceneSystems;
impl RobotSceneSystems {
    pub fn system_spawn_standard_robot_set(mut commands: Commands,
                                           asset_server: Res<AssetServer>,
                                           mut materials: ResMut<Assets<StandardMaterial>>,
                                           immut_vars: Res<OTFImmutVars>,
                                           frame_count: Res<FrameCount>) {
        if frame_count.0 == 1 {
            let robot_set = immut_vars.ref_robot_set();
            let robot_set_joint_state = robot_set.robot_set_joint_state_module().spawn_zeros_robot_set_joint_state(RobotSetJointStateType::Full);
            RobotSceneActions::action_spawn_robot_set(&mut commands, &asset_server, &mut materials, robot_set, &robot_set_joint_state, Color::rgba(0.6, 0.6, 0.7, 1.0), false, 0, None);
        }
    }
    pub fn system_spawn_standard_robot_geometric_shape_scene(mut commands: Commands,
                                                             asset_server: Res<AssetServer>,
                                                             mut materials: ResMut<Assets<StandardMaterial>>,
                                                             immut_vars: Res<OTFImmutVars>,
                                                             frame_count: Res<FrameCount>) {
        if frame_count.0 == 1 {
            let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();
            let robot_set_joint_state = robot_geometric_shape_scene.robot_set().robot_set_joint_state_module().spawn_zeros_robot_set_joint_state(RobotSetJointStateType::Full);
            RobotSceneActions::action_spawn_robot_geometric_shape_scene(&mut commands, &asset_server, &mut materials, robot_geometric_shape_scene, &robot_set_joint_state, Color::rgba(0.6, 0.6, 0.7, 1.0), Color::rgba(0.8, 0.2, 0.8, 1.0), false, false, 0, None);
        }
    }
    pub fn system_set_robot_set_joint_states(immut_vars: Res<OTFImmutVars>,
                                             mut query: Query<&mut RobotSetJointStateBevyComponent, Changed<RobotSetJointStateBevyComponent>>,
                                             mut query2: Query<(&RobotLinkSpawn, &mut Transform)>) {
        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();
        for r in query.iter_mut() {
            RobotSceneActions::action_set_robot_set_joint_state(&mut query2, robot_geometric_shape_scene, &*r);
        }
    }
    pub fn system_robot_set_joint_sliders_egui(mut query: Query<&mut RobotSetJointStateBevyComponent>,
                                               immut_vars: Res<OTFImmutVars>,
                                               windows: Res<Windows>,
                                               mut egui_context: ResMut<EguiContext>,
                                               mut egui_window_state_container: ResMut<EguiWindowStateContainer>,
                                               mut gui_global_info: ResMut<GuiGlobalInfo>) {
        let robot_set = immut_vars.ref_robot_set();

        let f = |ui: &mut Ui| { RobotSceneActions::action_robot_set_joint_sliders_egui(ui, robot_set, &mut query); };

        EguiActions::action_egui_container_generic(f, &EguiContainerMode::LeftPanel { resizable: true, default_width: 300.0 }, "sliders", &windows, &mut egui_context, &mut egui_window_state_container, &mut gui_global_info);
    }
    pub fn system_robot_set_link_info_egui(query: Query<&RobotSetJointStateBevyComponent>,
                                           immut_vars: Res<OTFImmutVars>,
                                           mut lines: ResMut<DebugLines>,
                                           mut robot_link_info_vars: ResMut<RobotLinkInfoVars>,
                                           windows: Res<Windows>,
                                           mut egui_context: ResMut<EguiContext>,
                                           mut egui_window_state_container: ResMut<EguiWindowStateContainer>,
                                           mut gui_global_info: ResMut<GuiGlobalInfo>,
                                           mut material_change_request_container: ResMut<MaterialChangeRequestContainer>) {
        let robot_set = immut_vars.ref_robot_set();

        let f = |ui: &mut Ui| { RobotSceneActions::action_robot_set_link_info_egui(ui, robot_set, &query, &mut lines, &mut robot_link_info_vars, &mut material_change_request_container); };

        EguiActions::action_egui_container_generic(f, &EguiContainerMode::LeftPanel { resizable: true, default_width: 300.0 }, "link_info", &windows, &mut egui_context, &mut egui_window_state_container, &mut gui_global_info);
    }
    pub fn system_robot_self_collision_calibrator_egui(robot: Res<Robot>,
                                                       mut robot_geometric_shape_module: ResMut<RobotGeometricShapeModule>,
                                                       query: Query<&RobotSetJointStateBevyComponent>,
                                                       windows: Res<Windows>,
                                                       mut egui_selection_block_container: ResMut<EguiSelectionBlockContainer>,
                                                       mut egui_context: ResMut<EguiContext>,
                                                       mut egui_window_state_container: ResMut<EguiWindowStateContainer>,
                                                       mut gui_global_info: ResMut<GuiGlobalInfo>,
                                                       mut material_change_request_container: ResMut<MaterialChangeRequestContainer>,
                                                       keys: Res<Input<KeyCode>>) {
        let f = |ui: &mut Ui| {
            ui.heading("Self-Collision Calibrator");

            ui.separator();

            if query.iter().len() != 1 {
                ui.label("Should only be one robot in robot self collision calibrator");
                return;
            }

            ui.label("Robot Link Shape Representation: ");
            EguiActions::action_egui_selection_over_enum::<RobotLinkShapeRepresentation>(ui, "robot_link_shape_representation", EguiSelectionMode::Checkboxes, None, 400.0, &mut egui_selection_block_container, &keys, false);
            let selection_block = egui_selection_block_container.get_selection_mut_ref("robot_link_shape_representation");
            let selection_vec = selection_block.unwrap_selections::<RobotLinkShapeRepresentation>();
            if selection_vec.is_empty() { return; }
            let selection = &selection_vec[0];

            for q in query.iter() {
                let joint_state = &q.joint_state;
                let robot_joint_state = robot.spawn_robot_joint_state(joint_state.concatenated_state().clone()).expect("error");
                let dis_threshold = match selection {
                    RobotLinkShapeRepresentation::Cubes => { 0.2 }
                    RobotLinkShapeRepresentation::ConvexShapes => { 0.15 }
                    RobotLinkShapeRepresentation::SphereSubcomponents => { 0.05 }
                    RobotLinkShapeRepresentation::CubeSubcomponents => { 0.05 }
                    RobotLinkShapeRepresentation::ConvexShapeSubcomponents => { 0.05 }
                    RobotLinkShapeRepresentation::TriangleMeshes => { 0.02 }
                };

                let input = RobotShapeCollectionQuery::Contact {
                    robot_joint_state: &robot_joint_state,
                    prediction: dis_threshold,
                    inclusion_list: &None,
                };

                let res = robot_geometric_shape_module.shape_collection_query(&input, selection.clone(), StopCondition::None, LogCondition::LogAll, true).expect("error");

                ui.separator();

                if ui.button("Reset Collision Grid").clicked() {
                    robot_geometric_shape_module.reset_robot_geometric_shape_collection(selection.clone(), false).expect("error");
                }

                ui.separator();

                if res.intersection_found() {
                    ui.horizontal(|ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::from_rgb(255, 0, 0));
                        ui.label("In collision!");
                        ui.visuals_mut().override_text_color = None;
                        let button = ui.button("Not a collision");
                        if button.clicked() {
                            robot_geometric_shape_module.set_robot_joint_state_as_non_collision(&robot_joint_state, &selection).expect("error");
                        }
                    });
                } else {
                    ui.visuals_mut().override_text_color = Some(Color32::from_rgb(0, 255, 0));
                    ui.label("Not in collision.");
                    ui.visuals_mut().override_text_color = None;
                }

                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("self_collision_calibration_grid")
                        .striped(false)
                        .num_columns(1)
                        .show(ui, |ui| {
                            let outputs = res.outputs();
                            for output in outputs {
                                let contact = output.raw_output().unwrap_contact().expect("error");
                                if let Some(contact) = &contact {
                                    let signatures = output.signatures();
                                    let signature0 = &signatures[0];
                                    let signature1 = &signatures[1];
                                    if contact.dist <= 0.0 {
                                        for signature in vec![signature0, signature1] {
                                            match signature {
                                                GeometricShapeSignature::RobotLink { link_idx, .. } => {
                                                    material_change_request_container.add_request(MaterialChangeRequest::new(GenericItemSignature::RobotLink {
                                                        robot_set_idx: 0,
                                                        robot_idx_in_set: 0,
                                                        link_idx_in_robot: *link_idx,
                                                    }, 0, MaterialChangeRequestType::ChangeButResetInNFrames { material: OptimaBevyMaterial::Color(Color::rgba(1.0, 0.0, 0.0, 0.7)), n: 1 }));
                                                }
                                                _ => { unreachable!() }
                                            }
                                        }
                                    }

                                    ui.group(|ui| {
                                        ui.vertical(|ui| {
                                            ui.label(format!("{:?}", signature0));
                                            ui.label(format!("{:?}", signature1));
                                            ui.label(format!("Dis: {:.5}", contact.dist));
                                            let avg_distance = robot_geometric_shape_module.robot_shape_collection(selection).expect("error").shape_collection().average_distance_from_signatures(&(signature0.clone(), signature1.clone()));
                                            ui.label(format!("Avg dis: {:.5}", avg_distance));
                                            ui.horizontal(|ui| {
                                                let radio = ui.radio(false, "Highlight");
                                                let button = ui.button("Set to always skip");

                                                if radio.is_pointer_button_down_on() || button.hovered() {
                                                    for signature in vec![signature0, signature1] {
                                                        match signature {
                                                            GeometricShapeSignature::RobotLink { link_idx, .. } => {
                                                                material_change_request_container.add_request(MaterialChangeRequest::new(GenericItemSignature::RobotLink {
                                                                    robot_set_idx: 0,
                                                                    robot_idx_in_set: 0,
                                                                    link_idx_in_robot: *link_idx,
                                                                }, 1, MaterialChangeRequestType::ChangeButResetInNFrames { material: OptimaBevyMaterial::Color(Color::rgba(0.8, 0.8, 0.1, 0.7)), n: 1 }));
                                                            }
                                                            _ => { unreachable!() }
                                                        }
                                                    }
                                                }

                                                if button.clicked() {
                                                    robot_geometric_shape_module.set_skip_between_link_pair(signature0, signature1, selection);
                                                }
                                            });
                                        });
                                    });

                                    ui.end_row();
                                }
                            }
                        });
                });
            }
        };

        EguiActions::action_egui_container_generic(f, &EguiContainerMode::LeftPanel { resizable: true, default_width: 300.0 }, "robot_self_collision_calibrator", &windows, &mut egui_context, &mut egui_window_state_container, &mut gui_global_info);
    }
    pub fn system_robot_jacobian_visualization_egui(robot: Res<Robot>,
                                                    query: Query<&RobotSetJointStateBevyComponent>,
                                                    windows: Res<Windows>,
                                                    mut egui_selection_block_container: ResMut<EguiSelectionBlockContainer>,
                                                    mut egui_context: ResMut<EguiContext>,
                                                    mut egui_window_state_container: ResMut<EguiWindowStateContainer>,
                                                    mut gui_global_info: ResMut<GuiGlobalInfo>,
                                                    mut material_change_request_container: ResMut<MaterialChangeRequestContainer>,
                                                    keys: Res<Input<KeyCode>>,
                                                    mut lines: ResMut<DebugLines>) {
        let f = |ui: &mut Ui| {
            ui.heading("Jacobian Vis");

            let links = robot.robot_configuration_module().robot_model_module().links();
            let mut start_link_display_strings = vec![];
            let mut start_link_idxs = vec![];
            for (idx, link) in links.iter().enumerate() {
                start_link_display_strings.push(format!("{}: {}", idx, link.name()));
                start_link_idxs.push(idx);
            }

            let mut end_link_display_strings = vec![];
            let mut end_link_idxs = vec![];
            for (idx, link) in links.iter().enumerate() {
                end_link_display_strings.push(format!("{}: {}", idx, link.name()));
                end_link_idxs.push(idx);
            }

            let joints = robot.robot_configuration_module().robot_model_module().joints();
            let mut dof_display_strings = vec![];
            let mut dof_idxs = vec![];
            let mut count = 0;
            for joint in joints {
                for axis in joint.joint_axes() {
                    dof_display_strings.push(format!("DOF idx {}", count));
                    dof_idxs.push((joint.joint_idx(), axis.joint_sub_dof_idx()));
                    count += 1;
                }
            }

            ui.label("Link start point: ");
            EguiActions::action_egui_selection_over_given_items(ui, "jacobian_vis_start_link", EguiSelectionMode::Checkboxes, start_link_idxs, Some(start_link_display_strings), 150.0, &mut egui_selection_block_container, &keys, false);
            ui.label("Link end point: ");
            EguiActions::action_egui_selection_over_given_items(ui, "jacobian_vis_end_link", EguiSelectionMode::Checkboxes, end_link_idxs, Some(end_link_display_strings), 150.0, &mut egui_selection_block_container, &keys, false);
            ui.label("DOF Column: ");
            ui.horizontal(|ui| {
                if ui.button("Select All").clicked() {
                    egui_selection_block_container
                        .get_selection_mut_ref("jacobian_vis_dofs")
                        .set_selections(dof_idxs.clone());
                }

                if ui.button("Deselect All").clicked() {
                    egui_selection_block_container
                        .get_selection_mut_ref("jacobian_vis_dofs")
                        .flush_selections();
                }
            });
            EguiActions::action_egui_selection_over_given_items(ui, "jacobian_vis_dofs", EguiSelectionMode::Checkboxes, dof_idxs, Some(dof_display_strings), 150.0, &mut egui_selection_block_container, &keys, true);

            let jacobian_types = vec!["Full".to_string(), "Translational".to_string(), "Rotational".to_string()];
            ui.label("Jacobian Type: ");
            EguiActions::action_egui_selection_combobox_dropdown_over_strings(ui, "jacobian_types", jacobian_types, None, &mut egui_selection_block_container);

            let start_link_selections = egui_selection_block_container.get_selection_mut_ref("jacobian_vis_start_link").unwrap_selections::<usize>();
            if start_link_selections.is_empty() { return; }
            let start_link_idx = start_link_selections[0];

            let end_link_selections = egui_selection_block_container.get_selection_mut_ref("jacobian_vis_end_link").unwrap_selections::<usize>();
            if end_link_selections.is_empty() { return; }
            let end_link_idx = end_link_selections[0];

            let chain = robot.robot_configuration_module().robot_model_module().get_link_chain(start_link_idx, end_link_idx).expect("error");
            if chain.is_none() { return; }
            let chain = chain.unwrap();

            let dof_columns = egui_selection_block_container.get_selection_mut_ref("jacobian_vis_dofs").unwrap_selections::<(usize, usize)>();

            let jacobian_type = egui_selection_block_container.get_selection_mut_ref("jacobian_types").unwrap_selections::<String>()[0].clone();

            for robot_set_joint_state_bevy_component in query.iter() {
                let robot_set_joint_state = &robot_set_joint_state_bevy_component.joint_state;
                let robot_joint_state = robot.spawn_robot_joint_state(robot_set_joint_state.concatenated_state().clone()).expect("error");
                let fk_res = robot.robot_kinematics_module().compute_fk(&robot_joint_state, &OptimaSE3PoseType::RotationMatrixAndTranslation).expect("error");

                let start_link_entry = fk_res.get_robot_fk_result_link_entry(start_link_idx);
                let end_link_entry = fk_res.get_robot_fk_result_link_entry(end_link_idx);

                // let start_frame = start_link_entry.pose().as_ref().unwrap().unwrap_rotation_and_translation().expect("error");
                let end_frame = end_link_entry.pose().as_ref().unwrap().unwrap_rotation_and_translation().expect("error");

                // let start_location = start_frame.translation();
                let end_location = end_frame.translation();
                // let start_rotation = start_frame.rotation().unwrap_rotation_matrix().expect("error");
                // let end_rotation = end_frame.rotation().unwrap_rotation_matrix().expect("error");

                let joints = robot.robot_configuration_module().robot_model_module().joints();
                for joint in joints {
                    // let parent_link = joint.preceding_link_idx().unwrap();
                    let child_link = joint.child_link_idx().unwrap();
                    if chain.contains(&child_link) {
                        for joint_axis in joint.joint_axes() {
                            if dof_columns.contains(&(joint.joint_idx(), joint_axis.joint_sub_dof_idx())) {
                                let parent_link_entry = fk_res.get_robot_fk_result_link_entry(child_link);
                                let parent_frame = parent_link_entry.pose().as_ref().unwrap().unwrap_rotation_and_translation().expect("error");

                                let parent_location = parent_frame.translation();
                                let parent_rotation = parent_frame.rotation().unwrap_rotation_matrix().expect("error");

                                let axis = joint_axis.axis();

                                if &jacobian_type == "Full" || &jacobian_type == "Translational" {
                                    match joint_axis.axis_primitive_type() {
                                        JointAxisPrimitiveType::Rotation => {
                                            let global_axis = parent_rotation * axis;
                                            lines.line_colored(TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(parent_location), TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(&(parent_location + global_axis)), 0.0, Color::rgb(1.0, 0.55, 0.01));
                                            lines.line_colored(TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(parent_location), TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(end_location), 0.0, Color::rgb(0.23, 0.79, 1.0));

                                            let connector = end_location - parent_location;
                                            let c = global_axis.cross(&connector);
                                            lines.line_colored(TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(end_location), TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(&(end_location + c)), 0.0, Color::rgb(0.25, 1.0, 0.49));
                                        }
                                        JointAxisPrimitiveType::Translation => {
                                            let local_axis = parent_rotation * axis;
                                            lines.line_colored(TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(end_location), TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(&(end_location + local_axis)), 0.0, Color::rgb(0.25, 1.0, 0.49));
                                        }
                                    }
                                }

                                if &jacobian_type == "Full" || &jacobian_type == "Rotational" {
                                    match joint_axis.axis_primitive_type() {
                                        JointAxisPrimitiveType::Rotation => {
                                            let local_axis = parent_rotation * axis;
                                            lines.line_colored(TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(end_location), TransformUtils::util_convert_z_up_vector3_to_y_up_bevy_vec3(&(end_location + local_axis)), 0.0, Color::rgb(0.97, 1.0, 0.1));
                                        }
                                        JointAxisPrimitiveType::Translation => { }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        EguiActions::action_egui_container_generic(f,&EguiContainerMode::RightPanel { resizable: true, default_width: 300.0 }, "bevy_robot_jacobian_visualization", &windows, &mut egui_context, &mut egui_window_state_container, &mut gui_global_info);
    }
}

#[derive(Component)]
pub struct RobotLinkSpawn {
    pub robot_set_idx: usize,
    pub robot_idx_in_set: usize,
    pub link_idx_in_robot: usize,
    /// Global offset is already in bevy y up space.
    pub global_offset: Vec3,
}

#[derive(Component)]
pub struct EnvObjSpawn {
    pub robot_geometric_shape_scene_idx: usize,
    pub env_obj_idx: usize,
    /// Global offset is already in bevy y up space.
    pub global_offset: Vec3,
}

#[derive(Component)]
pub struct RobotSetJointStateBevyComponent {
    robot_set_idx: usize,
    joint_state: RobotSetJointState,
}

pub struct RobotLinkInfoVars {
    pub frame_display_size: f64,
    pub link_frame_display: Vec<Vec<bool>>,
}

impl RobotLinkInfoVars {
    pub fn new(robot_set: &RobotSet) -> Self {
        let mut link_frame_display = vec![];

        let robot_configuration_modules = robot_set.robot_set_configuration_module().robot_configuration_modules();
        for (robot_idx_in_set, robot_configuration_module) in robot_configuration_modules.iter().enumerate() {
            link_frame_display.push(vec![]);
            let links = robot_configuration_module.robot_model_module().links();
            for _ in links { link_frame_display[robot_idx_in_set].push(false); }
        }

        Self {
            frame_display_size: 0.1,
            link_frame_display,
        }
    }
}