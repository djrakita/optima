use std::path::PathBuf;
use bevy::asset::Assets;
use bevy::pbr::PbrBundle;
use bevy::prelude::{AssetServer, Color, Commands, Res, ResMut, StandardMaterial};
use crate::optima_bevy::optima_bevy_utils::engine::FrameCount;
use crate::optima_bevy::optima_bevy_utils::transform::TransformUtils;
use crate::robot_set_modules::robot_set::RobotSet;
use crate::robot_set_modules::robot_set_joint_state_module::{RobotSetJointState, RobotSetJointStateType};
use crate::utils::utils_files::optima_path::path_buf_from_string_components;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;

pub struct RobotActions;
impl RobotActions {
    pub fn action_spawn_robot_set(commands: &mut Commands,
                                  asset_server: &Res<AssetServer>,
                                  materials: &mut ResMut<Assets<StandardMaterial>>,
                                  robot_set: &RobotSet,
                                  robot_set_joint_state: &RobotSetJointState,
                                  color: Color,
                                  wireframe: bool) {
        let robot_set_configuration_module = robot_set.robot_set_configuration_module();
        let robot_configuration_modules = robot_set_configuration_module.robot_configuration_modules();
        let robot_set_mesh_file_manager = robot_set.robot_set_mesh_file_manager();
        let robot_mesh_file_managers = robot_set_mesh_file_manager.robot_mesh_file_manager_modules();

        let robot_set_fk_res = robot_set.robot_set_kinematics_module().compute_fk(robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");

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
                        let transform = TransformUtils::util_convert_se3_pose_to_y_up_bevy_transform(pose);

                        commands.spawn_bundle(PbrBundle {
                            mesh: asset_server.load(&path),
                            material: materials.add(color.into()),
                            transform,
                            ..Default::default()
                        });
                    }
                }
            }
        }
    }
}

pub struct RobotSystems;
impl RobotSystems {
    pub fn system_spawn_robot_set(mut commands: Commands,
                                  asset_server: Res<AssetServer>,
                                  mut materials: ResMut<Assets<StandardMaterial>>,
                                  robot_set: Res<RobotSet>,
                                  frame_count: Res<FrameCount>) {
        if frame_count.0 == 1 {
            let robot_set_joint_state = robot_set.robot_set_joint_state_module().spawn_zeros_robot_set_joint_state(RobotSetJointStateType::Full);
            RobotActions::action_spawn_robot_set(&mut commands, &asset_server, &mut materials, & *robot_set, &robot_set_joint_state, Color::rgb(0.6, 0.6, 0.7), false);
        }
    }
}