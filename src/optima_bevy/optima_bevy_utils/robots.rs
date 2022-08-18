use std::path::PathBuf;
use bevy::asset::Assets;
use bevy::math::Vec3;
use bevy::pbr::PbrBundle;
use bevy::prelude::{AssetServer, Color, Commands, Res, ResMut, StandardMaterial};
use crate::optima_bevy::optima_bevy_utils::engine::FrameCount;
use crate::optima_bevy::optima_bevy_utils::transform::TransformUtils;
use crate::robot_set_modules::GetRobotSet;
use crate::robot_set_modules::robot_set::RobotSet;
use crate::robot_set_modules::robot_set_joint_state_module::{RobotSetJointState, RobotSetJointStateType};
use crate::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaPathMatchingPattern, OptimaPathMatchingStopCondition, OptimaStemCellPath, path_buf_from_string_components};
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
    pub fn action_spawn_robot_geometric_shape_scene(commands: &mut Commands,
                                                    asset_server: &Res<AssetServer>,
                                                    materials: &mut ResMut<Assets<StandardMaterial>>,
                                                    robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                                    robot_set_joint_state: &RobotSetJointState,
                                                    robot_color: Color,
                                                    environment_color: Color,
                                                    robot_wireframe: bool,
                                                    environment_wireframe: bool) {
        Self::action_spawn_robot_set(commands, asset_server, materials, robot_geometric_shape_scene.get_robot_set(), robot_set_joint_state, robot_color, robot_wireframe);

        let poses = robot_geometric_shape_scene.recover_poses(robot_set_joint_state, None).expect("error");

        let env_object_spawners = robot_geometric_shape_scene.env_obj_spawners();
        for (env_obj_idx, env_object_spawner) in env_object_spawners.iter().enumerate() {
            let asset_name = env_object_spawner.asset_name().to_string();
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

            let shape_idxs = robot_geometric_shape_scene.get_shape_idxs_from_env_obj_idx(env_obj_idx).expect("error");
            let shape_idx = shape_idxs[0];
            let pose = poses.poses()[shape_idx].clone().unwrap();
            let mut transform = TransformUtils::util_convert_se3_pose_to_y_up_bevy_transform(&pose);
            transform.scale = match env_object_spawner.scale() {
                None => { Vec3::new(1.,1.,1.) }
                Some(s) => { Vec3::new(s as f32,s as f32,s as f32) }
            };

            commands.spawn_bundle(PbrBundle {
                mesh: asset_server.load(&path),
                material: materials.add(environment_color.into()),
                transform,
                ..Default::default()
            });
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
    pub fn system_spawn_robot_geometric_shape_scene(mut commands: Commands,
                                                    asset_server: Res<AssetServer>,
                                                    mut materials: ResMut<Assets<StandardMaterial>>,
                                                    robot_geometric_shape_scene: Res<RobotGeometricShapeScene>,
                                                    frame_count: Res<FrameCount>) {
        if frame_count.0 == 1 {
            let robot_set_joint_state = robot_geometric_shape_scene.get_robot_set().robot_set_joint_state_module().spawn_zeros_robot_set_joint_state(RobotSetJointStateType::Full);
            RobotActions::action_spawn_robot_geometric_shape_scene(&mut commands, &asset_server, &mut materials, & *robot_geometric_shape_scene, &robot_set_joint_state, Color::rgb(0.6, 0.6, 0.7), Color::rgb(0.8, 0.2, 0.8), false, false);
        }
    }
}