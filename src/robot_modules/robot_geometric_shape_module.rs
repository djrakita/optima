#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::time::{Duration, Instant};
use nalgebra::{DVector, Vector3};
use parry3d_f64::query::Ray;
use serde::{Deserialize, Serialize};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_mesh_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_modules::robot_kinematics_module::{RobotFKResult, RobotKinematicsModule};
use crate::robot_modules::robot_joint_state_module::{RobotJointState, RobotJointStateModule, RobotJointStateType};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::utils_console::{get_default_progress_bar, ConsoleInputUtils};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaAssetLocation, RobotModuleJsonType};
use crate::utils::utils_generic_data_structures::{AveragingFloat, SquareArray2D};
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShapeSignature, LogCondition, StopCondition};
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::utils_shape_geometry::geometric_shape::GeometricShapeQueryGroupOutputPy;
use crate::utils::utils_shape_geometry::shape_collection::{ShapeCollection, ShapeCollectionInputPoses, ShapeCollectionQuery, ShapeCollectionQueryList, ShapeCollectionQueryOutput, ShapeCollectionQueryPairsList};
use crate::utils::utils_traits::{AssetSaveAndLoadable, SaveAndLoadable, ToAndFromRonString};

/// Robot module that provides useful functions over geometric shapes.  For example, the module is
/// able to compute if a robot is in collision given a particular robot joint state.  For all geometry
/// query types, refer to the `RobotShapeCollectionQuery` enum.
///
/// The most important function here is `RobotGeometricShapeModule.shape_collection_query`.  This
/// function takes in a `RobotShapeCollectionQuery` as input and outputs a
/// corresponding `GeometricShapeQueryGroupOutput`.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotGeometricShapeModule {
    robot_joint_state_module: RobotJointStateModule,
    robot_kinematics_module: RobotKinematicsModule,
    robot_mesh_file_manager_module: RobotMeshFileManagerModule,
    robot_shape_collections: Vec<RobotShapeCollection>
}
impl RobotGeometricShapeModule {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(robot_configuration_module: RobotConfigurationModule, force_preprocessing: bool) -> Result<Self, OptimaError> {
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        let robot_kinematics_module = RobotKinematicsModule::new(robot_configuration_module.clone());
        let robot_mesh_file_manager_module = RobotMeshFileManagerModule::new_from_name(robot_configuration_module.robot_name())?;
        return if force_preprocessing {
            let mut out_self = Self {
                robot_joint_state_module,
                robot_kinematics_module,
                robot_mesh_file_manager_module,
                robot_shape_collections: vec![]
            };
            out_self.preprocessing()?;
            Ok(out_self)
        } else {
            let robot_name = robot_kinematics_module.robot_name().to_string();
            let res = Self::load_as_asset(OptimaAssetLocation::RobotModuleJson { robot_name, t: RobotModuleJsonType::ShapeGeometryModule });
            match res {
                Ok(res) => { Ok(res) }
                Err(_) => { Self::new(robot_configuration_module, true) }
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    pub fn new(robot_configuration_module: RobotConfigurationModule, force_preprocessing: bool) -> Result<Self, OptimaError> {
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        let robot_kinematics_module = RobotKinematicsModule::new(robot_configuration_module.clone());
        let robot_mesh_file_manager_module = RobotMeshFileManagerModule::new_from_name(robot_configuration_module.robot_name())?;
        return if force_preprocessing {
            let mut out_self = Self {
                robot_joint_state_module,
                robot_kinematics_module,
                robot_mesh_file_manager_module,
                robot_shape_collections: vec![]
            };
            Err(OptimaError::new_generic_error_str("Cannot preprocess geometric shape module from WASM.", file!(), line!()))
        } else {
            let robot_name = robot_kinematics_module.robot_name().to_string();
            let res = Self::load_as_asset(OptimaAssetLocation::RobotModuleJson { robot_name, t: RobotModuleJsonType::ShapeGeometryModule });
            match res {
                Ok(res) => { Ok(res) }
                Err(_) => { Self::new(robot_configuration_module, true) }
            }
        }
    }
    pub fn new_from_names(robot_names: RobotNames, force_preprocessing: bool) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_names)?;
        Self::new(robot_configuration_module, force_preprocessing)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn preprocessing(&mut self) -> Result<(), OptimaError> {
        let robot_link_shape_representations = vec![
            RobotLinkShapeRepresentation::Cubes,
            RobotLinkShapeRepresentation::ConvexShapes,
            RobotLinkShapeRepresentation::SphereSubcomponents,
            RobotLinkShapeRepresentation::CubeSubcomponents,
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents,
            RobotLinkShapeRepresentation::TriangleMeshes
        ];

        for robot_link_shape_representation in &robot_link_shape_representations {
            self.preprocessing_robot_geometric_shape_collection(robot_link_shape_representation)?;
        }

        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn preprocessing_robot_geometric_shape_collection(&mut self,
                                                      robot_link_shape_representation: &RobotLinkShapeRepresentation) -> Result<(), OptimaError> {
        optima_print(&format!("Setup on {:?}...", robot_link_shape_representation), PrintMode::Println, PrintColor::Blue, true);
        // Base model modules must be used as these computations apply to all derived configuration
        // variations of this model, not just particular configurations.
        let robot_name = self.robot_kinematics_module.robot_name();
        let base_robot_model_module = RobotModelModule::new(robot_name)?;
        let base_robot_kinematics_module = RobotKinematicsModule::new_from_names(RobotNames::new_base(robot_name))?;
        let base_robot_joint_state_module = RobotJointStateModule::new_from_names(RobotNames::new_base(robot_name))?;
        let num_links = base_robot_model_module.links().len();

        // Initialize GeometricShapeCollision.
        let mut shape_collection = ShapeCollection::new_empty();
        let geometric_shapes = self.robot_mesh_file_manager_module.get_geometric_shapes(&robot_link_shape_representation)?;
        for geometric_shape in geometric_shapes {
            if let Some(geometric_shape) = geometric_shape {
                shape_collection.add_geometric_shape(geometric_shape.clone());
            }
        }
        let num_shapes = shape_collection.shapes().len();

        // Initialize the RobotGeometricShapeCollection with the GeometricShapeCollection.
        let mut robot_shape_collection = RobotShapeCollection::new(num_links, robot_link_shape_representation.clone(), shape_collection)?;

        // These SquareArray2Ds will hold information to determine the average distances between links
        // as well as whether links always intersect or never collide.
        let mut distance_average_array = SquareArray2D::<AveragingFloat>::new(num_shapes, true, None);
        let mut collision_counter_array = SquareArray2D::<f64>::new(num_shapes, true, None);

        // This loop takes random robot joint state samples and determines intersection and average
        // distance information between links.
        let start = Instant::now();
        let mut count = 0.0;
        let max_samples = 100_000;
        let min_samples = 70;

        let mut pb = get_default_progress_bar(1000);

        // Where distances and intersections are actually checked at each joint state sample.
        for i in 0..max_samples {
            count += 1.0;
            let sample = base_robot_joint_state_module.sample_joint_state(&RobotJointStateType::Full);
            let fk_res = base_robot_kinematics_module.compute_fk(&sample, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
            let poses = robot_shape_collection.recover_poses(&fk_res)?;
            let input = ShapeCollectionQuery::Distance { poses: &poses, inclusion_list: &None };

            let res = robot_shape_collection.shape_collection.shape_collection_query(&input, StopCondition::None, LogCondition::LogAll, false)?;

            let outputs = res.unwrap_geometric_shape_query_group_output().outputs();
            for output in outputs {
                let signatures = output.signatures();
                let signature1 = &signatures[0];
                let signature2 = &signatures[1];
                let shape_idx1 = robot_shape_collection.shape_collection.get_shape_idx_from_signature(signature1)?;
                let shape_idx2 = robot_shape_collection.shape_collection.get_shape_idx_from_signature(signature2)?;
                let dis = output.raw_output().unwrap_distance()?;
                distance_average_array.adjust_data(|x| x.add_new_value(dis.clone()), shape_idx1, shape_idx2 )?;
                if dis <= 0.0 {
                    collision_counter_array.adjust_data(|x| *x += 1.0, shape_idx1, shape_idx2)?;
                }
            }

            let duration = start.elapsed();
            let duration_ratio = duration.as_secs_f64() / self.stop_at_min_sample_duration(robot_link_shape_representation).as_secs_f64();
            let max_sample_ratio = i as f64 / max_samples as f64;
            let min_sample_ratio = i as f64 / min_samples as f64;
            let ratio = duration_ratio.max(max_sample_ratio).min(min_sample_ratio);
            pb.set((ratio * 1000.0) as u64);
            pb.message(&format!("sample {} ", i));

            if duration > self.stop_at_min_sample_duration(robot_link_shape_representation) && i >= min_samples { break; }
        }

        // Determines average distances and decides if links should be skipped based on previous
        // computations.  These reesults are saved in the RobotGeometricShapeCollection.
        for i in 0..num_shapes {
            for j in 0..num_shapes {
                // Retrieves and saves the average distance between the given pair of links.
                let averaging_float = distance_average_array.data_cell(i, j)?;
                robot_shape_collection.shape_collection.replace_average_distance_from_idxs(averaging_float.value(), i, j)?;

                // Pairwise checks should never happen between the same shape.
                if i == j { robot_shape_collection.shape_collection.replace_skip_from_idxs(true, i, j)?; }

                let shapes = robot_shape_collection.shape_collection.shapes();
                let signature1 = shapes[i].signature();
                let signature2 = shapes[j].signature();
                match signature1 {
                    GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: _ } => {
                        let link_idx1 = link_idx.clone();
                        match signature2 {
                            GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: _ } => {
                                let link_idx2 = link_idx.clone();
                                if link_idx1 == link_idx2 {
                                    robot_shape_collection.shape_collection.replace_skip_from_idxs(true, i, j)?;
                                }
                            }
                            _ => { }
                        }
                    }
                    _ => { }
                }

                // Checks if links are always in intersecting.
                let ratio_of_checks_in_collision = collision_counter_array.data_cell(i, j)? / count;
                if count >= min_samples as f64 && ratio_of_checks_in_collision > 0.99 {
                    robot_shape_collection.shape_collection.replace_skip_from_idxs(true, i, j)?;
                }

                // Checks if links are never in collision
                if count >= 1000.0 && ratio_of_checks_in_collision == 0.0 {
                    robot_shape_collection.shape_collection.replace_skip_from_idxs(true, i, j)?;
                }
            }
        }

        pb.finish();
        println!();

        self.robot_shape_collections.push(robot_shape_collection);
        self.save_as_asset(OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ShapeGeometryModule })?;
        self.save_as_asset(OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.to_string(), t: RobotModuleJsonType::ShapeGeometryModulePermanent })?;

        Ok(())
    }
    fn get_all_robot_link_shape_representations() -> Vec<RobotLinkShapeRepresentation> {
        let robot_link_shape_representations = vec![
            RobotLinkShapeRepresentation::Cubes,
            RobotLinkShapeRepresentation::ConvexShapes,
            RobotLinkShapeRepresentation::SphereSubcomponents,
            RobotLinkShapeRepresentation::CubeSubcomponents,
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents,
            RobotLinkShapeRepresentation::TriangleMeshes
        ];
        robot_link_shape_representations
    }
    pub fn robot_shape_collection(&self, shape_representation: &RobotLinkShapeRepresentation) -> Result<&RobotShapeCollection, OptimaError> {
        for s in &self.robot_shape_collections {
            if &s.robot_link_shape_representation == shape_representation { return Ok(s) }
        }
        unreachable!()
    }
    fn robot_geometric_shape_collection_mut(&mut self, shape_representation: &RobotLinkShapeRepresentation) -> Result<&mut RobotShapeCollection, OptimaError> {
        for s in &mut self.robot_shape_collections {
            if &s.robot_link_shape_representation == shape_representation { return Ok(s) }
        }
        unreachable!()
    }
    pub fn shape_collection_query<'a>(&'a self,
                                      input: &'a RobotShapeCollectionQuery,
                                      robot_link_shape_representation: RobotLinkShapeRepresentation,
                                      stop_condition: StopCondition,
                                      log_condition: LogCondition,
                                      sort_outputs: bool) -> Result<ShapeCollectionQueryOutput, OptimaError> {
        return match input {
            RobotShapeCollectionQuery::ProjectPoint { robot_joint_state, point, solid, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::ProjectPoint {
                    poses: &poses,
                    point,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::ContainsPoint { robot_joint_state, point, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::ContainsPoint {
                    poses: &poses,
                    point,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::DistanceToPoint { robot_joint_state, point, solid, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::DistanceToPoint {
                    poses: &poses,
                    point,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::IntersectsRay { robot_joint_state, ray, max_toi, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::IntersectsRay {
                    poses: &poses,
                    ray,
                    max_toi: *max_toi,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::CastRay { robot_joint_state, ray, max_toi, solid, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::CastRay {
                    poses: &poses,
                    ray,
                    max_toi: *max_toi,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::CastRayAndGetNormal { robot_joint_state, ray, max_toi, solid, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::CastRayAndGetNormal {
                    poses: &poses,
                    ray,
                    max_toi: *max_toi,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::IntersectionTest { robot_joint_state, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::IntersectionTest {
                    poses: &poses,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::Distance { robot_joint_state, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::Distance {
                    poses: &poses,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::ClosestPoints { robot_joint_state, max_dis, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::ClosestPoints {
                    poses: &poses,
                    max_dis: *max_dis,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::Contact { robot_joint_state, prediction, inclusion_list } => {
                let res = self.robot_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::Contact {
                    poses: &poses,
                    prediction: *prediction,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotShapeCollectionQuery::CCD { robot_joint_state_t1, robot_joint_state_t2, inclusion_list } => {
                let res_t1 = self.robot_kinematics_module.compute_fk(robot_joint_state_t1, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let res_t2 = self.robot_kinematics_module.compute_fk(robot_joint_state_t2, &OptimaSE3PoseType::ImplicitDualQuaternion)?;

                let collection = self.robot_shape_collection(&robot_link_shape_representation)?;
                let poses_t1 = collection.recover_poses(&res_t1)?;
                let poses_t2 = collection.recover_poses(&res_t2)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::CCD {
                    poses_t1: &poses_t1,
                    poses_t2: &poses_t2,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
        }
    }
    pub fn set_robot_joint_state_as_non_collision(&mut self, robot_joint_state: &RobotJointState) -> Result<(), OptimaError> {
        let all_robot_link_shape_representations = Self::get_all_robot_link_shape_representations();

        for robot_link_shape_representation in &all_robot_link_shape_representations {
            let input = RobotShapeCollectionQuery::Contact {
                robot_joint_state,
                prediction: 0.01,
                inclusion_list: &None
            };

            let res = self.shape_collection_query(&input,
                                                  robot_link_shape_representation.clone(),
                                                  StopCondition::None,
                                                  LogCondition::LogAll,
                                                  false)?;

            let collection = self.robot_geometric_shape_collection_mut(robot_link_shape_representation)?;

            let outputs = res.unwrap_geometric_shape_query_group_output().outputs();
            for output in outputs {
                let signatures = output.signatures();
                let contact = output.raw_output().unwrap_contact()?;
                if let Some(contact) = &contact {

                    // Does not mark as skip if the penetration depth is greater than 0.12 meters.
                    if contact.dist <= 0.0 && contact.dist > -0.12 {
                        let signature1 = &signatures[0];
                        let signature2 = &signatures[1];
                        let idx1 = collection.shape_collection.get_shape_idx_from_signature(signature1)?;
                        let idx2 = collection.shape_collection.get_shape_idx_from_signature(signature2)?;
                        collection.shape_collection.replace_skip_from_idxs(true, idx1, idx2)?;
                    }
                }
            }
        }

        self.save_as_asset(OptimaAssetLocation::RobotModuleJson { robot_name: self.robot_kinematics_module.robot_configuration_module().robot_name().to_string(), t: RobotModuleJsonType::ShapeGeometryModule })?;

        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn reset_robot_geometric_shape_collection(&mut self, robot_link_shape_representation: RobotLinkShapeRepresentation) -> Result<(), OptimaError> {
        let response = ConsoleInputUtils::get_console_input_string("About to reset robot geometric shape collections.  Confirm? (y or n).", PrintColor::Blue)?;
        if response == "y" {
            let permanent = Self::load_as_asset(OptimaAssetLocation::RobotModuleJson { robot_name: self.robot_kinematics_module.robot_configuration_module().robot_name().to_string(), t: RobotModuleJsonType::ShapeGeometryModulePermanent })?;
            for (i, r) in self.robot_shape_collections.iter_mut().enumerate() {
            if &r.robot_link_shape_representation == &robot_link_shape_representation {
                *r = permanent.robot_shape_collections[i].clone();
                self.save_as_asset(OptimaAssetLocation::RobotModuleJson { robot_name: self.robot_kinematics_module.robot_configuration_module().robot_name().to_string(), t: RobotModuleJsonType::ShapeGeometryModule })?;
                return Ok(());
            }
        }
        }
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn reset_all_robot_geometric_shape_collections(&mut self) -> Result<(), OptimaError> {
        let response = ConsoleInputUtils::get_console_input_string("About to reset all robot geometric shape collections.  Confirm? (y or n).", PrintColor::Blue)?;
        if response == "y" {
            let all = Self::get_all_robot_link_shape_representations();
            for r in &all {
                self.reset_robot_geometric_shape_collection(r.clone())?;
            }
        }
        Ok(())
    }
    fn stop_at_min_sample_duration(&self, robot_link_shape_representation: &RobotLinkShapeRepresentation) -> Duration {
        match robot_link_shape_representation {
            RobotLinkShapeRepresentation::Cubes => { Duration::from_secs(20) }
            RobotLinkShapeRepresentation::ConvexShapes => { Duration::from_secs(30) }
            RobotLinkShapeRepresentation::SphereSubcomponents => { Duration::from_secs(30) }
            RobotLinkShapeRepresentation::CubeSubcomponents => { Duration::from_secs(30) }
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents => { Duration::from_secs(60) }
            RobotLinkShapeRepresentation::TriangleMeshes => { Duration::from_secs(120) }
        }
    }
}
impl SaveAndLoadable for RobotGeometricShapeModule {
    type SaveType = (String, String, String);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_kinematics_module.robot_configuration_module().get_serialization_string(), self.robot_mesh_file_manager_module.get_serialization_string(), self.robot_shape_collections.get_serialization_string())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let robot_configuration_module = RobotConfigurationModule::load_from_json_string(&load.0)?;
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        let robot_kinematics_module = RobotKinematicsModule::new(robot_configuration_module);
        let robot_mesh_file_manager_module = RobotMeshFileManagerModule::load_from_json_string(&load.1)?;
        // let robot_shape_collections: Vec<RobotShapeCollection> = SaveAndLoadableVec::load_from_json_string(&load.2)?;
        let robot_shape_collections: Vec<RobotShapeCollection> = Vec::load_from_json_string(&load.2)?;

        Ok(Self {
            robot_joint_state_module,
            robot_kinematics_module,
            robot_mesh_file_manager_module,
            robot_shape_collections
        })
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotGeometricShapeModule {
    #[new]
    pub fn new_py(robot_name: &str, configuration_name: Option<&str>) -> RobotGeometricShapeModule {
        return Self::new_from_names(RobotNames::new(robot_name, configuration_name), false).expect("error");
    }
    #[args(robot_link_shape_representation = "\"Cubes\"", stop_condition = "\"Intersection\"", log_condition = "\"BelowMinDistance(0.5)\"", sort_outputs = "true", include_full_output_json_string = "true")]
    pub fn intersection_test_query_py(&self,
                                      joint_state: Vec<f64>,
                                      robot_link_shape_representation: &str,
                                      stop_condition: &str,
                                      log_condition: &str,
                                      sort_outputs: bool,
                                      include_full_output_json_string: bool) -> GeometricShapeQueryGroupOutputPy {
        let joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state)).expect("error");
        let input = RobotShapeCollectionQuery::IntersectionTest {
            robot_joint_state: &joint_state,
            inclusion_list: None
        };
        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        let py_output = res.unwrap_geometric_shape_query_group_output().convert_to_py_output(include_full_output_json_string);
        py_output
    }
    #[args(robot_link_shape_representation = "\"Cubes\"", stop_condition = "\"Intersection\"", log_condition = "\"BelowMinDistance(0.5)\"", sort_outputs = "true", include_full_output_json_string = "true")]
    pub fn distance_query_py(&self,
                             joint_state: Vec<f64>,
                             robot_link_shape_representation: &str,
                             stop_condition: &str,
                             log_condition: &str,
                             sort_outputs: bool,
                             include_full_output_json_string: bool) -> GeometricShapeQueryGroupOutputPy {
        let joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state)).expect("error");
        let input = RobotShapeCollectionQuery::Distance {
            robot_joint_state: &joint_state,
            inclusion_list: &None
        };
        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        let py_output = res.unwrap_geometric_shape_query_group_output().convert_to_py_output(include_full_output_json_string);
        py_output
    }
    #[args(robot_link_shape_representation = "\"Cubes\"", stop_condition = "\"Intersection\"", log_condition = "\"BelowMinDistance(0.5)\"", sort_outputs = "true", include_full_output_json_string = "true")]
    pub fn contact_query_py(&self,
                            joint_state: Vec<f64>,
                            prediction: f64,
                            robot_link_shape_representation: &str,
                            stop_condition: &str,
                            log_condition: &str,
                            sort_outputs: bool,
                            include_full_output_json_string: bool) -> GeometricShapeQueryGroupOutputPy {
        let joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state)).expect("error");
        let input = RobotShapeCollectionQuery::Contact {
            robot_joint_state: &joint_state,
            prediction,
            inclusion_list: &None
        };
        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        let py_output = res.unwrap_geometric_shape_query_group_output().convert_to_py_output(include_full_output_json_string);
        py_output
    }
    #[args(robot_link_shape_representation = "\"Cubes\"", stop_condition = "\"Intersection\"", log_condition = "\"BelowMinDistance(0.5)\"", sort_outputs = "true", include_full_output_json_string = "true")]
    pub fn ccd_query_py(&self,
                        joint_state_t1: Vec<f64>,
                        joint_state_t2: Vec<f64>,
                        robot_link_shape_representation: &str,
                        stop_condition: &str,
                        log_condition: &str,
                        sort_outputs: bool,
                        include_full_output_json_string: bool) -> GeometricShapeQueryGroupOutputPy {
        let joint_state_t1 = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state_t1)).expect("error");
        let joint_state_t2 = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state_t2)).expect("error");

        let input = RobotShapeCollectionQuery::CCD {
            robot_joint_state_t1: &joint_state_t1,
            robot_joint_state_t2: &joint_state_t2,
            inclusion_list: &None
        };
        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        let py_output = res.unwrap_geometric_shape_query_group_output().convert_to_py_output(include_full_output_json_string);
        py_output
    }
    pub fn set_robot_joint_state_as_non_collision_py(&mut self, robot_joint_state: Vec<f64>) {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(robot_joint_state)).expect("error");
        self.set_robot_joint_state_as_non_collision(&robot_joint_state).expect("error");
    }
    pub fn reset_robot_geometric_shape_collection_py(&mut self, robot_link_shape_representation: &str) {
        self.reset_robot_geometric_shape_collection(RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error")).expect("error");
    }
    pub fn reset_all_robot_geometric_shape_collections_py(&mut self) {
        self.reset_all_robot_geometric_shape_collections().expect("error");
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotGeometricShapeModule {
    #[wasm_bindgen(constructor)]
    pub fn new_wasm(robot_name: String, configuration_name: Option<String>) -> RobotGeometricShapeModule {
        return match configuration_name {
            None => { Self::new_from_names(RobotNames::new(&robot_name, None), false).expect("error") }
            Some(c) => { Self::new_from_names(RobotNames::new(&robot_name, Some(&c)), false).expect("error") }
        }
    }
    pub fn intersection_test_query_wasm(&self, joint_state: Vec<f64>, robot_link_shape_representation: &str, stop_condition: &str, log_condition: &str, sort_outputs: bool) -> JsValue {
        let joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state)).expect("error");
        let input = RobotShapeCollectionQuery::IntersectionTest {
            robot_joint_state: &joint_state
        };

        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        JsValue::from_serde(&res).unwrap()
    }
    pub fn distance_query_wasm(&self, joint_state: Vec<f64>, robot_link_shape_representation: &str, stop_condition: &str, log_condition: &str, sort_outputs: bool) -> JsValue {
        let joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state)).expect("error");
        let input = RobotShapeCollectionQuery::Distance {
            robot_joint_state: &joint_state
        };

        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        JsValue::from_serde(&res).unwrap()
    }
    pub fn contact_query_wasm(&self, joint_state: Vec<f64>, prediction: f64, robot_link_shape_representation: &str, stop_condition: &str, log_condition: &str, sort_outputs: bool) -> JsValue {
        let joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state)).expect("error");
        let input = RobotShapeCollectionQuery::Contact {
            robot_joint_state: &joint_state,
            prediction
        };

        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        JsValue::from_serde(&res).unwrap()
    }
    pub fn ccd_query_wasm(&self, joint_state_t1: Vec<f64>, joint_state_t2: Vec<f64>, robot_link_shape_representation: &str, stop_condition: &str, log_condition: &str, sort_outputs: bool) -> JsValue {
        let joint_state_t1 = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state_t1)).expect("error");
        let joint_state_t2 = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::from_vec(joint_state_t2)).expect("error");

        let input = RobotShapeCollectionQuery::CCD {
            robot_joint_state_t1: &joint_state_t1,
            robot_joint_state_t2: &joint_state_t2
        };

        let res = self.shape_collection_query(&input,
                                              RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).expect("error"),
                                              StopCondition::from_ron_string(stop_condition).expect("error"),
                                              LogCondition::from_ron_string(log_condition).expect("error"),
                                              sort_outputs).expect("error");
        JsValue::from_serde(&res).unwrap()
    }
}

/// A robot specific version of a `ShapeCollection`.  All shapes in the underlying `ShapeCollection`
/// refers to geometry representing some part of a robot link.  This also includes information on
/// the shape representation of the links as well as a nice way to map from a robot link index to
/// all shape indices corresponding to shapes that are rigidly attached to that link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotShapeCollection {
    robot_link_shape_representation: RobotLinkShapeRepresentation,
    shape_collection: ShapeCollection,
    link_idx_to_shape_idxs_mapping: Vec<Vec<usize>>
}
impl RobotShapeCollection {
    pub fn new(num_robot_links: usize, robot_link_shape_representation: RobotLinkShapeRepresentation, shape_collection: ShapeCollection) -> Result<Self, OptimaError> {
        let mut robot_link_idx_to_shape_idxs_mapping = vec![];

        for _ in 0..num_robot_links { robot_link_idx_to_shape_idxs_mapping.push(vec![]); }

        let shapes = shape_collection.shapes();
        for (shape_idx, shape) in shapes.iter().enumerate() {
            match shape.signature() {
                GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: _ } => {
                    robot_link_idx_to_shape_idxs_mapping[*link_idx].push(shape_idx);
                }
                _ => { }
            }
        }

        Ok(Self {
            robot_link_shape_representation,
            shape_collection: shape_collection,
            link_idx_to_shape_idxs_mapping: robot_link_idx_to_shape_idxs_mapping
        })
    }
    pub fn robot_link_shape_representation(&self) -> &RobotLinkShapeRepresentation {
        &self.robot_link_shape_representation
    }
    pub fn shape_collection(&self) -> &ShapeCollection {
        &self.shape_collection
    }
    pub fn link_idx_to_shape_idxs_mapping(&self) -> &Vec<Vec<usize>> {
        &self.link_idx_to_shape_idxs_mapping
    }
    pub fn get_shape_idxs_from_link_idx(&self, link_idx: usize) -> Result<&Vec<usize>, OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(link_idx, self.link_idx_to_shape_idxs_mapping.len(), file!(), line!())?;
        return Ok(&self.link_idx_to_shape_idxs_mapping[link_idx]);
    }
    pub fn recover_poses(&self, robot_fk_result: &RobotFKResult) -> Result<ShapeCollectionInputPoses, OptimaError> {
        let mut geometric_shape_collection_input_poses = ShapeCollectionInputPoses::new(&self.shape_collection);
        let link_entries = robot_fk_result.link_entries();
        for (link_idx, link_entry) in link_entries.iter().enumerate() {
            let pose = link_entry.pose();
            if let Some(pose) = pose {
                let shape_idxs = self.get_shape_idxs_from_link_idx(link_idx)?;
                for shape_idx in shape_idxs {
                    geometric_shape_collection_input_poses.insert_or_replace_pose_by_idx(*shape_idx, pose.clone())?;
                }
            }
        }

        Ok(geometric_shape_collection_input_poses)
    }
}
impl SaveAndLoadable for RobotShapeCollection {
    type SaveType = (RobotLinkShapeRepresentation, String, Vec<Vec<usize>>);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_link_shape_representation.clone(), self.shape_collection.get_serialization_string(), self.link_idx_to_shape_idxs_mapping.clone())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let shape_collection = ShapeCollection::load_from_json_string(&load.1)?;
        Ok(Self {
            robot_link_shape_representation: load.0.clone(),
            shape_collection,
            link_idx_to_shape_idxs_mapping: load.2.clone()
        })
    }
}

/// A robot specific version of a `ShapeCollectionQuery`.  Is basically the same but trades out
/// shape pose information with `RobotJointState` structs.  The SE(3) poses can then automatically
/// be resolved using forward kinematics.
pub enum RobotShapeCollectionQuery<'a> {
    ProjectPoint { robot_joint_state: &'a RobotJointState, point: &'a Vector3<f64>, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    ContainsPoint { robot_joint_state: &'a RobotJointState, point: &'a Vector3<f64>, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    DistanceToPoint { robot_joint_state: &'a RobotJointState, point: &'a Vector3<f64>, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    IntersectsRay { robot_joint_state: &'a RobotJointState, ray: &'a Ray, max_toi: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    CastRay { robot_joint_state: &'a RobotJointState, ray: &'a Ray, max_toi: f64, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    CastRayAndGetNormal { robot_joint_state: &'a RobotJointState, ray: &'a Ray, max_toi: f64, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    IntersectionTest { robot_joint_state: &'a RobotJointState, inclusion_list: Option<&'a ShapeCollectionQueryPairsList> },
    Distance { robot_joint_state: &'a RobotJointState, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    ClosestPoints { robot_joint_state: &'a RobotJointState, max_dis: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    Contact { robot_joint_state: &'a RobotJointState, prediction: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    CCD { robot_joint_state_t1: &'a RobotJointState, robot_joint_state_t2: &'a RobotJointState, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> }
}
impl <'a> RobotShapeCollectionQuery<'a> {
    pub fn get_robot_joint_state(&self) -> Result<Vec<&'a RobotJointState>, OptimaError> {
        match self {
            RobotShapeCollectionQuery::ProjectPoint { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::ContainsPoint { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::DistanceToPoint { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::IntersectsRay { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::CastRay { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::CastRayAndGetNormal { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::IntersectionTest { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::Distance { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::ClosestPoints { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::Contact { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotShapeCollectionQuery::CCD { robot_joint_state_t1, robot_joint_state_t2, inclusion_list: _ } => { Ok(vec![robot_joint_state_t1, robot_joint_state_t2]) }
        }
    }
}

/// The representation of the robot link geometry objects.
/// - `Cubes`: wraps all links in best fitting cubes (essentially oriented bounding boxes)
/// - `ConvexShapes`: wraps all links in convex shapes
/// - `SphereSubcomponents`: decomposes each link into convex subcomponents and wraps each in a best fitting sphere.
/// - `CubeSubcomponents`: decomposes each link into convex subcomponents and wraps each in a best fitting cube.
/// - `ConvexShapeSubcomponents`: decomposes each link into convex subcomponents.
/// - `TriangleMeshes`: directly uses the given meshes as geometry.
#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Serialize, Deserialize)]
pub enum RobotLinkShapeRepresentation {
    Cubes,
    ConvexShapes,
    SphereSubcomponents,
    CubeSubcomponents,
    ConvexShapeSubcomponents,
    TriangleMeshes
}
