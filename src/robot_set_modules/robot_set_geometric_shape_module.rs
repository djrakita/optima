use nalgebra::Vector3;
use parry3d_f64::query::Ray;
use serde::{Deserialize, Serialize};
use crate::robot_modules::robot_geometric_shape_module::{RobotGeometricShapeModule, RobotLinkShapeRepresentation};
use crate::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
use crate::robot_set_modules::robot_set_kinematics_module::{RobotSetFKResult, RobotSetKinematicsModule};
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::load_object_from_json_string;
use crate::utils::utils_generic_data_structures::{MemoryCell, SquareArray2D};
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShapeQueryGroupOutput, GeometricShapeSignature, LogCondition, StopCondition};
use crate::utils::utils_shape_geometry::shape_collection::{ShapeCollection, ShapeCollectionInputPoses, ShapeCollectionQuery};
use crate::utils::utils_traits::SaveAndLoadable;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetGeometricShapeModule {
    robot_set_kinematics_module: RobotSetKinematicsModule,
    robot_set_shape_collections: Vec<RobotSetShapeCollection>
}
impl RobotSetGeometricShapeModule {
    pub fn new(robot_set_configuration_module: &RobotSetConfigurationModule) -> Result<Self, OptimaError> {
        let robot_set_kinematics_module = RobotSetKinematicsModule::new(robot_set_configuration_module);

        let mut out_self = Self {
            robot_set_kinematics_module,
            robot_set_shape_collections: vec![]
        };

        out_self.setup_robot_set_shape_collections(robot_set_configuration_module)?;

        Ok(out_self)
    }
    pub fn new_from_set_name(set_name: &str) -> Result<Self, OptimaError> {
        let robot_set_configuration_module = RobotSetConfigurationModule::new_from_set_name(set_name)?;
        Self::new(&robot_set_configuration_module)
    }
    pub fn shape_collection_query<'a>(&'a self,
                                      input: &'a RobotSetShapeCollectionQuery,
                                      robot_link_shape_representation: RobotLinkShapeRepresentation,
                                      stop_condition: StopCondition,
                                      log_condition: LogCondition,
                                      sort_outputs: bool) -> Result<GeometricShapeQueryGroupOutput, OptimaError> {
        return match input {
            RobotSetShapeCollectionQuery::ProjectPoint { robot_joint_state, point, solid } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::ProjectPoint {
                    poses: &poses,
                    point,
                    solid: *solid
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::ContainsPoint { robot_joint_state, point } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::ContainsPoint {
                    poses: &poses,
                    point
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::DistanceToPoint { robot_joint_state, point, solid } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::DistanceToPoint {
                    poses: &poses,
                    point,
                    solid: *solid
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::IntersectsRay { robot_joint_state, ray, max_toi } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::IntersectsRay {
                    poses: &poses,
                    ray,
                    max_toi: *max_toi
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::CastRay { robot_joint_state, ray, max_toi, solid } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::CastRay {
                    poses: &poses,
                    ray,
                    max_toi: *max_toi,
                    solid: *solid
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::CastRayAndGetNormal { robot_joint_state, ray, max_toi, solid } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::CastRayAndGetNormal {
                    poses: &poses,
                    ray,
                    max_toi: *max_toi,
                    solid: *solid
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::IntersectionTest { robot_joint_state } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::IntersectionTest {
                    poses: &poses,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::Distance { robot_joint_state } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::Distance {
                    poses: &poses,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::ClosestPoints { robot_joint_state, max_dis } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::ClosestPoints {
                    poses: &poses,
                    max_dis: *max_dis
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::Contact { robot_joint_state, prediction } => {
                let res = self.robot_set_kinematics_module.compute_fk(robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses = collection.recover_poses(&res)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::Contact {
                    poses: &poses,
                    prediction: *prediction
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotSetShapeCollectionQuery::CCD { robot_joint_state_t1, robot_joint_state_t2 } => {
                let res_t1 = self.robot_set_kinematics_module.compute_fk(robot_joint_state_t1, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
                let res_t2 = self.robot_set_kinematics_module.compute_fk(robot_joint_state_t2, &OptimaSE3PoseType::ImplicitDualQuaternion)?;

                let collection = self.robot_set_shape_collection(&robot_link_shape_representation)?;
                let poses_t1 = collection.recover_poses(&res_t1)?;
                let poses_t2 = collection.recover_poses(&res_t2)?;
                collection.shape_collection.shape_collection_query(&ShapeCollectionQuery::CCD {
                    poses_t1: &poses_t1,
                    poses_t2: &poses_t2
                }, stop_condition, log_condition, sort_outputs)
            }
        }
    }
    pub fn robot_set_shape_collection(&self, shape_representation: &RobotLinkShapeRepresentation) -> Result<&RobotSetShapeCollection, OptimaError> {
        for s in &self.robot_set_shape_collections {
            if &s.robot_link_shape_representation == shape_representation { return Ok(s) }
        }
        Err(OptimaError::UnreachableCode)
    }
    fn setup_robot_set_shape_collections(&mut self, robot_set_configuration_module: &RobotSetConfigurationModule) -> Result<(), OptimaError> {
        let mut robot_geometric_shape_modules = vec![];
        for r in robot_set_configuration_module.robot_configuration_modules() {
            let robot_geometric_shape_module = RobotGeometricShapeModule::new_from_names(RobotNames::new(r.robot_name(), None), false)?;
            robot_geometric_shape_modules.push(robot_geometric_shape_module);
        }

        self.setup_from_robot_geometric_shape_modules(&robot_geometric_shape_modules)?;

        Ok(())
    }
    fn setup_from_robot_geometric_shape_modules(&mut self, robot_geometric_shape_modules: &Vec<RobotGeometricShapeModule>) -> Result<(), OptimaError> {
        let all_link_shape_representations = self.get_all_robot_link_shape_representations();

        for robot_link_shape_representation in &all_link_shape_representations {
            let mut robot_geometric_shape_collections = vec![];
            for robot_geometric_shape_module in robot_geometric_shape_modules {
                robot_geometric_shape_collections.push(robot_geometric_shape_module.robot_shape_collection(robot_link_shape_representation)?);
            }

            let mut curr_idx = 0;
            let mut robot_and_link_idx_to_shape_idxs_mapping = vec![];
            let mut shape_collection = ShapeCollection::new_empty();
            let mut skips: Option<SquareArray2D<MemoryCell<bool>>> = None;
            let mut average_distances: Option<SquareArray2D<MemoryCell<f64>>> = None;
            for (robot_idx_in_set, robot_geometric_shape_collection) in robot_geometric_shape_collections.iter().enumerate() {
                let mut mapping = robot_geometric_shape_collection.link_idx_to_shape_idxs_mapping().clone();
                for v in &mut mapping {
                    for m in v {
                        *m += curr_idx;
                    }
                }
                robot_and_link_idx_to_shape_idxs_mapping.push(mapping);

                for shape in robot_geometric_shape_collection.shape_collection().shapes() {
                    let mut new_shape = shape.clone();
                    let new_signature = match shape.signature() {
                        GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link } => {
                            GeometricShapeSignature::RobotSetLink {
                                robot_idx_in_set,
                                link_idx_in_robot: *link_idx,
                                shape_idx_in_link: *shape_idx_in_link
                            }
                        }
                        _ => { GeometricShapeSignature::None }
                    };
                    new_shape.set_signature(new_signature);
                    shape_collection.add_geometric_shape(new_shape);
                    curr_idx += 1;
                }

                let shape_collection = robot_geometric_shape_collection.shape_collection();
                match &mut skips {
                    None => { skips = Some(shape_collection.skips().clone()) }
                    Some(skips) => { skips.concatenate_in_place(shape_collection.skips(), None) }
                }

                match &mut average_distances {
                    None => { average_distances = Some(shape_collection.average_distances().clone()) }
                    Some(average_distances) => { average_distances.concatenate_in_place(shape_collection.average_distances(), Some(MemoryCell::new(1.0))) }
                }
            }

            shape_collection.set_skips(skips.unwrap())?;
            shape_collection.set_average_distances(average_distances.unwrap())?;

            let robot_set_shape_collection = RobotSetShapeCollection {
                robot_link_shape_representation: robot_link_shape_representation.clone(),
                shape_collection,
                robot_and_link_idx_to_shape_idxs_mapping
            };

            self.robot_set_shape_collections.push(robot_set_shape_collection);
        }

        Ok(())
    }
    fn get_all_robot_link_shape_representations(&self) -> Vec<RobotLinkShapeRepresentation> {
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
}
impl SaveAndLoadable for RobotSetGeometricShapeModule {
    type SaveType = (String, String);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_set_kinematics_module.get_serialization_string(), self.robot_set_shape_collections.get_serialization_string())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let robot_set_kinematics_module = RobotSetKinematicsModule::load_from_json_string(&load.0)?;
        let robot_set_shape_collections = Vec::load_from_json_string(&load.1)?;

        Ok(Self {
            robot_set_kinematics_module,
            robot_set_shape_collections
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotSetShapeCollection {
    robot_link_shape_representation: RobotLinkShapeRepresentation,
    shape_collection: ShapeCollection,
    robot_and_link_idx_to_shape_idxs_mapping: Vec<Vec<Vec<usize>>>
}
impl RobotSetShapeCollection {
    pub fn robot_link_shape_representation(&self) -> &RobotLinkShapeRepresentation {
        &self.robot_link_shape_representation
    }
    pub fn shape_collection(&self) -> &ShapeCollection {
        &self.shape_collection
    }
    pub fn robot_and_link_idx_to_shape_idxs_mapping(&self) -> &Vec<Vec<Vec<usize>>> {
        &self.robot_and_link_idx_to_shape_idxs_mapping
    }
    pub fn get_shape_idxs_from_robot_idx_and_link_idx(&self, robot_idx_in_set: usize, link_idx_in_robot: usize) -> Result<&Vec<usize>, OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(robot_idx_in_set, self.robot_and_link_idx_to_shape_idxs_mapping.len(), file!(), line!())?;
        OptimaError::new_check_for_idx_out_of_bound_error(link_idx_in_robot, self.robot_and_link_idx_to_shape_idxs_mapping[robot_idx_in_set].len(), file!(), line!())?;

        return Ok(&self.robot_and_link_idx_to_shape_idxs_mapping[robot_idx_in_set][link_idx_in_robot]);
    }
    pub fn recover_poses(&self, robot_set_fk_result: &RobotSetFKResult) -> Result<ShapeCollectionInputPoses, OptimaError> {
        let mut out = ShapeCollectionInputPoses::new(&self.shape_collection);
        let mapping = &self.robot_and_link_idx_to_shape_idxs_mapping;

        let robot_fk_results = robot_set_fk_result.robot_fk_results();
        for (robot_idx_in_set, fk) in robot_fk_results.iter().enumerate() {
            let link_entries = fk.link_entries();
            for (link_idx, link_entry) in link_entries.iter().enumerate() {
                if let Some(pose) = link_entry.pose() {
                    let shape_idxs = mapping.get(robot_idx_in_set).unwrap().get(link_idx).unwrap();
                    for shape_idx in shape_idxs {
                        out.insert_or_replace_pose_by_idx(*shape_idx, pose.clone())?;
                    }
                }
            }
        }

        Ok(out)
    }
}
impl SaveAndLoadable for RobotSetShapeCollection {
    type SaveType = (RobotLinkShapeRepresentation, String, Vec<Vec<Vec<usize>>>);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.robot_link_shape_representation.clone(), self.shape_collection.get_serialization_string(), self.robot_and_link_idx_to_shape_idxs_mapping.clone())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let shape_collection = ShapeCollection::load_from_json_string(&load.1)?;

        Ok(Self {
            robot_link_shape_representation: load.0,
            shape_collection,
            robot_and_link_idx_to_shape_idxs_mapping: load.2
        })
    }
}

/// A robot set specific version of a `ShapeCollectionQuery`.  Is basically the same but trades out
/// shape pose information with `RobotJointState` structs.  The SE(3) poses can then automatically
/// be resolved using forward kinematics.
pub enum RobotSetShapeCollectionQuery<'a> {
    ProjectPoint { robot_joint_state: &'a RobotSetJointState, point: &'a Vector3<f64>, solid: bool },
    ContainsPoint { robot_joint_state: &'a RobotSetJointState, point: &'a Vector3<f64> },
    DistanceToPoint { robot_joint_state: &'a RobotSetJointState, point: &'a Vector3<f64>, solid: bool },
    IntersectsRay { robot_joint_state: &'a RobotSetJointState, ray: &'a Ray, max_toi: f64 },
    CastRay { robot_joint_state: &'a RobotSetJointState, ray: &'a Ray, max_toi: f64, solid: bool },
    CastRayAndGetNormal { robot_joint_state: &'a RobotSetJointState, ray: &'a Ray, max_toi: f64, solid: bool },
    IntersectionTest { robot_joint_state: &'a RobotSetJointState },
    Distance { robot_joint_state: &'a RobotSetJointState },
    ClosestPoints { robot_joint_state: &'a RobotSetJointState, max_dis: f64 },
    Contact { robot_joint_state: &'a RobotSetJointState, prediction: f64 },
    CCD { robot_joint_state_t1: &'a RobotSetJointState, robot_joint_state_t2: &'a RobotSetJointState }
}
impl <'a> RobotSetShapeCollectionQuery<'a> {
    pub fn get_robot_joint_state(&self) -> Result<Vec<&'a RobotSetJointState>, OptimaError> {
        match self {
            RobotSetShapeCollectionQuery::ProjectPoint { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::ContainsPoint { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::DistanceToPoint { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::IntersectsRay { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::CastRay { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::CastRayAndGetNormal { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::IntersectionTest { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::Distance { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::ClosestPoints { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::Contact { robot_joint_state, .. } => { Ok(vec![robot_joint_state]) }
            RobotSetShapeCollectionQuery::CCD { robot_joint_state_t1, robot_joint_state_t2 } => { Ok(vec![robot_joint_state_t1, robot_joint_state_t2]) }
        }
    }
}