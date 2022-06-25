#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use nalgebra::{DVector, Vector3};
use parry3d_f64::query::Ray;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_geometric_shape_module::RobotLinkShapeRepresentation;
use crate::robot_set_modules::GetRobotSet;
use crate::robot_set_modules::robot_set::{RobotSet};
#[cfg(not(target_arch = "wasm32"))]
use crate::robot_set_modules::robot_set::{RobotSetPy};
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::scenes::GetRobotGeometricShapeScene;
use crate::utils::utils_console::{optima_print, optima_print_new_line, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaAssetLocation, OptimaStemCellPath};
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PosePy, OptimaSE3PoseType};
use crate::utils::utils_shape_geometry::geometric_shape::{BVHCombinableShape, BVHCombinableShapeAABB, GeometricShape, GeometricShapeQueryGroupOutput, GeometricShapeSignature, LogCondition, StopCondition};
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShapeQueryGroupOutputPy};
use crate::utils::utils_shape_geometry::shape_collection::{BVH, BVHSceneFilterOutput, BVHVisit, ProximaBudget, ProximaEngine, ProximaProximityOutput, ProximaSceneFilterOutput, ShapeCollection, ShapeCollectionBVH, ShapeCollectionBVHAABB, ShapeCollectionInputPoses, ShapeCollectionQuery, ShapeCollectionQueryList, ShapeCollectionQueryPairsList, SignedDistanceLossFunction};
use crate::utils::utils_shape_geometry::trimesh_engine::ConvexDecompositionResolution;
use crate::utils::utils_traits::{SaveAndLoadable, ToAndFromRonString};

/// Combines a `RobotSet` with geometric shapes around the robot set to form a scene.  This struct
/// can be used to perform geometric queries over all of the robot link and environment object
/// geometric shapes in the scene.
///
/// # Example
/// ```
/// use optima::robot_modules::robot_configuration_module::{ContiguousChainMobilityMode, RobotConfigurationModule};
/// use optima::robot_modules::robot_geometric_shape_module::RobotLinkShapeRepresentation;
/// use optima::robot_set_modules::robot_set::RobotSet;
/// use optima::robot_set_modules::robot_set_configuration_module::RobotSetConfigurationModule;
/// use optima::robot_set_modules::robot_set_joint_state_module::RobotSetJointStateType;
/// use optima::scenes::robot_geometric_shape_scene::{EnvObjSpawner, RobotGeometricShapeScene, RobotGeometricShapeSceneQuery};
/// use optima::utils::utils_robot::robot_module_utils::RobotNames;
/// use optima::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
/// use optima::utils::utils_shape_geometry::geometric_shape::{LogCondition, StopCondition};
///
/// // Initializes a RobotSetConfigurationModule that will be used to create a RobotSet below.
/// let mut r = RobotSetConfigurationModule::new_empty();
///
/// r.add_robot_configuration_from_names(RobotNames::new_base("ur5"))?;
/// let mut sawyer_configuration = RobotConfigurationModule::new_from_names(RobotNames::new_base("sawyer"))?;
/// sawyer_configuration.set_mobile_base_mode(ContiguousChainMobilityMode::PlanarTranslation {x_bounds: (-2.0, 2.0),y_bounds: (-2.0, 2.0)})?;
/// sawyer_configuration.set_base_offset(&OptimaSE3Pose::new_from_euler_angles(0.,0.,0.,1.0,0.,0., &OptimaSE3PoseType::ImplicitDualQuaternion))?;
/// r.add_robot_configuration(sawyer_configuration)?;
///
/// // Initializes a RobotSet.
/// let robot_set = RobotSet::new_from_robot_set_configuration_module(r)?;
///
/// // Initializes a RobotScene with the robot set
/// let mut scene = RobotGeometricShapeScene::new(robot_set, RobotLinkShapeRepresentation::ConvexShapes, vec![])?;
/// scene.add_environment_object(EnvObjSpawner::new("sphere", Some(0.7), None, None, None), false)?;
/// let joint_state = scene.robot_set().robot_set_joint_state_module().spawn_zeros_robot_set_joint_state(RobotSetJointStateType::DOF);
///
/// // Query input.
/// let input = RobotGeometricShapeSceneQuery::Contact {
///     robot_set_joint_state: &joint_state,
///     env_obj_pose_constraint_group_input: None,
///     prediction: 0.2 ,
///     inclusion_list: &None
/// };
///
/// // Runs the Contact query on the scene and prints a summary of the result.
/// let res = scene.shape_collection_query(&input, StopCondition::None, LogCondition::LogAll, true)?;
/// res.print_summary();
///
/// ```
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotGeometricShapeScene {
    robot_set: RobotSet,
    robot_link_shape_representation: RobotLinkShapeRepresentation,
    shape_collection: ShapeCollection,
    robot_and_link_idx_to_shape_idxs_mapping: Vec<Vec<Vec<usize>>>,
    env_obj_idx_to_shape_idxs_mapping: Vec<Vec<usize>>,
    env_obj_idx_to_pose_constraint: Vec<EnvObjPoseConstraint>,
    last_robot_link_shape_idx: usize,
    env_obj_count: usize,
    env_obj_spawners: Vec<EnvObjSpawner>
}
impl RobotGeometricShapeScene {
    pub fn new(robot_set: RobotSet, robot_link_shape_representation: RobotLinkShapeRepresentation, env_obj_spawners: Vec<EnvObjSpawner>) -> Result<Self, OptimaError> {
        let robot_set_geometric_shape_module = robot_set.generate_robot_set_geometric_shape_module()?;
        let robot_set_shape_collection = robot_set_geometric_shape_module.robot_set_shape_collection(&robot_link_shape_representation)?;
        
        let shape_collection = robot_set_shape_collection.shape_collection().clone();
        let robot_and_link_idx_to_shape_idxs_mapping = robot_set_shape_collection.robot_and_link_idx_to_shape_idxs_mapping().clone();

        let last_robot_link_shape_idx = robot_set_shape_collection.shape_collection().shapes().len() - 1;

        let mut out_self = Self {
            robot_set,
            robot_link_shape_representation,
            shape_collection,
            robot_and_link_idx_to_shape_idxs_mapping,
            env_obj_idx_to_shape_idxs_mapping: vec![],
            env_obj_idx_to_pose_constraint: vec![],
            last_robot_link_shape_idx,
            env_obj_count: 0,
            env_obj_spawners: vec![]
        };

        for s in env_obj_spawners { out_self.add_environment_object(s, false)?; }

        Ok(out_self)
    }
    /// Adds an environment object to the scene.  Returns environment object index.
    pub fn add_environment_object(&mut self,
                                  spawner: EnvObjSpawner,
                                  force_preprocessing: bool) -> Result<usize, OptimaError> {
        self.env_obj_spawners.push( spawner.clone());

        self.preprocess_object_shape_if_necessary(&spawner.asset_name, spawner.decomposition_resolution, force_preprocessing)?;
        let geometric_shapes = self.get_geometric_shapes_to_add_to_environment(&spawner.asset_name, spawner.scale, spawner.shape_representation)?;
        return self.add_env_obj_geometric_shapes_to_scene(&geometric_shapes, spawner.pose_constraint);
    }
    fn get_path_to_mesh_file(&self, name: &str) -> Result<OptimaStemCellPath, OptimaError> {
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::SceneMeshFile {name: name.to_string()});
        OptimaError::new_check_for_stem_cell_path_does_not_exist(&path, file!(), line!())?;

        let files_in_directory = path.get_all_items_in_directory(false, false);

        if files_in_directory.len() == 0 { return Err(OptimaError::new_generic_error_str(&format!("Scene mesh file {:?} does not contain file in directory.", name), file!(), line!())); }

        if files_in_directory.len() > 1 {
            optima_print(&format!("WARNING: Scene mesh file {:?} contains more than one file.  Will arbitrarily try reading the first as the mesh file.", name), PrintMode::Println, PrintColor::Yellow, true);
        }

        path.append(&files_in_directory[0]);

        return Ok(path);
    }
    fn preprocess_object_shape_if_necessary(&self, name: &str, decomposition_resolution: Option<ConvexDecompositionResolution>, force_preprocessing: bool) -> Result<(), OptimaError> {
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::SceneMeshFile {name: name.to_string()});
        OptimaError::new_check_for_stem_cell_path_does_not_exist(&path, file!(), line!())?;

        let mut path = OptimaStemCellPath::new_asset_path().expect("error");
        path.append_file_location(&OptimaAssetLocation::SceneMeshFilePreprocessing {name: name.to_string()});
        if !path.exists() || force_preprocessing {
            optima_print(&format!("Preprocessing environment {}...", name), PrintMode::Println, PrintColor::Blue, true);
            let decomposition_resolution = match decomposition_resolution {
                None => { ConvexDecompositionResolution::Medium }
                Some(d) => { d }
            };

            let path_to_mesh_file = self.get_path_to_mesh_file(name)?;
            let trimesh_engine = path_to_mesh_file.load_file_to_trimesh_engine()?;

            let convex_shape = trimesh_engine.compute_convex_hull();
            let convex_shape_subcomponents = trimesh_engine.compute_convex_decomposition(decomposition_resolution);

            let mut output_path = OptimaStemCellPath::new_asset_path()?;
            output_path.append_file_location(&OptimaAssetLocation::SceneMeshFileConvexShape {name: name.to_string()});
            output_path.append("convex_shape.stl");
            output_path.save_trimesh_engine_to_stl(&convex_shape)?;

            let mut output_path = OptimaStemCellPath::new_asset_path()?;
            output_path.append_file_location(&OptimaAssetLocation::SceneMeshFileConvexShapeSubcomponents {name: name.to_string()});

            for (i, c) in convex_shape_subcomponents.iter().enumerate() {
                let mut output_path_local = output_path.clone();
                output_path_local.append(&format!("{}.stl", i));
                output_path_local.save_trimesh_engine_to_stl(c)?;
            }
        }

        Ok(())
    }
    fn get_geometric_shapes_to_add_to_environment(&self, name: &str, scale: Option<f64>, shape_representation: Option<EnvObjShapeRepresentation>) -> Result<Vec<GeometricShape>, OptimaError> {
        let mut out_vec = vec![];

        let shape_representation = match shape_representation {
            None => {EnvObjShapeRepresentation::default()}
            Some(s) => {s}
        };

        let mut base_trimesh_engines = match shape_representation {
            EnvObjShapeRepresentation::BestFitSphere |
            EnvObjShapeRepresentation::BestFitCube |
            EnvObjShapeRepresentation::BestFitConvexShape  => {
                let mut p = OptimaStemCellPath::new_asset_path()?;
                p.append_file_location(&OptimaAssetLocation::SceneMeshFileConvexShape {name: name.to_string()});
                p.load_all_possible_files_in_directory_to_trimesh_engines()?
            }
            EnvObjShapeRepresentation::SphereSubcomponents |
            EnvObjShapeRepresentation::CubeSubcomponents |
            EnvObjShapeRepresentation::ConvexShapeSubcomponents => {
                let mut p = OptimaStemCellPath::new_asset_path()?;
                p.append_file_location(&OptimaAssetLocation::SceneMeshFileConvexShapeSubcomponents {name: name.to_string()});
                p.load_all_possible_files_in_directory_to_trimesh_engines()?
            }
            EnvObjShapeRepresentation::TriangleMesh => {
                let mut p = OptimaStemCellPath::new_asset_path()?;
                p.append_file_location(&OptimaAssetLocation::SceneMeshFile {name: name.to_string()});
                p.load_all_possible_files_in_directory_to_trimesh_engines()?
            }
        };

        if let Some(scale) = scale {
            if scale != 1.0 { for t in &mut base_trimesh_engines { t.scale_vertices(scale); } }
        }

        let add_idx = self.env_obj_count;

        let mut base_shapes = vec![];
        for (i, t) in base_trimesh_engines.iter().enumerate() {
            let signature = GeometricShapeSignature::EnvironmentObject { environment_object_idx: add_idx, shape_idx_in_object: i };
            base_shapes.push(GeometricShape::new_triangle_mesh_from_trimesh_engine(t, signature))
        }

        for s in &base_shapes {
            match shape_representation {
                EnvObjShapeRepresentation::BestFitSphere => {
                    out_vec.push(s.to_best_fit_sphere());
                }
                EnvObjShapeRepresentation::BestFitCube => {
                    out_vec.push(s.to_best_fit_cube());
                }
                EnvObjShapeRepresentation::BestFitConvexShape => {
                    out_vec.push(s.clone());
                }
                EnvObjShapeRepresentation::SphereSubcomponents => {
                    out_vec.push(s.to_best_fit_sphere());
                }
                EnvObjShapeRepresentation::CubeSubcomponents => {
                    out_vec.push(s.to_best_fit_cube());
                }
                EnvObjShapeRepresentation::ConvexShapeSubcomponents => {
                    out_vec.push(s.clone());
                }
                EnvObjShapeRepresentation::TriangleMesh => {
                    out_vec.push(s.clone());
                }
            }
        }

        Ok(out_vec)
    }
    fn add_env_obj_geometric_shapes_to_scene(&mut self, shapes: &Vec<GeometricShape>, pose_constraint: Option<EnvObjPoseConstraint>) -> Result<usize, OptimaError> {
        let add_idx = self.env_obj_count.clone();

        let mut mapping_vec = vec![];
        for s in shapes.iter() {
            let shape_idx = self.shape_collection.shapes().len();
            self.shape_collection.add_geometric_shape(s.clone());
            mapping_vec.push(shape_idx);
        }

        let num_shapes = self.shape_collection.shapes().len();

        for shape_idx in &mapping_vec {
            for env_obj_shape_idx in self.last_robot_link_shape_idx+1..num_shapes {
                self.shape_collection.set_base_skip_from_idxs(true, *shape_idx, env_obj_shape_idx)?;
                self.shape_collection.replace_skip_from_idxs(true, *shape_idx, env_obj_shape_idx)?;
            }
        }

        self.env_obj_idx_to_shape_idxs_mapping.push(mapping_vec);

        let pose_constraint = match pose_constraint {
            None => { EnvObjPoseConstraint::default()}
            Some(p) => {p}
        };
        self.env_obj_idx_to_pose_constraint.push(pose_constraint);

        self.env_obj_count += 1;
        Ok(add_idx)
    }
    fn check_if_pose_constraint_will_cause_cycle(&self, env_obj_idx: usize, pose_constraint: &EnvObjPoseConstraint) -> bool {
        let mut curr_pose_constraint = pose_constraint;
        loop {
            match curr_pose_constraint {
                EnvObjPoseConstraint::RelativeOffset { parent_signature, offset:_ } => {
                    match parent_signature {
                        GeometricShapeSignature::EnvironmentObject { environment_object_idx, shape_idx_in_object: _ } => {
                            if *environment_object_idx == env_obj_idx {
                                return true;
                            } else {
                                curr_pose_constraint = &self.env_obj_idx_to_pose_constraint[*environment_object_idx];
                            }
                        }
                        _ => { return false; }
                    }
                }
                EnvObjPoseConstraint::Absolute(_) => { return false; }
            }
        }

    }
    pub fn robot_set(&self) -> &RobotSet {
        &self.robot_set
    }
    pub fn robot_link_shape_representation(&self) -> &RobotLinkShapeRepresentation {
        &self.robot_link_shape_representation
    }
    pub fn shape_collection(&self) -> &ShapeCollection {
        &self.shape_collection
    }
    pub fn get_shape_idxs_from_robot_idx_and_link_idx(&self, robot_idx_in_set: usize, link_idx_in_robot: usize) -> Result<&Vec<usize>, OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(robot_idx_in_set, self.robot_and_link_idx_to_shape_idxs_mapping.len(), file!(), line!())?;
        OptimaError::new_check_for_idx_out_of_bound_error(link_idx_in_robot, self.robot_and_link_idx_to_shape_idxs_mapping[robot_idx_in_set].len(), file!(), line!())?;

        return Ok(&self.robot_and_link_idx_to_shape_idxs_mapping[robot_idx_in_set][link_idx_in_robot]);
    }
    pub fn get_shape_idxs_from_env_obj_idx(&self, env_obj_idx: usize) -> Result<&Vec<usize>, OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(env_obj_idx, self.env_obj_idx_to_shape_idxs_mapping.len(), file!(), line!())?;

        return Ok(&self.env_obj_idx_to_shape_idxs_mapping[env_obj_idx])
    }
    /// Updates the pose constraint on a given environment object in the scene.
    pub fn update_env_obj_pose_constraint(&mut self, env_obj_idx: usize, pose_constraint: EnvObjPoseConstraint) -> Result<(), OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(env_obj_idx, self.env_obj_idx_to_pose_constraint.len(), file!(), line!())?;

        let causes_cycle = self.check_if_pose_constraint_will_cause_cycle(env_obj_idx, &pose_constraint);
        if causes_cycle {
            optima_print(&format!("WARNING: Adding constraint {:?} to environment object {:?} would cause constraint cycle.  Could not add.", pose_constraint, env_obj_idx), PrintMode::Println, PrintColor::Yellow, true);
            return Ok(());
        }

        self.env_obj_idx_to_pose_constraint[env_obj_idx] = pose_constraint;
        Ok(())
    }
    pub fn recover_poses(&self,
                         set_joint_state: &RobotSetJointState,
                         pose_constraint_group_input: Option<&EnvObjPoseConstraintGroupInput>) -> Result<ShapeCollectionInputPoses, OptimaError> {
        let mut out_poses = ShapeCollectionInputPoses::new(&self.shape_collection);

        let fk_res = self.robot_set.robot_set_kinematics_module().compute_fk(set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
        let robot_fk_results = fk_res.robot_fk_results();

        for (robot_idx_in_set, robot_fk_result) in robot_fk_results.iter().enumerate() {
            for (link_idx_in_robot, link_entry) in robot_fk_result.link_entries().iter().enumerate() {
                if let Some(pose) = link_entry.pose() {
                    let shape_idxs = self.get_shape_idxs_from_robot_idx_and_link_idx(robot_idx_in_set, link_idx_in_robot)?;
                    for shape_idx in shape_idxs {
                        out_poses.insert_or_replace_pose_by_idx(*shape_idx, pose.clone())?;
                    }
                }
            }
        }

        let mut pose_constraints = vec![];
        let pose_constraint_group_input = pose_constraint_group_input;
        match pose_constraint_group_input {
            None => { pose_constraints = self.env_obj_idx_to_pose_constraint.clone(); }
            Some(pose_constraint_group_input) => {
                let inputs = &pose_constraint_group_input.inputs;
                for (i, input) in inputs.iter().enumerate() {
                    match input {
                        None => { pose_constraints.push(self.env_obj_idx_to_pose_constraint[i].clone()) }
                        Some(input) => { pose_constraints.push(input.clone()) }
                    }
                }
            }
        }

        let mut loop_count = 0;
        let num_env_objs = self.env_obj_count;
        let mut env_obj_is_done = vec![];
        for _ in 0..num_env_objs { env_obj_is_done.push(false); }

        loop {
            let mut complete = true;
            for env_obj_idx in 0..num_env_objs {
                if env_obj_is_done[env_obj_idx] { continue; }
                let pose_constraint = &pose_constraints[env_obj_idx];
                match pose_constraint {
                    EnvObjPoseConstraint::Absolute(p) => {
                        let shape_idxs = self.get_shape_idxs_from_env_obj_idx(env_obj_idx)?;
                        for shape_idx in shape_idxs {
                            out_poses.insert_or_replace_pose_by_idx(*shape_idx, p.clone())?;
                        }
                        env_obj_is_done[env_obj_idx] = true;
                    }
                    EnvObjPoseConstraint::RelativeOffset { parent_signature, offset } => {
                        let parent_shape_idx = self.shape_collection.get_shape_idx_from_signature(parent_signature)?;
                        OptimaError::new_check_for_idx_out_of_bound_error(parent_shape_idx, out_poses.poses().len(), file!(), line!())?;
                        let parent_shape_pose_option = out_poses.poses().get(parent_shape_idx).unwrap();
                        match parent_shape_pose_option {
                            None => { complete = false; }
                            Some(parent_shape_pose) => {
                                let out_pose = parent_shape_pose.multiply(offset, true)?;
                                let shape_idxs = self.get_shape_idxs_from_env_obj_idx(env_obj_idx)?;
                                for shape_idx in shape_idxs {
                                    out_poses.insert_or_replace_pose_by_idx(*shape_idx, out_pose.clone())?;
                                }
                                env_obj_is_done[env_obj_idx] = true;
                            }
                        }
                    }
                }
            }
            loop_count += 1;
            if loop_count > 10_000 {
                optima_print("ERROR: Problem in RobotGeometricShapeScene recover_poses.  Seems like there is a loop in a pose constraint, so go fix the loop detection function.", PrintMode::Println, PrintColor::Red, true);
                panic!()
            }
            if complete { break; }
        }

        Ok(out_poses)
    }
    /*
    pub fn set_attachment_between_env_obj_and_robot_link(&mut self,
                                                         env_obj_idx: usize,
                                                         robot_idx_in_set: usize,
                                                         link_idx_in_robot: usize,
                                                         set_joint_state: &RobotSetJointState,
                                                         pose_constraint_group_input: Option<&EnvObjPoseConstraintGroupInput>) -> Result<(), OptimaError> {
        self.reset_skips_on_given_env_obj(env_obj_idx)?;

        let poses = self.recover_poses(set_joint_state, pose_constraint_group_input)?;

        let shape_idxs = self.get_shape_idxs_from_robot_idx_and_link_idx(robot_idx_in_set, link_idx_in_robot)?;
        let shape_idx = shape_idxs[0];
        let link_pose = poses.poses()[shape_idx].as_ref().unwrap();

        let shape_idxs = self.get_shape_idxs_from_env_obj_idx(env_obj_idx)?;
        let shape_idx = shape_idxs[0];
        let env_obj_pose = poses.poses()[shape_idx].as_ref().unwrap();

        // set new constraint
        let disp = link_pose.displacement(&env_obj_pose, true)?;
        self.env_obj_idx_to_pose_constraint[env_obj_idx] = EnvObjPoseConstraint::RelativeOffset {
            parent_signature: GeometricShapeSignature::RobotSetLink {
                robot_idx_in_set,
                link_idx_in_robot,
                shape_idx_in_link: 0
            },
            offset: disp
        };

        // set skips such that the given env obj is now checked for collision against other env objects.


        // set skips such that the given env obj is not checked against robot links that were close in
        // distance at time of attachment.

        Ok(())
    }
    pub fn reset_skips_on_given_env_obj(&mut self, env_obj_idx: usize) -> Result<(), OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(env_obj_idx, self.env_obj_count, file!(), line!())?;

        let shape_idxs = self.get_shape_idxs_from_env_obj_idx(env_obj_idx)?.clone();
        let num_shapes = self.shape_collection.shapes().len();
        for given_env_obj_shape_idx in &shape_idxs {
            for env_obj_shape_idx in 0..num_shapes {
                self.shape_collection.reset_skip_to_base_from_idxs(*given_env_obj_shape_idx, env_obj_shape_idx)?;
            }
        }

        Ok(())
    }
    */
    pub fn shape_collection_query<'a>(&'a self,
                                      input: &'a RobotGeometricShapeSceneQuery,
                                      stop_condition: StopCondition,
                                      log_condition: LogCondition,
                                      sort_outputs: bool) -> Result<GeometricShapeQueryGroupOutput, OptimaError> {
        return match input {
            RobotGeometricShapeSceneQuery::ProjectPoint { robot_set_joint_state, env_obj_pose_constraint_group_input, point, solid, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::ProjectPoint {
                    poses: &poses,
                    point,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::ContainsPoint { robot_set_joint_state, env_obj_pose_constraint_group_input, point, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::ContainsPoint {
                    poses: &poses,
                    point,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::DistanceToPoint { robot_set_joint_state, env_obj_pose_constraint_group_input, point, solid, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::DistanceToPoint {
                    poses: &poses,
                    point,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::IntersectsRay { robot_set_joint_state, env_obj_pose_constraint_group_input, ray, max_toi, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::IntersectsRay {
                    poses: &poses,
                    ray: *ray,
                    max_toi: *max_toi,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::CastRay { robot_set_joint_state, env_obj_pose_constraint_group_input, ray, max_toi, solid, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::CastRay {
                    poses: &poses,
                    ray: *ray,
                    max_toi: *max_toi,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::CastRayAndGetNormal { robot_set_joint_state, env_obj_pose_constraint_group_input, ray, max_toi, solid, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::CastRayAndGetNormal {
                    poses: &poses,
                    ray: *ray,
                    max_toi: *max_toi,
                    solid: *solid,
                    inclusion_list,
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::IntersectionTest { robot_set_joint_state, env_obj_pose_constraint_group_input, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::IntersectionTest {
                    poses: &poses,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::Distance { robot_set_joint_state, env_obj_pose_constraint_group_input, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::Distance {
                    poses: &poses,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::ClosestPoints { robot_set_joint_state, env_obj_pose_constraint_group_input, max_dis, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::ClosestPoints {
                    poses: &poses,
                    max_dis: *max_dis,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::Contact { robot_set_joint_state, env_obj_pose_constraint_group_input, prediction, inclusion_list } => {
                let poses = self.recover_poses(robot_set_joint_state, *env_obj_pose_constraint_group_input)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::Contact {
                    poses: &poses,
                    prediction: *prediction,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
            RobotGeometricShapeSceneQuery::CCD { robot_set_joint_state_t1, env_obj_pose_constraint_group_input_t1, robot_set_joint_state_t2, env_obj_pose_constraint_group_input_t2, inclusion_list } => {
                let poses_t1 = self.recover_poses(robot_set_joint_state_t1, *env_obj_pose_constraint_group_input_t1)?;
                let poses_t2 = self.recover_poses(robot_set_joint_state_t2, *env_obj_pose_constraint_group_input_t2)?;
                self.shape_collection.shape_collection_query(&ShapeCollectionQuery::CCD {
                    poses_t1: &poses_t1,
                    poses_t2: &poses_t2,
                    inclusion_list
                }, stop_condition, log_condition, sort_outputs)
            }
        }
    }

    pub fn spawn_query_list(&self) -> ShapeCollectionQueryList {
        return self.shape_collection.spawn_query_list();
    }
    pub fn spawn_query_pairs_list(&self, override_all_skips: bool) -> ShapeCollectionQueryPairsList {
        return self.shape_collection.spawn_query_pairs_list(override_all_skips);
    }
    pub fn spawn_proxima_engine(&self) -> ProximaEngine {
        return self.shape_collection.spawn_proxima_engine();
    }
    pub fn spawn_bvh<T: BVHCombinableShape>(&self, robot_set_joint_state: &RobotSetJointState, env_obj_pose_constraint_group_input: Option<&EnvObjPoseConstraintGroupInput>, branch_factor: usize) -> ShapeCollectionBVH<T> {
        let poses = self.recover_poses(robot_set_joint_state, env_obj_pose_constraint_group_input).expect("error");

        return self.shape_collection.spawn_bvh(&poses, branch_factor);
    }
    pub fn update_bvh<T: BVHCombinableShape>(&self, bvh: &mut ShapeCollectionBVH<T>, robot_set_joint_state: &RobotSetJointState, env_obj_pose_constraint_group_input: Option<&EnvObjPoseConstraintGroupInput>) {
        let poses = self.recover_poses(robot_set_joint_state, env_obj_pose_constraint_group_input).expect("error");

        self.shape_collection.update_bvh(bvh, &poses);
    }

    pub fn proxima_proximity_query(&self,
                                   robot_set_joint_state: &RobotSetJointState,
                                   env_obj_pose_constraint_group_input: Option<&EnvObjPoseConstraintGroupInput>,
                                   proxima_engine: &mut ProximaEngine,
                                   d_max: f64,
                                   a_max: f64,
                                   loss_function: SignedDistanceLossFunction,
                                   r: f64,
                                   proxima_budget: ProximaBudget,
                                   inclusion_list: &Option<&ShapeCollectionQueryPairsList>) -> Result<ProximaProximityOutput, OptimaError> {
        let poses = self.recover_poses(robot_set_joint_state, env_obj_pose_constraint_group_input)?;
        return self.shape_collection.proxima_proximity_query(&poses, proxima_engine, d_max, a_max, loss_function, r, proxima_budget, inclusion_list);
    }
    pub fn proxima_scene_filter(&self,
                                   robot_set_joint_state: &RobotSetJointState,
                                   env_obj_pose_constraint_group_input: Option<&EnvObjPoseConstraintGroupInput>,
                                   proxima_engine: &mut ProximaEngine,
                                   d_max: f64,
                                   a_max: f64,
                                   loss_function: SignedDistanceLossFunction,
                                   r: f64,
                                   inclusion_list: &Option<&ShapeCollectionQueryPairsList>) -> Result<ProximaSceneFilterOutput, OptimaError> {
        let poses = self.recover_poses(robot_set_joint_state, env_obj_pose_constraint_group_input)?;
        return self.shape_collection.proxima_scene_filter(&poses, proxima_engine, d_max, a_max, &loss_function, r, inclusion_list);
    }
    pub fn bvh_scene_filter<T: BVHCombinableShape>(&self, bvh: &mut ShapeCollectionBVH<T>, robot_set_joint_state: &RobotSetJointState, env_obj_pose_constraint_group_input: Option<&EnvObjPoseConstraintGroupInput>, visit: BVHVisit) -> BVHSceneFilterOutput {
        let poses = self.recover_poses(robot_set_joint_state, env_obj_pose_constraint_group_input).expect("error");

        return self.shape_collection.bvh_scene_filter(bvh, &poses, visit);
    }

    pub fn print_summary(&self) {
        self.robot_set.print_summary();
        optima_print_new_line();

        let num_objects = self.env_obj_count;
        optima_print(&format!("{} objects.", num_objects), PrintMode::Println, PrintColor::Cyan, true);
        for i in 0..num_objects {
            optima_print(&format!(" Object {} ---> ", i), PrintMode::Println, PrintColor::Cyan, false);
            optima_print(&format!("    Object Info: {:?}", self.env_obj_spawners[i].to_self_no_nones()), PrintMode::Println, PrintColor::None, false);
            optima_print(&format!("    Object Pose: {:?}", self.env_obj_idx_to_pose_constraint[i]), PrintMode::Println, PrintColor::None, false);
        }
    }
}
impl SaveAndLoadable for RobotGeometricShapeScene {
    type SaveType = Self;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        self.clone()
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        return Ok(load);
    }
}
impl GetRobotSet for RobotGeometricShapeScene {
    fn get_robot_set(&self) -> &RobotSet {
        &self.robot_set
    }
}
impl GetRobotGeometricShapeScene for RobotGeometricShapeScene {
    fn get_robot_geometric_shape_scene(&self) -> &RobotGeometricShapeScene {
        self
    }
}

/// (You probably want to use RobotGeometricShapeScenePy instead.)
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotGeometricShapeScene {
    #[new]
    pub fn new_py(robot_set_py: RobotSetPy, robot_link_shape_representation: &str) -> Self {
        let robot_set = robot_set_py.get_robot_set().clone();
        let self_res = Self::new(robot_set, RobotLinkShapeRepresentation::from_ron_string(robot_link_shape_representation).unwrap(), vec![]);
        return self_res.unwrap();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pyclass]
pub struct RobotGeometricShapeScenePy {
    #[pyo3(get)]
    robot_set_py: Py<RobotSetPy>,
    robot_geometric_shape_scene: RobotGeometricShapeScene
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotGeometricShapeScenePy {
    #[new]
    #[args(robot_link_shape_representation="\"Cubes\"")]
    pub fn new(robot_set_py: RobotSetPy, robot_link_shape_representation: &str, py: Python) -> Self {
        let robot_geometric_shape_scene = RobotGeometricShapeScene::new_py(robot_set_py.clone(), robot_link_shape_representation);
        Self {
            robot_set_py: Py::new(py, robot_set_py).expect("error"),
            robot_geometric_shape_scene
        }
    }
    #[args(scale="1.0", shape_representation="\"CubeSubcomponents\"", decomposition_resolution="\"Medium\"", force_preprocessing="false")]
    pub fn add_environment_object_py(&mut self, asset_name: &str, scale: f64, shape_representation: &str, decomposition_resolution: &str, force_preprocessing: bool, pose: Option<OptimaSE3PosePy>) -> usize {
        let env_obj_spawner = EnvObjSpawner::new(
            asset_name,
            Some(scale),
            Some(EnvObjShapeRepresentation::from_ron_string(shape_representation).unwrap()),
            Some(ConvexDecompositionResolution::from_ron_string(decomposition_resolution).unwrap()),
            Some(EnvObjPoseConstraint::Absolute(match &pose {
                None => { OptimaSE3Pose::default() }
                Some(p) => { p.pose().clone() }
            })));

        let idx = self.robot_geometric_shape_scene.add_environment_object(env_obj_spawner, force_preprocessing).expect("error");

        return idx;
    }
    pub fn update_env_obj_pose_constraint_py(&mut self, env_obj_idx: usize, pose: OptimaSE3PosePy, parent_signature: Option<&str>) {
        match parent_signature {
            None => {
                let pose_constraint = EnvObjPoseConstraint::Absolute(pose.pose().clone());
                self.robot_geometric_shape_scene.update_env_obj_pose_constraint(env_obj_idx, pose_constraint).expect("error");
            }
            Some(parent_signature) => {
                let parent_signature = GeometricShapeSignature::from_ron_string(parent_signature).expect("error");
                let pose_constraint = EnvObjPoseConstraint::RelativeOffset { parent_signature: parent_signature, offset:  pose.pose().clone() };
                self.robot_geometric_shape_scene.update_env_obj_pose_constraint(env_obj_idx, pose_constraint).expect("error");
            }
        }
    }
    pub fn print_summary_py(&self) {
        self.robot_geometric_shape_scene.print_summary();
    }
    #[args(stop_condition="\"None\"", log_condition="\"LogAll\"", sort_outputs="true", include_full_output_json_string="true")]
    pub fn contact_query_py(&self, robot_set_joint_state: Vec<f64>, prediction: f64, stop_condition: &str, log_condition: &str, sort_outputs: bool, include_full_output_json_string: bool) -> GeometricShapeQueryGroupOutputPy {
        let stop_condition = StopCondition::from_ron_string(stop_condition).expect("error");
        let log_condition = LogCondition::from_ron_string(log_condition).expect("error");

        let robot_set_joint_state = self.robot_geometric_shape_scene.robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let input = RobotGeometricShapeSceneQuery::Contact {
            robot_set_joint_state: &robot_set_joint_state,
            env_obj_pose_constraint_group_input: None,
            prediction,
            inclusion_list: &None
        };
        let res = self.robot_geometric_shape_scene.shape_collection_query(&input, stop_condition, log_condition, sort_outputs).expect("error");
        let py_output = res.convert_to_py_output(include_full_output_json_string);
        return py_output;
    }

    pub fn spawn_proxima_engine_py(&self) -> ProximaEngine {
        self.robot_geometric_shape_scene.spawn_proxima_engine()
    }
    pub fn proxima_proximity_query_py(&self,
                                      robot_set_joint_state: Vec<f64>,
                                      proxima_engine: &mut ProximaEngine,
                                      d_max: f64,
                                      a_max: f64,
                                      loss_function: &str,
                                      r: f64,
                                      proxima_budget: &str) -> ProximaProximityOutput {
        let robot_set_joint_state = self.robot_geometric_shape_scene.robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let res = self.robot_geometric_shape_scene.proxima_proximity_query(
            &robot_set_joint_state,
            None,
            proxima_engine,
            d_max,
            a_max,
            SignedDistanceLossFunction::from_ron_string(loss_function).expect("error"), r,
            ProximaBudget::from_ron_string(proxima_budget).expect("error"), &None).expect("error");
        return res;
    }
    pub fn update_aabb_bvh(&self, bvh_aabb: &mut ShapeCollectionBVHAABB, robot_set_joint_state: Vec<f64>) {
        let robot_set_joint_state = self.robot_geometric_shape_scene.robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let poses = self.robot_geometric_shape_scene.recover_poses(&robot_set_joint_state, None).expect("error");
        bvh_aabb.bvh.bvh_mut().update(&self.robot_geometric_shape_scene.shape_collection.shapes(), &poses);
    }

    pub fn spawn_bvh_aabb_py(&self, robot_set_joint_state: Vec<f64>, branch_factor: usize) -> ShapeCollectionBVHAABB {
        let robot_set_joint_state = self.robot_geometric_shape_scene.robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let bvh = self.robot_geometric_shape_scene.spawn_bvh::<BVHCombinableShapeAABB>(&robot_set_joint_state, None, branch_factor);
        ShapeCollectionBVHAABB {
            bvh
        }
    }
    #[args(stop_condition="\"None\"", log_condition="\"LogAll\"", sort_outputs="true", include_full_output_json_string="true")]
    pub fn bvh_aabb_contact_query_py(&self, bvh_aabb: &mut ShapeCollectionBVHAABB, robot_set_joint_state: Vec<f64>, prediction: f64, stop_condition: &str, log_condition: &str, sort_outputs: bool, include_full_output_json_string: bool) -> GeometricShapeQueryGroupOutputPy {
        let robot_set_joint_state = self.robot_geometric_shape_scene.robot_set.robot_set_joint_state_module().spawn_robot_set_joint_state_try_auto_type(DVector::from_vec(robot_set_joint_state)).expect("error");
        let filter = self.robot_geometric_shape_scene.bvh_scene_filter(&mut bvh_aabb.bvh, &robot_set_joint_state, None, BVHVisit::Distance { margin: prediction });
        let input = RobotGeometricShapeSceneQuery::Contact {
            robot_set_joint_state: &robot_set_joint_state,
            env_obj_pose_constraint_group_input: None,
            prediction,
            inclusion_list: &Some(filter.pairs_list())
        };

        let stop_condition = StopCondition::from_ron_string(stop_condition).expect("error");
        let log_condition = LogCondition::from_ron_string(log_condition).expect("error");

        let res = self.robot_geometric_shape_scene.shape_collection_query(&input, stop_condition, log_condition, sort_outputs).expect("error");
        let py_output = res.convert_to_py_output(include_full_output_json_string);
        return py_output;
    }
}

/// Used to spawn environment objects in the scene.  These spawners can also be saved to
/// load the same environment at a later time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvObjSpawner {
    asset_name: String,
    scale: Option<f64>,
    shape_representation: Option<EnvObjShapeRepresentation>,
    decomposition_resolution: Option<ConvexDecompositionResolution>,
    pose_constraint: Option<EnvObjPoseConstraint>
}
impl EnvObjSpawner {
    pub fn new(asset_name: &str,
               scale: Option<f64>,
               shape_representation: Option<EnvObjShapeRepresentation>,
               decomposition_resolution: Option<ConvexDecompositionResolution>,
               pose_constraint: Option<EnvObjPoseConstraint>) -> Self {
        Self {
            asset_name: asset_name.to_string(),
            scale,
            shape_representation,
            decomposition_resolution,
            pose_constraint
        }
    }
    fn to_self_no_nones(&self) -> Self {
        Self {
            asset_name: self.asset_name.clone(),
            scale: match self.scale {
                None => { Some(1.0) }
                Some(s) => {Some(s)}
            },
            shape_representation: match &self.shape_representation {
                None => { Some(EnvObjShapeRepresentation::default()) }
                Some(s) => { Some(s.clone())}
            },
            decomposition_resolution: match &self.decomposition_resolution {
                None => { Some(ConvexDecompositionResolution::Low) }
                Some(d) => {Some(d.clone())}
            },
            pose_constraint: match &self.pose_constraint {
                None => { Some(EnvObjPoseConstraint::default()) }
                Some(p) => { Some(p.clone()) }
            }
        }
    }
}
impl Default for EnvObjSpawner {
    fn default() -> Self {
        Self {
            asset_name: "sphere".to_string(),
            scale: None,
            shape_representation: None,
            decomposition_resolution: None,
            pose_constraint: None
        }
    }
}

/// Used to specify a pose for a given environment object in a scene.  For example, a pose constraint
/// can be an absolue pose in the scene, or a pose that is parented to another shape with a local
/// pose offset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EnvObjPoseConstraint {
    Absolute(OptimaSE3Pose),
    RelativeOffset { parent_signature: GeometricShapeSignature, offset: OptimaSE3Pose }
}
impl Default for EnvObjPoseConstraint {
    fn default() -> Self {
        Self::Absolute(OptimaSE3Pose::default())
    }
}

/// Used to specify the geometric shape representation of a given environment object.
#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Serialize, Deserialize)]
pub enum EnvObjShapeRepresentation {
    BestFitSphere,
    BestFitCube,
    BestFitConvexShape,
    SphereSubcomponents,
    CubeSubcomponents,
    ConvexShapeSubcomponents,
    TriangleMesh
}
impl Default for EnvObjShapeRepresentation {
    fn default() -> Self {
        Self::BestFitConvexShape
    }
}

/// Can be used as an optional input into a `RobotGeometricShapeSceneQuery`.  If this object is not
/// provided in the query input, the previously specified pose constraints saved in `RobotGeometricShapeScene`
/// will be used in the query.
#[derive(Clone, Debug)]
pub struct EnvObjPoseConstraintGroupInput {
    inputs: Vec<Option<EnvObjPoseConstraint>>
}
impl EnvObjPoseConstraintGroupInput {
    pub fn new_empty_some(robot_geometric_shape_scene: &RobotGeometricShapeScene) -> Self {
        let num_env_objects = robot_geometric_shape_scene.env_obj_count;
        let mut out_vec = vec![];
        for _ in 0..num_env_objects { out_vec.push(None); }

        Self {
            inputs: out_vec
        }
    }
    pub fn inject_new_pose_constraint(&mut self, env_obj_idx: usize, pose_constraint: EnvObjPoseConstraint) -> Result<(), OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(env_obj_idx, self.inputs.len(), file!(), line!())?;
        self.inputs[env_obj_idx] = Some(pose_constraint);

        Ok(())
    }
}

/// Used as an input into the powerful RobotGeometricShapeScene::shape_collection_query function.
#[derive(Clone, Debug)]
pub enum RobotGeometricShapeSceneQuery<'a> {
    ProjectPoint { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, point: &'a Vector3<f64>, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    ContainsPoint { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, point: &'a Vector3<f64>, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    DistanceToPoint { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, point: &'a Vector3<f64>, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    IntersectsRay { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, ray: &'a Ray, max_toi: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    CastRay { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, ray: &'a Ray, max_toi: f64, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    CastRayAndGetNormal { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, ray: &'a Ray, max_toi: f64, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    IntersectionTest { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    Distance { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    ClosestPoints { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, max_dis: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    Contact { robot_set_joint_state: &'a RobotSetJointState, env_obj_pose_constraint_group_input: Option<&'a EnvObjPoseConstraintGroupInput>, prediction: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    CCD { robot_set_joint_state_t1: &'a RobotSetJointState, env_obj_pose_constraint_group_input_t1: Option<&'a EnvObjPoseConstraintGroupInput>, robot_set_joint_state_t2: &'a RobotSetJointState, env_obj_pose_constraint_group_input_t2: Option<&'a EnvObjPoseConstraintGroupInput>, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> }
}
