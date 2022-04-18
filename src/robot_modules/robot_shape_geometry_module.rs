use std::time::{Duration, Instant};
use pbr::ProgressBar;
use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_modules::robot_fk_module::{RobotFKModule, RobotFKResult};
use crate::robot_modules::robot_joint_state_module::{RobotJointStateModule, RobotJointStateType};
use crate::robot_modules::robot_model_module::RobotModelModule;
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{OptimaAssetLocation, OptimaStemCellPath, RobotModuleJsonType};
use crate::utils::utils_generic_data_structures::{AveragingFloat, SquareArray2D};
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeSignature, LogCondition, StopCondition};
use crate::utils::utils_shape_geometry::geometric_shape_collection::{GeometricShapeCollection, GeometricShapeCollectionInputPoses, GeometricShapeCollectionQueryInput};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotShapeGeometryModule {
    robot_fk_module: RobotFKModule,
    robot_mesh_file_manager_module: RobotMeshFileManagerModule,
    robot_geometric_shape_collections: Vec<RobotGeometricShapeCollection>
}
impl RobotShapeGeometryModule {
    pub fn new(robot_configuration_module: &RobotConfigurationModule) -> Result<Self, OptimaError> {
        let out_self = Self::new_try_loaded(&robot_configuration_module);
        match out_self {
            Ok(out_self) => { Ok(out_self) }
            Err(_) => { return Self::new_not_loaded(&robot_configuration_module) }
        }
    }
    pub fn new_from_names(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_name, configuration_name)?;
        return Self::new(&robot_configuration_module);
    }
    pub fn new_not_loaded(robot_configuration_module: &RobotConfigurationModule) -> Result<Self, OptimaError> {
        let robot_name = robot_configuration_module.robot_model_module().robot_name().to_string();

        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        let robot_fk_module = RobotFKModule::new(robot_configuration_module.clone(), robot_joint_state_module);
        let robot_mesh_file_manager_module = RobotMeshFileManagerModule::new(&robot_configuration_module.robot_model_module())?;

        let mut out_self = Self {
            robot_fk_module,
            robot_mesh_file_manager_module,
            robot_geometric_shape_collections: vec![]
        };

        let representations = vec![
            RobotLinkShapeRepresentation::Cubes,
            RobotLinkShapeRepresentation::ConvexShapes,
            RobotLinkShapeRepresentation::SphereSubcomponents,
            RobotLinkShapeRepresentation::CubeSubcomponents,
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents,
            RobotLinkShapeRepresentation::TriangleMeshes
        ];

        for r in &representations {
            out_self.setup_robot_geometric_shape_collection(&robot_name, r)?;
        }

        Ok(out_self)
    }
    pub fn new_try_loaded(robot_configuration_module: &RobotConfigurationModule) -> Result<Self, OptimaError> {
        let robot_name = robot_configuration_module.robot_model_module().robot_name().to_string();
        let mut path = OptimaStemCellPath::new_asset_path()?;
        path.append_file_location(&OptimaAssetLocation::RobotModuleJson { robot_name: robot_name.clone(), t: RobotModuleJsonType::ShapeGeometryModule });
        let mut out_self: Self = path.load_object_from_json_file()?;
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());
        out_self.robot_fk_module = RobotFKModule::new(robot_configuration_module.clone(), robot_joint_state_module);

        Ok(out_self)
    }
    fn setup_robot_geometric_shape_collection(&mut self,
                                              robot_name: &str,
                                              robot_link_shape_representation: &RobotLinkShapeRepresentation) -> Result<(), OptimaError> {
        optima_print(&format!("Setup on {:?}...", robot_link_shape_representation), PrintMode::Println, PrintColor::Blue, true);
        // Base model modules must be used as these computations apply to all derived configuration
        // variations of this model, not just particular configurations.
        let base_robot_model_module = RobotModelModule::new(robot_name)?;
        let base_robot_fk_module = RobotFKModule::new_from_names(robot_name, None)?;
        let base_robot_joint_state_module = RobotJointStateModule::new_from_names(robot_name, None)?;
        let num_links = base_robot_model_module.links().len();

        // Initialize GeometricShapeCollision.
        let mut geometric_shape_collection = GeometricShapeCollection::new_empty();
        let geometric_shapes = robot_link_shape_representation.get_geometric_shapes(&self.robot_mesh_file_manager_module)?;
        for geometric_shape in geometric_shapes {
            if let Some(geometric_shape) = geometric_shape {
                geometric_shape_collection.add_geometric_shape(geometric_shape.clone());
            }
        }
        let num_shapes = geometric_shape_collection.shapes().len();

        // Initialize the RobotGeometricShapeCollection with the GeometricShapeCollection.
        let mut robot_geometric_shape_collection = RobotGeometricShapeCollection::new(num_links, robot_link_shape_representation.clone(), geometric_shape_collection)?;

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

        let mut pb = ProgressBar::new(1000);
        pb.format("╢▌▌░╟");
        pb.show_counter = false;

        // Where distances and intersections are actually checked at each joint state sample.
        for i in 0..max_samples {
            count += 1.0;
            let sample = base_robot_joint_state_module.sample_joint_state(&RobotJointStateType::Full);
            let fk_res = base_robot_fk_module.compute_fk(&sample, &OptimaSE3PoseType::ImplicitDualQuaternion)?;
            let poses = fk_res.to_geometric_shape_collection_input_poses(&robot_geometric_shape_collection)?;
            let input = GeometricShapeCollectionQueryInput::Distance { poses: &poses };

            let res = robot_geometric_shape_collection.geometric_shape_collection.generic_group_query(&input, StopCondition::None, LogCondition::LogAll, false)?;

            let outputs = res.outputs();
            for output in outputs {
                let signatures = output.signatures();
                let signature1 = &signatures[0];
                let signature2 = &signatures[1];
                let shape_idx1 = robot_geometric_shape_collection.geometric_shape_collection.get_shape_idx_from_signature(signature1)?;
                let shape_idx2 = robot_geometric_shape_collection.geometric_shape_collection.get_shape_idx_from_signature(signature2)?;
                let dis = output.raw_output().unwrap_distance()?;
                distance_average_array.data_cell_mut(shape_idx1, shape_idx2)?.absorb(dis);
                distance_average_array.data_cell_mut(shape_idx2, shape_idx1)?.absorb(dis);
                if dis <= 0.0 {
                    *collision_counter_array.data_cell_mut(shape_idx1, shape_idx2)? += 1.0;
                    *collision_counter_array.data_cell_mut(shape_idx2, shape_idx1)? += 1.0;
                }
            }

            let duration = start.elapsed();
            let duration_ratio = duration.as_secs_f64() / robot_link_shape_representation.stop_at_min_sample_duration().as_secs_f64();
            let max_sample_ratio = i as f64 / max_samples as f64;
            let min_sample_ratio = i as f64 / min_samples as f64;
            let ratio = duration_ratio.max(max_sample_ratio).min(min_sample_ratio);
            pb.set((ratio * 1000.0) as u64);

            if duration > robot_link_shape_representation.stop_at_min_sample_duration() && i >= min_samples { break; }
        }

        // Determines average distances and decides if links should be skipped based on previous
        // computations.  These reesults are saved in the RobotGeometricShapeCollection.
        for i in 0..num_shapes {
            for j in 0..num_shapes {
                // Retrieves and saves the average distance between the given pair of links.
                let averaging_float = distance_average_array.data_cell(i, j)?;
                robot_geometric_shape_collection.geometric_shape_collection.set_average_distance_from_idxs(averaging_float.value(), i, j)?;

                // Pairwise checks should never happen between the same shape.
                if i == j { robot_geometric_shape_collection.geometric_shape_collection.set_skip_from_idxs(true, i, j)?; }

                let shapes = robot_geometric_shape_collection.geometric_shape_collection.shapes();
                let signature1 = shapes[i].signature();
                let signature2 = shapes[j].signature();
                match signature1 {
                    GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: _ } => {
                        let link_idx1 = link_idx.clone();
                        match signature2 {
                            GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: _ } => {
                                let link_idx2 = link_idx.clone();
                                if link_idx1 == link_idx2 {
                                    robot_geometric_shape_collection.geometric_shape_collection.set_skip_from_idxs(true, i, j)?;
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
                    robot_geometric_shape_collection.geometric_shape_collection.set_skip_from_idxs(true, i, j)?;
                }

                // Checks if links are never in collision
                if count >= 1000.0 && ratio_of_checks_in_collision == 0.0 {
                    robot_geometric_shape_collection.geometric_shape_collection.set_skip_from_idxs(true, i, j)?;
                }
            }
        }

        pb.finish();
        optima_print(&format!("{} samples.", count), PrintMode::Println, PrintColor::None, false);

        self.robot_geometric_shape_collections.push(robot_geometric_shape_collection);

        Ok(())
    }
    pub fn get_robot_geometric_shape_collection(&self, shape_representation: &RobotLinkShapeRepresentation) -> Result<&RobotGeometricShapeCollection, OptimaError> {
        for s in &self.robot_geometric_shape_collections {
            if &s.robot_link_shape_representation == shape_representation { return Ok(s) }
        }
        Err(OptimaError::UnreachableCode)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotGeometricShapeCollection {
    robot_link_shape_representation: RobotLinkShapeRepresentation,
    geometric_shape_collection: GeometricShapeCollection,
    robot_link_idx_to_shape_idxs_mapping: Vec<Vec<usize>>
}
impl RobotGeometricShapeCollection {
    pub fn new(num_robot_links: usize, robot_link_shape_representation: RobotLinkShapeRepresentation, geometric_shape_collection: GeometricShapeCollection) -> Result<Self, OptimaError> {
        let mut robot_link_idx_to_shape_idxs_mapping = vec![];

        for _ in 0..num_robot_links { robot_link_idx_to_shape_idxs_mapping.push(vec![]); }

        let shapes = geometric_shape_collection.shapes();
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
            geometric_shape_collection,
            robot_link_idx_to_shape_idxs_mapping
        })
    }
    pub fn get_shape_idxs_from_link_idx(&self, link_idx: usize) -> Result<&Vec<usize>, OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(link_idx, self.robot_link_idx_to_shape_idxs_mapping.len(), file!(), line!())?;
        return Ok(&self.robot_link_idx_to_shape_idxs_mapping[link_idx]);
    }
    pub fn robot_link_shape_representation(&self) -> &RobotLinkShapeRepresentation {
        &self.robot_link_shape_representation
    }
    pub fn geometric_shape_collection(&self) -> &GeometricShapeCollection {
        &self.geometric_shape_collection
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Serialize, Deserialize)]
pub enum RobotLinkShapeRepresentation {
    Cubes,
    ConvexShapes,
    SphereSubcomponents,
    CubeSubcomponents,
    ConvexShapeSubcomponents,
    TriangleMeshes
}
impl RobotLinkShapeRepresentation {
    pub fn get_link_idx_to_shape_idxs(&self) {
        // TODO
    }
    pub fn get_geometric_shapes(&self, robot_mesh_file_manager_module: &RobotMeshFileManagerModule) -> Result<Vec<Option<GeometricShape>>, OptimaError> {
        let mut out_vec = vec![];

        match self {
            RobotLinkShapeRepresentation::Cubes => {
                let paths = robot_mesh_file_manager_module.get_paths_to_meshes()?;
                for (link_idx, path) in paths.iter().enumerate() {
                    match path {
                        None => { out_vec.push(None); }
                        Some(path) => {
                            let trimesh_engine = path.load_stl_to_trimesh_engine()?;
                            let base_shape = GeometricShape::new_triangle_mesh(&trimesh_engine, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: 0 });
                            let cube_shape = base_shape.to_best_fit_cube();
                            out_vec.push(Some(cube_shape));
                        }
                    }
                }
            }
            RobotLinkShapeRepresentation::ConvexShapes => {
                let paths = robot_mesh_file_manager_module.get_paths_to_convex_shape_meshes()?;
                for (link_idx, path) in paths.iter().enumerate() {
                    match path {
                        None => { out_vec.push(None); }
                        Some(path) => {
                            let trimesh_engine = path.load_stl_to_trimesh_engine()?;
                            let base_shape = GeometricShape::new_convex_shape(&trimesh_engine, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: 0 });
                            out_vec.push(Some(base_shape));
                        }
                    }
                }
            }
            RobotLinkShapeRepresentation::SphereSubcomponents => {
                let paths = robot_mesh_file_manager_module.get_paths_to_convex_shape_subcomponent_meshes()?;
                for (link_idx, v) in paths.iter().enumerate() {
                    if v.len() == 0 { out_vec.push(None); }
                    for (shape_idx_in_link, path) in v.iter().enumerate() {
                        let trimesh_engine = path.load_stl_to_trimesh_engine()?;
                        let base_shape = GeometricShape::new_convex_shape(&trimesh_engine, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link });
                        let sphere_shape = base_shape.to_best_fit_sphere();
                        out_vec.push(Some(sphere_shape));
                    }
                }
            }
            RobotLinkShapeRepresentation::CubeSubcomponents => {
                let paths = robot_mesh_file_manager_module.get_paths_to_convex_shape_subcomponent_meshes()?;
                for (link_idx, v) in paths.iter().enumerate() {
                    if v.len() == 0 { out_vec.push(None); }
                    for (shape_idx_in_link, path) in v.iter().enumerate() {
                        let trimesh_engine = path.load_stl_to_trimesh_engine()?;
                        let base_shape = GeometricShape::new_convex_shape(&trimesh_engine, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link });
                        let cube_shape = base_shape.to_best_fit_cube();
                        out_vec.push(Some(cube_shape));
                    }
                }
            }
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents => {
                let paths = robot_mesh_file_manager_module.get_paths_to_convex_shape_subcomponent_meshes()?;
                for (link_idx, v) in paths.iter().enumerate() {
                    if v.len() == 0 { out_vec.push(None); }
                    for (shape_idx_in_link, path) in v.iter().enumerate() {
                        let trimesh_engine = path.load_stl_to_trimesh_engine()?;
                        let base_shape = GeometricShape::new_convex_shape(&trimesh_engine, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link });
                        out_vec.push(Some(base_shape));
                    }
                }
            }
            RobotLinkShapeRepresentation::TriangleMeshes => {
                let paths = robot_mesh_file_manager_module.get_paths_to_convex_shape_meshes()?;
                for (link_idx, path) in paths.iter().enumerate() {
                    match path {
                        None => { out_vec.push(None); }
                        Some(path) => {
                            let trimesh_engine = path.load_stl_to_trimesh_engine()?;
                            let base_shape = GeometricShape::new_triangle_mesh(&trimesh_engine, GeometricShapeSignature::RobotLink { link_idx, shape_idx_in_link: 0 });
                            out_vec.push(Some(base_shape));
                        }
                    }
                }
            }
        }

        Ok(out_vec)
    }
    fn stop_at_min_sample_duration(&self) -> Duration {
        match self {
            RobotLinkShapeRepresentation::Cubes => { Duration::from_secs(20) }
            RobotLinkShapeRepresentation::ConvexShapes => { Duration::from_secs(30) }
            RobotLinkShapeRepresentation::SphereSubcomponents => { Duration::from_secs(30) }
            RobotLinkShapeRepresentation::CubeSubcomponents => { Duration::from_secs(30) }
            RobotLinkShapeRepresentation::ConvexShapeSubcomponents => { Duration::from_secs(60) }
            RobotLinkShapeRepresentation::TriangleMeshes => { Duration::from_secs(120) }
        }
    }
}

/// private implementation that is just accessible by `RobotShapeGeometryModule`.
impl RobotFKResult {
    pub fn to_geometric_shape_collection_input_poses(&self, robot_geometric_shape_collection: &RobotGeometricShapeCollection) -> Result<GeometricShapeCollectionInputPoses, OptimaError> {
        let mut geometric_shape_collection_input_poses = GeometricShapeCollectionInputPoses::new(&robot_geometric_shape_collection.geometric_shape_collection);
        let link_entries = self.link_entries();
        for (link_idx, link_entry) in link_entries.iter().enumerate() {
            let pose = link_entry.pose();
            if let Some(pose) = pose {
                let shape_idxs = robot_geometric_shape_collection.get_shape_idxs_from_link_idx(link_idx)?;
                for shape_idx in shape_idxs {
                    geometric_shape_collection_input_poses.insert_or_replace_pose_by_idx(*shape_idx, pose.clone())?;
                }
            }
        }

        Ok(geometric_shape_collection_input_poses)
    }
}