use serde::{Serialize, Deserialize};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_file_manager_module::RobotMeshFileManagerModule;
use crate::robot_modules::robot_fk_module::RobotFKModule;
use crate::robot_modules::robot_joint_state_module::RobotJointStateModule;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_generic_data_structures::SquareArray2D;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeSignature};
use crate::utils::utils_shape_geometry::geometric_shape_collection::GeometricShapeCollection;

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
    fn new_not_loaded(robot_configuration_module: &RobotConfigurationModule) -> Result<Self, OptimaError> {
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
        ];

        for r in &representations {
            out_self.setup_robot_geometric_shape_collection(&robot_name, r)?;
        }

        Ok(out_self)
    }
    fn new_try_loaded(_robot_configuration_module: &RobotConfigurationModule) -> Result<Self, OptimaError> {
        todo!()
    }
    fn setup_robot_geometric_shape_collection(&mut self,
                                              robot_name: &str,
                                              robot_link_shape_representation: &RobotLinkShapeRepresentation) -> Result<(), OptimaError> {
        let mut geometric_shape_collection = GeometricShapeCollection::new_empty();
        let geometric_shapes = robot_link_shape_representation.get_geometric_shapes(&self.robot_mesh_file_manager_module)?;
        for geometric_shape in geometric_shapes {
            if let Some(geometric_shape) = geometric_shape {
                geometric_shape_collection.add_geometric_shape(geometric_shape.clone());
            }
        }

        let num_shapes = geometric_shape_collection.shapes().len();

        let mut output_skips_array = SquareArray2D::<bool>::new(num_shapes, true, None);
        let mut distance_aggregation_array = SquareArray2D::<f64>::new(num_shapes, true, Some(0.0));
        let mut collision_counter_array = SquareArray2D::<f64>::new(num_shapes, true, None);

        let base_robot_fk_module = RobotFKModule::new_from_names(robot_name, None)?;
        let base_robot_joint_state_module = RobotJointStateModule::new_from_names(robot_name, None)?;

        // TODO: keep going here!!!

        let robot_geometric_shape_collection = RobotGeometricShapeCollection {
            robot_link_shape_representation: robot_link_shape_representation.clone(),
            geometric_shape_collection
        };

        self.robot_geometric_shape_collections.push(robot_geometric_shape_collection);

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotGeometricShapeCollection {
    robot_link_shape_representation: RobotLinkShapeRepresentation,
    geometric_shape_collection: GeometricShapeCollection
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
}