use nalgebra::{Vector3};
use parry3d_f64::query::Ray;
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string};
use crate::utils::utils_generic_data_structures::{MemoryCell, SquareArray2D};
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeQueries, GeometricShapeQueryGroupOutput, GeometricShapeQuery, GeometricShapeSignature, LogCondition, StopCondition};
use crate::utils::utils_traits::{SaveAndLoadable};

/// A collection of `GeometricShape` objects.  Contains the vector of shapes as well as information
/// on the relationship between shapes.  The most important function in this struct is
/// `shape_collection_query`.  This function takes in a `ShapeCollectionQuery` input, resolves
/// all poses of the geometric shapes in the scenes, and automatically invokes the
/// `GeometricShapeQueries::generic_group_query` function with the correct, corresponding inputs.
///
/// The `skips` field is a two dimensional square array that specifies whether a particular pair of shapes
/// should be skipped in a pairwise geometry query (e.g., intersection checking, distance checking, etc).
/// Also, the `average_distances` field is a two dimensional square array that allows for thte saving and recall of
/// precomputed average distances between pairs of shapes (will be 1.0 by default for all shapes
/// until changed).  A `ShapeCollection` allows for dynamic adding of shapes as well.
///
/// The ordering of shapes in the `shapes` field is important; the index that a particular shape is
/// at in this list correspond to its "shape index".  For example, shapes\[0\] would have a "shape index"
/// of 0, shapes\[1\] would have a "shape index" of 1, etc.  These shape indices also correspond to all
/// `SquareArray2D` fields in this object.  For example, the skips.data_cell(3, 6) would access whether
/// any pairwise geometric shape query by GeometricShapeQueries::generic_group_query should skip the
/// computation between shape with index 3 and shape with index 6.  Use the `get_shape_idx_from_signature`
/// function to map a signature to a shape index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeCollection {
    shapes: Vec<GeometricShape>,
    skips: SquareArray2D<MemoryCell<bool>>,
    average_distances: SquareArray2D<MemoryCell<f64>>,
    sorted_signatures_with_shape_idxs: Vec<(GeometricShapeSignature, usize)>
}
impl ShapeCollection {
    pub fn new_empty() -> Self {
        Self {
            shapes: vec![],
            skips: SquareArray2D::new(0, true, None),
            average_distances: SquareArray2D::new(0, true, None),
            sorted_signatures_with_shape_idxs: vec![]
        }
    }
    pub fn add_geometric_shape(&mut self, geometric_shape: GeometricShape) {
        let add_idx = self.shapes.len();
        let sorted_idx = self.sorted_signatures_with_shape_idxs.binary_search_by(|x| geometric_shape.signature().partial_cmp(&x.0).unwrap() );
        let sorted_idx = match sorted_idx { Ok(idx) => {idx} Err(idx) => {idx} };
        self.sorted_signatures_with_shape_idxs.insert(sorted_idx, (geometric_shape.signature().clone(), add_idx));
        self.shapes.push(geometric_shape);
        self.skips.append_new_row_and_column(Some(MemoryCell::new(false)));
        self.average_distances.append_new_row_and_column(Some(MemoryCell::new(1.0)));
    }
    pub fn set_base_skip(&mut self, skip: bool, signature1: &GeometricShapeSignature, signature2: &GeometricShapeSignature) -> Result<(), OptimaError> {
        let idx1 = self.get_shape_idx_from_signature(signature1)?;
        let idx2 = self.get_shape_idx_from_signature(signature2)?;

        self.skips.adjust_data(|x| x.replace_base_value(skip), idx1, idx2)?;

        Ok(())
    }
    pub fn replace_skip(&mut self, skip: bool, signature1: &GeometricShapeSignature, signature2: &GeometricShapeSignature) -> Result<(), OptimaError> {
        let idx1 = self.get_shape_idx_from_signature(signature1)?;
        let idx2 = self.get_shape_idx_from_signature(signature2)?;

        self.replace_skip_from_idxs(skip, idx1, idx2)
    }
    pub fn replace_skip_from_idxs(&mut self, skip: bool, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.skips.adjust_data(|x| x.replace_value(skip, false), idx1, idx2)
    }
    pub fn set_base_average_distance(&mut self, dis: f64, signature1: &GeometricShapeSignature, signature2: &GeometricShapeSignature) -> Result<(), OptimaError> {
        let idx1 = self.get_shape_idx_from_signature(signature1)?;
        let idx2 = self.get_shape_idx_from_signature(signature2)?;

        // self.average_distances.replace_data(dis, idx1, idx2)?;
        self.average_distances.adjust_data(|x| x.replace_base_value(dis), idx1, idx2 )?;

        Ok(())
    }
    pub fn replace_average_distance(&mut self, dis: f64, signature1: &GeometricShapeSignature, signature2: &GeometricShapeSignature) -> Result<(), OptimaError> {
        let idx1 = self.get_shape_idx_from_signature(signature1)?;
        let idx2 = self.get_shape_idx_from_signature(signature2)?;

        self.replace_average_distance_from_idxs(dis, idx1, idx2)
    }
    pub fn replace_average_distance_from_idxs(&mut self, dis: f64, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.average_distances.adjust_data(|x| x.replace_value(dis, false), idx1, idx2 )
    }
    pub fn shapes(&self) -> &Vec<GeometricShape> {
        &self.shapes
    }
    pub fn skips(&self) -> &SquareArray2D<MemoryCell<bool>> {
        &self.skips
    }
    pub fn average_distances(&self) -> &SquareArray2D<MemoryCell<f64>> {
        &self.average_distances
    }
    pub fn get_shape_idx_from_signature(&self, signature: &GeometricShapeSignature) -> Result<usize, OptimaError> {
        let binary_search_res = self.sorted_signatures_with_shape_idxs.binary_search_by(|x| signature.partial_cmp(&x.0).unwrap());
        return match binary_search_res {
            Ok(idx) => {
                Ok(self.sorted_signatures_with_shape_idxs[idx].1)
            }
            Err(_) => {
                Err(OptimaError::new_generic_error_str(&format!("Shape with signature {:?} not found in GeometricShapeCollection.", signature), file!(), line!()))
            }
        };
    }
    pub fn get_geometric_shape_query_input_vec<'a>(&'a self, input: &'a ShapeCollectionQuery) -> Result<Vec<GeometricShapeQuery<'a>>, OptimaError> {
        return match input {
            ShapeCollectionQuery::ProjectPoint { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::ContainsPoint { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::DistanceToPoint { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::IntersectsRay { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::CastRay { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::CastRayAndGetNormal { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::IntersectionTest { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::Distance { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::ClosestPoints { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::Contact { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            ShapeCollectionQuery::CCD { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
        }
    }
    fn get_single_object_geometric_shape_query_input_vec<'a>(&'a self, input: &'a ShapeCollectionQuery) -> Result<Vec<GeometricShapeQuery<'a>>, OptimaError> {
        let mut out_vec = vec![];

        let poses = &input.get_poses()?[0].poses;
        for (i, s) in self.shapes.iter().enumerate() {
            let pose = &poses[i];
            if let Some(pose) = pose {
                match input {
                    ShapeCollectionQuery::ProjectPoint { poses: _, point, solid } => {
                        out_vec.push(GeometricShapeQuery::ProjectPoint {
                            object: s,
                            pose: pose.clone(),
                            point,
                            solid: *solid
                        });
                    }
                    ShapeCollectionQuery::ContainsPoint { poses: _, point } => {
                        out_vec.push(GeometricShapeQuery::ContainsPoint {
                            object: s,
                            pose: pose.clone(),
                            point
                        });
                    }
                    ShapeCollectionQuery::DistanceToPoint { poses: _, point, solid } => {
                        out_vec.push(GeometricShapeQuery::DistanceToPoint {
                            object: s,
                            pose: pose.clone(),
                            point,
                            solid: *solid
                        });
                    }
                    ShapeCollectionQuery::IntersectsRay { poses: _, ray, max_toi } => {
                        out_vec.push(GeometricShapeQuery::IntersectsRay {
                            object: s,
                            pose: pose.clone(),
                            ray,
                            max_toi: *max_toi
                        });
                    }
                    ShapeCollectionQuery::CastRay { poses: _, ray, max_toi, solid } => {
                        out_vec.push(GeometricShapeQuery::CastRay {
                            object: s,
                            pose: pose.clone(),
                            ray,
                            max_toi: *max_toi,
                            solid: *solid
                        });
                    }
                    ShapeCollectionQuery::CastRayAndGetNormal { poses: _, ray, max_toi, solid } => {
                        out_vec.push(GeometricShapeQuery::CastRayAndGetNormal {
                            object: s,
                            pose: pose.clone(),
                            ray,
                            max_toi: *max_toi,
                            solid: *solid
                        });
                    }
                    _ => { return Err(OptimaError::UnreachableCode) }
                }
            }
        }
        Ok(out_vec)
    }
    fn get_pairwise_objects_geometric_shape_query_input_vec<'a>(&'a self, input: &'a ShapeCollectionQuery) -> Result<Vec<GeometricShapeQuery<'a>>, OptimaError> {
        let mut out_vec = vec![];

        let poses = &input.get_poses()?[0].poses;

        for (i, shape1) in self.shapes.iter().enumerate() {
            let pose1 = &poses[i];
            if let Some(pose1) = pose1 {
                for (j, shape2) in self.shapes.iter().enumerate() {
                    let pose2 = &poses[j];
                    if let Some(pose2) = pose2 {
                        if i <= j {
                            let skip = self.skips.data_cell(i, j)?.curr_value();
                            if !*skip {
                                match input {
                                    ShapeCollectionQuery::IntersectionTest { .. } => {
                                        out_vec.push(GeometricShapeQuery::IntersectionTest {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone()
                                        });
                                    }
                                    ShapeCollectionQuery::Distance { .. } => {
                                        out_vec.push(GeometricShapeQuery::Distance {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone()
                                        });
                                    }
                                    ShapeCollectionQuery::ClosestPoints { poses: _, max_dis } => {
                                        out_vec.push(GeometricShapeQuery::ClosestPoints {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone(),
                                            max_dis: *max_dis
                                        });
                                    }
                                    ShapeCollectionQuery::Contact { poses: _, prediction } => {
                                        out_vec.push(GeometricShapeQuery::Contact {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone(),
                                            prediction: *prediction
                                        });
                                    }
                                    ShapeCollectionQuery::CCD { poses_t1: _, poses_t2 } => {
                                        let pose1_t2 = &poses_t2.poses[i];
                                        let pose2_t2 = &poses_t2.poses[j];
                                        if let Some(pose1_t2) = pose1_t2 {
                                            if let Some(pose2_t2) = pose2_t2 {
                                                out_vec.push(GeometricShapeQuery::CCD {
                                                    object1: shape1,
                                                    object1_pose_t1: pose1.clone(),
                                                    object1_pose_t2: pose1_t2.clone(),
                                                    object2: shape2,
                                                    object2_pose_t1: pose2.clone(),
                                                    object2_pose_t2: pose2_t2.clone()
                                                });
                                            }
                                        }
                                    }
                                    _ => { return Err(OptimaError::UnreachableCode) }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(out_vec)
    }
    pub fn shape_collection_query<'a>(&'a self,
                                      input: &'a ShapeCollectionQuery,
                                      stop_condition: StopCondition,
                                      log_condition: LogCondition,
                                      sort_outputs: bool) -> Result<GeometricShapeQueryGroupOutput, OptimaError> {
        let input_vec = self.get_geometric_shape_query_input_vec(input)?;
        Ok(GeometricShapeQueries::generic_group_query(input_vec, stop_condition, log_condition, sort_outputs))
    }
    pub fn set_skips(&mut self, skips: SquareArray2D<MemoryCell<bool>>) -> Result<(), OptimaError> {
        if skips.side_length() != self.skips.side_length() {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to set skips with incorrect size matrix."), file!(), line!()));
        }
        self.skips = skips;
        Ok(())
    }
    pub fn set_average_distances(&mut self, average_distances: SquareArray2D<MemoryCell<f64>>) -> Result<(), OptimaError> {
        if average_distances.side_length() != self.average_distances.side_length() {
            return Err(OptimaError::new_generic_error_str(&format!("Tried to set average distances with incorrect size matrix."), file!(), line!()));
        }
        self.average_distances = average_distances;
        Ok(())
    }
}
impl SaveAndLoadable for ShapeCollection {
    type SaveType = (String, String, String, Vec<(GeometricShapeSignature, usize)>);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.shapes.get_serialization_string(),
         self.skips.convert_to_standard_cells().get_serialization_string(),
         self.average_distances.convert_to_standard_cells().get_serialization_string(),
         self.sorted_signatures_with_shape_idxs.clone())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let shapes = Vec::load_from_json_string(&load.0)?;
        let skips_standard: SquareArray2D<bool> = load_object_from_json_string(&load.1)?;
        let average_distances_standard: SquareArray2D<f64> = load_object_from_json_string(&load.2)?;
        let skips = skips_standard.convert_to_memory_cells();
        let average_distances = average_distances_standard.convert_to_memory_cells();
        let sorted_signatures_with_shape_idxs = load.3.clone();

        Ok(Self {
            shapes,
            skips,
            average_distances,
            sorted_signatures_with_shape_idxs
        })
    }
}

/// An input into the important `ShapeCollection::shape_collection_query` function.
pub enum ShapeCollectionQuery<'a> {
    ProjectPoint { poses: &'a ShapeCollectionInputPoses, point: &'a Vector3<f64>, solid: bool },
    ContainsPoint { poses: &'a ShapeCollectionInputPoses, point: &'a Vector3<f64> },
    DistanceToPoint { poses: &'a ShapeCollectionInputPoses, point: &'a Vector3<f64>, solid: bool },
    IntersectsRay { poses: &'a ShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64 },
    CastRay { poses: &'a ShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64, solid: bool },
    CastRayAndGetNormal { poses: &'a ShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64, solid: bool },
    IntersectionTest { poses: &'a ShapeCollectionInputPoses },
    Distance { poses: &'a ShapeCollectionInputPoses },
    ClosestPoints { poses: &'a ShapeCollectionInputPoses, max_dis: f64 },
    Contact { poses: &'a ShapeCollectionInputPoses, prediction: f64 },
    /// Continuous collision detection.
    CCD { poses_t1: &'a ShapeCollectionInputPoses, poses_t2: &'a ShapeCollectionInputPoses }
}
impl <'a> ShapeCollectionQuery<'a> {
    fn get_poses(&self) -> Result<Vec<&'a ShapeCollectionInputPoses>, OptimaError> {
        match self {
            ShapeCollectionQuery::ProjectPoint { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::ContainsPoint { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::DistanceToPoint { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::IntersectsRay { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::CastRay { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::CastRayAndGetNormal { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::IntersectionTest { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::Distance { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::ClosestPoints { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::Contact { poses, .. } => { Ok(vec![poses]) }
            ShapeCollectionQuery::CCD { poses_t1, poses_t2 } => { Ok(vec![poses_t1, poses_t2]) }
        }
    }
}

/// A convenient way to pass SE(3) pose information into a `ShapeCollectionQuery` object.  The length
/// of the `poses` field vector will be the same length as the `ShapeCollection shapes` field.  If a
/// particular pose is `None` in this list, the shape at the corresponding index in `ShapeCollection.shapes`
/// will be omitted from any computation that uses this object.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeCollectionInputPoses {
    poses: Vec<Option<OptimaSE3Pose>>
}
impl ShapeCollectionInputPoses {
    pub fn new(shape_collection: &ShapeCollection) -> Self {
        let num_shapes = shape_collection.shapes.len();
        let mut poses = vec![];
        for _ in 0..num_shapes { poses.push(None); }
        Self {
            poses
        }
    }
    pub fn insert_or_replace_pose(&mut self,
                                  signature: &GeometricShapeSignature,
                                  pose: OptimaSE3Pose,
                                  shape_collection: &ShapeCollection) -> Result<(), OptimaError> {
        let idx = shape_collection.get_shape_idx_from_signature(signature)?;

        self.poses[idx] = Some(pose);

        Ok(())
    }
    pub fn insert_or_replace_pose_by_idx(&mut self, idx: usize, pose: OptimaSE3Pose) -> Result<(), OptimaError> {
        OptimaError::new_check_for_idx_out_of_bound_error(idx, self.poses.len(), file!(), line!())?;

        self.poses[idx] = Some(pose);

        Ok(())
    }
    pub fn poses(&self) -> &Vec<Option<OptimaSE3Pose>> {
        &self.poses
    }
    /// Returns true if all poses in this object are `Some` and not `None`.
    pub fn is_full(&self) -> bool {
        for p in &self.poses {
            if p.is_none() { return false; }
        }
        return true;
    }
}

