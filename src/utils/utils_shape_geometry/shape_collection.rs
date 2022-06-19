use nalgebra::{UnitQuaternion, Vector3};
use parry3d_f64::query::{Ray};
use serde::{Serialize, Deserialize};
use instant::{Duration, Instant};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string};
use crate::utils::utils_generic_data_structures::{Array1D, MemoryCell, Mixable, SquareArray2D};
use crate::utils::utils_sampling::SimpleSamplers;
use crate::utils::utils_se3::optima_rotation::OptimaRotation;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeQueries, GeometricShapeQueryGroupOutput, GeometricShapeQuery, GeometricShapeSignature, LogCondition, StopCondition, ContactWrapper};
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
    sorted_signatures_with_shape_idxs: Vec<(GeometricShapeSignature, usize)>,
    /// The id will be updated each time a geometric shape is added.  This will help track whether 
    /// mutable objects given out by the shape collection (intended to be updated throughout runtime)
    /// are still valid.
    id: f64 
}
impl ShapeCollection {
    pub fn new_empty() -> Self {
        Self {
            shapes: vec![],
            skips: SquareArray2D::new(0, true, None),
            average_distances: SquareArray2D::new(0, true, None),
            sorted_signatures_with_shape_idxs: vec![],
            id: SimpleSamplers::uniform_sample((-1.0, 1.0))
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
        self.id = SimpleSamplers::uniform_sample((-1.0, 1.0));
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
    pub fn skips_mut(&mut self) -> &mut SquareArray2D<MemoryCell<bool>> {
        &mut self.skips
    }
    pub fn average_distances_mut(&mut self) -> &mut SquareArray2D<MemoryCell<f64>> {
        &mut self.average_distances
    }

    pub fn set_base_skip_from_idxs(&mut self, skip: bool, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        if idx1 == idx2 {
            return self.skips.adjust_data(|x| x.replace_base_value(true), idx1, idx2 )
        }
        self.skips.adjust_data(|x| x.replace_base_value(skip), idx1, idx2 )
    }
    pub fn replace_skip_from_idxs(&mut self, skip: bool, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.skips.adjust_data(|x| x.replace_value(skip, false), idx1, idx2)
    }
    pub fn reset_skip_to_base_from_idxs(&mut self, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.skips.adjust_data(|x| x.reset_to_base_value(false), idx1, idx2 )
    }

    pub fn set_base_average_distance_from_idxs(&mut self, dis: f64, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.average_distances.adjust_data(|x| x.replace_base_value(dis), idx1, idx2 )
    }
    pub fn replace_average_distance_from_idxs(&mut self, dis: f64, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.average_distances.adjust_data(|x| x.replace_value(dis, false), idx1, idx2 )
    }
    pub fn reset_average_distance_to_base_from_idxs(&mut self, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.average_distances.adjust_data(|x| x.reset_to_base_value(false), idx1, idx2 )
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
            _ => { panic!("Should not be reached on inputs that are not composed of smaller, atomic queries.  \
            If you believe that your input query should be composed of an input vec of GeometricShapeQuery objects,\
            add it as an arm in the match statement above. ") }
        }
    }

    pub fn spawn_query_list(&self) -> ShapeCollectionQueryList {
        return ShapeCollectionQueryList { list: vec![], id: self.id };
    }
    pub fn spawn_query_pairs_list(&self, override_all_skips: bool) -> ShapeCollectionQueryPairsList {
        return ShapeCollectionQueryPairsList { pairs: vec![], override_all_skips, id: self.id };
    }
    pub fn spawn_proxima_engine(&self) -> ProximaEngine {
        let num_shapes = self.shapes.len();

        let mut grid: SquareArray2D<Option<ProximaPairwiseBlock>> = SquareArray2D::new(num_shapes, false, Some(None));
        for i in 0..num_shapes {
            for j in 0..num_shapes {
                if i < j {
                    let proxima_pairwise_block = ProximaPairwiseBlock {
                        object_1_signature: self.shapes[i].signature().clone(),
                        object_2_signature: self.shapes[j].signature().clone(),
                        ..Default::default()
                    };
                    grid.replace_data(Some(proxima_pairwise_block), i, j).expect("error");
                }
            }
        }

        ProximaEngine {
            grid,
            id: self.id
        }
    }

    /// This is the workhorse function of this struct.  It does lots of kinds of geometric shape queries
    /// over collections of shapes.
    pub fn shape_collection_query<'a>(&'a self,
                                      input: &'a ShapeCollectionQuery,
                                      stop_condition: StopCondition,
                                      log_condition: LogCondition,
                                      sort_outputs: bool) -> Result<ShapeCollectionQueryOutput, OptimaError> {
        return match input {
            _ => {
                // This is the "standard loop" for shape collection queries.  They all involve the
                // same strategy: loop through the given shape queries, possibly stopping early if
                // a stop condition is met, and return the answer.  Any ShapeCollectionQuery that
                // does not meet this strategy should form a different arm of the match statement.
                let input_vec = self.get_geometric_shape_query_input_vec(input)?;
                let g = GeometricShapeQueries::generic_group_query(input_vec, stop_condition, log_condition, sort_outputs);
                Ok(ShapeCollectionQueryOutput::GeometricShapeQueryGroupOutput(g))
            }
        }
    }

    pub fn proxima_query(&self,
                     poses: &ShapeCollectionInputPoses,
                     proxima_engine: &mut ProximaEngine,
                     d_max: f64,
                     a_max: f64,
                     loss_function: SignedDistanceLossFunction,
                     r: f64,
                     proxima_budget: ProximaBudget,
                     inclusion_list: &Option<&ShapeCollectionQueryPairsList>) -> Result<ProximaOutput, OptimaError> {
        assert_eq!(self.id, proxima_engine.id);
        assert!(0.0 <= r && r <= 1.0);

        let start = instant::Instant::now();

        let mut output = ProximaOutput {
            output_sum: 0.0,
            maximum_possible_error: 0.0,
            ground_truth_check_signatures: vec![],
            duration: Default::default()
        };

        let grid = proxima_engine.grid_mut_ref();
        let poses = &poses.poses;

        let mut maximum_possible_error = 0.0;
        let mut output_sum = 0.0;
        let mut proxima_outputs = vec![];

        if let Some(inclusion_list) = inclusion_list {
            for pair in &inclusion_list.pairs {
                let i = pair.0;
                let j = pair.1;
                if let Some(pose1) = &poses[i] {
                    if let Some(pose2) = &poses[j] {
                        let skip = self.skips.data_cell(i, j)?.curr_value();
                        if !*skip {
                            let data_cell_mut = grid.data_cell_mut(i, j)?;
                            if let Some(data_cell_mut) = data_cell_mut {
                                let shape1 = &self.shapes[i];
                                let shape2 = &self.shapes[j];
                                let shape_average_distance = self.average_distances.data_cell(i, j)?.curr_value();
                                let proxima_result = ProximaFunctions::proxima_single_comparison(data_cell_mut, shape1, shape2, *shape_average_distance, i, j, pose1, pose2, d_max, a_max, r, &loss_function, &mut output)?;
                                if let Some(proxima_result) = proxima_result {
                                    maximum_possible_error += proxima_result.max_possible_error;
                                    output_sum += proxima_result.d_c;
                                    proxima_outputs.push(proxima_result);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            let num_shapes = self.shapes.len();
            for i in 0..num_shapes {
                for j in 0..num_shapes {
                    if i < j {
                        if let Some(pose1) = &poses[i] {
                            if let Some(pose2) = &poses[j] {
                                let skip = self.skips.data_cell(i, j)?.curr_value();
                                if !*skip {
                                    let data_cell_mut = grid.data_cell_mut(i, j)?;
                                    if let Some(data_cell_mut) = data_cell_mut {
                                        let shape1 = &self.shapes[i];
                                        let shape2 = &self.shapes[j];
                                        let shape_average_distance = self.average_distances.data_cell(i, j)?.curr_value();
                                        let proxima_result = ProximaFunctions::proxima_single_comparison(data_cell_mut, shape1, shape2, *shape_average_distance, i, j, pose1, pose2, d_max, a_max, r, &loss_function, &mut output)?;
                                        if let Some(proxima_result) = proxima_result {
                                            maximum_possible_error += proxima_result.max_possible_error;
                                            output_sum += proxima_result.d_c;
                                            proxima_outputs.push(proxima_result);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        proxima_outputs.sort_by(|x, y| y.max_possible_error.partial_cmp(&x.max_possible_error).unwrap());

        'f: for proxima_output in &proxima_outputs {
            match proxima_budget {
                ProximaBudget::Accuracy(budget) => {
                    if maximum_possible_error < budget { break 'f; }
                }
                ProximaBudget::Time(budget) => {
                    let duration = start.elapsed();
                    if duration > budget { break 'f; }
                }
            }

            let shape_idx1 = proxima_output.shape_idxs.0;
            let shape_idx2 = proxima_output.shape_idxs.1;
            let shape1 = &self.shapes[shape_idx1];
            let shape2 = &self.shapes[shape_idx2];
            let pose1 = poses[shape_idx1].as_ref().unwrap();
            let pose2 = poses[shape_idx2].as_ref().unwrap();
            let shape_average_distance = self.average_distances.data_cell(shape_idx1, shape_idx2)?.curr_value();

            let data_cell_mut = grid.data_cell_mut(shape_idx1, shape_idx2)?.as_mut().unwrap();

            ProximaFunctions::proxima_ground_truth_check_and_update_block(data_cell_mut, shape1, pose1, shape2, pose2, &mut output)?;
            let d_c = ProximaFunctions::proxima_loss_with_cutoff(data_cell_mut.contact_j.dist, d_max, a_max, *shape_average_distance, &loss_function);

            maximum_possible_error -= proxima_output.max_possible_error;
            output_sum -= proxima_output.d_c;
            output_sum += d_c;
        }

        output.output_sum = output_sum;
        output.maximum_possible_error = maximum_possible_error;
        output.duration = start.elapsed();

        Ok(output)
    }

    fn get_single_object_geometric_shape_query_input_vec<'a>(&'a self, input: &'a ShapeCollectionQuery) -> Result<Vec<GeometricShapeQuery<'a>>, OptimaError> {
        let mut out_vec = vec![];

        let poses = &input.get_shape_collection_input_poses()?[0].poses;
        let inclusion_list = input.get_inclusion_list();
        if let Some(inclusion_list) = inclusion_list {
            assert_eq!(inclusion_list.id, self.id, "id must match ShapeCollection.");
            let list = &inclusion_list.list;

            for i in list {
                let pose = &poses[*i];
                if let Some(pose) = pose {
                    match input {
                        ShapeCollectionQuery::ProjectPoint { poses: _, point, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::ProjectPoint {
                                object: &self.shapes[*i],
                                pose: pose.clone(),
                                point,
                                solid: *solid
                            });
                        }
                        ShapeCollectionQuery::ContainsPoint { poses: _, point, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::ContainsPoint {
                                object: &self.shapes[*i],
                                pose: pose.clone(),
                                point
                            });
                        }
                        ShapeCollectionQuery::DistanceToPoint { poses: _, point, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::DistanceToPoint {
                                object: &self.shapes[*i],
                                pose: pose.clone(),
                                point,
                                solid: *solid
                            });
                        }
                        ShapeCollectionQuery::IntersectsRay { poses: _, ray, max_toi, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::IntersectsRay {
                                object: &self.shapes[*i],
                                pose: pose.clone(),
                                ray,
                                max_toi: *max_toi
                            });
                        }
                        ShapeCollectionQuery::CastRay { poses: _, ray, max_toi, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::CastRay {
                                object: &self.shapes[*i],
                                pose: pose.clone(),
                                ray,
                                max_toi: *max_toi,
                                solid: *solid
                            });
                        }
                        ShapeCollectionQuery::CastRayAndGetNormal { poses: _, ray, max_toi, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::CastRayAndGetNormal {
                                object: &self.shapes[*i],
                                pose: pose.clone(),
                                ray,
                                max_toi: *max_toi,
                                solid: *solid
                            });
                        }
                        _ => { unreachable!() }
                    }
                }
            }

            return Ok(out_vec);
        } else {
            for (i, s) in self.shapes.iter().enumerate() {
                let pose = &poses[i];
                if let Some(pose) = pose {
                    match input {
                        ShapeCollectionQuery::ProjectPoint { poses: _, point, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::ProjectPoint {
                                object: s,
                                pose: pose.clone(),
                                point,
                                solid: *solid
                            });
                        }
                        ShapeCollectionQuery::ContainsPoint { poses: _, point, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::ContainsPoint {
                                object: s,
                                pose: pose.clone(),
                                point
                            });
                        }
                        ShapeCollectionQuery::DistanceToPoint { poses: _, point, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::DistanceToPoint {
                                object: s,
                                pose: pose.clone(),
                                point,
                                solid: *solid
                            });
                        }
                        ShapeCollectionQuery::IntersectsRay { poses: _, ray, max_toi, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::IntersectsRay {
                                object: s,
                                pose: pose.clone(),
                                ray,
                                max_toi: *max_toi
                            });
                        }
                        ShapeCollectionQuery::CastRay { poses: _, ray, max_toi, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::CastRay {
                                object: s,
                                pose: pose.clone(),
                                ray,
                                max_toi: *max_toi,
                                solid: *solid
                            });
                        }
                        ShapeCollectionQuery::CastRayAndGetNormal { poses: _, ray, max_toi, solid, inclusion_list: _ } => {
                            out_vec.push(GeometricShapeQuery::CastRayAndGetNormal {
                                object: s,
                                pose: pose.clone(),
                                ray,
                                max_toi: *max_toi,
                                solid: *solid
                            });
                        }
                        _ => { unreachable!() }
                    }
                }
            }

            return Ok(out_vec);
        }
    }
    fn get_pairwise_objects_geometric_shape_query_input_vec<'a>(&'a self, input: &'a ShapeCollectionQuery) -> Result<Vec<GeometricShapeQuery<'a>>, OptimaError> {
        let mut out_vec = vec![];

        let poses = &input.get_shape_collection_input_poses()?[0].poses;
        let inclusion_list = input.get_inclusion_pairs_list();
        if let Some(inclusion_list) = inclusion_list {
            assert_eq!(inclusion_list.id, self.id, "id must ShapeCollection.");
            let list = &inclusion_list.pairs;

            for (i, j) in list {
                let pose1 = &poses[*i];
                let pose2 = &poses[*j];
                if let Some(pose1) = pose1 {
                    if let Some(pose2) = pose2 {
                        if inclusion_list.override_all_skips || !*self.skips.data_cell(*i, *j)?.curr_value() {
                            match input {
                                ShapeCollectionQuery::IntersectionTest { .. } => {
                                    out_vec.push(GeometricShapeQuery::IntersectionTest {
                                        object1: &self.shapes[*i],
                                        object1_pose: pose1.clone(),
                                        object2: &self.shapes[*j],
                                        object2_pose: pose2.clone()
                                    });
                                }
                                ShapeCollectionQuery::Distance { .. } => {
                                    out_vec.push(GeometricShapeQuery::Distance {
                                        object1: &self.shapes[*i],
                                        object1_pose: pose1.clone(),
                                        object2: &self.shapes[*j],
                                        object2_pose: pose2.clone()
                                    });
                                }
                                ShapeCollectionQuery::ClosestPoints { poses: _, max_dis, inclusion_list: _ } => {
                                    out_vec.push(GeometricShapeQuery::ClosestPoints {
                                        object1: &self.shapes[*i],
                                        object1_pose: pose1.clone(),
                                        object2: &self.shapes[*j],
                                        object2_pose: pose2.clone(),
                                        max_dis: *max_dis
                                    });
                                }
                                ShapeCollectionQuery::Contact { poses: _, prediction, inclusion_list: _ } => {
                                    out_vec.push(GeometricShapeQuery::Contact {
                                        object1: &self.shapes[*i],
                                        object1_pose: pose1.clone(),
                                        object2: &self.shapes[*j],
                                        object2_pose: pose2.clone(),
                                        prediction: *prediction
                                    });
                                }
                                ShapeCollectionQuery::CCD { poses_t1: _, poses_t2, inclusion_list: _ } => {
                                    let pose1_t2 = &poses_t2.poses[*i];
                                    let pose2_t2 = &poses_t2.poses[*j];
                                    if let Some(pose1_t2) = pose1_t2 {
                                        if let Some(pose2_t2) = pose2_t2 {
                                            out_vec.push(GeometricShapeQuery::CCD {
                                                object1: &self.shapes[*i],
                                                object1_pose_t1: pose1.clone(),
                                                object1_pose_t2: pose1_t2.clone(),
                                                object2: &self.shapes[*j],
                                                object2_pose_t1: pose2.clone(),
                                                object2_pose_t2: pose2_t2.clone()
                                            });
                                        }
                                    }
                                }
                                _ => { unreachable!() }
                            }
                        }
                    }
                }
            }

            return Ok(out_vec);
        } else {
            for (i, shape1) in self.shapes.iter().enumerate() {
            let pose1 = &poses[i];
            if let Some(pose1) = pose1 {
                for (j, shape2) in self.shapes.iter().enumerate() {
                    let pose2 = &poses[j];
                    if let Some(pose2) = pose2 {
                        if i < j {
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
                                    ShapeCollectionQuery::ClosestPoints { poses: _, max_dis, inclusion_list: _ } => {
                                        out_vec.push(GeometricShapeQuery::ClosestPoints {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone(),
                                            max_dis: *max_dis
                                        });
                                    }
                                    ShapeCollectionQuery::Contact { poses: _, prediction, inclusion_list: _ } => {
                                        out_vec.push(GeometricShapeQuery::Contact {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone(),
                                            prediction: *prediction
                                        });
                                    }
                                    ShapeCollectionQuery::CCD { poses_t1: _, poses_t2, inclusion_list: _ } => {
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
                                    _ => { unreachable!() }
                                }
                            }
                        }
                    }
                }
            }
        }
            return Ok(out_vec);
        }
    }
}
impl SaveAndLoadable for ShapeCollection {
    type SaveType = (String, String, String, Vec<(GeometricShapeSignature, usize)>);

    fn get_save_serialization_object(&self) -> Self::SaveType {
        (self.shapes.get_serialization_string(),
         self.skips.get_serialization_string(),
         self.average_distances.get_serialization_string(),
         self.sorted_signatures_with_shape_idxs.clone())
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let load: Self::SaveType = load_object_from_json_string(json_str)?;
        let shapes = Vec::load_from_json_string(&load.0)?;
        let skips = load_object_from_json_string(&load.1)?;
        let average_distances = load_object_from_json_string(&load.2)?;
        let sorted_signatures_with_shape_idxs = load.3.clone();

        Ok(Self {
            shapes,
            skips,
            average_distances,
            sorted_signatures_with_shape_idxs,
            id: SimpleSamplers::uniform_sample((-1.0,1.0))
        })
    }
}

/// An input into the important `ShapeCollection::shape_collection_query` function.
pub enum ShapeCollectionQuery<'a> {
    ProjectPoint { poses: &'a ShapeCollectionInputPoses, point: &'a Vector3<f64>, solid: bool ,inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    ContainsPoint { poses: &'a ShapeCollectionInputPoses, point: &'a Vector3<f64>, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    DistanceToPoint { poses: &'a ShapeCollectionInputPoses, point: &'a Vector3<f64>, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    IntersectsRay { poses: &'a ShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    CastRay { poses: &'a ShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    CastRayAndGetNormal { poses: &'a ShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64, solid: bool, inclusion_list: &'a Option<&'a ShapeCollectionQueryList> },
    IntersectionTest { poses: &'a ShapeCollectionInputPoses, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    Distance { poses: &'a ShapeCollectionInputPoses, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    ClosestPoints { poses: &'a ShapeCollectionInputPoses, max_dis: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    Contact { poses: &'a ShapeCollectionInputPoses, prediction: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    /// Continuous collision detection.
    CCD { poses_t1: &'a ShapeCollectionInputPoses, poses_t2: &'a ShapeCollectionInputPoses, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> },
    Proxima { poses: &'a ShapeCollectionInputPoses, prediction: f64, inclusion_list: &'a Option<&'a ShapeCollectionQueryPairsList> }
}
impl <'a> ShapeCollectionQuery<'a> {
    fn get_shape_collection_input_poses(&self) -> Result<Vec<&'a ShapeCollectionInputPoses>, OptimaError> {
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
            ShapeCollectionQuery::CCD { poses_t1, poses_t2, inclusion_list: _ } => { Ok(vec![poses_t1, poses_t2]) }
            ShapeCollectionQuery::Proxima { poses, .. } => { Ok(vec![poses]) }
        }
    }
    fn get_inclusion_list(&self) -> &Option<&'a ShapeCollectionQueryList> {
        return match self {
            ShapeCollectionQuery::ProjectPoint { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::ContainsPoint { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::DistanceToPoint { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::IntersectsRay { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::CastRay { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::CastRayAndGetNormal { inclusion_list, .. } => { inclusion_list }
            _ => { panic!("wrong type.  If you think this is a type that should have an inclusion_list, add it to the match above.") }
        }
    }
    fn get_inclusion_pairs_list(&self) -> &Option<&'a ShapeCollectionQueryPairsList> {
        return match self {
            ShapeCollectionQuery::IntersectionTest { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::Distance { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::ClosestPoints { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::Contact { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::CCD { inclusion_list, .. } => { inclusion_list }
            ShapeCollectionQuery::Proxima { inclusion_list, .. } => { inclusion_list }
            _ => { panic!("wrong type.  If you think this is a type that should have an inclusion pairs list, add it to the match above.") }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ShapeCollectionQueryOutput {
    GeometricShapeQueryGroupOutput(GeometricShapeQueryGroupOutput)
}
impl ShapeCollectionQueryOutput {
    pub fn unwrap_geometric_shape_query_group_output(&self) -> &GeometricShapeQueryGroupOutput {
        match self {
            ShapeCollectionQueryOutput::GeometricShapeQueryGroupOutput(g) => {g}
            _ => { panic!("wrong type.") }
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeCollectionQueryList {
    list: Vec<usize>,
    id: f64
}
impl ShapeCollectionQueryList {
    pub fn add_idx(&mut self, idx: usize) {
        self.list.push(idx);
    }
    pub fn add_idxs(&mut self, idxs: Vec<usize>) {
        for idx in idxs {
            self.add_idx(idx);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeCollectionQueryPairsList {
    pairs: Vec<(usize, usize)>,
    override_all_skips: bool,
    id: f64
}
impl ShapeCollectionQueryPairsList {
    pub fn add_pair(&mut self, pair: (usize, usize)) {
        self.pairs.push(pair);
    }
    pub fn add_pairs(&mut self, pairs: Vec<(usize, usize)>) {
        for pair in pairs {
            self.add_pair(pair);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProximaEngine {
    grid: SquareArray2D<Option<ProximaPairwiseBlock>>,
    id: f64
}
impl ProximaEngine {
    fn grid_mut_ref(&mut self) -> &mut SquareArray2D<Option<ProximaPairwiseBlock>> {
        &mut self.grid
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProximaSingleObjectBlock {
    object_signature: GeometricShapeSignature,
    /// Pose from last update, not the last time it was ground truth checked.
    previous_se3_pose: Option<OptimaSE3Pose>
}
impl Mixable for ProximaSingleObjectBlock {
    /// Mixing doesn't really make sense for a ProximaSingleObjectBlock, so this is just a dummy implementation
    /// so that it can be used in an `Array1D`.
    fn mix(&self, _other: &Self) -> Self {
        self.clone()
    }
}
impl Default for ProximaSingleObjectBlock {
    fn default() -> Self {
        Self {
            object_signature: GeometricShapeSignature::None,
            previous_se3_pose: None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProximaPairwiseBlock {
    initialized: bool,
    object_1_signature: GeometricShapeSignature,
    object_2_signature: GeometricShapeSignature,
    a_transform_j: OptimaSE3Pose,
    b_transform_j: OptimaSE3Pose,
    rotation_disp_j: OptimaRotation,
    translation_disp_j: f64,
    contact_j: ContactWrapper,
    f: f64
}
impl Mixable for ProximaPairwiseBlock {
    /// Mixing doesn't really make sense for a ProximaPairwiseBlock, so this is just a dummy implementation
    /// so that it can be used in a `SquareArray2D`.
    fn mix(&self, _other: &Self) -> Self {
        self.clone()
    }
}
impl Default for ProximaPairwiseBlock {
    fn default() -> Self {
        Self {
            initialized: false,
            object_1_signature: GeometricShapeSignature::None,
            object_2_signature: GeometricShapeSignature::None,
            a_transform_j: Default::default(),
            b_transform_j: Default::default(),
            rotation_disp_j: OptimaRotation::new_unit_quaternion_identity(),
            translation_disp_j: 0.0,
            contact_j: ContactWrapper::default(),
            f: 0.0
        }
    }
}

pub struct ProximaFunctions;
impl ProximaFunctions {
    pub fn proxima_single_comparison(data_cell_mut: &mut ProximaPairwiseBlock,
                                     shape1: &GeometricShape,
                                     shape2: &GeometricShape,
                                     shape_average_distance: f64,
                                     shape1_idx: usize,
                                     shape2_idx: usize,
                                     pose1: &OptimaSE3Pose,
                                     pose2: &OptimaSE3Pose,
                                     d_max: f64,
                                     a_max: f64,
                                     r: f64,
                                     loss_function: &SignedDistanceLossFunction,
                                     output: &mut ProximaOutput) -> Result<Option<ProximaSingleComparisonOutput>, OptimaError> {
        return if data_cell_mut.initialized {
            let bounds = Self::proxima_compute_bounds(data_cell_mut, shape1_idx, shape2_idx, shape_average_distance, pose1, pose2, d_max, a_max)?;
            match bounds {
                ProximaSignedDistanceBoundsResult::Completed { lower_bound, upper_bound } => {
                    let d_hat = (1.0 - r) * lower_bound + r * upper_bound;
                    let l_c = Self::proxima_loss_with_cutoff(lower_bound, d_max, a_max, shape_average_distance, loss_function);
                    let u_c = Self::proxima_loss_with_cutoff(upper_bound, d_max, a_max, shape_average_distance, loss_function);
                    let d_c = Self::proxima_loss_with_cutoff(d_hat, d_max, a_max, shape_average_distance, loss_function);

                    let max_possible_error = (l_c - d_c).max(d_c - u_c);

                    Ok(Some(ProximaSingleComparisonOutput {
                        max_possible_error,
                        d_c,
                        shape_idxs: (shape1_idx, shape2_idx)
                    }))
                }
                _ => { Ok(None) }
            }
        } else {
            Self::proxima_ground_truth_check_and_update_block(data_cell_mut, shape1, pose1, shape2, pose2, output)?;
            let d_c = Self::proxima_loss_with_cutoff(data_cell_mut.contact_j.dist, d_max, a_max, shape_average_distance, loss_function);

            Ok(Some(ProximaSingleComparisonOutput {
                max_possible_error: 0.0,
                d_c,
                shape_idxs: (shape1_idx, shape2_idx)
            }))
        }
    }
    pub fn proxima_compute_bounds(data_cell_mut: &mut ProximaPairwiseBlock,
                                  _shape1_idx: usize,
                                  _shape2_idx: usize,
                                  shape_average_distance: f64,
                                  pose1: &OptimaSE3Pose,
                                  pose2: &OptimaSE3Pose,
                                  d_max: f64,
                                  a_max: f64,) -> Result<ProximaSignedDistanceBoundsResult, OptimaError> {
        let a_translation_k = pose1.translation();
        let b_translation_k = pose2.translation();
        let translation_disp_k = (a_translation_k - b_translation_k).norm();
        let delta_m = (data_cell_mut.translation_disp_j - translation_disp_k);

        let a_rotation_k = pose1.rotation();
        let b_rotation_k = pose2.rotation();
        let rotation_disp_k = a_rotation_k.displacement(&b_rotation_k, false)?;
        let delta_r = data_cell_mut.rotation_disp_j.displacement(&rotation_disp_k, false)?.angle();

        let l = data_cell_mut.contact_j.dist - delta_m - Self::proxima_phi(data_cell_mut.f, delta_r);
        let l_wrt_average = l / shape_average_distance;

        if l > d_max { return Ok(ProximaSignedDistanceBoundsResult::Pruned); }
        if l_wrt_average > a_max { return Ok(ProximaSignedDistanceBoundsResult::Pruned);  }

        let a_transform_j = &data_cell_mut.a_transform_j;
        let b_transform_j = &data_cell_mut.b_transform_j;

        let a_rotation_j = a_transform_j.rotation();
        let b_rotation_j = b_transform_j.rotation();

        let a_translation_j = a_transform_j.translation();
        let b_translation_j = b_transform_j.translation();

        let a_c_j = &data_cell_mut.contact_j.point1;
        let b_c_j = &data_cell_mut.contact_j.point2;

        let a_c_k = a_rotation_k.multiply_by_point(&(&a_rotation_j.inverse().multiply_by_point(&(a_c_j - a_translation_j)) + a_translation_k));
        let b_c_k = b_rotation_k.multiply_by_point(&(&b_rotation_j.inverse().multiply_by_point(&(b_c_j - b_translation_j)) + b_translation_k));

        let u = (&a_c_k - &b_c_k).norm();

        return Ok(ProximaSignedDistanceBoundsResult::Completed { lower_bound: l, upper_bound: u });
    }
    pub fn proxima_phi(h: f64, theta: f64) -> f64 {
        (2.0*h*h*(1.0 - theta.cos())).sqrt()
    }
    pub fn proxima_loss_with_cutoff(d_hat: f64,
                                    d_max: f64,
                                    a_max: f64,
                                    shape_average_distance: f64,
                                    loss_function: &SignedDistanceLossFunction) -> f64 {
        return if d_hat >= d_max || (d_hat / shape_average_distance) >= a_max { 0.0 } else { loss_function.loss(d_hat / shape_average_distance, a_max) }
    }
    pub fn proxima_ground_truth_check_and_update_block(data_cell_mut: &mut ProximaPairwiseBlock,
                                                       shape1: &GeometricShape,
                                                       pose1: &OptimaSE3Pose,
                                                       shape2: &GeometricShape,
                                                       pose2: &OptimaSE3Pose,
                                                       output: &mut ProximaOutput) -> Result<(), OptimaError> {
        let contact_wrapper = GeometricShapeQueries::contact(shape1, pose1, shape2, pose2, f64::INFINITY).unwrap();

        let a_translation_k = pose1.translation();
        let b_translation_k = pose2.translation();
        let translation_disp = (a_translation_k - b_translation_k).norm();

        let a_rotation_k = pose1.rotation();
        let b_rotation_k = pose2.rotation();
        let rotation_disp = a_rotation_k.displacement(&b_rotation_k, false)?;

        data_cell_mut.contact_j = contact_wrapper;
        data_cell_mut.a_transform_j = pose1.clone();
        data_cell_mut.b_transform_j = pose2.clone();
        data_cell_mut.translation_disp_j = translation_disp;
        data_cell_mut.rotation_disp_j = rotation_disp;
        data_cell_mut.initialized = true;

        output.ground_truth_check_signatures.push( (shape1.signature().clone(), shape2.signature().clone()) );

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum ProximaSignedDistanceBoundsResult {
    Pruned,
    Completed { lower_bound: f64, upper_bound: f64 }
}

#[derive(Clone, Debug)]
pub struct ProximaSingleComparisonOutput {
    max_possible_error: f64,
    d_c: f64,
    shape_idxs: (usize, usize)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProximaOutput {
    output_sum: f64,
    maximum_possible_error: f64,
    ground_truth_check_signatures: Vec<(GeometricShapeSignature, GeometricShapeSignature)>,
    duration: Duration
}

#[derive(Clone, Debug)]
pub enum ProximaBudget {
    Accuracy(f64),
    Time(Duration)
}

#[derive(Clone, Debug)]
pub enum SignedDistanceLossFunction {
    GaussianHinge,
    Hinge
}
impl SignedDistanceLossFunction {
    pub fn loss(&self, x: f64, cutoff: f64) -> f64 {
        return match self {
            SignedDistanceLossFunction::GaussianHinge => {
                if x > 0.0 { (-x.powi(2) / (0.2 * cutoff * cutoff)).exp() } else { -x + 1.0 }
            }
            SignedDistanceLossFunction::Hinge => {
                if x > cutoff { 0.0 } else { -(1.0 / cutoff) * (x - cutoff) }
            }
        }
    }
}

