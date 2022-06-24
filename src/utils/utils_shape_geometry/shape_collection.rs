#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use nalgebra::{Vector3};
use parry3d_f64::query::{Ray};
use serde::{Serialize, Deserialize};
use instant::{Duration};
use crate::utils::utils_combinations::comb;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string};
use crate::utils::utils_generic_data_structures::{MemoryCell, Mixable, SquareArray2D};
use crate::utils::utils_sampling::SimpleSamplers;
use crate::utils::utils_se3::optima_rotation::OptimaRotation;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeQueries, GeometricShapeQueryGroupOutput, GeometricShapeQuery, GeometricShapeSignature, LogCondition, StopCondition, ContactWrapper, BVHCombinableShape, BVHCombinableShapeAABB};
use crate::utils::utils_traits::{SaveAndLoadable, ToAndFromJsonString};

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
    pub fn spawn_bvh<T: BVHCombinableShape>(&self, poses: &ShapeCollectionInputPoses, branch_factor: usize) -> ShapeCollectionBVH<T> {
        let out_bvh = BVH::construct_new(&self.shapes, &poses, branch_factor);
        return ShapeCollectionBVH {
            bvh: out_bvh,
            id: self.id
        };
    }
    pub fn update_bvh<T: BVHCombinableShape>(&self, bvh: &mut ShapeCollectionBVH<T>, poses: &ShapeCollectionInputPoses) {
        assert_eq!(self.id, bvh.id);
        bvh.bvh_mut().update(&self.shapes, poses);
    }

    /// This is the workhorse function of this struct.  It does lots of kinds of geometric shape queries
    /// over collections of shapes.
    pub fn shape_collection_query<'a>(&'a self,
                                      input: &'a ShapeCollectionQuery,
                                      stop_condition: StopCondition,
                                      log_condition: LogCondition,
                                      sort_outputs: bool) -> Result<GeometricShapeQueryGroupOutput, OptimaError> {
        let input_vec = self.get_geometric_shape_query_input_vec(input)?;
        let g = GeometricShapeQueries::generic_group_query(input_vec, stop_condition, log_condition, sort_outputs);
        Ok(g)
    }

    pub fn proxima_proximity_query(&self,
                                   poses: &ShapeCollectionInputPoses,
                                   proxima_engine: &mut ProximaEngine,
                                   d_max: f64,
                                   a_max: f64,
                                   loss_function: SignedDistanceLossFunction,
                                   r: f64,
                                   proxima_budget: ProximaBudget,
                                   inclusion_list: &Option<&ShapeCollectionQueryPairsList>) -> Result<ProximaProximityOutput, OptimaError> {
        assert_eq!(self.id, proxima_engine.id);
        assert!(0.0 <= r && r <= 1.0);

        let start = instant::Instant::now();

        let mut output = ProximaProximityOutput {
            output_sum: 0.0,
            maximum_possible_error: 0.0,
            ground_truth_check_signatures: vec![],
            single_comparison_outputs: vec![],
            duration: Default::default(),
            query_pairs_list: self.spawn_query_pairs_list(false)
        };

        let filter_output = self.proxima_scene_filter(poses, proxima_engine, d_max, a_max, &loss_function, r, inclusion_list).expect("error");
        let mut output_sum = filter_output.output_sum;
        let mut maximum_possible_error = filter_output.maximum_possible_error;
        let mut single_comparison_outputs = filter_output.single_comparison_outputs.clone();
        for s in &filter_output.ground_truth_check_signatures {
            output.ground_truth_check_signatures.push(s.clone());
        }

        let grid = proxima_engine.grid_mut_ref();
        let poses = &poses.poses;

        'f: for single_comparison_output in &mut single_comparison_outputs {
            match proxima_budget {
                ProximaBudget::Accuracy(budget) => {
                    if maximum_possible_error < budget { break 'f; }
                }
                ProximaBudget::Time(budget) => {
                    let duration = start.elapsed();
                    if duration > budget { break 'f; }
                }
            }

            if single_comparison_output.ground_truth_check { continue; }

            let shape_idx1 = single_comparison_output.shape_idxs.0;
            let shape_idx2 = single_comparison_output.shape_idxs.1;
            let shape1 = &self.shapes[shape_idx1];
            let shape2 = &self.shapes[shape_idx2];
            let pose1 = poses[shape_idx1].as_ref().unwrap();
            let pose2 = poses[shape_idx2].as_ref().unwrap();
            let shape_average_distance = self.average_distances.data_cell(shape_idx1, shape_idx2)?.curr_value();

            let data_cell_mut = grid.data_cell_mut(shape_idx1, shape_idx2)?.as_mut().unwrap();

            ProximaFunctions::proxima_ground_truth_check_and_update_block(data_cell_mut, shape1, pose1, shape2, pose2)?;
            single_comparison_output.ground_truth_check = true;
            output.ground_truth_check_signatures.push((shape1.signature().clone(), shape2.signature().clone()));
            let d_c = ProximaFunctions::proxima_loss_with_cutoff(data_cell_mut.contact_j.dist, d_max, a_max, *shape_average_distance, &loss_function);

            maximum_possible_error -= single_comparison_output.max_possible_error;
            output_sum -= single_comparison_output.d_c;
            output_sum += d_c;
        }

        output.output_sum = output_sum;
        output.maximum_possible_error = maximum_possible_error;
        output.duration = start.elapsed();
        output.single_comparison_outputs = single_comparison_outputs;
        output.query_pairs_list = filter_output.query_pairs_list;

        Ok(output)
    }
    pub fn proxima_scene_filter(&self,
                                poses: &ShapeCollectionInputPoses,
                                proxima_engine: &mut ProximaEngine,
                                d_max: f64,
                                a_max: f64,
                                loss_function: &SignedDistanceLossFunction,
                                r: f64,
                                inclusion_list: &Option<&ShapeCollectionQueryPairsList>) -> Result<ProximaSceneFilterOutput, OptimaError> {
        let start = instant::Instant::now();

        let grid = proxima_engine.grid_mut_ref();
        let poses = &poses.poses;

        let mut filter_output = ProximaSceneFilterOutput {
            single_comparison_outputs: vec![],
            query_pairs_list: self.spawn_query_pairs_list(false),
            output_sum: 0.0,
            maximum_possible_error: 0.0,
            ground_truth_check_signatures: vec![],
            duration: Default::default()
        };

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
                                let proxima_single_comparison_output = ProximaFunctions::proxima_single_comparison(data_cell_mut, shape1, shape2, *shape_average_distance, i, j, pose1, pose2, d_max, a_max, r, &loss_function)?;
                                if let Some(p) = &proxima_single_comparison_output {
                                    filter_output.maximum_possible_error += p.max_possible_error;
                                    filter_output.output_sum += p.d_c;
                                    filter_output.single_comparison_outputs.push(p.clone());
                                    filter_output.query_pairs_list.add_pair((i,j));
                                    if p.ground_truth_check { filter_output.ground_truth_check_signatures.push((shape1.signature().clone(), shape2.signature().clone())) }
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
                                        let proxima_single_comparision_output = ProximaFunctions::proxima_single_comparison(data_cell_mut, shape1, shape2, *shape_average_distance, i, j, pose1, pose2, d_max, a_max, r, &loss_function)?;
                                        if let Some(p) = &proxima_single_comparision_output {
                                            filter_output.maximum_possible_error += p.max_possible_error;
                                            filter_output.output_sum += p.d_c;
                                            filter_output.single_comparison_outputs.push(p.clone());
                                            filter_output.query_pairs_list.add_pair((i,j));
                                            if p.ground_truth_check { filter_output.ground_truth_check_signatures.push((shape1.signature().clone(), shape2.signature().clone())) }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        filter_output.single_comparison_outputs.sort_by(|x, y| y.max_possible_error.partial_cmp(&x.max_possible_error).unwrap());
        filter_output.duration = start.elapsed();

        Ok(filter_output)
    }
    pub fn bvh_scene_filter<T: BVHCombinableShape>(&self, bvh: &mut ShapeCollectionBVH<T>, poses: &ShapeCollectionInputPoses, visit: BVHVisit) -> BVHSceneFilterOutput {
        assert_eq!(self.id, bvh.id);
        self.update_bvh(bvh, poses);
        let res = BVH::filter(&bvh.bvh, &bvh.bvh, visit, true);

        let mut pairs_list = self.spawn_query_pairs_list(false);
        pairs_list.add_pairs(res.idxs);

        return BVHSceneFilterOutput {
            pairs_list,
            num_visits: res.num_visits,
            duration: res.duration
        }
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

/*
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
*/

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
    pub fn set_override_all_skips(&mut self, b: bool) {
        self.override_all_skips = b;
    }
    pub fn pairs(&self) -> &Vec<(usize, usize)> {
        &self.pairs
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
                                     loss_function: &SignedDistanceLossFunction) -> Result<Option<ProximaSingleComparisonOutput>, OptimaError> {
        return if data_cell_mut.initialized {
            let bounds_result = Self::proxima_compute_bounds(data_cell_mut, shape_average_distance, pose1, pose2, d_max, a_max)?;
            match bounds_result {
                ProximaSignedDistanceBoundsResult::ComputedBothLowerAndUpperBound { lower_bound, upper_bound, modified_upper_bound_points } => {
                    let d_hat = (1.0 - r) * lower_bound + r * upper_bound;
                    let l_c = Self::proxima_loss_with_cutoff(lower_bound, d_max, a_max, shape_average_distance, loss_function);
                    let u_c = Self::proxima_loss_with_cutoff(upper_bound, d_max, a_max, shape_average_distance, loss_function);
                    let d_c = Self::proxima_loss_with_cutoff(d_hat, d_max, a_max, shape_average_distance, loss_function);

                    let max_possible_error = (l_c - d_c).max(d_c - u_c);

                    Ok(Some(ProximaSingleComparisonOutput {
                        max_possible_error,
                        d_c,
                        shape_idxs: (shape1_idx, shape2_idx),
                        shape_signatures: (shape1.signature().clone(), shape2.signature().clone()),
                        lower_bound_signed_distance: lower_bound,
                        upper_bound_signed_distance: upper_bound,
                        modified_upper_bound_points,
                        ground_truth_check: false
                    }))
                }
                _ => { Ok(None) }
            }
        } else {
            Self::proxima_ground_truth_check_and_update_block(data_cell_mut, shape1, pose1, shape2, pose2)?;
            let d_c = Self::proxima_loss_with_cutoff(data_cell_mut.contact_j.dist, d_max, a_max, shape_average_distance, loss_function);

            let modified_upper_bound_points = (data_cell_mut.contact_j.point1, data_cell_mut.contact_j.point2);
            
            Ok(Some(ProximaSingleComparisonOutput {
                max_possible_error: 0.0,
                d_c,
                shape_idxs: (shape1_idx, shape2_idx),
                shape_signatures: (shape1.signature().clone(), shape2.signature().clone()),
                lower_bound_signed_distance: data_cell_mut.contact_j.dist,
                upper_bound_signed_distance: data_cell_mut.contact_j.dist,
                modified_upper_bound_points,
                ground_truth_check: true
            }))
        }
    }
    pub fn proxima_compute_bounds(data_cell_mut: &mut ProximaPairwiseBlock,
                                  shape_average_distance: f64,
                                  pose1: &OptimaSE3Pose,
                                  pose2: &OptimaSE3Pose,
                                  d_max: f64,
                                  a_max: f64) -> Result<ProximaSignedDistanceBoundsResult, OptimaError> {
        let a_translation_k = pose1.translation();
        let b_translation_k = pose2.translation();
        let translation_disp_k = (a_translation_k - b_translation_k).norm();
        let delta_m = data_cell_mut.translation_disp_j - translation_disp_k;

        let a_rotation_k = pose1.rotation();
        let b_rotation_k = pose2.rotation();
        let rotation_disp_k = a_rotation_k.displacement(&b_rotation_k, false).expect("error");
        let delta_r = data_cell_mut.rotation_disp_j.displacement(&rotation_disp_k, false).expect("error").angle();

        let l = data_cell_mut.contact_j.dist - delta_m - Self::proxima_phi(data_cell_mut.f, delta_r);
        let l_wrt_average = l / shape_average_distance;

        if l > d_max { return Ok(ProximaSignedDistanceBoundsResult::PrunedAfterLowerBound { lower_bound: l }); }
        if l_wrt_average > a_max { return Ok(ProximaSignedDistanceBoundsResult::PrunedAfterLowerBound { lower_bound: l_wrt_average });  }

        let a_transform_j = &data_cell_mut.a_transform_j;
        let b_transform_j = &data_cell_mut.b_transform_j;

        let a_rotation_j = a_transform_j.rotation();
        let b_rotation_j = b_transform_j.rotation();

        let a_translation_j = a_transform_j.translation();
        let b_translation_j = b_transform_j.translation();

        let a_c_j = &data_cell_mut.contact_j.point1;
        let b_c_j = &data_cell_mut.contact_j.point2;

        let a_c_k = a_rotation_k.multiply_by_point( &(a_rotation_j.inverse().multiply_by_point(&(a_c_j - a_translation_j)))) + a_translation_k;
        let b_c_k = b_rotation_k.multiply_by_point( &(b_rotation_j.inverse().multiply_by_point(&(b_c_j - b_translation_j)))) + b_translation_k;

        let u = (&a_c_k - &b_c_k).norm();

        return Ok(ProximaSignedDistanceBoundsResult::ComputedBothLowerAndUpperBound { lower_bound: l, upper_bound: u, modified_upper_bound_points: (a_c_k, b_c_k) });
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
                                                       pose2: &OptimaSE3Pose) -> Result<(), OptimaError> {
        let contact_wrapper = GeometricShapeQueries::contact(shape1, pose1, shape2, pose2, f64::INFINITY).unwrap();

        let a_translation_k = pose1.translation();
        let b_translation_k = pose2.translation();
        let translation_disp = (a_translation_k - b_translation_k).norm();

        let a_rotation_k = pose1.rotation();
        let b_rotation_k = pose2.rotation();
        let rotation_disp = a_rotation_k.displacement(&b_rotation_k, false).expect("error");

        data_cell_mut.contact_j = contact_wrapper;
        data_cell_mut.a_transform_j = pose1.clone();
        data_cell_mut.b_transform_j = pose2.clone();
        data_cell_mut.translation_disp_j = translation_disp;
        data_cell_mut.rotation_disp_j = rotation_disp;
        data_cell_mut.initialized = true;

        Ok(())
    }
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProximaSignedDistanceBoundsResult {
    PrunedAfterLowerBound { lower_bound: f64 },
    ComputedBothLowerAndUpperBound { lower_bound: f64, upper_bound: f64, modified_upper_bound_points: (Vector3<f64>, Vector3<f64>) },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProximaSingleComparisonOutput {
    max_possible_error: f64,
    d_c: f64,
    shape_idxs: (usize, usize),
    shape_signatures: (GeometricShapeSignature, GeometricShapeSignature),
    lower_bound_signed_distance: f64,
    upper_bound_signed_distance: f64,
    modified_upper_bound_points: (Vector3<f64>, Vector3<f64>),
    ground_truth_check: bool
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct ProximaProximityOutput {
    output_sum: f64,
    maximum_possible_error: f64,
    duration: Duration,
    ground_truth_check_signatures: Vec<(GeometricShapeSignature, GeometricShapeSignature)>,
    single_comparison_outputs: Vec<ProximaSingleComparisonOutput>,
    query_pairs_list: ShapeCollectionQueryPairsList
}
impl ProximaProximityOutput {
    pub fn output_witness_points_collection(&self) -> WitnessPointsCollection {
        let mut out = WitnessPointsCollection { collection: vec![] };

        for s in &self.single_comparison_outputs {
            out.collection.push(WitnessPoints {
                witness_points: s.modified_upper_bound_points,
                shape_signatures: s.shape_signatures.clone(),
                witness_points_type: match s.ground_truth_check {
                    true => { WitnessPointsType::GroundTruth }
                    false => { WitnessPointsType::ProximaUpperBoundApproximations }
                }
            });
        }

        out
    }
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl ProximaProximityOutput {
    pub fn output_witness_points_collection_py(&self) -> WitnessPointsCollection {
        self.output_witness_points_collection()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct ProximaSceneFilterOutput {
    output_sum: f64,
    maximum_possible_error: f64,
    duration: Duration,
    ground_truth_check_signatures: Vec<(GeometricShapeSignature, GeometricShapeSignature)>,
    single_comparison_outputs: Vec<ProximaSingleComparisonOutput>,
    query_pairs_list: ShapeCollectionQueryPairsList
}
impl ProximaSceneFilterOutput {
    pub fn output_witness_points_collection(&self) -> WitnessPointsCollection {
        let mut out = WitnessPointsCollection { collection: vec![] };

        for s in &self.single_comparison_outputs {
            out.collection.push(WitnessPoints {
                witness_points: s.modified_upper_bound_points,
                shape_signatures: s.shape_signatures.clone(),
                witness_points_type: match s.ground_truth_check {
                    true => { WitnessPointsType::GroundTruth }
                    false => { WitnessPointsType::ProximaUpperBoundApproximations }
                }
            });
        }

        out
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProximaBudget {
    Accuracy(f64),
    Time(Duration)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct WitnessPointsCollection {
    collection: Vec<WitnessPoints>
}
impl WitnessPointsCollection {
    pub fn new() -> Self {
        Self {
            collection: vec![]
        }
    }
    pub fn insert(&mut self, witness_points: WitnessPoints) {
        self.collection.push(witness_points);
    }
    pub fn collection(&self) -> &Vec<WitnessPoints> {
        &self.collection
    }
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl WitnessPointsCollection {
    pub fn to_json_string_py(&self) -> String {
        self.to_json_string()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct WitnessPoints {
    witness_points: ( Vector3<f64>, Vector3<f64> ),
    shape_signatures: ( GeometricShapeSignature, GeometricShapeSignature ),
    witness_points_type: WitnessPointsType
}
impl WitnessPoints {
    pub fn new(witness_points: ( Vector3<f64>, Vector3<f64> ), shape_signatures: ( GeometricShapeSignature, GeometricShapeSignature ), witness_points_type: WitnessPointsType) -> Self {
        Self {
            witness_points,
            shape_signatures,
            witness_points_type
        }
    }
    pub fn witness_points(&self) -> (Vector3<f64>, Vector3<f64>) {
        self.witness_points
    }
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub enum WitnessPointsType {
    GroundTruth,
    ProximaUpperBoundApproximations
}

#[derive(Clone, Debug)]
pub struct BVH<T: BVHCombinableShape> {
    /// layer 0 is the leaf layer of the tree, layer len-1 is the root
    layers: Vec<Vec<BVHCombinableShapeTreeNode<T>>>
}
impl <T: BVHCombinableShape> BVH <T> {
    pub fn construct_new(shapes: &Vec<GeometricShape>, poses: &ShapeCollectionInputPoses, branch_factor: usize) -> Self {
        assert!(branch_factor > 1 && branch_factor <= 4);
        assert_eq!(shapes.len(), poses.poses.len());

        let mut base_layer = vec![];
        for (s, p) in shapes.iter().zip(&poses.poses) {
            match p {
                None => { panic!("All poses must be included as Some for BVH construction.") }
                Some(pose) => {
                    let combinable_shape = T::new_from_shape_and_pose(s, pose);
                    base_layer.push(BVHCombinableShapeTreeNode {
                        combinable_shape,
                        layer_idx: 0,
                        children_idxs_in_child_layer: vec![],
                        parent_idx_in_parent_layer: None
                    });
                }
            }
        }

        let mut out_self = Self {
            layers: vec![ base_layer ]
        };

        let mut layer_idx = 1;
        loop {
            let res = out_self.add_new_layer(branch_factor, layer_idx);
            if !res { return out_self; }
            layer_idx += 1;
        }
    }
    pub fn update(&mut self, shapes: &Vec<GeometricShape>, poses: &ShapeCollectionInputPoses) {
        assert_eq!(shapes.len(), poses.poses.len());

        let poses = &poses.poses;

        // update leaf layer
        for (i, pose) in poses.iter().enumerate() {
            match pose {
                None => { panic!("poses must all be Some in BVH.") }
                Some(pose) => {
                    self.layers[0][i].combinable_shape = T::new_from_shape_and_pose(&shapes[i], pose);
                }
            }
        }

        let num_layers = self.layers.len();

        for layer_idx in 1..num_layers {
            self.update_layer(layer_idx);
        }
    }
    /// Returns usize tuples of shape idxs from BVH a and b, respectively, that cannot be
    /// culled by the BVH and should be further inspected.
    pub fn filter(a: &Self, b: &Self, visit: BVHVisit, a_and_b_are_the_same: bool) -> BVHFilterOutput {
        let start = instant::Instant::now();

        let num_layers_a = a.layers.len();
        let num_layers_b = b.layers.len();

        assert!(num_layers_a >= 1);
        assert!(num_layers_b >= 1);

        let mut curr_layer_idx_a = num_layers_a - 1;
        let mut curr_layer_idx_b = num_layers_b - 1;

        let mut out_vec = vec![];
        let mut num_visits = 0;

        let mut queue = vec![ ((curr_layer_idx_a, 0), (curr_layer_idx_b, 0)) ];

        loop {
            let pop = queue.pop();
            if pop.is_none() {
                return BVHFilterOutput {
                    idxs: out_vec,
                    num_visits,
                    duration: start.elapsed()
                };
            }
            let pop = pop.unwrap();

            let (layer_idx_a, node_idx_a) = pop.0;
            let (layer_idx_b, node_idx_b) = pop.1;

            let node_a = &a.layers[layer_idx_a][node_idx_a];
            let node_b = &b.layers[layer_idx_b][node_idx_b];

            let cull = match &visit {
                BVHVisit::Intersection => {
                    let intersection = T::intersection_test(&node_a.combinable_shape, &node_b.combinable_shape);
                    !intersection
                }
                BVHVisit::Distance { margin } => {
                    let distance = T::distance(&node_a.combinable_shape, &node_b.combinable_shape);
                    distance > *margin
                }
            };
            num_visits += 1;

            if !cull {
                if layer_idx_a == 0 && layer_idx_b == 0 {
                    if node_idx_a < node_idx_b || !a_and_b_are_the_same {
                        out_vec.push((node_idx_a, node_idx_b));
                    }
                }
                else if layer_idx_a == 0 {
                    let children_idxs_b = &node_b.children_idxs_in_child_layer;
                    for c in children_idxs_b {
                        queue.push(( (layer_idx_a, node_idx_a), (layer_idx_b-1, *c) ) );
                    }
                }
                else if layer_idx_b == 0 {
                    let children_idxs_a = &node_a.children_idxs_in_child_layer;
                    for c in children_idxs_a {
                        queue.push(( (layer_idx_a-1, *c), (layer_idx_b, node_idx_b) ) );
                    }
                }
                else {
                    let children_idxs_a = &node_a.children_idxs_in_child_layer;
                    let children_idxs_b = &node_b.children_idxs_in_child_layer;

                    for c_a in children_idxs_a {
                        for c_b in children_idxs_b {
                            queue.push(( (layer_idx_a-1, *c_a), (layer_idx_b-1, *c_b) ) );
                        }
                    }
                }
            }
        }
    }
    fn add_new_layer(&mut self, branch_factor: usize, layer_idx: usize) -> bool {
        let mut new_layer = vec![];

        let child_layer_idx = self.layers.len() - 1;
        let num_nodes_in_child_layer = self.layers[child_layer_idx].len();
        if num_nodes_in_child_layer == 1 { return false; }

        let v = (0..num_nodes_in_child_layer).collect::<Vec<_>>();
        let combinations = comb(&v, branch_factor.min(num_nodes_in_child_layer));

        let mut volume_pack = vec![];
        for c in combinations {
            let mut tmp = vec![];
            for cc in &c {
                tmp.push(&self.layers[child_layer_idx][*cc].combinable_shape);
            }
            let volume_if_combined = T::volume_if_combined(tmp);
            volume_pack.push((c.clone(), volume_if_combined));
        }
        volume_pack.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

        let mut already_done_idxs = vec![];
        let mut remaining_idxs = vec![];
        for _ in 0..num_nodes_in_child_layer { already_done_idxs.push(false); }
        for i in 0..num_nodes_in_child_layer { remaining_idxs.push(i); }

        'f: for v in &volume_pack {
            for vv in &v.0 { if already_done_idxs[*vv] { continue 'f; } }

            let mut shapes = vec![];
            for vv in &v.0 { shapes.push(&self.layers[child_layer_idx][*vv].combinable_shape); }

            let new_node_idx = new_layer.len();
            let new_combinable_shape = T::combine(shapes);
            let mut new_node = BVHCombinableShapeTreeNode {
                combinable_shape: new_combinable_shape,
                layer_idx,
                children_idxs_in_child_layer: v.0.clone(),
                parent_idx_in_parent_layer: None
            };
            new_layer.push(new_node);

            for vv in &v.0 {
                self.layers[child_layer_idx][*vv].parent_idx_in_parent_layer = Some(new_node_idx);

                already_done_idxs[*vv] = true;
                let binary_search_res = remaining_idxs.binary_search(vv);
                match binary_search_res {
                    Ok(idx) => { remaining_idxs.remove(idx); }
                    Err(_) => { unreachable!() }
                }
            }
        }

        if remaining_idxs.len() > 0 {
            let mut shapes = vec![];
            for r in &remaining_idxs { shapes.push(&self.layers[child_layer_idx][*r].combinable_shape) }
            let new_combinable_shape = T::combine(shapes);
            let new_node_idx = new_layer.len();
            let mut new_node = BVHCombinableShapeTreeNode {
                combinable_shape: new_combinable_shape,
                layer_idx,
                children_idxs_in_child_layer: remaining_idxs.clone(),
                parent_idx_in_parent_layer: None
            };
            new_layer.push(new_node);

            for r in remaining_idxs {
                self.layers[child_layer_idx][r].parent_idx_in_parent_layer = Some(new_node_idx);
            }
        }

        self.layers.push(new_layer);

        return true;
    }
    fn update_layer(&mut self, layer_idx: usize) {
        let num_nodes_in_layer = self.layers[layer_idx].len();
        for node_idx in 0..num_nodes_in_layer {
            let children_idxs = &self.layers[layer_idx][node_idx].children_idxs_in_child_layer;
            let mut children_shapes = vec![];
            for c in children_idxs {
                children_shapes.push(&self.layers[layer_idx-1][*c].combinable_shape);
            }
            let updated_shape = T::combine(children_shapes);
            self.layers[layer_idx][node_idx].combinable_shape = updated_shape;
        }
    }
}

#[derive(Clone, Debug)]
pub enum BVHVisit {
    Intersection,
    Distance { margin: f64 }
}

#[derive(Clone, Debug)]
pub struct BVHFilterOutput {
    idxs: Vec<(usize, usize)>,
    num_visits: usize,
    duration: Duration
}

#[derive(Clone, Debug)]
pub struct BVHSceneFilterOutput {
    pairs_list: ShapeCollectionQueryPairsList,
    num_visits: usize,
    duration: Duration
}
impl BVHSceneFilterOutput {
    pub fn pairs_list(&self) -> &ShapeCollectionQueryPairsList {
        &self.pairs_list
    }
    pub fn num_visits(&self) -> usize {
        self.num_visits
    }
    pub fn duration(&self) -> Duration {
        self.duration
    }
}

#[derive(Clone, Debug)]
pub struct BVHCombinableShapeTreeNode<T: BVHCombinableShape> {
    combinable_shape: T,
    layer_idx: usize,
    children_idxs_in_child_layer: Vec<usize>,
    parent_idx_in_parent_layer: Option<usize>
}

#[derive(Clone, Debug)]
pub struct ShapeCollectionBVH<T: BVHCombinableShape> {
    bvh: BVH<T>,
    id: f64
}
impl <T: BVHCombinableShape> ShapeCollectionBVH<T> {
    pub fn bvh(&self) -> &BVH<T> {
        &self.bvh
    }
    pub fn bvh_mut(&mut self) -> &mut BVH<T> {
        &mut self.bvh
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pyclass]
pub struct ShapeCollectionBVHAABB {
    pub bvh: ShapeCollectionBVH<BVHCombinableShapeAABB>
}