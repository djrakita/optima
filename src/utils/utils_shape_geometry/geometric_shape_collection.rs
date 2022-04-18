use nalgebra::{Vector3};
use parry3d_f64::query::Ray;
use serde::{Serialize, Deserialize};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_generic_data_structures::SquareArray2D;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeQueries, GeometricShapeQueryGroupOutput, GeometricShapeQueryInput, GeometricShapeSignature, LogCondition, StopCondition};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricShapeCollection {
    shapes: Vec<GeometricShape>,
    skips: SquareArray2D<bool>,
    average_distances: SquareArray2D<f64>,
    sorted_signatures_with_shape_idxs: Vec<(GeometricShapeSignature, usize)>
}
impl GeometricShapeCollection {
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
        self.skips.append_new_row_and_column(Some(false));
        self.average_distances.append_new_row_and_column(Some(1.0));
    }
    pub fn set_skip(&mut self, skip: bool, signature1: &GeometricShapeSignature, signature2: &GeometricShapeSignature) -> Result<(), OptimaError> {
        let idx1 = self.get_shape_idx_from_signature(signature1)?;
        let idx2 = self.get_shape_idx_from_signature(signature2)?;

        self.skips.replace_data(skip, idx1, idx2)?;

        Ok(())
    }
    pub fn set_average_distance(&mut self, dis: f64, signature1: &GeometricShapeSignature, signature2: &GeometricShapeSignature) -> Result<(), OptimaError> {
        let idx1 = self.get_shape_idx_from_signature(signature1)?;
        let idx2 = self.get_shape_idx_from_signature(signature2)?;

        self.average_distances.replace_data(dis, idx1, idx2)?;

        Ok(())
    }
    pub fn set_skip_from_idxs(&mut self, skip: bool, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.skips.replace_data(skip, idx1, idx2)
    }
    pub fn set_average_distance_from_idxs(&mut self, dis: f64, idx1: usize, idx2: usize) -> Result<(), OptimaError> {
        self.average_distances.replace_data(dis, idx1, idx2)
    }
    pub fn shapes(&self) -> &Vec<GeometricShape> {
        &self.shapes
    }
    pub fn skips(&self) -> &SquareArray2D<bool> {
        &self.skips
    }
    pub fn average_distances(&self) -> &SquareArray2D<f64> {
        &self.average_distances
    }
    pub fn get_shape_idx_from_signature(&self, signature: &GeometricShapeSignature) -> Result<usize, OptimaError> {
        let binary_search_res = self.sorted_signatures_with_shape_idxs.binary_search_by(|x| signature.partial_cmp(&x.0).unwrap());
        return match binary_search_res {
            Ok(idx) => {
                Ok(idx)
            }
            Err(_) => {
                Err(OptimaError::new_generic_error_str(&format!("Shape with signature {:?} not found in GeometricShapeCollection.", signature), file!(), line!()))
            }
        };
    }
    pub fn get_geometric_shape_query_input_vec<'a>(&'a self, input: &'a GeometricShapeCollectionQueryInput) -> Result<Vec<GeometricShapeQueryInput<'a>>, OptimaError> {
        return match input {
            GeometricShapeCollectionQueryInput::ProjectPoint { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::ContainsPoint { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::DistanceToPoint { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::IntersectsRay { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::CastRay { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::CastRayAndGetNormal { .. } => { self.get_single_object_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::IntersectionTest { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::Distance { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::ClosestPoints { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::Contact { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
            GeometricShapeCollectionQueryInput::CCD { .. } => { self.get_pairwise_objects_geometric_shape_query_input_vec(input) }
        }
    }
    fn get_single_object_geometric_shape_query_input_vec<'a>(&'a self, input: &'a GeometricShapeCollectionQueryInput) -> Result<Vec<GeometricShapeQueryInput<'a>>, OptimaError> {
        let mut out_vec = vec![];

        let poses = &input.get_poses()?[0].poses;
        for (i, s) in self.shapes.iter().enumerate() {
            let pose = &poses[i];
            if let Some(pose) = pose {
                match input {
                    GeometricShapeCollectionQueryInput::ProjectPoint { poses: _, point, solid } => {
                        out_vec.push(GeometricShapeQueryInput::ProjectPoint {
                            object: s,
                            pose: pose.clone(),
                            point,
                            solid: *solid
                        });
                    }
                    GeometricShapeCollectionQueryInput::ContainsPoint { poses: _, point } => {
                        out_vec.push(GeometricShapeQueryInput::ContainsPoint {
                            object: s,
                            pose: pose.clone(),
                            point
                        });
                    }
                    GeometricShapeCollectionQueryInput::DistanceToPoint { poses: _, point, solid } => {
                        out_vec.push(GeometricShapeQueryInput::DistanceToPoint {
                            object: s,
                            pose: pose.clone(),
                            point,
                            solid: *solid
                        });
                    }
                    GeometricShapeCollectionQueryInput::IntersectsRay { poses: _, ray, max_toi } => {
                        out_vec.push(GeometricShapeQueryInput::IntersectsRay {
                            object: s,
                            pose: pose.clone(),
                            ray,
                            max_toi: *max_toi
                        });
                    }
                    GeometricShapeCollectionQueryInput::CastRay { poses: _, ray, max_toi, solid } => {
                        out_vec.push(GeometricShapeQueryInput::CastRay {
                            object: s,
                            pose: pose.clone(),
                            ray,
                            max_toi: *max_toi,
                            solid: *solid
                        });
                    }
                    GeometricShapeCollectionQueryInput::CastRayAndGetNormal { poses: _, ray, max_toi, solid } => {
                        out_vec.push(GeometricShapeQueryInput::CastRayAndGetNormal {
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
    fn get_pairwise_objects_geometric_shape_query_input_vec<'a>(&'a self, input: &'a GeometricShapeCollectionQueryInput) -> Result<Vec<GeometricShapeQueryInput<'a>>, OptimaError> {
        let mut out_vec = vec![];

        let poses = &input.get_poses()?[0].poses;

        for (i, shape1) in self.shapes.iter().enumerate() {
            let pose1 = &poses[i];
            if let Some(pose1) = pose1 {
                for (j, shape2) in self.shapes.iter().enumerate() {
                    let pose2 = &poses[j];
                    if let Some(pose2) = pose2 {
                        if i <= j {
                            let skip = self.skips.data_cell(i, j)?;
                            if !*skip {
                                match input {
                                    GeometricShapeCollectionQueryInput::IntersectionTest { .. } => {
                                        out_vec.push(GeometricShapeQueryInput::IntersectionTest {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone()
                                        });
                                    }
                                    GeometricShapeCollectionQueryInput::Distance { .. } => {
                                        out_vec.push(GeometricShapeQueryInput::Distance {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone()
                                        });
                                    }
                                    GeometricShapeCollectionQueryInput::ClosestPoints { poses: _, max_dis } => {
                                        out_vec.push(GeometricShapeQueryInput::ClosestPoints {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone(),
                                            max_dis: *max_dis
                                        });
                                    }
                                    GeometricShapeCollectionQueryInput::Contact { poses: _, prediction } => {
                                        out_vec.push(GeometricShapeQueryInput::Contact {
                                            object1: shape1,
                                            object1_pose: pose1.clone(),
                                            object2: shape2,
                                            object2_pose: pose2.clone(),
                                            prediction: *prediction
                                        });
                                    }
                                    GeometricShapeCollectionQueryInput::CCD { poses_t1: _, poses_t2 } => {
                                        let pose1_t2 = &poses_t2.poses[i];
                                        let pose2_t2 = &poses_t2.poses[j];
                                        if let Some(pose1_t2) = pose1_t2 {
                                            if let Some(pose2_t2) = pose2_t2 {
                                                out_vec.push(GeometricShapeQueryInput::CCD {
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
    pub fn generic_group_query<'a>(&'a self,
                                   input: &'a GeometricShapeCollectionQueryInput,
                                   stop_condition: StopCondition,
                                   log_condition: LogCondition,
                                   sort_outputs: bool) -> Result<GeometricShapeQueryGroupOutput, OptimaError> {
        let input_vec = self.get_geometric_shape_query_input_vec(input)?;
        Ok(GeometricShapeQueries::generic_group_query(input_vec, stop_condition, log_condition, sort_outputs))
    }
}

pub enum GeometricShapeCollectionQueryInput<'a> {
    ProjectPoint { poses: &'a GeometricShapeCollectionInputPoses, point: &'a Vector3<f64>, solid: bool },
    ContainsPoint { poses: &'a GeometricShapeCollectionInputPoses, point: &'a Vector3<f64> },
    DistanceToPoint { poses: &'a GeometricShapeCollectionInputPoses, point: &'a Vector3<f64>, solid: bool },
    IntersectsRay { poses: &'a GeometricShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64 },
    CastRay { poses: &'a GeometricShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64, solid: bool },
    CastRayAndGetNormal { poses: &'a GeometricShapeCollectionInputPoses, ray: &'a Ray, max_toi: f64, solid: bool },
    IntersectionTest { poses: &'a GeometricShapeCollectionInputPoses },
    Distance { poses: &'a GeometricShapeCollectionInputPoses },
    ClosestPoints { poses: &'a GeometricShapeCollectionInputPoses, max_dis: f64 },
    Contact { poses: &'a GeometricShapeCollectionInputPoses, prediction: f64 },
    CCD { poses_t1: &'a GeometricShapeCollectionInputPoses, poses_t2: &'a GeometricShapeCollectionInputPoses }
}
impl <'a> GeometricShapeCollectionQueryInput<'a> {
    pub fn get_poses(&self) -> Result<Vec<&'a GeometricShapeCollectionInputPoses>, OptimaError> {
        match self {
            GeometricShapeCollectionQueryInput::ProjectPoint { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::ContainsPoint { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::DistanceToPoint { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::IntersectsRay { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::CastRay { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::CastRayAndGetNormal { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::IntersectionTest { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::Distance { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::ClosestPoints { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::Contact { poses, .. } => { Ok(vec![poses]) }
            GeometricShapeCollectionQueryInput::CCD { poses_t1, poses_t2 } => { Ok(vec![poses_t1, poses_t2]) }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricShapeCollectionInputPoses {
    poses: Vec<Option<OptimaSE3Pose>>
}
impl GeometricShapeCollectionInputPoses {
    pub fn new(geometric_shape_collection: &GeometricShapeCollection) -> Self {
        let num_shapes = geometric_shape_collection.shapes.len();
        let mut poses = vec![];
        for _ in 0..num_shapes { poses.push(None); }
        Self {
            poses
        }
    }
    pub fn insert_or_replace_pose(&mut self,
                                  signature: &GeometricShapeSignature,
                                  pose: OptimaSE3Pose,
                                  geometric_shape_collection: &GeometricShapeCollection) -> Result<(), OptimaError> {
        let idx = geometric_shape_collection.get_shape_idx_from_signature(signature)?;

        self.poses[idx] = Some(pose);

        Ok(())
    }
    pub fn insert_or_replace_pose_by_idx(&mut self, idx: usize, pose: OptimaSE3Pose) -> Result<(), OptimaError> {
        OptimaError::new_check_for_out_of_bound_error(idx, self.poses.len(), file!(), line!())?;

        self.poses[idx] = Some(pose);

        Ok(())
    }
    pub fn poses(&self) -> &Vec<Option<OptimaSE3Pose>> {
        &self.poses
    }
    pub fn is_full(&self) -> bool {
        for p in &self.poses {
            if p.is_none() { return false; }
        }
        return true;
    }
}

