#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::time::{Duration};
use std::vec;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use nalgebra::{Isometry3, Point3, Unit, Vector3};
use parry3d_f64::query::{ClosestPoints, Contact, NonlinearRigidMotion, PointProjection, Ray, RayIntersection};
use parry3d_f64ad::query::{ClosestPoints as ClosestPointsAS, Contact as ContactAD, NonlinearRigidMotion as NonlinearRigidMotionAD, PointProjection as PointProjectionAD, Ray as RayAD, RayIntersection as RayIntersectionAD};
use parry3d_f64::shape::{Ball, ConvexPolyhedron, Cuboid, Shape, TriMesh};
use parry3d_f64ad::shape::{Ball as BallAD, ConvexPolyhedron as ConvexPolyhedronAD, Cuboid as CuboidAD, Shape as ShapeAD, TriMesh as TriMeshAD};
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaStemCellPath};
use crate::utils::utils_generic_data_structures::EnumMapToType;
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseAll, OptimaSE3PoseType};
use crate::utils::utils_shape_geometry::shape_collection::{WitnessPoints, WitnessPointsCollection, WitnessPointsType};
use crate::utils::utils_shape_geometry::trimesh_engine::TrimeshEngine;
use crate::utils::utils_traits::{SaveAndLoadable, ToAndFromJsonString};

/// A `GeometricShapeObject` contains useful functions for computing intersection, distances,
/// contacts, raycasting, etc between geometric objects in a scenes.
///
/// This object has a few  fields:
/// - shape: The geometric shape object from the parry3d library.
/// - signature: A `GeometricShapeSignature` used to recognize a particular geometric shape object.
/// - initial_pose_of_shape: The SE(3) pose of the underlying shape when it is initialized.  This is a
/// very important field to understand in order to accurately pose a geometric shape in an environment,
/// so we go into more detail on this below.
///
/// The `initial_pose_of_shape` object is an `OptimaSE3PoseAll` and is the key to understanding how
/// to accurately pose a geometric shape in an environment.  This pose is not meant to be mutable whatsoever
/// so it should not be thought of as stateful.  Instead, it represents only the static pose of the
/// underlying geometric shape when it is first initialized.  To illustrate, suppose a simple cube geometric
/// shape is initialized as a `GeometricShapeObject` and the `initial_pose_of_shape` is set as an SE(3) pose
/// composed of a unit quaternion \[0,0.383,0,0.924\] with a translation of \[1,0,0\].  Thus, this cube is
/// initially placed 1 unit forward on the x axis and exhibits a rotation of 45 degrees around the y axis.
/// Now, ALL GeometricShapeObject transformations for geometric queries will be made with respect to this initial
/// pose!  For example, if a geometric query is invoked on our cube shape above with an SE(3) pose composed
/// of an identity unit quaternion \[0,0,0,1\] and translation of \[0,1,0\], the geometric operation will
/// take place on the underlying cube with unit quaternion rotation \[0,0.383,0,0.924\] * \[0,0,0,1\] = \[0,0.383,0,0.924\]
/// and translation \[1,0,0\] + \[0,1,0\] = \[1,1,0\].
///
/// NOTE: If `initial_pose_of_shape` is None, this "initial pose" is assumed to be the identity rotation
/// and translation such that the pose is initial posed at the origin with no rotation.
pub struct GeometricShape {
    shape: Box<Arc<dyn Shape>>,
    signature: GeometricShapeSignature,
    initial_pose_of_shape: Option<OptimaSE3PoseAll>,
    /// The farthest distance from any point on the shape to the shape's local origin point
    f: f64,
    spawner: GeometricShapeSpawner
}
impl GeometricShape {
    pub fn new_cube(half_extent_x: f64,
                    half_extent_y: f64,
                    half_extent_z: f64,
                    signature: GeometricShapeSignature,
                    initial_pose_of_shape: Option<OptimaSE3Pose>) -> Self {
        let spawner = GeometricShapeSpawner::Cube {
            half_extent_x,
            half_extent_y,
            half_extent_z,
            signature: signature.clone(),
            initial_pose_of_shape: initial_pose_of_shape.clone()
        };
        let cube = Cuboid::new(Vector3::new(half_extent_x,half_extent_y,half_extent_z));
        let mut f = Vector3::new(half_extent_x, half_extent_y, half_extent_z).norm();
        if let Some(initial_pose_of_shape) = &initial_pose_of_shape {
            f += initial_pose_of_shape.unwrap_implicit_dual_quaternion().expect("error").translation().norm();
        }

        Self {
            shape: Box::new(Arc::new(cube)),
            signature,
            initial_pose_of_shape: Self::recover_initial_pose_all_of_shape_from_option(initial_pose_of_shape),
            f,
            spawner
        }
    }
    pub fn new_sphere(radius: f64, signature: GeometricShapeSignature, initial_pose_of_shape: Option<OptimaSE3Pose>) -> Self {
        let spawner = GeometricShapeSpawner::Sphere {
            radius,
            signature: signature.clone(),
            initial_pose_of_shape: initial_pose_of_shape.clone()
        };
        let sphere = Ball::new(radius);
        let mut f = sphere.radius;
        if let Some(initial_pose_of_shape) = &initial_pose_of_shape {
            f += initial_pose_of_shape.unwrap_implicit_dual_quaternion().expect("error").translation().norm();
        }

        Self {
            shape: Box::new(Arc::new(sphere)),
            signature,
            initial_pose_of_shape: Self::recover_initial_pose_all_of_shape_from_option(initial_pose_of_shape),
            f,
            spawner
        }
    }
    pub fn new_convex_shape(trimesh_engine_path: &OptimaStemCellPath, signature: GeometricShapeSignature) -> Self {
        let trimesh_engine= trimesh_engine_path.load_file_to_trimesh_engine().expect("error");
        Self::new_convex_shape_from_trimesh_engine(&trimesh_engine, signature)
    }
    pub fn new_triangle_mesh(trimesh_engine_path: &OptimaStemCellPath, signature: GeometricShapeSignature) -> Self {
        let trimesh_engine= trimesh_engine_path.load_file_to_trimesh_engine().expect("error");
        Self::new_triangle_mesh_from_trimesh_engine(&trimesh_engine, signature)
    }
    pub fn new_convex_shape_from_trimesh_engine(trimesh_engine: &TrimeshEngine, signature: GeometricShapeSignature) -> Self {
        let spawner = GeometricShapeSpawner::ConvexShape {
            path_string_components: trimesh_engine.path_string_components().clone(),
            trimesh_engine: if trimesh_engine.path_string_components().is_empty() { Some(trimesh_engine.clone()) } else { None },
            signature: signature.clone()
        };

        let points: Vec<Point3<f64>> = trimesh_engine.vertices().iter().map(|v| NalgebraConversions::vector3_to_point3(v)).collect();
        let convex_shape = ConvexPolyhedron::from_convex_hull(&points).expect("error");
        let f = trimesh_engine.compute_f();
        // let f = convex_shape.bounding_sphere(&Isometry3::identity()).radius * 2.0;

        Self {
            shape: Box::new(Arc::new(convex_shape)),
            signature,
            initial_pose_of_shape: None,
            f,
            spawner
        }
    }
    pub fn new_triangle_mesh_from_trimesh_engine(trimesh_engine: &TrimeshEngine, signature: GeometricShapeSignature) -> Self {
        let spawner = GeometricShapeSpawner::TriangleMesh {
            path_string_components: trimesh_engine.path_string_components().clone(),
            trimesh_engine: if trimesh_engine.path_string_components().is_empty() { Some(trimesh_engine.clone()) } else { None },
            signature: signature.clone()
        };

        let points: Vec<Point3<f64>> = trimesh_engine.vertices().iter().map(|v| NalgebraConversions::vector3_to_point3(v)).collect();
        let indices: Vec<[u32; 3]> = trimesh_engine.indices().iter().map(|i| [i[0] as u32, i[1] as u32, i[2] as u32] ).collect();
        let f = trimesh_engine.compute_f();

        let tri_mesh = TriMesh::new(points, indices);
        // let f = tri_mesh.bounding_sphere(&Isometry3::identity()).radius * 2.0;

        Self {
            shape: Box::new(Arc::new(tri_mesh)),
            signature,
            initial_pose_of_shape: None,
            f,
            spawner
        }
    }
    pub fn to_best_fit_cube(&self) -> Self {
        let aabb = self.shape.compute_aabb(&Isometry3::identity());
        let center = aabb.center();
        let maxs = aabb.maxs;

        let init_pose_of_shape = OptimaSE3Pose::new_from_euler_angles(0.,0.,0., center[0], center[1], center[2], &OptimaSE3PoseType::ImplicitDualQuaternion);
        return Self::new_cube(maxs[0] - center[0], maxs[1] - center[1], maxs[2] - center[2], self.signature.clone(), Some(init_pose_of_shape));
    }
    pub fn to_best_fit_sphere(&self) -> Self {
        let sphere = self.shape.compute_bounding_sphere(&Isometry3::identity());
        let center = sphere.center();
        let radius = sphere.radius() / 2.0;

        let init_pose_of_shape = OptimaSE3Pose::new_from_euler_angles(0.,0.,0., center[0], center[1], center[2], &OptimaSE3PoseType::ImplicitDualQuaternion);
        return Self::new_sphere(radius, self.signature.clone(), Some(init_pose_of_shape));
    }
    pub fn project_point(&self, pose: &OptimaSE3Pose, point: &Vector3<f64>, solid: bool) -> PointProjection {
        let point = Point3::from_slice(point.data.as_slice());
        self.shape.project_point(&self.recover_transformed_pose_wrt_initial_pose(pose).to_nalgebra_isometry(), &point, solid)
    }
    pub fn contains_point(&self, pose: &OptimaSE3Pose, point: &Vector3<f64>) -> bool {
        let point = Point3::from_slice(point.data.as_slice());
        self.shape.contains_point(&self.recover_transformed_pose_wrt_initial_pose(pose).to_nalgebra_isometry(), &point)
    }
    pub fn distance_to_point(&self, pose: &OptimaSE3Pose, point: &Vector3<f64>, solid: bool) -> f64 {
        let pt = Point3::from_slice(point.as_slice());
        self.shape.distance_to_point(&self.recover_transformed_pose_wrt_initial_pose(pose).to_nalgebra_isometry(), &pt, solid)
    }
    pub fn intersects_ray(&self, pose: &OptimaSE3Pose, ray: &Ray, max_toi: f64) -> bool {
        self.shape.intersects_ray(&self.recover_transformed_pose_wrt_initial_pose(pose).to_nalgebra_isometry(), ray, max_toi)
    }
    /// Computes the time of impact between this transform shape and a ray.
    pub fn cast_ray(&self, pose: &OptimaSE3Pose, ray: &Ray, max_toi: f64, solid: bool) -> Option<f64> {
        self.shape.cast_ray(&self.recover_transformed_pose_wrt_initial_pose(pose).to_nalgebra_isometry(), ray, max_toi, solid)
    }
    pub fn cast_ray_and_get_normal(&self, pose: &OptimaSE3Pose, ray: &Ray, max_toi: f64, solid: bool) -> Option<RayIntersection> {
        self.shape.cast_ray_and_get_normal(&self.recover_transformed_pose_wrt_initial_pose(pose).to_nalgebra_isometry(), ray, max_toi, solid)
    }
    fn recover_initial_pose_all_of_shape_from_option(initial_pose_of_shape: Option<OptimaSE3Pose>) -> Option<OptimaSE3PoseAll> {
        return match initial_pose_of_shape {
            None => { None }
            Some(initial_pose_of_shape) => { Some(OptimaSE3PoseAll::new(&initial_pose_of_shape)) }
        }
    }
    fn recover_transformed_pose_wrt_initial_pose(&self, pose: &OptimaSE3Pose) -> OptimaSE3Pose {
        return match &self.initial_pose_of_shape {
            None => { pose.clone() }
            Some(initial_pose_of_shape) => {
                let initial_pose_of_shape = initial_pose_of_shape.get_pose_by_type(pose.map_to_pose_type());
                let a = pose.multiply(initial_pose_of_shape, false).expect("error");
                a
            }
        }
    }
    pub fn signature(&self) -> &GeometricShapeSignature {
        &self.signature
    }
    pub fn spawner(&self) -> &GeometricShapeSpawner {
        &self.spawner
    }
    pub fn f(&self) -> f64 {
        self.f
    }
    pub fn set_signature(&mut self, signature: GeometricShapeSignature) {
        self.spawner.set_signature(signature.clone());
        self.signature = signature;
    }
}
impl Clone for GeometricShape {
    fn clone(&self) -> Self {
        self.spawner.spawn()
    }
}
impl Serialize for GeometricShape {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        serializer.serialize_some(self.spawner())
    }
}
impl<'de> Deserialize<'de> for GeometricShape {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        let s: GeometricShapeSpawner = Deserialize::deserialize(deserializer)?;
        Ok(s.spawn())
    }
}
impl Debug for GeometricShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self.spawner))
    }
}
impl SaveAndLoadable for GeometricShape {
    type SaveType = GeometricShapeSpawner;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        self.spawner.clone()
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let spawner: Self::SaveType = load_object_from_json_string(json_str)?;
        return Ok(spawner.spawn());
    }
}

pub struct GeometricShapeAD {
    shape: Box<Arc<dyn ShapeAD>>,
    signature: GeometricShapeSignature,
    initial_pose_of_shape: Option<OptimaSE3PoseAll>,
    /// The farthest distance from any point on the shape to the shape's local origin point
    f: f64,
    spawner: GeometricShapeSpawner
}

/// Utility class that holds important geometric shape query functions.
pub struct GeometricShapeQueries;
impl GeometricShapeQueries {
    pub fn generic_group_query(inputs: Vec<GeometricShapeQuery>, stop_condition: StopCondition, log_condition: LogCondition, sort_outputs: bool) -> GeometricShapeQueryGroupOutput {
        let start = instant::Instant::now();
        let mut outputs = vec![];
        let mut output_distances: Vec<f64> = vec![];
        let mut num_queries = 0;
        let mut intersection_found = false;
        let mut minimum_distance = f64::INFINITY;

        for input in &inputs {
            let output = Self::generic_query(input);
            num_queries += 1;
            let proxy_dis = output.raw_output.proxy_dis();

            if proxy_dis <= 0.0 { intersection_found = true; }
            if proxy_dis < minimum_distance { minimum_distance = proxy_dis; }

            let stop = output.raw_output.trigger_stop(&stop_condition);

            if output.raw_output.trigger_log(&log_condition) {
                if sort_outputs {
                    let binary_search_res = output_distances.binary_search_by(|x| x.partial_cmp(&proxy_dis).unwrap() );
                    let idx = match binary_search_res { Ok(i) => {i} Err(i) => {i} };
                    output_distances.insert(idx, proxy_dis);
                    outputs.insert(idx, output);
                } else {
                    outputs.push(output);
                }
            }

            if stop { break; }
        }

        return GeometricShapeQueryGroupOutput {
            outputs,
            duration: start.elapsed(),
            num_queries,
            intersection_found,
            minimum_distance
        }
    }
    pub fn generic_query(input: &GeometricShapeQuery) -> GeometricShapeQueryOutput {
        let start = instant::Instant::now();
        let raw_output = match input {
            GeometricShapeQuery::ProjectPoint { object, pose, point, solid } => {
                GeometricShapeQueryRawOutput::ProjectPoint(PointProjectionWrapper::new(&object.project_point(pose, point, *solid)))
            }
            GeometricShapeQuery::ContainsPoint { object, pose, point } => {
                GeometricShapeQueryRawOutput::ContainsPoint(object.contains_point(pose, point))
            }
            GeometricShapeQuery::DistanceToPoint { object, pose, point, solid } => {
                GeometricShapeQueryRawOutput::DistanceToPoint(object.distance_to_point(pose, point, *solid))
            }
            GeometricShapeQuery::IntersectsRay { object, pose, ray, max_toi } => {
                GeometricShapeQueryRawOutput::IntersectsRay(object.intersects_ray(pose, ray, *max_toi))
            }
            GeometricShapeQuery::CastRay { object, pose, ray, max_toi, solid } => {
                GeometricShapeQueryRawOutput::CastRay(object.cast_ray(pose, ray, *max_toi, *solid))
            }
            GeometricShapeQuery::CastRayAndGetNormal { object, pose, ray, max_toi, solid } => {
                let res = &object.cast_ray_and_get_normal(pose, ray, *max_toi, *solid);
                let out = match res {
                    None => { None }
                    Some(res) => { Some(RayIntersectionWrapper::new(res)) }
                };
                GeometricShapeQueryRawOutput::CastRayAndGetNormal(out)
            }
            GeometricShapeQuery::IntersectionTest { object1, object1_pose, object2, object2_pose } => {
                GeometricShapeQueryRawOutput::IntersectionTest(Self::intersection_test(object1, object1_pose, object2, object2_pose))
            }
            GeometricShapeQuery::Distance { object1, object1_pose, object2, object2_pose } => {
                GeometricShapeQueryRawOutput::Distance(Self::distance(object1, object1_pose, object2, object2_pose))
            }
            GeometricShapeQuery::ClosestPoints { object1, object1_pose, object2, object2_pose, max_dis } => {
                GeometricShapeQueryRawOutput::ClosestPoints(ClosestPointsWrapper::new(&Self::closest_points(object1, object1_pose, object2, object2_pose, *max_dis)))
            }
            GeometricShapeQuery::Contact { object1, object1_pose, object2, object2_pose, prediction } => {
                let out = Self::contact(object1, object1_pose, object2, object2_pose, *prediction);
                GeometricShapeQueryRawOutput::Contact(out)
            }
            GeometricShapeQuery::CCD { object1, object1_pose_t1, object1_pose_t2, object2, object2_pose_t1, object2_pose_t2 } => {
                GeometricShapeQueryRawOutput::CCD(Self::ccd(object1, object1_pose_t1, object1_pose_t2, object2, object2_pose_t1, object2_pose_t2))
            }
        };

        GeometricShapeQueryOutput {
            raw_output,
            duration: start.elapsed(),
            signatures: input.get_signatures()
        }
    }

    pub fn intersection_test(object1: &GeometricShape,
                             object1_pose: &OptimaSE3Pose,
                             object2: &GeometricShape,
                             object2_pose: &OptimaSE3Pose) -> bool {
        let pos1 = object1.recover_transformed_pose_wrt_initial_pose(object1_pose).to_nalgebra_isometry();
        let pos2 = object2.recover_transformed_pose_wrt_initial_pose(object2_pose).to_nalgebra_isometry();

        parry3d_f64::query::intersection_test(&pos1, &**object1.shape, &pos2, &**object2.shape).expect("error")
    }
    pub fn distance(object1: &GeometricShape,
                    object1_pose: &OptimaSE3Pose,
                    object2: &GeometricShape,
                    object2_pose: &OptimaSE3Pose) -> f64 {
        let pos1 = object1.recover_transformed_pose_wrt_initial_pose(object1_pose).to_nalgebra_isometry();
        let pos2 = object2.recover_transformed_pose_wrt_initial_pose(object2_pose).to_nalgebra_isometry();

        parry3d_f64::query::distance(&pos1, &**object1.shape, &pos2, &**object2.shape).expect("error")
    }
    /// Computes the pair of closest points between two shapes.
    /// Returns `ClosestPoints::Disjoint` if the objects are separated by a distance greater than `max_dist`.
    /// The result points in `ClosestPoints::WithinMargin` are expressed in world-space.
    pub fn closest_points(object1: &GeometricShape,
                          object1_pose: &OptimaSE3Pose,
                          object2: &GeometricShape,
                          object2_pose: &OptimaSE3Pose,
                          max_dis: f64) -> ClosestPoints {
        let pos1 = object1.recover_transformed_pose_wrt_initial_pose(object1_pose).to_nalgebra_isometry();
        let pos2 = object2.recover_transformed_pose_wrt_initial_pose(object2_pose).to_nalgebra_isometry();

        parry3d_f64::query::closest_points(&pos1, &**object1.shape, &pos2, &**object2.shape, max_dis).expect(&format!("ERROR: {:?}, {:?}, {:?}, {:?}", pos1, pos2, object1, object2))
    }
    /// Returns None if the objects are separated by a distance greater than prediction. The result is given in world-space
    pub fn contact(object1: &GeometricShape,
                   object1_pose: &OptimaSE3Pose,
                   object2: &GeometricShape,
                   object2_pose: &OptimaSE3Pose,
                   prediction: f64) -> Option<ContactWrapper> {
        let pos1 = object1.recover_transformed_pose_wrt_initial_pose(object1_pose).to_nalgebra_isometry();
        let pos2 = object2.recover_transformed_pose_wrt_initial_pose(object2_pose).to_nalgebra_isometry();

        let contact = parry3d_f64::query::contact(&pos1, &**object1.shape, &pos2, &**object2.shape, prediction).expect("error");
        return match &contact {
            None => { None }
            Some(contact) => { Some(ContactWrapper::new(contact)) }
        }
    }
    /// Continuous collision detection.
    /// Returns None if the objects will never collide.  The CCDResult collision point is provided
    /// in world-space.
    pub fn ccd(object1: &GeometricShape,
               object1_pose_t1: &OptimaSE3Pose,
               object1_pose_t2: &OptimaSE3Pose,
               object2: &GeometricShape,
               object2_pose_t1: &OptimaSE3Pose,
               object2_pose_t2: &OptimaSE3Pose) -> Option<CCDResult> {
        let object1_pose_t1 = object1.recover_transformed_pose_wrt_initial_pose(object1_pose_t1);
        let object1_pose_t2 = object1.recover_transformed_pose_wrt_initial_pose(object1_pose_t2);
        let object2_pose_t1 = object2.recover_transformed_pose_wrt_initial_pose(object2_pose_t1);
        let object2_pose_t2 = object2.recover_transformed_pose_wrt_initial_pose(object2_pose_t2);

        let disp1 = object1_pose_t1.displacement_separate_rotation_and_translation(&object1_pose_t2, false).expect("error");
        let axis_angle = disp1.0.to_axis_angle();
        let angvel1 = axis_angle.1 * axis_angle.0;
        let linvel1 = disp1.1;

        let disp2 = object2_pose_t1.displacement_separate_rotation_and_translation(&object2_pose_t2, false).expect("error");
        let axis_angle = disp2.0.to_axis_angle();
        let angvel2 = axis_angle.1 * axis_angle.0;
        let linvel2 = disp2.1;

        let motion1 = NonlinearRigidMotion::new(object1_pose_t1.to_nalgebra_isometry(), Point3::origin(), linvel1, angvel1);
        let motion2 = NonlinearRigidMotion::new(object2_pose_t1.to_nalgebra_isometry(), Point3::origin(), linvel2, angvel2);

        let res = parry3d_f64::query::nonlinear_time_of_impact(&motion1, &**object1.shape, &motion2, &**object2.shape, 0.0, 1.0, true).expect("error");

        return match &res {
            None => { None }
            Some(res) => {
                let t = res.toi;
                let slerp_pose1 = object1_pose_t1.slerp(&object1_pose_t2, t, false).ok()?;

                let witness1_vec = Vector3::new(res.witness1[0], res.witness1[1], res.witness1[2]);

                let witness1_global = slerp_pose1.multiply_by_point(&witness1_vec);

                Some(CCDResult {
                    toi: t,
                    collision_point: witness1_global,
                    normal1: res.normal1,
                    normal2: res.normal2
                })
            }
        }
    }
}

/// A `GeometricShapeSignature` is used to identify a particular `GeometricShape`.  Importantly,
/// a signature is serializable, can be equal or non-equal via PartialEq and Eq, and able to be
/// sorted via PartialOrd and Ord.
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum GeometricShapeSignature {
    None,
    RobotLink { link_idx: usize, shape_idx_in_link: usize },
    RobotSetLink { robot_idx_in_set: usize, link_idx_in_robot: usize, shape_idx_in_link: usize },
    EnvironmentObject { environment_object_idx: usize, shape_idx_in_object: usize }
}
impl EnumMapToType<GeometricShapeSignatureType> for GeometricShapeSignature {
    fn map_to_type(&self) -> GeometricShapeSignatureType {
        match self {
            GeometricShapeSignature::None => { GeometricShapeSignatureType::None }
            GeometricShapeSignature::RobotLink { .. } => { GeometricShapeSignatureType::RobotLink }
            GeometricShapeSignature::RobotSetLink { .. } => { GeometricShapeSignatureType::RobotSetLink }
            GeometricShapeSignature::EnvironmentObject { .. } => { GeometricShapeSignatureType::EnvironmentObject }
        }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum GeometricShapeSignatureType {
    None,
    RobotLink,
    RobotSetLink,
    EnvironmentObject
}

/// A `GeometricShapeSpawner` is the main object that allows a `GeometricShape` to be serializable
/// and deserializable.  This spawner object contains all necessary information in order to construct
/// a `GeometricShape` from scratch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GeometricShapeSpawner {
    Cube { half_extent_x: f64, half_extent_y: f64, half_extent_z: f64, signature: GeometricShapeSignature, initial_pose_of_shape: Option<OptimaSE3Pose> },
    Sphere { radius: f64, signature: GeometricShapeSignature, initial_pose_of_shape: Option<OptimaSE3Pose> },
    ConvexShape { path_string_components: Vec<String>, trimesh_engine: Option<TrimeshEngine>, signature: GeometricShapeSignature },
    TriangleMesh { path_string_components: Vec<String>, trimesh_engine: Option<TrimeshEngine>, signature: GeometricShapeSignature }
}
impl GeometricShapeSpawner {
    pub fn spawn(&self) -> GeometricShape {
        match self {
            GeometricShapeSpawner::Cube { half_extent_x, half_extent_y, half_extent_z, signature, initial_pose_of_shape } => {
                GeometricShape::new_cube(*half_extent_x, *half_extent_y, *half_extent_z, signature.clone(), initial_pose_of_shape.clone())
            }
            GeometricShapeSpawner::Sphere { radius, signature, initial_pose_of_shape } => {
                GeometricShape::new_sphere( *radius, signature.clone(), initial_pose_of_shape.clone() )
            }
            GeometricShapeSpawner::ConvexShape { path_string_components, trimesh_engine, signature } => {
                if let Some(trimesh_engine) = trimesh_engine {
                    return GeometricShape::new_convex_shape_from_trimesh_engine(trimesh_engine, signature.clone());
                }
                let path = OptimaStemCellPath::new_asset_path_from_string_components(path_string_components).expect("error");
                GeometricShape::new_convex_shape( &path, signature.clone() )
            }
            GeometricShapeSpawner::TriangleMesh { path_string_components, trimesh_engine, signature } => {
                if let Some(trimesh_engine) = trimesh_engine {
                    return GeometricShape::new_convex_shape_from_trimesh_engine(trimesh_engine, signature.clone());
                }
                let path = OptimaStemCellPath::new_asset_path_from_string_components(path_string_components).expect("error");
                GeometricShape::new_triangle_mesh( &path, signature.clone() )
            }
        }
    }
    pub fn set_signature(&mut self, input_signature: GeometricShapeSignature) {
        match self {
            GeometricShapeSpawner::Cube { half_extent_x: _, half_extent_y: _, half_extent_z: _, signature, initial_pose_of_shape: _ } => { *signature = input_signature.clone() }
            GeometricShapeSpawner::Sphere { radius: _, signature, initial_pose_of_shape: _ } => { *signature = input_signature.clone() }
            GeometricShapeSpawner::ConvexShape { path_string_components: _, trimesh_engine: _, signature } => { *signature = input_signature.clone() }
            GeometricShapeSpawner::TriangleMesh { path_string_components: _, trimesh_engine: _, signature } => { *signature = input_signature.clone() }
        }
    }
}

/// A `GeometricShapeSpawner` is the main object that allows a `GeometricShape` to be serializable
/// and deserializable.  This spawner object contains all necessary information in order to construct
/// a `GeometricShape` from scratch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GeometricShapeSpawnerAD {

}

/// Holds all possible inputs into the `GeometricShapeQueries::generic_group_query` and
/// `GeometricShapeQueries::generic_query` functions.
pub enum GeometricShapeQuery<'a> {
    ProjectPoint { object: &'a GeometricShape, pose: OptimaSE3Pose, point: &'a Vector3<f64>, solid: bool },
    ContainsPoint { object: &'a GeometricShape, pose: OptimaSE3Pose, point: &'a Vector3<f64> },
    DistanceToPoint { object: &'a GeometricShape, pose: OptimaSE3Pose, point: &'a Vector3<f64>, solid: bool },
    IntersectsRay { object: &'a GeometricShape, pose: OptimaSE3Pose, ray: &'a Ray, max_toi: f64 },
    CastRay { object: &'a GeometricShape, pose: OptimaSE3Pose, ray: &'a Ray, max_toi: f64, solid: bool },
    CastRayAndGetNormal { object: &'a GeometricShape, pose: OptimaSE3Pose, ray: &'a Ray, max_toi: f64, solid: bool },
    IntersectionTest { object1: &'a GeometricShape, object1_pose: OptimaSE3Pose, object2: &'a GeometricShape, object2_pose: OptimaSE3Pose },
    Distance { object1: &'a GeometricShape, object1_pose: OptimaSE3Pose, object2: &'a GeometricShape, object2_pose: OptimaSE3Pose },
    ClosestPoints { object1: &'a GeometricShape, object1_pose: OptimaSE3Pose, object2: &'a GeometricShape, object2_pose: OptimaSE3Pose, max_dis: f64 },
    Contact { object1: &'a GeometricShape, object1_pose: OptimaSE3Pose, object2: &'a GeometricShape, object2_pose: OptimaSE3Pose, prediction: f64 },
    CCD { object1: &'a GeometricShape, object1_pose_t1: OptimaSE3Pose, object1_pose_t2: OptimaSE3Pose, object2: &'a GeometricShape, object2_pose_t1: OptimaSE3Pose, object2_pose_t2: OptimaSE3Pose }
}
impl <'a> GeometricShapeQuery<'a> {
    pub fn get_signatures(&self) -> Vec<GeometricShapeSignature> {
        let mut out_vec = vec![];
        match self {
            GeometricShapeQuery::ProjectPoint { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQuery::ContainsPoint { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQuery::DistanceToPoint { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQuery::IntersectsRay { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQuery::CastRay { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQuery::CastRayAndGetNormal { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQuery::IntersectionTest { object1, object1_pose: _, object2, object2_pose: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQuery::Distance { object1, object1_pose: _, object2, object2_pose: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQuery::ClosestPoints { object1, object1_pose: _, object2, object2_pose: _, max_dis: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQuery::Contact { object1, object1_pose: _, object2, object2_pose: _, prediction: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQuery::CCD { object1, object1_pose_t1: _, object1_pose_t2: _, object2, object2_pose_t1: _, object2_pose_t2: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
        }
        out_vec
    }
}

/// A raw output from a single `GeometricShapeQuery`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GeometricShapeQueryRawOutput {
    ProjectPoint(PointProjectionWrapper),
    ContainsPoint(bool),
    DistanceToPoint(f64),
    IntersectsRay(bool),
    CastRay(Option<f64>),
    CastRayAndGetNormal(Option<RayIntersectionWrapper>),
    IntersectionTest(bool),
    Distance(f64),
    ClosestPoints(ClosestPointsWrapper),
    Contact(Option<ContactWrapper>),
    CCD(Option<CCDResult>)
}
impl GeometricShapeQueryRawOutput {
    pub fn unwrap_project_point(&self) -> Result<PointProjectionWrapper, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::ProjectPoint(p) => { Ok(p.clone()) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_contains_point(&self) -> Result<bool, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::ContainsPoint(c) => { Ok(*c) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_distance_to_point(&self) -> Result<f64, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::DistanceToPoint(d) => { Ok(*d) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_intersects_ray(&self) -> Result<bool, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::IntersectsRay(b) => { Ok(*b) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_cast_ray(&self) -> Result<Option<f64>, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::CastRay(b) => { Ok(*b) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_cast_ray_and_get_normal(&self) -> Result<Option<RayIntersectionWrapper>, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::CastRayAndGetNormal(b) => { Ok(b.clone()) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_intersection_test(&self) -> Result<bool, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::IntersectionTest(b) => { Ok(*b) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_distance(&self) -> Result<f64, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::Distance(d) => { Ok(*d) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_closest_points(&self) -> Result<&ClosestPointsWrapper, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::ClosestPoints(c) => { Ok(c) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_contact(&self) -> Result<Option<ContactWrapper>, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::Contact(c) => { Ok(c.clone()) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_ccd(&self) -> Result<&Option<CCDResult>, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::CCD(c) => { Ok(c) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn trigger_stop(&self, stop_condition: &StopCondition) -> bool {
        let proxy_dis = self.proxy_dis();
        return match stop_condition {
            StopCondition::None => { false }
            StopCondition::Intersection => { proxy_dis <= 0.0 }
            StopCondition::BelowMinDistance(d) => { proxy_dis < *d }
        }
    }
    pub fn trigger_log(&self, log_condition: &LogCondition) -> bool {
        let proxy_dis = self.proxy_dis();
        return match log_condition {
            LogCondition::LogAll => { true }
            LogCondition::Intersection => { proxy_dis <= 0.0 }
            LogCondition::BelowMinDistance(d) => { proxy_dis < *d }
        }
    }
    fn proxy_dis(&self) -> f64 {
        return match self {
            GeometricShapeQueryRawOutput::ProjectPoint(r) => {
                if r.is_inside { -f64::INFINITY } else { f64::INFINITY }
            }
            GeometricShapeQueryRawOutput::ContainsPoint(r) => {
                if *r { -f64::INFINITY } else { f64::INFINITY }
            }
            GeometricShapeQueryRawOutput::DistanceToPoint(r) => { return *r }
            GeometricShapeQueryRawOutput::IntersectsRay(r) => {
                if *r { -f64::INFINITY } else { f64::INFINITY }
            }
            GeometricShapeQueryRawOutput::CastRay(r) => {
                if r.is_some() { -f64::INFINITY } else { f64::INFINITY }
            }
            GeometricShapeQueryRawOutput::CastRayAndGetNormal(r) => {
                if r.is_some() { -f64::INFINITY } else { f64::INFINITY }
            }
            GeometricShapeQueryRawOutput::IntersectionTest(r) => {
                if *r { -f64::INFINITY } else { f64::INFINITY }
            }
            GeometricShapeQueryRawOutput::Distance(r) => {
                *r
            }
            GeometricShapeQueryRawOutput::ClosestPoints(r) => {
                match r {
                    ClosestPointsWrapper::Intersecting => { -f64::INFINITY }
                    ClosestPointsWrapper::WithinMargin(p1, p2) => {
                        let dis = (p1 - p2).norm();
                        dis
                    }
                    ClosestPointsWrapper::Disjoint => { f64::INFINITY }
                }
            }
            GeometricShapeQueryRawOutput::Contact(r) => {
                match r {
                    None => { f64::INFINITY }
                    Some(c) => { c.dist }
                }
            }
            GeometricShapeQueryRawOutput::CCD(r) => {
                if r.is_some() { -f64::INFINITY } else { f64::INFINITY }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointProjectionWrapper {
    pub point: Vector3<f64>,
    pub is_inside: bool
}
impl PointProjectionWrapper {
    pub fn new(point_projection: &PointProjection) -> Self {
        let p = &point_projection.point;
        let point = Vector3::new(p[0], p[1], p[2]);
        Self {
            point,
            is_inside: point_projection.is_inside
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClosestPointsWrapper {
    Intersecting,
    WithinMargin(Vector3<f64>, Vector3<f64>),
    Disjoint
}
impl ClosestPointsWrapper {
    pub fn new(closest_points: &ClosestPoints) -> Self {
        return match closest_points {
            ClosestPoints::Intersecting => { Self::Intersecting }
            ClosestPoints::WithinMargin(a, b) => { Self::WithinMargin( Vector3::new(a[0], a[1], a[2]), Vector3::new(b[0], b[1], b[2]) ) }
            ClosestPoints::Disjoint => { Self::Disjoint }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RayIntersectionWrapper {
    pub toi: f64,
    pub normal: Vector3<f64>,
    pub edge_idx: u32,
    pub face_idx: u32,
    pub vertex_idx: u32
}
impl RayIntersectionWrapper {
    pub fn new(r: &RayIntersection) -> Self {
        Self {
            toi: r.toi,
            normal: Vector3::new(r.normal[0], r.normal[1], r.normal[2]),
            edge_idx: r.feature.unwrap_edge(),
            face_idx: r.feature.unwrap_face(),
            vertex_idx: r.feature.unwrap_vertex()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContactWrapper {
    pub dist: f64,
    pub normal1: Vector3<f64>,
    pub normal2: Vector3<f64>,
    pub point1: Vector3<f64>,
    pub point2: Vector3<f64>
}
impl ContactWrapper {
    pub fn new(contact: &Contact) -> Self {
        let n1 = &contact.normal1;
        let n2 = &contact.normal1;
        let p1 = &contact.point1;
        let p2 = &contact.point2;
        Self {
            dist: contact.dist,
            normal1: Vector3::new(n1[0], n1[1], n1[2]),
            normal2: Vector3::new(n2[0], n2[1], n2[2]),
            point1: Vector3::new(p1[0], p1[1], p1[2]),
            point2: Vector3::new(p2[0], p2[1], p2[2])
        }
    }
}
impl Default for ContactWrapper {
    fn default() -> Self {
        Self {
            dist: 0.0,
            normal1: Default::default(),
            normal2: Default::default(),
            point1: Default::default(),
            point2: Default::default()
        }
    }
}

/// Holds a result from a continuous collision detection (CCD) computation.
/// Here, `toi` refers to time of impact, `collision_point` is the point in global space that the
/// shapes collide at, and `normal1` and `normal2` vectors are the direction that shapes should
/// instantaneously move to alleviate the collision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CCDResult {
    toi: f64,
    collision_point: Vector3<f64>,
    normal1: Unit<Vector3<f64>>,
    normal2: Unit<Vector3<f64>>
}
impl CCDResult {
    pub fn toi(&self) -> f64 { self.toi }
    pub fn collision_point(&self) -> Vector3<f64> {
        self.collision_point
    }
    pub fn normal1(&self) -> Unit<Vector3<f64>> {
        self.normal1
    }
    pub fn normal2(&self) -> Unit<Vector3<f64>> {
        self.normal2
    }
}

/// Output from the `GeometricShapeQueries::generic_query` function. Contains a `GeometricShapeQueryRawOutput`,
/// the signatures of the shape (or shapes) involved in the query, and the amount of time it took
/// to complete the query.
///
/// The `signatures` field is a vector that will hold either one or two `GeometricShapeSignature`
/// objects depending on how many shapes are involved in the computation. For example, `ProjectPoint`
/// only needs one object while `IntersectionTest` needs two objects.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricShapeQueryOutput {
    duration: Duration,
    signatures: Vec<GeometricShapeSignature>,
    raw_output: GeometricShapeQueryRawOutput
}
impl GeometricShapeQueryOutput {
    pub fn duration(&self) -> Duration {
        self.duration
    }
    pub fn signatures(&self) -> &Vec<GeometricShapeSignature> {
        &self.signatures
    }
    pub fn raw_output(&self) -> &GeometricShapeQueryRawOutput {
        &self.raw_output
    }
}

/// Output from the `GeometricShapeQueries::generic_group_query` function.  Contains a vector of
/// `GeometricShapeQueryOutput` objects, the minimum distance found in the query, if an intersection
/// was found on the query, the number of total queries involved in the group query, and the total
/// amount of time needed to compute the group query.
///
/// For reference on what a "distance" means for a particular output type, look at what
/// is returned by the `GeometricShapeQueryRawOutput proxy_dis` function.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricShapeQueryGroupOutput {
    duration: Duration,
    num_queries: usize,
    intersection_found: bool,
    minimum_distance: f64,
    outputs: Vec<GeometricShapeQueryOutput>
}
impl GeometricShapeQueryGroupOutput {
    pub fn duration(&self) -> Duration {
        self.duration
    }
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }
    pub fn intersection_found(&self) -> bool {
        self.intersection_found
    }
    pub fn minimum_distance(&self) -> f64 {
        self.minimum_distance
    }
    pub fn outputs(&self) -> &Vec<GeometricShapeQueryOutput> {
        &self.outputs
    }
    pub fn print_summary(&self) {
        let len = self.outputs.len();
        for i in 0..len {
            let o = &self.outputs[len - i - 1];
            optima_print(&format!("   Raw output: {:?}", o.raw_output), PrintMode::Println, PrintColor::None, false, 0, None, vec![]);
            optima_print(&format!("   Duration: {:?}", o.duration), PrintMode::Println, PrintColor::None, false, 0, None, vec![]);
            optima_print(&format!("   Signatures: {:?}", o.signatures), PrintMode::Println, PrintColor::None, false, 0, None, vec![]);
            optima_print(&format!(" Output {} --- ^", len - i - 1), PrintMode::Println, PrintColor::Cyan, true, 0, None, vec![]);
        }
        optima_print("Outputs --- ^ ", PrintMode::Println, PrintColor::Blue, false, 0, None, vec![]);
        optima_print("------------", PrintMode::Println, PrintColor::None, false, 0, None, vec![]);
        optima_print(&format!("Duration: {:?}", self.duration), PrintMode::Println, PrintColor::Blue, true, 0, None, vec![]);
        optima_print(&format!("Num Queries: {:?}", self.num_queries), PrintMode::Println, PrintColor::Blue, true, 0, None, vec![]);
        optima_print(&format!("Intersection Found: {:?}", self.intersection_found), PrintMode::Println, PrintColor::Blue, true, 0, None, vec![]);
        optima_print(&format!("Minimum Distance: {:?}", self.minimum_distance), PrintMode::Println, PrintColor::Blue, true, 0, None, vec![]);

    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn convert_to_py_output(&self, include_full_output_json_string: bool) -> GeometricShapeQueryGroupOutputPy {
        let full_output_json_string = match include_full_output_json_string {
            true => { self.to_json_string() }
            false => { "".to_string() }
        };

        GeometricShapeQueryGroupOutputPy {
            duration: self.duration.as_secs_f64(),
            num_queries: self.num_queries,
            intersection_found: self.intersection_found,
            minimum_distance: self.minimum_distance,
            witness_points_collection: self.output_witness_points_collection(),
            full_output_json_string
        }
    }
    pub fn output_witness_points_collection(&self) -> WitnessPointsCollection {
        let mut witness_points_collection = WitnessPointsCollection::new();
        for output in &self.outputs {
            match &output.raw_output {
                GeometricShapeQueryRawOutput::ProjectPoint(_) => {}
                GeometricShapeQueryRawOutput::ContainsPoint(_) => {}
                GeometricShapeQueryRawOutput::DistanceToPoint(_) => {}
                GeometricShapeQueryRawOutput::IntersectsRay(_) => {}
                GeometricShapeQueryRawOutput::CastRay(_) => {}
                GeometricShapeQueryRawOutput::CastRayAndGetNormal(_) => {}
                GeometricShapeQueryRawOutput::IntersectionTest(_) => {}
                GeometricShapeQueryRawOutput::Distance(_) => {}
                GeometricShapeQueryRawOutput::ClosestPoints(_) => {
                    todo!()
                }
                GeometricShapeQueryRawOutput::Contact(c) => {
                    match c {
                        None => {}
                        Some(c) => {
                            witness_points_collection.insert(WitnessPoints::new(c.dist,(c.point1, c.point2), (output.signatures[0].clone(), output.signatures[1].clone()), WitnessPointsType::GroundTruth));
                        }
                    }
                }
                GeometricShapeQueryRawOutput::CCD(_) => {}
            }
        }
        witness_points_collection
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct GeometricShapeQueryGroupOutputPy {
    #[pyo3(get)]
    duration: f64,
    #[pyo3(get)]
    num_queries: usize,
    #[pyo3(get)]
    intersection_found: bool,
    #[pyo3(get)]
    minimum_distance: f64,
    #[pyo3(get)]
    witness_points_collection: WitnessPointsCollection,
    #[pyo3(get)]
    full_output_json_string: String
}

/// Allows for control over when the `GeometricShapeQueries::generic_group_query` function should
/// be early terminated.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StopCondition {
    None,
    Intersection,
    BelowMinDistance(f64)
}

/// Allows for control over when the `GeometricShapeQueries::generic_group_query` function should
/// log a `GeometricShapeQueryOutput` into the outputs field in `GeometricShapeQueryGroupOutput`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LogCondition {
    LogAll,
    /// Only logs the output when it is an intersection.
    Intersection,
    /// Only logs the output when it is below the given distance.
    BelowMinDistance(f64)
}

pub trait BVHCombinableShape where Self: Sized {
    fn new_from_shape_and_pose(shape: &GeometricShape, pose: &OptimaSE3Pose) -> Self;
    fn volume(&self) -> f64;
    fn volume_if_combined(shapes: Vec<&Self>) -> f64;
    fn combine(shapes: Vec<&Self>) -> Self;
    fn intersection_test(a: &Self, b: &Self) -> bool;
    fn distance(a: &Self, b: &Self) -> f64;
}

#[derive(Clone, Debug)]
pub struct BVHCombinableShapeAABB {
    cuboid: Cuboid,
    maxs: Vector3<f64>,
    mins: Vector3<f64>,
    half_extents: Vector3<f64>,
    center: Vector3<f64>
}
impl BVHCombinableShapeAABB {
    pub fn new(maxs: Vector3<f64>, mins: Vector3<f64>) -> Self {
        for i in 0..3 { assert!(maxs[i] > mins[i]); }

        let mut half_extents = Vector3::default();
        let mut center = Vector3::default();
        for i in 0..3 {
            half_extents[i] = (maxs[i] - mins[i]) / 2.0;
            center[i] = (maxs[i] + mins[i]) / 2.0;
        }

        let cuboid = Cuboid::new(half_extents.clone());

        Self {
            cuboid,
            maxs,
            mins,
            half_extents,
            center
        }
    }
    fn vector3_min(vs: &Vec<&Vector3<f64>>) -> Vector3<f64> {
        assert!(vs.len() > 0);
        let mut out = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for i in 0..3 {
            for v in vs {
                out[i] = v[i].min(out[i]);
            }
        }
        out
    }
    fn vector3_max(vs: &Vec<&Vector3<f64>>) -> Vector3<f64> {
        assert!(vs.len() > 0);
        let mut out = Vector3::new(-f64::INFINITY, -f64::INFINITY, -f64::INFINITY);
        for i in 0..3 {
            for v in vs {
                out[i] = v[i].max(out[i]);
            }
        }
        out
    }
    pub fn maxs(&self) -> Vector3<f64> {
        self.maxs
    }
    pub fn mins(&self) -> Vector3<f64> {
        self.mins
    }
    pub fn half_extents(&self) -> Vector3<f64> {
        self.half_extents
    }
    pub fn center(&self) -> Vector3<f64> {
        self.center
    }
}
impl BVHCombinableShape for BVHCombinableShapeAABB {
    fn new_from_shape_and_pose(shape: &GeometricShape, pose: &OptimaSE3Pose) -> Self {
        let recovered_pose = shape.recover_transformed_pose_wrt_initial_pose(pose);
        let iso = recovered_pose.to_nalgebra_isometry();
        let bounding_box = shape.shape.compute_aabb(&iso);
        let maxs = bounding_box.maxs.clone();
        let mins = bounding_box.mins.clone();
        Self::new(Vector3::new(maxs[0], maxs[1], maxs[2]),Vector3::new(mins[0], mins[1], mins[2]))
    }

    fn volume(&self) -> f64 {
        let mut volume = 0.0;
        for h in self.half_extents.iter() { volume *= 2.0 * *h; }
        return volume;
    }

    fn volume_if_combined(shapes: Vec<&Self>) -> f64 {
        let all_mins: Vec<&Vector3<f64>> = shapes.iter().map(|x| &x.mins).collect();
        let all_maxs: Vec<&Vector3<f64>> = shapes.iter().map(|x| &x.maxs).collect();

        let new_mins = Self::vector3_min(&all_mins);
        let new_maxs = Self::vector3_max(&all_maxs);

        let mut half_extents = Vector3::default();
        for i in 0..3 {
            half_extents[i] = (new_maxs[i] - new_mins[i]) / 2.0;
        }

        let mut volume = 1.0;
        for h in half_extents.iter() { volume *= 2.0 * *h; }

        return volume;
    }

    fn combine(shapes: Vec<&Self>) -> Self {
        let all_mins: Vec<&Vector3<f64>> = shapes.iter().map(|x| &x.mins).collect();
        let all_maxs: Vec<&Vector3<f64>> = shapes.iter().map(|x| &x.maxs).collect();

        let new_mins = Self::vector3_min(&all_mins);
        let new_maxs = Self::vector3_max(&all_maxs);

        return Self::new(new_maxs, new_mins);
    }

    fn intersection_test(a: &Self, b: &Self) -> bool {
        let mut pos1 = Isometry3::identity();
        pos1.translation = a.center.into();
        let mut pos2 = Isometry3::identity();
        pos2.translation = b.center.into();

        return parry3d_f64::query::intersection_test(&pos1, &a.cuboid, &pos2, &b.cuboid).expect("error");
    }

    fn distance(a: &Self, b: &Self) -> f64 {
        let mut pos1 = Isometry3::identity();
        pos1.translation = a.center.into();
        let mut pos2 = Isometry3::identity();
        pos2.translation = b.center.into();

        return parry3d_f64::query::distance(&pos1, &a.cuboid, &pos2, &b.cuboid).expect("error");
    }
}

// TODO: Finish BVHCombinableShapeSphere
/*
pub struct BVHCombinableShapeSphere {
    ball: Ball,
    center: Vector3<f64>,
    radius: f64
}
impl BVHCombinableShapeSphere {
    pub fn new(center: Vector3<f64>, radius: f64) -> Self {
        Self {
            ball: Ball::new(radius),
            center,
            radius
        }
    }
}
impl BVHCombinableShape for BVHCombinableShapeSphere {
    fn new_from_shape_and_pose(shape: &GeometricShape, pose: &OptimaSE3Pose) -> Self {
        todo!()
    }

    fn volume(&self) -> f64 {
        (4.0/3.0) * std::f64::consts::PI * self.radius * self.radius * self.radius
    }

    fn volume_if_combined(shapes: Vec<&Self>) -> f64 {
        todo!()
    }

    fn combine(shapes: Vec<&Self>) -> Self {
        todo!()
    }

    fn intersection_test(a: &Self, b: &Self) -> bool {
        todo!()
    }

    fn distance(a: &Self, b: &Self) -> f64 {
        todo!()
    }
}
*/
