use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use nalgebra::{Isometry3, Point3, Unit, Vector3};
use parry3d_f64::query::{ClosestPoints, Contact, NonlinearRigidMotion, PointProjection, Ray, RayIntersection};
use parry3d_f64::shape::{Cuboid, Shape, Ball, ConvexPolyhedron, TriMesh};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseAll, OptimaSE3PoseType};
use crate::utils::utils_shape_geometry::trimesh_engine::TrimeshEngine;

/// A `GeometricShapeObject` contains useful functions for computing intersection, distances,
/// contacts, raycasting, etc between geometric objects in a scene.  This object has a few  fields:
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
/// composed of a unit quaternion [0,0.383,0,0.924] with a translation of [1,0,0].  Thus, this cube is
/// initially placed 1 unit forward on the x axis and exhibits a rotation of 45 degrees around the y axis.
/// Now, ALL GeometricShapeObject transformations for geometric queries will be made with respect to this initial
/// pose!  For example, if a geometric query is invoked on our cube shape above with an SE(3) pose composed
/// of an identity unit quaternion [0,0,0,1] and translation of [0,1,0], the geometric operation will
/// take place on the underlying cube with unit quaternion rotation [0,0.383,0,0.924] * [0,0,0,1] = [0,0.383,0,0.924]
/// and translation [1,0,0] + [0,1,0] = [1,1,0].
///
/// NOTE: If `initial_pose_of_shape` is None, this "initial pose" is assumed to be the identity rotation
/// and translation such that the pose is initial posed at the origin with no rotation.
pub struct GeometricShape {
    shape: Box<Arc<dyn Shape>>,
    signature: GeometricShapeSignature,
    initial_pose_of_shape: Option<OptimaSE3PoseAll>,
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

        Self {
            shape: Box::new(Arc::new(cube)),
            signature,
            initial_pose_of_shape: Self::recover_initial_pose_all_of_shape_from_option(initial_pose_of_shape),
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

        Self {
            shape: Box::new(Arc::new(sphere)),
            signature,
            initial_pose_of_shape: Self::recover_initial_pose_all_of_shape_from_option(initial_pose_of_shape),
            spawner
        }
    }
    pub fn new_convex_shape(trimesh_engine: &TrimeshEngine, signature: GeometricShapeSignature) -> Self {
        let spawner = GeometricShapeSpawner::ConvexShape {
            trimesh_engine: trimesh_engine.clone(),
            signature: signature.clone()
        };
        
        let points: Vec<Point3<f64>> = trimesh_engine.vertices().iter().map(|v| NalgebraConversions::vector3_to_point3(v)).collect();
        let convex_shape = ConvexPolyhedron::from_convex_hull(&points).expect("error");

        Self {
            shape: Box::new(Arc::new(convex_shape)),
            signature,
            initial_pose_of_shape: None,
            spawner
        }
    }
    pub fn new_triangle_mesh(trimesh_engine: &TrimeshEngine, signature: GeometricShapeSignature) -> Self {
        let spawner = GeometricShapeSpawner::TriangleMesh {
            trimesh_engine: trimesh_engine.clone(),
            signature: signature.clone()
        };
        
        let points: Vec<Point3<f64>> = trimesh_engine.vertices().iter().map(|v| NalgebraConversions::vector3_to_point3(v)).collect();
        let indices: Vec<[u32; 3]> = trimesh_engine.indices().iter().map(|i| [i[0] as u32, i[1] as u32, i[2] as u32] ).collect();

        let tri_mesh = TriMesh::new(points, indices);

        Self {
            shape: Box::new(Arc::new(tri_mesh)),
            signature,
            initial_pose_of_shape: None,
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

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum GeometricShapeSignature {
    None,
    RobotLink { link_idx: usize, shape_idx_in_link: usize }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GeometricShapeSpawner {
    Cube { half_extent_x: f64, half_extent_y: f64, half_extent_z: f64, signature: GeometricShapeSignature, initial_pose_of_shape: Option<OptimaSE3Pose> },
    Sphere { radius: f64, signature: GeometricShapeSignature, initial_pose_of_shape: Option<OptimaSE3Pose> },
    ConvexShape { trimesh_engine: TrimeshEngine, signature: GeometricShapeSignature },
    TriangleMesh { trimesh_engine: TrimeshEngine, signature: GeometricShapeSignature }
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
            GeometricShapeSpawner::ConvexShape { trimesh_engine, signature } => {
                GeometricShape::new_convex_shape( &trimesh_engine, signature.clone() )
            }
            GeometricShapeSpawner::TriangleMesh { trimesh_engine, signature } => {
                GeometricShape::new_triangle_mesh( &trimesh_engine, signature.clone() )
            }
        }
    }
}

pub struct GeometricShapeQueries;
impl GeometricShapeQueries {
    pub fn generic_group_query(inputs: Vec<GeometricShapeQueryInput>, stop_condition: StopCondition, log_condition: LogCondition, sort_outputs: bool) -> GeometricShapeQueryGroupOutput {
        let start = Instant::now();
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
    pub fn generic_query(input: &GeometricShapeQueryInput) -> GeometricShapeQueryOutput {
        let start = Instant::now();
        let raw_output = match input {
            GeometricShapeQueryInput::ProjectPoint { object, pose, point, solid } => {
                GeometricShapeQueryRawOutput::ProjectPoint(object.project_point(pose, point, *solid))
            }
            GeometricShapeQueryInput::ContainsPoint { object, pose, point } => {
                GeometricShapeQueryRawOutput::ContainsPoint(object.contains_point(pose, point))
            }
            GeometricShapeQueryInput::DistanceToPoint { object, pose, point, solid } => {
                GeometricShapeQueryRawOutput::DistanceToPoint(object.distance_to_point(pose, point, *solid))
            }
            GeometricShapeQueryInput::IntersectsRay { object, pose, ray, max_toi } => {
                GeometricShapeQueryRawOutput::IntersectsRay(object.intersects_ray(pose, ray, *max_toi))
            }
            GeometricShapeQueryInput::CastRay { object, pose, ray, max_toi, solid } => {
                GeometricShapeQueryRawOutput::CastRay(object.cast_ray(pose, ray, *max_toi, *solid))
            }
            GeometricShapeQueryInput::CastRayAndGetNormal { object, pose, ray, max_toi, solid } => {
                GeometricShapeQueryRawOutput::CastRayAndGetNormal(object.cast_ray_and_get_normal(pose, ray, *max_toi, *solid))
            }
            GeometricShapeQueryInput::IntersectionTest { object1, object1_pose, object2, object2_pose } => {
                GeometricShapeQueryRawOutput::IntersectionTest(Self::intersection_test(object1, object1_pose, object2, object2_pose))
            }
            GeometricShapeQueryInput::Distance { object1, object1_pose, object2, object2_pose } => {
                GeometricShapeQueryRawOutput::Distance(Self::distance(object1, object1_pose, object2, object2_pose))
            }
            GeometricShapeQueryInput::ClosestPoints { object1, object1_pose, object2, object2_pose, max_dis } => {
                GeometricShapeQueryRawOutput::ClosestPoints(Self::closest_points(object1, object1_pose, object2, object2_pose, *max_dis))
            }
            GeometricShapeQueryInput::Contact { object1, object1_pose, object2, object2_pose, prediction } => {
                GeometricShapeQueryRawOutput::Contact(Self::contact(object1, object1_pose, object2, object2_pose, *prediction))
            }
            GeometricShapeQueryInput::CCD { object1, object1_pose_t1, object1_pose_t2, object2, object2_pose_t1, object2_pose_t2 } => {
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

        parry3d_f64::query::closest_points(&pos1, &**object1.shape, &pos2, &**object2.shape, max_dis).expect("error")
    }
    /// Returns None if the objects are separated by a distance greater than prediction. The result is given in world-space
    pub fn contact(object1: &GeometricShape,
                   object1_pose: &OptimaSE3Pose,
                   object2: &GeometricShape,
                   object2_pose: &OptimaSE3Pose,
                   prediction: f64) -> Option<Contact> {
        let pos1 = object1.recover_transformed_pose_wrt_initial_pose(object1_pose).to_nalgebra_isometry();
        let pos2 = object2.recover_transformed_pose_wrt_initial_pose(object2_pose).to_nalgebra_isometry();

        parry3d_f64::query::contact(&pos1, &**object1.shape, &pos2, &**object2.shape, prediction).expect("error")
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

#[derive(Clone, Debug)]
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

pub enum GeometricShapeQueryInput<'a> {
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
impl <'a> GeometricShapeQueryInput<'a> {
    pub fn get_signatures(&self) -> Vec<GeometricShapeSignature> {
        let mut out_vec = vec![];
        match self {
            GeometricShapeQueryInput::ProjectPoint { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQueryInput::ContainsPoint { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQueryInput::DistanceToPoint { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQueryInput::IntersectsRay { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQueryInput::CastRay { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQueryInput::CastRayAndGetNormal { object, .. } => { out_vec.push(object.signature.clone()) }
            GeometricShapeQueryInput::IntersectionTest { object1, object1_pose: _, object2, object2_pose: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQueryInput::Distance { object1, object1_pose: _, object2, object2_pose: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQueryInput::ClosestPoints { object1, object1_pose: _, object2, object2_pose: _, max_dis: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQueryInput::Contact { object1, object1_pose: _, object2, object2_pose: _, prediction: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
            GeometricShapeQueryInput::CCD { object1, object1_pose_t1: _, object1_pose_t2: _, object2, object2_pose_t1: _, object2_pose_t2: _ } => {
                out_vec.push(object1.signature.clone());
                out_vec.push(object2.signature.clone());
            }
        }
        out_vec
    }
}

#[derive(Clone, Debug)]
pub enum GeometricShapeQueryRawOutput {
    ProjectPoint(PointProjection),
    ContainsPoint(bool),
    DistanceToPoint(f64),
    IntersectsRay(bool),
    CastRay(Option<f64>),
    CastRayAndGetNormal(Option<RayIntersection>),
    IntersectionTest(bool),
    Distance(f64),
    ClosestPoints(ClosestPoints),
    Contact(Option<Contact>),
    CCD(Option<CCDResult>)
}
impl GeometricShapeQueryRawOutput {
    pub fn unwrap_project_point(&self) -> Result<PointProjection, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::ProjectPoint(p) => { Ok(*p) }
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
    pub fn unwrap_cast_ray_and_get_normal(&self) -> Result<Option<RayIntersection>, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::CastRayAndGetNormal(b) => { Ok(*b) }
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
    pub fn unwrap_closest_points(&self) -> Result<&ClosestPoints, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::ClosestPoints(c) => { Ok(c) }
            _ => { return Err(OptimaError::new_generic_error_str("Incompatible type.", file!(), line!())) }
        }
    }
    pub fn unwrap_contact(&self) -> Result<Option<Contact>, OptimaError> {
        return match self {
            GeometricShapeQueryRawOutput::Contact(c) => { Ok(*c) }
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
                    ClosestPoints::Intersecting => { -f64::INFINITY }
                    ClosestPoints::WithinMargin(p1, p2) => {
                        let dis = (p1 - p2).norm();
                        dis
                    }
                    ClosestPoints::Disjoint => { f64::INFINITY }
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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
}

#[derive(Clone, Debug)]
pub enum StopCondition {
    None,
    Intersection,
    BelowMinDistance(f64)
}

#[derive(Clone, Debug)]
pub enum LogCondition {
    LogAll,
    Intersection,
    BelowMinDistance(f64)
}