use std::fmt::Debug;
use nalgebra::{Point3, Scalar, Vector3};

pub struct NalgebraConversions;
impl NalgebraConversions {
    pub fn vector3_to_point3<T>(v: &Vector3<T>) -> Point3<T> where T: Copy + Clone + PartialEq + Scalar + Debug {
    return Point3::new(v[0], v[1], v[2]);
}

    pub fn point3_to_vector3<T>(p: &Point3<T>) -> Vector3<T> where T: Copy + Clone + PartialEq + Scalar + Debug {
    return Vector3::new(p[0], p[1], p[2]);
}
}
