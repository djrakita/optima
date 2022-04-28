use std::fmt::Debug;
use nalgebra::{DMatrix, DVector, Point3, Scalar, Vector3};

pub struct NalgebraConversions;
impl NalgebraConversions {
    pub fn vector3_to_point3<T>(v: &Vector3<T>) -> Point3<T> where T: Copy + Clone + PartialEq + Scalar + Debug {
        return Point3::new(v[0], v[1], v[2]);
    }

    pub fn point3_to_vector3<T>(p: &Point3<T>) -> Vector3<T> where T: Copy + Clone + PartialEq + Scalar + Debug {
        return Vector3::new(p[0], p[1], p[2]);
    }

    pub fn dvector_to_vec<T>(d: &DVector<T>) -> Vec<T> where T: Copy + Clone + PartialEq + Scalar + Debug + num_traits::identities::Zero {
        let mut v = vec![];
        for dd in d {
            v.push(*dd);
        }
        return v;
    }

    pub fn vec_to_dvector<T>(v: &Vec<T>) -> DVector<T> where T: Copy + Clone + PartialEq + Scalar + Debug + num_traits::identities::Zero {
        let mut d = DVector::zeros(v.len());
        for (i, vv) in v.iter().enumerate() {
            d[i] = *vv;
        }
        return d;
    }

    pub fn dmatrix_to_vecs(d: &DMatrix<f64>) -> Vec<Vec<f64>> {
        let mut out_vec = vec![];
        let num_rows = d.nrows();
        let num_cols = d.ncols();
        for i in 0..num_rows {
            out_vec.push(vec![]);
            for j in 0..num_cols {
                out_vec[i].push(d[(i,j)])
            }
        }
        out_vec
    }
}
