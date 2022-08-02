use nalgebra::{DMatrix, DVector};
use factorial::Factorial;

pub struct FiniteDifferenceUtils;
impl FiniteDifferenceUtils {
    /// Time stencils should be with respect to the current time (i.e., current time should be 0.0,
    /// a time 1 second ago should be -1.0, etc.)
    pub fn get_fd_coefficients(time_stencils: &Vec<f64>, derivative_order: usize) -> Vec<f64> {
        let n = time_stencils.len();
        assert!(derivative_order < n);

        let mut m = DMatrix::<f64>::zeros(n, n);
        let mut v = DVector::<f64>::zeros(n);

        v[derivative_order] = derivative_order.factorial() as f64;

        for i in 0..n {
            for j in 0..n {
                m[(i,j)] = time_stencils[j].powi(i as i32);
            }
        }

        let m_inv = m.pseudo_inverse(0.00001).unwrap();
        let res = m_inv * v;

        return res.data.as_slice().to_vec();
    }
}

