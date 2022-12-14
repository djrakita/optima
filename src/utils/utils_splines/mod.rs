use std::fmt::Debug;
use nalgebra::{DVector};
use crate::utils::utils_sampling::SimpleSamplers;

#[derive(Clone, Debug)]
pub struct ArclengthParameterizedSpline {
    spline: Spline,
    /// Each element takes the form of (length, t_value)
    arclength_markers: Vec<(f64, f64)>,
    total_arclength: f64
}
impl ArclengthParameterizedSpline {
    pub fn new(spline: Spline, num_arclength_markers: usize) -> Self {
        assert!(num_arclength_markers > 10);

        let mut arclength_markers = vec![];

        let mut t = 0.0;
        let max_allowable_t_value = spline.max_allowable_t_value();
        let step_size = max_allowable_t_value / num_arclength_markers as f64;
        let mut accumulated_distance = 0.0;

        let mut prev_point = spline.interpolate(0.0);

        let mut passed_m = false;
        while !passed_m {
            if t >= max_allowable_t_value {
                passed_m = true;
                t = max_allowable_t_value;
            }
            let curr_point = spline.interpolate(t);
            let dis = (&curr_point - &prev_point).norm();
            accumulated_distance += dis;

            arclength_markers.push((accumulated_distance, t));

            prev_point = curr_point;
            t += step_size;
        }

        Self {
            spline,
            arclength_markers,
            total_arclength: accumulated_distance
        }
    }
    pub fn interpolate(&self, s: f64) -> DVector<f64> {
        assert!( 0.0 <= s && s <= 1.0 );

        let r = s * self.total_arclength;

        let binary_search_res = self.arclength_markers.binary_search_by(|x| x.0.partial_cmp(&r).unwrap());

        return match binary_search_res {
            Ok(idx) => {
                self.spline.interpolate(self.arclength_markers[idx].1)
            }
            Err(idx) => {
                let upper_bound_idx = idx + 1;
                let lower_bound_idx = idx;

                let upper_bound_dis = self.arclength_markers[upper_bound_idx].0;
                let lower_bound_dis = self.arclength_markers[lower_bound_idx].0;

                let upper_bound_t = self.arclength_markers[upper_bound_idx].1;
                let lower_bound_t = self.arclength_markers[lower_bound_idx].1;

                let dis_ratio = (r - lower_bound_dis) / (upper_bound_dis - lower_bound_dis);

                let t = lower_bound_t + dis_ratio * (upper_bound_t - lower_bound_t);

                self.spline.interpolate(t)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Spline {
    InterpolatingSpline(InterpolatingSpline),
    BSpline(BSpline)
}
impl Spline {
    pub fn interpolate(&self, t: f64) -> DVector<f64> {
        match self {
            Spline::InterpolatingSpline(s) => { s.interpolate(t) }
            Spline::BSpline(s) => { s.interpolate(t) }
        }
    }
    pub fn max_allowable_t_value(&self) -> f64 {
        match self {
            Spline::InterpolatingSpline(s) => { s.max_allowable_t_value() }
            Spline::BSpline(s) => { s.max_allowable_t_value() }
        }
    }
}

#[derive(Clone, Debug)]
pub struct InterpolatingSpline {
    control_points: Vec<DVector<f64>>,
    spline_segment_a_coefficients: Vec<Vec<DVector<f64>>>,
    spline_type: InterpolatingSplineType,
    num_spline_segments: usize
}
impl InterpolatingSpline {
    pub fn new(control_points: Vec<DVector<f64>>, spline_type: InterpolatingSplineType) -> Self {
        let num_points = control_points.len();
        assert!(num_points > 0);
        assert_eq!((num_points - spline_type.num_overlap_between_segments()) % (spline_type.num_control_points_per_segment() - spline_type.num_overlap_between_segments()), 0);
        let num_spline_segments = (num_points - spline_type.num_overlap_between_segments()) / (spline_type.num_control_points_per_segment() - spline_type.num_overlap_between_segments());
        let control_point_dim = control_points[0].len();

        let spline_segment_a_coefficients = vec![vec![DVector::from_vec(vec![0.0; control_point_dim]); spline_type.num_control_points_per_segment()]; num_spline_segments];

        let mut out_self = Self {
            control_points,
            spline_segment_a_coefficients,
            spline_type,
            num_spline_segments
        };

        for i in 0..num_spline_segments {
            out_self.calculate_a_vector_coefficients(i);
        }

        out_self
    }
    pub fn new_random(spline_type: InterpolatingSplineType, num_spline_segments: usize, control_point_dim: usize) -> Self {
        assert!(num_spline_segments > 0);
        let num_points = spline_type.num_overlap_between_segments() + num_spline_segments * (spline_type.num_control_points_per_segment() - spline_type.num_overlap_between_segments());

        let mut control_points = vec![];
        for _ in 0..num_points {
            let control_point = DVector::from_vec(SimpleSamplers::uniform_samples(&vec![(-1.0, 1.0); control_point_dim], None));
            control_points.push(control_point);
        }

        Self::new(control_points, spline_type)
    }
    pub fn new_random_3d_on_2d_plane(spline_type: InterpolatingSplineType, num_spline_segments: usize) -> Self {
        assert!(num_spline_segments > 0);
        let num_points = spline_type.num_overlap_between_segments() + num_spline_segments * (spline_type.num_control_points_per_segment() - spline_type.num_overlap_between_segments());

        let mut control_points = vec![];
        for _ in 0..num_points {
            let s = SimpleSamplers::uniform_samples(&vec![(-1.0, 1.0); 2], None);
            let control_point = DVector::from_vec(vec![s[0], s[1], 0.0]);
            control_points.push(control_point);
        }

        Self::new(control_points, spline_type)
    }
    pub fn add_random_segment(&mut self) {
        for _ in 0..self.spline_type.num_control_points_per_segment() - self.spline_type.num_overlap_between_segments() {
            let control_point = DVector::from_vec(SimpleSamplers::uniform_samples(&vec![(-1.0, 1.0); self.control_points[0].len()], None));
            self.control_points.push(control_point);
        }
        self.num_spline_segments += 1;

        self.spline_segment_a_coefficients.push(vec![DVector::from_vec(vec![0.0; self.control_points[0].len()]); self.spline_type.num_control_points_per_segment()]);

        self.calculate_a_vector_coefficients(self.num_spline_segments-1);
    }
    #[inline]
    pub fn update_control_point(&mut self, idx: usize, control_point: DVector<f64>) {
        self.control_points[idx] = control_point;
        let spline_segment_idxs = self.map_control_point_idx_to_spline_segment_idxs(idx);
        for s in spline_segment_idxs {
            self.calculate_a_vector_coefficients(s);
        }
    }
    #[inline]
    pub fn interpolate(&self, t: f64) -> DVector<f64> {
        if t == self.max_allowable_t_value() { return self.interpolate(t - 0.00000001); }

        assert!(t >= 0.0);
        let rt = t.fract();
        let spline_segment_idx = t.floor() as usize;
        assert!(spline_segment_idx < self.num_spline_segments, "t: {}", t);

        let a_vecs = &self.spline_segment_a_coefficients[spline_segment_idx];

        let mut out = DVector::from_vec(vec![0.0; a_vecs[0].len()]);
        for (i, a) in a_vecs.iter().enumerate() {
            out += a * rt.powi(i as i32);
        }

        return out;
    }
    #[inline]
    pub fn map_control_point_idx_to_spline_segment_idxs(&self, control_point_idx: usize) -> Vec<usize> {
        assert!(control_point_idx < self.control_points.len());

        let num_control_points_per_segment = self.spline_type.num_control_points_per_segment();
        let num_overlap_between_segments = self.spline_type.num_overlap_between_segments();

        let a = control_point_idx % (num_control_points_per_segment - num_overlap_between_segments);
        let b = control_point_idx / (num_control_points_per_segment - num_overlap_between_segments);

        let dis_from_segment_edge_option1 = a;
        let dis_from_segment_edge_option2 = num_control_points_per_segment - a;
        let dis_from_either_edge_of_segment = usize::min(dis_from_segment_edge_option1, dis_from_segment_edge_option2);

        if dis_from_either_edge_of_segment >= num_overlap_between_segments { return vec![b]; }

        assert_ne!(dis_from_segment_edge_option1, dis_from_segment_edge_option2);
        if dis_from_segment_edge_option1 < dis_from_segment_edge_option2 && b == 0 {
            return vec![b]
        }
        if dis_from_segment_edge_option1 < dis_from_segment_edge_option2 && b == self.num_spline_segments {
            return vec![b-1];
        }
        if dis_from_segment_edge_option2 < dis_from_segment_edge_option1 && b == self.num_spline_segments {
            return vec![b-1];
        }

        return vec![b-1, b];
    }
    #[inline]
    pub fn map_control_point_idx_to_idx_in_spline_segments(&self, control_point_idx: usize) -> Vec<(usize, usize)> {
        let mut out = vec![];
        let spline_segment_idxs = self.map_control_point_idx_to_spline_segment_idxs(control_point_idx);

        for spline_segment_idx in spline_segment_idxs {
            let control_point_idxs = self.map_spline_segment_idx_to_control_point_idxs(spline_segment_idx);
            let idx = control_point_idxs.iter().position(|x| *x == control_point_idx ).unwrap();
            out.push((spline_segment_idx, idx));
        }

        out
    }
    #[inline]
    pub fn map_spline_segment_idx_to_control_point_idxs(&self, spline_segment_idx: usize) -> Vec<usize> {
        assert!(spline_segment_idx < self.num_spline_segments, "idx {}, num_spline_segments {}", spline_segment_idx, self.num_spline_segments);

        let num_control_points_per_segment = self.spline_type.num_control_points_per_segment();
        let num_overlap_between_segments = self.spline_type.num_overlap_between_segments();

        let start = (num_control_points_per_segment - num_overlap_between_segments) * spline_segment_idx;

        let mut out_vec = vec![];
        for i in 0..num_control_points_per_segment {
            out_vec.push(start + i);
        }

        out_vec
    }
    #[inline]
    fn calculate_a_vector_coefficients(&mut self, spline_segment_idx: usize) {
        let control_point_idxs = self.map_spline_segment_idx_to_control_point_idxs(spline_segment_idx);
        let control_point_refs: Vec<&DVector<f64>> = control_point_idxs.iter().map(|x| &self.control_points[*x]).collect();
        let basis_matrix = self.spline_type.basis_matrix();

        match self.spline_type {
            InterpolatingSplineType::Linear => {
                let a_vecs = calculate_a_vector_coefficients_generic(&control_point_refs, &basis_matrix);
                self.spline_segment_a_coefficients[spline_segment_idx] = a_vecs;
            }
            InterpolatingSplineType::Quadratic => {
                let a_vecs = calculate_a_vector_coefficients_generic(&control_point_refs, &basis_matrix);
                self.spline_segment_a_coefficients[spline_segment_idx] = a_vecs;
            }
            InterpolatingSplineType::HermiteCubic => {
                let a_vecs = calculate_a_vector_coefficients_generic(&control_point_refs, &basis_matrix);
                self.spline_segment_a_coefficients[spline_segment_idx] = a_vecs;
            }
            InterpolatingSplineType::NaturalCubic => {
                let a_vecs = calculate_a_vector_coefficients_generic(&control_point_refs, &basis_matrix);
                self.spline_segment_a_coefficients[spline_segment_idx] = a_vecs;
            }
            InterpolatingSplineType::CardinalCubic { w } => {
                let p1 = control_point_refs[0];
                let d_p1_d_t = control_point_refs[1];
                let p2 = control_point_refs[2];
                let d_p2_d_t = control_point_refs[3];

                let p0 = p2 - (2.0/(1.0-w))* d_p1_d_t;
                let p3 = p1 + (2.0/(1.0-w))* d_p2_d_t;

                let control_points = vec![&p0, p1, p2, &p3];
                let a_vecs = calculate_a_vector_coefficients_generic(&control_points, &basis_matrix);
                self.spline_segment_a_coefficients[spline_segment_idx] = a_vecs;
            }
            InterpolatingSplineType::BezierCubic => {
                let p0 = control_point_refs[0];
                let d_p0_d_t = control_point_refs[1];
                let p3 = control_point_refs[2];
                let d_p3_d_t = control_point_refs[3];

                let p1 = (1./3.)*d_p0_d_t + p0;
                let p2 = p3 - (1./3.)*d_p3_d_t;

                let control_points = vec![p0, &p1, &p2, p3];
                let a_vecs = calculate_a_vector_coefficients_generic(&control_points, &basis_matrix);
                self.spline_segment_a_coefficients[spline_segment_idx] = a_vecs;
            }
        }

    }
    #[inline]
    pub fn spline_type(&self) -> InterpolatingSplineType {
        self.spline_type
    }
    #[inline]
    pub fn num_spline_segments(&self) -> usize {
        self.num_spline_segments
    }
    #[inline]
    pub fn control_points(&self) -> &Vec<DVector<f64>> {
        &self.control_points
    }
    #[inline]
    pub fn max_allowable_t_value(&self) -> f64 {
        return self.num_spline_segments as f64;
    }
}

fn calculate_a_vector_coefficients_generic(p: &Vec<&DVector<f64>>, basis_matrix: &Vec<Vec<f64>>) -> Vec<DVector<f64>> {
    assert!(p.len() > 0);
    let mut out_vec = vec![];

    for row in basis_matrix {
        assert_eq!(row.len(), p.len());
        let mut a = DVector::from_vec(vec![0.0; p[0].len()]);
        for (i, value) in row.iter().enumerate() {
            a += *value * p[i];
        }
        out_vec.push(a);
    }

    out_vec
}

#[derive(Clone, Debug, Copy)]
pub enum InterpolatingSplineType {
    Linear,
    Quadratic,
    HermiteCubic,
    NaturalCubic,
    CardinalCubic{ w: f64 },
    BezierCubic
}
impl InterpolatingSplineType {
    #[inline(always)]
    pub fn num_control_points_per_segment(&self) -> usize {
        match self {
            InterpolatingSplineType::Linear => { 2 }
            InterpolatingSplineType::Quadratic => { 3 }
            InterpolatingSplineType::HermiteCubic => { 4 }
            InterpolatingSplineType::NaturalCubic => { 4 }
            InterpolatingSplineType::CardinalCubic { .. } => { 4 }
            InterpolatingSplineType::BezierCubic => { 4 }
        }
    }
    #[inline(always)]
    pub fn num_overlap_between_segments(&self) -> usize {
        match self {
            InterpolatingSplineType::Linear => { 1 }
            InterpolatingSplineType::Quadratic => { 1 }
            InterpolatingSplineType::HermiteCubic => { 2 }
            InterpolatingSplineType::NaturalCubic => { 1 }
            InterpolatingSplineType::CardinalCubic { .. } => { 2 }
            InterpolatingSplineType::BezierCubic => { 2 }
        }
    }
    /// Returns this "matrix" in rows
    #[inline(always)]
    pub fn basis_matrix(&self) -> Vec<Vec<f64>> {
        match self {
            InterpolatingSplineType::Linear => { vec![vec![1.0, 0.0], vec![-1.0, 1.0]] }
            InterpolatingSplineType::Quadratic => { vec![vec![1.0, 0.0, 0.0], vec![-3.0, 4.0, -1.0], vec![2.0, -4.0, 2.0]] }
            InterpolatingSplineType::HermiteCubic => { vec![vec![1.,0.,0.,0.], vec![0.,1.,0.,0.], vec![-3.,-2.,3.,-1.], vec![2.,1.,-2.,1.]] }
            InterpolatingSplineType::NaturalCubic => { vec![vec![1.,0.,0.,0.], vec![0.,1.,0.,0.], vec![0.,0.,0.5,0.], vec![-1.,-1.,-0.5,1.]] }
            InterpolatingSplineType::CardinalCubic { w } => { vec![vec![0.,1.,0.,0.], vec![(*w-1.0)/2.0,0.,(1.0 - *w)/2.,0.], vec![1. -*w,0.5*(-*w - 5.),*w+2.,(*w-1.)/2.], vec![(*w-1.)/2.,(*w+3.)/2.,0.5*(-*w - 3.),(1.-*w)/2.]]  }
            InterpolatingSplineType::BezierCubic => { vec![vec![1.,0.,0.,0.], vec![-3.,3.,0.,0.], vec![3.,-6.,3.,0.], vec![-1.,3.,-3.,1.]] }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BSpline {
    control_points: Vec<DVector<f64>>,
    knot_vector: Vec<f64>,
    k: usize,
    k_2: f64
}
impl BSpline {
    pub fn new(control_points: Vec<DVector<f64>>, k: usize) -> Self {
        assert!(k > 1);
        let mut knot_vector = vec![];
        let k_2 = k as f64 / 2.0;
        for i in 0..(k+control_points.len()) {
            knot_vector.push( -k_2 + i as f64)
        }
        Self {
            control_points,
            knot_vector,
            k,
            k_2
        }
    }
    pub fn new_random(num_points: usize, control_point_dim: usize, k: usize) -> Self {
        let mut control_points = vec![];
        for _ in 0..num_points {
            let control_point = DVector::from_vec(SimpleSamplers::uniform_samples(&vec![(-1.0, 1.0); control_point_dim], None));
            control_points.push(control_point);
        }

        Self::new(control_points, k)
    }
    pub fn new_random_3d_on_2d_plane(num_points: usize, k: usize) -> Self {

        let mut control_points = vec![];
        for _ in 0..num_points {
            let s = SimpleSamplers::uniform_samples(&vec![(-1.0, 1.0); 2], None);
            let control_point = DVector::from_vec(vec![s[0], s[1], 0.0]);
            control_points.push(control_point);
        }

        Self::new(control_points, k)
    }
    #[inline]
    pub fn cox_de_boor_recurrence(&self, i: usize, k: usize, t: f64) -> f64 {
        assert!(k > 0);
        if k == 1 {
            return if self.knot_vector[i] <= t && t < self.knot_vector[i + 1] { 1.0 } else { 0.0 }
        }

        let c0 = (t - self.knot_vector[i]) / (self.knot_vector[i+k-1] - self.knot_vector[i]);
        let c1 = (self.knot_vector[i+k] - t) / (self.knot_vector[i+k] - self.knot_vector[i+1]);

        return c0 * self.cox_de_boor_recurrence(i, k-1, t) + c1 * self.cox_de_boor_recurrence(i+1, k-1, t);
    }
    #[allow(dead_code)]
    pub (crate) fn interpolate_naive(&self, t: f64) -> DVector<f64> {
        let mut out_sum = DVector::from_vec(vec![0.0; self.control_points[0].len()]);
        for (control_point_idx, control_point) in self.control_points.iter().enumerate() {
            out_sum += control_point*self.cox_de_boor_recurrence(control_point_idx, self.k, t);
        }
        out_sum
    }
    #[inline]
    pub fn interpolate(&self, t: f64) -> DVector<f64> {
        let k_2 = self.k_2;

        let mut out_sum = DVector::from_vec(vec![0.0; self.control_points[0].len()]);
        for (control_point_idx, control_point) in self.control_points.iter().enumerate() {
            let c = control_point_idx as f64;
            if c - k_2 <= t && t < c + k_2 {
                out_sum += control_point*self.cox_de_boor_recurrence(control_point_idx, self.k, t);
            }
        }
        out_sum
    }
    #[inline]
    pub fn update_control_point(&mut self, idx: usize, control_point: DVector<f64>) {
        self.control_points[idx] = control_point;
    }
    #[inline]
    pub fn control_points(&self) -> &Vec<DVector<f64>> {
        &self.control_points
    }
    #[inline]
    pub fn max_allowable_t_value(&self) -> f64 {
        return self.control_points.len() as f64 - 1.0;
    }
}
