use nalgebra::{DVector, Vector2, Vector3};
use crate::utils::utils_math::interpolation::get_range;
use crate::utils::utils_math::vector::get_orthonormal_basis;
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;


pub fn closest_point_on_2_lines_dvecs(u1: &DVector<f64>, u2: &DVector<f64>, v1: &DVector<f64>, v2: &DVector<f64>) -> (f64, f64) {
    let u = u2 - u1;
    let v = v2 - v1;
    let rho = v1 - u1;
    let uv = u.dot(&v);
    let uu = u.dot(&u);
    let vv = v.dot(&v);
    let urho = u.dot(&rho);
    let vrho = v.dot(&rho);

    let vt = (vrho*uu - urho*uv) / (uv*uv - vv*uu).max(0.00000000001);
    let ut = (uv * vt + urho) / uu.max(0.00000000001);

    // ut = ut.max(0.0).min(1.0);
    // vt = vt.max(0.0).min(1.0);

    return (ut, vt)
}

pub fn closest_point_on_2_lines(u1: &Vec<f64>, u2: &Vec<f64>, v1: &Vec<f64>, v2: &Vec<f64>) -> (f64, f64)  {
    closest_point_on_2_lines_dvecs(&NalgebraConversions::vec_to_dvector(u1), &NalgebraConversions::vec_to_dvector(u2), &NalgebraConversions::vec_to_dvector(v1), &NalgebraConversions::vec_to_dvector(v2))
}

pub fn pt_dis_to_line_seg_dvecs(pt: &DVector<f64>, a: &DVector<f64>, b: &DVector<f64>) -> (f64, DVector<f64>) {
    let tmp = (a-b).norm();
    let mut u = (pt - a).dot( &(b - a) ) / (tmp * tmp).max(0.00000000001);
    u = u.max(0.).min(1.);
    let p = a + u*(b-a);
    let dis = (&p - pt).norm();
    return (dis, p.clone());
}

pub fn pt_dis_to_line_seg(pt: &Vec<f64>, a: &Vec<f64>, b: &Vec<f64>) -> (f64, DVector<f64>) {
    pt_dis_to_line_seg_dvecs(&NalgebraConversions::vec_to_dvector(&pt), &NalgebraConversions::vec_to_dvector(&a), &NalgebraConversions::vec_to_dvector(&b))
}

pub fn pt_dis_to_to_line_dvecs(pt: &DVector<f64>, a: &DVector<f64>, b: &DVector<f64>) -> (f64, DVector<f64>) {
    let tmp = (a-b).norm();
    let u = (pt - a).dot( &(b - a) ) / (tmp * tmp).max(0.00000000001);
    let p = a + u*(b-a);
    let dis = (&p - pt).norm();
    return (dis, p.clone());
}

pub fn pt_dis_to_line(pt: &Vec<f64>, a: &Vec<f64>, b: &Vec<f64>) -> (f64, DVector<f64>) {
    pt_dis_to_to_line_dvecs(&NalgebraConversions::vec_to_dvector(&pt), &NalgebraConversions::vec_to_dvector(&a), &NalgebraConversions::vec_to_dvector(&b))
}

pub fn pt_dis_to_line_and_seg_dvecs(pt: &DVector<f64>, a: &DVector<f64>, b: &DVector<f64>) -> (f64, DVector<f64>, f64, f64, DVector<f64>, f64) {
    let tmp = (a-b).norm();
    let u1 = (pt - a).dot( &(b - a) ) / (tmp * tmp).max(0.00000000001);
    let u2 = u1.max(0.).min(1.);
    let p1 = a + u1*(b-a);
    let p2 = a + u2*(b-a);
    let dis1 = (&p1 - pt).norm();
    let dis2 = (&p2 - pt).norm();
    return (dis1, p1.clone(), u1, dis2, p2.clone(), u2)
}

pub fn pt_dis_to_line_and_seg(pt: &Vec<f64>, a: &Vec<f64>, b: &Vec<f64>) -> (f64, DVector<f64>, f64, f64, DVector<f64>, f64) {
    pt_dis_to_line_and_seg_dvecs(&NalgebraConversions::vec_to_dvector(&pt), &NalgebraConversions::vec_to_dvector(&a), &NalgebraConversions::vec_to_dvector(&b))
}

pub fn area_of_triangle(a: &Vector2<f64>, b: &Vector2<f64>, c: &Vector2<f64>) -> f64 {
    area_of_triangle_from_sidelengths((a-b).norm(), (b-c).norm(), (a-c).norm())
}

pub fn area_of_triangle_from_sidelengths(a_len: f64, b_len: f64, c_len: f64) -> f64 {
    let s = (a_len + b_len + c_len) / 2.0;
    (s*(s-a_len)*(s-b_len)*(s-c_len)).sqrt()
}

pub fn quadratic_barycentric_coordinates(pt: &Vector2<f64>, v1: &Vector2<f64>, v2: &Vector2<f64>, v3: &Vector2<f64>, v4: &Vector2<f64>)  -> (f64, f64, f64, f64) {
    let a = v1 - pt;
    let b = v2 - v1;
    let c = v4 - v1;
    let d = v1 - v2 - v4 + v3;

    let a3 = Vector3::new(a[0], a[1], 0.0);
    let b3 = Vector3::new(b[0], b[1], 0.0);
    let c3 = Vector3::new(c[0], c[1], 0.0);
    let d3 = Vector3::new(d[0], d[1], 0.0);

    let a_cross = c3.cross(&d3);
    let b_cross = c3.cross(&b3) + a3.cross(&d3);
    let c_cross = a3.cross(&b3);

    let aa = a_cross[2];
    let bb = b_cross[2];
    let cc = c_cross[2];

    let u1;
    let u2;

    if aa.abs() < 0.00000000000001 {
        u1 = -cc / bb;
        u2 = u1;
    } else {
        if bb*bb - 4.0 * aa * cc > 0.0 {
            u1 = (-bb + (bb*bb - 4.0*aa*cc).sqrt()) / 2.0*aa;
            u2 = (-bb - (bb*bb - 4.0*aa*cc).sqrt()) / 2.0*aa;
        } else {
            u1 = -1000.0;
            u2 = u1;
        }
    }

    let mut mu = -100000.0;
    if u1 >= 0.0 && u1 <= 1.0 {
        mu = u1;
    }
    if u2 >= 0.0 && u2 <= 1.0 {
        mu = u2;
    }

    let a_cross = b3.cross(&d3);
    let b_cross = b3.cross(&c3) + a3.cross(&d3);
    let c_cross = a3.cross(&c3);

    let aa = a_cross[2];
    let bb = b_cross[2];
    let cc = c_cross[2];

    let w1;
    let w2;

    if aa.abs() <  0.00000000000001 {
        w1 = -cc / bb;
        w2 = w1;
    } else {
        if bb*bb - 4.0 * aa * cc > 0.0 {
            w1 = (-bb + (bb*bb - 4.0*aa*cc).sqrt()) / 2.0*aa;
            w2 = (-bb - (bb*bb - 4.0*aa*cc).sqrt()) / 2.0*aa;
        } else {
            w1 = -1000.0;
            w2 = w1;
        }
    }

    let mut lambda = -10000.0;
    if w1 >= 0.0 && w1 <= 1.0 {
        lambda = w1;
    }
    if w2 >= 0.0 && w2 <= 1.0 {
        lambda = w2;
    }

    let alpha1 = (1.0-mu) * (1.0-lambda);
    let alpha2 = lambda * (1.0-mu);
    let alpha3 = mu * lambda;
    let alpha4 = (1.0-lambda) * mu;

    (alpha1, alpha2, alpha3, alpha4)
}

pub fn signed_volume_of_triangle(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> f64 {
    let v321 = c[0]*b[1]*a[2];
    let v231 = b[0]*c[1]*a[2];
    let v312 = c[0]*a[1]*b[2];
    let v132 = a[0]*c[1]*b[2];
    let v213 = b[0]*a[1]*c[2];
    let v123 = a[0]*b[1]*c[2];

    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123);
}

pub fn get_points_around_circle(center_point: &DVector<f64>, rotation_axis: &DVector<f64>, circle_radius: f64, num_samples: usize, seed: Option<u64>) -> Vec<DVector<f64>> {
    assert!(center_point.len() > 2);
    assert_eq!(center_point.len(), rotation_axis.len());

    let basis = get_orthonormal_basis(rotation_axis, 3, seed);

    let local_x = basis[1].clone();
    let local_y = basis[2].clone();

    let step_size = 2.0*std::f64::consts::PI / (num_samples as f64);
    let range = get_range(0.0, 2.0*std::f64::consts::PI, step_size);

    let mut out_points: Vec<DVector<f64>> = Vec::new();

    let l = range.len();
    for i in 0..l {
        let point = circle_radius * range[i].cos() * &local_x + circle_radius * range[i].sin() * &local_y;
        out_points.push(&point + center_point);
    }

    return out_points;
}

pub fn get_points_around_horizontal_circle(center_point: &DVector<f64>, starting_point: &DVector<f64>, radians: f64, num_samples: usize) -> Vec<DVector<f64>> {
    assert!(center_point.len() > 2);
    assert_eq!(center_point.len(), starting_point.len());

    let mut out_points: Vec<DVector<f64>> = Vec::new();

    let mut curr_point = starting_point.clone();

    let up_vector = &NalgebraConversions::vec_to_dvector(&vec![0.,0.,1.]);

    let radius = (&NalgebraConversions::vec_to_dvector(&vec![center_point[0], center_point[1]]) - &NalgebraConversions::vec_to_dvector(&vec![starting_point[0], starting_point[1]])).norm();

    // let radius = (center_point - starting_point).norm();

    let angular_step_size = radians / (num_samples as f64);
    let step_size = (2.0 * radius * radius * (1.0 - angular_step_size.cos())).sqrt();
    // let step_size = angular_step_size;

    for _ in 0..num_samples {
        out_points.push(curr_point.clone());
        let facing_vector = center_point - &curr_point;
        let facing_vector_normalized = &facing_vector / facing_vector.norm();
        let dir = facing_vector_normalized.cross(&up_vector);
        let dir_normalized = &dir / dir.norm();
        curr_point = &curr_point + (step_size * &dir_normalized);
    }

    return out_points;
}