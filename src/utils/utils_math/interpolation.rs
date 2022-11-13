use nalgebra::DVector;

pub struct SimpleInterpolationUtils;
impl SimpleInterpolationUtils {
    pub fn linear_interpolation(start_point: &DVector<f64>,
                                end_point: &DVector<f64>,
                                mode: &LinearInterpolationMode) -> Vec<DVector<f64>> {
        let mut out_vec = vec![];

        let mut dir = end_point - start_point;
        let n = dir.norm();
        dir /= n;

        match mode {
            LinearInterpolationMode::FixedNumKnots { num_knots } => {
                let spacing_between_knots = n / (*num_knots as f64 - 1.0);

                let mut curr_point = start_point.clone();
                for _ in 0..*num_knots-1 {
                    out_vec.push(curr_point.clone());
                    curr_point = &curr_point + spacing_between_knots * &dir;
                }
                out_vec.push(end_point.clone());
            }
            LinearInterpolationMode::FixedL2NormSpacing { spacing } => {
                if *spacing > n { out_vec.push(start_point.clone()); out_vec.push(end_point.clone()); }
                else {
                    let num_knots = (n / *spacing).ceil() as usize;

                    let mut curr_point = start_point.clone();
                    for _ in 0..num_knots {
                        out_vec.push(curr_point.clone());
                        curr_point = &curr_point + *spacing * &dir;
                    }
                    out_vec.push(end_point.clone());
                }
            }
        }

        out_vec
    }
}

pub enum LinearInterpolationMode {
    FixedNumKnots { num_knots: usize },
    FixedL2NormSpacing { spacing: f64 }
}



pub fn get_range(range_start: f64, range_stop: f64, step_size: f64) -> Vec<f64> {
    let mut out_range = Vec::new();
    out_range.push(range_start);
    let mut last_added_val = range_start;

    while !( (range_stop - last_added_val).abs() < step_size ) {
        if range_stop > range_start {
            last_added_val = last_added_val + step_size;
        } else {
            last_added_val = last_added_val - step_size;
        }
        out_range.push(last_added_val);
    }

    out_range.push(range_stop);

    out_range
}