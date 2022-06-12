use rand::Rng;
use rand_distr::{Normal, Distribution};

pub struct SimpleSamplers;
impl SimpleSamplers {
    pub fn uniform_samples(bounds: &Vec<(f64, f64)>) -> Vec<f64> {
        let mut out_vec = vec![];
        let mut rng = rand::thread_rng();
        for b in bounds {
            if b.0 == b.1 {
                out_vec.push(b.0);
            } else {
                out_vec.push(rng.gen_range(b.0..b.1));
            }
        }
        out_vec
    }
    pub fn uniform_sample(bounds: (f64, f64)) -> f64 {
        let mut rng = rand::thread_rng();
        return rng.gen_range(bounds.0..bounds.1)
    }
    pub fn normal_samples(means_and_standard_deviations: &Vec<(f64, f64)>) -> Vec<f64> {
        let mut out_vec = vec![];
        let mut rng = rand::thread_rng();
        for (mean, standard_deviation) in means_and_standard_deviations {
            let distribution = Normal::new(*mean, *standard_deviation).expect("error");
            out_vec.push(distribution.sample(&mut rng));
        }
        out_vec
    }
    pub fn uniform_samples_i32(bounds: &Vec<(i32, i32)>) -> Vec<i32> {
        let bounds: Vec<(f64, f64)> = bounds.iter().map(|x| (x.0 as f64, x.1 as f64) ).collect();
        let float_samples = Self::uniform_samples(&bounds);
        return float_samples.iter().map(|x| x.round() as i32).collect();
    }
}