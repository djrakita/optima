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
    pub fn normal_samples(means_and_standard_deviations: &Vec<(f64, f64)>) -> Vec<f64> {
        let mut out_vec = vec![];
        let mut rng = rand::thread_rng();
        for (mean, standard_deviation) in means_and_standard_deviations {
            let distribution = Normal::new(*mean, *standard_deviation).expect("error");
            out_vec.push(distribution.sample(&mut rng));
        }
        out_vec
    }
}