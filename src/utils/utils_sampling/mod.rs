use rand::Rng;
use rand_distr::{Normal, Distribution};
use rand_chacha::ChaCha20Rng;
use rand::prelude::*;

pub fn get_rng(random_seed: Option<u64>) -> ChaCha20Rng {
    match random_seed {
        None => { ChaCha20Rng::from_entropy() }
        Some(seed) => { ChaCha20Rng::seed_from_u64(seed) }
    }
}

pub struct SimpleSamplers;
impl SimpleSamplers {
    pub fn uniform_samples(bounds: &Vec<(f64, f64)>, seed: Option<u64>) -> Vec<f64> {
        let mut out_vec = vec![];
        let mut rng = get_rng(seed);
        for b in bounds {
            if b.0 == b.1 {
                out_vec.push(b.0);
            } else {
                out_vec.push(rng.gen_range(b.0..b.1));
            }
        }
        out_vec
    }
    pub fn uniform_sample(bounds: (f64, f64), seed: Option<u64>) -> f64 {
        let mut rng = get_rng(seed);
        return rng.gen_range(bounds.0..bounds.1)
    }
    pub fn normal_samples(means_and_standard_deviations: &Vec<(f64, f64)>, seed: Option<u64>) -> Vec<f64> {
        let mut out_vec = vec![];
        let mut rng = get_rng(seed);
        for (mean, standard_deviation) in means_and_standard_deviations {
            let distribution = Normal::new(*mean, *standard_deviation).expect("error");
            out_vec.push(distribution.sample(&mut rng));
        }
        out_vec
    }
    pub fn uniform_samples_i32(bounds: &Vec<(i32, i32)>, seed: Option<u64>) -> Vec<i32> {
        let bounds: Vec<(f64, f64)> = bounds.iter().map(|x| (x.0 as f64, x.1 as f64) ).collect();
        let float_samples = Self::uniform_samples(&bounds, seed);
        return float_samples.iter().map(|x| x.round() as i32).collect();
    }
}