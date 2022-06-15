use std::sync::Mutex;
use std::time::Duration;
use nalgebra::DVector;
use optimization_engine::{constraints, Optimizer, Problem, SolverError};
use optimization_engine::alm::{AlmCache, AlmFactory, AlmOptimizer, AlmProblem, NO_JACOBIAN_MAPPING, NO_MAPPING, NO_SET};
use optimization_engine::core::ExitStatus;
use optimization_engine::panoc::{PANOCCache, PANOCOptimizer};
use crate::optima_tensor_function::{OptimaTensor, OptimaTensorFunction, OTFImmutVars, OTFMutVars};
use crate::optima_tensor_function::standard_functions::{OTFComposition, OTFMaxZero, OTFWeightedSum};
#[cfg(not(target_arch = "wasm32"))]
use nlopt::*;

#[derive(Clone)]
pub enum NonlinearOptimizer {
    OpEn(OpEnNonlinearOptimizer),
    #[cfg(not(target_arch = "wasm32"))]
    Nlopt(NLoptNonlinearOptimizer)
}
impl NonlinearOptimizer {
    pub fn new<F: OptimaTensorFunction + Clone + 'static>(cost: F, problem_size: usize, t: NonlinearOptimizerType) -> Self {
        return match t {
            NonlinearOptimizerType::OpEn => { Self::OpEn(OpEnNonlinearOptimizer::new(cost, problem_size)) }
            #[cfg(not(target_arch = "wasm32"))]
            NonlinearOptimizerType::NloptSLSQP => { Self::Nlopt(NLoptNonlinearOptimizer::new_slsqp(cost, problem_size)) }
        }
    }
    pub fn add_equality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        match self {
            NonlinearOptimizer::OpEn(n) => { n.add_equality_constraint(f); }
            #[cfg(not(target_arch = "wasm32"))]
            NonlinearOptimizer::Nlopt(_) => {}
        }
    }
    pub fn add_less_than_zero_inequality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        match self {
            NonlinearOptimizer::OpEn(n) => { n.add_less_than_zero_inequality_constraint(f); }
            #[cfg(not(target_arch = "wasm32"))]
            NonlinearOptimizer::Nlopt(_) => {}
        }
    }
    pub fn set_bounds(&mut self, bounds: Vec<(f64, f64)>) {
        match self {
            NonlinearOptimizer::OpEn(n) => { n.set_bounds(bounds); }
            #[cfg(not(target_arch = "wasm32"))]
            NonlinearOptimizer::Nlopt(_) => {}
        }
    }
    pub fn optimize(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters) -> OptimizerResult {
        return match self {
            NonlinearOptimizer::OpEn(n) => { n.optimize(init_condition, immut_vars, mut_vars, parameters) }
            #[cfg(not(target_arch = "wasm32"))]
            NonlinearOptimizer::Nlopt(_) => { todo!() }
        }
    }
}

#[derive(Clone, Debug)]
pub enum NonlinearOptimizerType {
    OpEn,
    #[cfg(not(target_arch = "wasm32"))]
    NloptSLSQP
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct OpEnNonlinearOptimizer {
    cost_function: Box<dyn OptimaTensorFunction>,
    constraint_function: Option<OTFWeightedSum>,
    problem_size: usize,
    bounds: (Vec<f64>, Vec<f64>),
}
impl OpEnNonlinearOptimizer {
    pub fn new<F: OptimaTensorFunction + Clone + 'static>(cost: F, problem_size: usize) -> Self {
        let mut lower_bounds = vec![];
        let mut upper_bounds = vec![];
        for _ in 0..problem_size { lower_bounds.push(-f64::INFINITY); upper_bounds.push(f64::INFINITY); }
        Self {
            cost_function: Box::new(cost),
            constraint_function: None,
            problem_size,
            bounds: (lower_bounds, upper_bounds)
        }
    }
    pub fn add_equality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        if self.constraint_function.is_none() {
            self.constraint_function = Some(OTFWeightedSum::new());
        }

        self.constraint_function.as_mut().unwrap().add_function(f, None);
    }
    pub fn add_less_than_zero_inequality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        if self.constraint_function.is_none() {
            self.constraint_function = Some(OTFWeightedSum::new());
        }

        let wrapped_f = OTFComposition::new(OTFMaxZero, f);

        self.constraint_function.as_mut().unwrap().add_function(wrapped_f, None);
    }
    pub fn set_bounds(&mut self, bounds: Vec<(f64, f64)>) {
        assert_eq!(self.problem_size, bounds.len());
        let mut lower_bounds = vec![];
        let mut upper_bounds = vec![];
        for b in bounds {
            lower_bounds.push(b.0);
            upper_bounds.push(b.1);
        }
        self.bounds = (lower_bounds, upper_bounds);
    }
    pub fn optimize(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters) -> OptimizerResult {
        return match self.constraint_function {
            None => { self.optimize_panoc(init_condition, immut_vars, mut_vars, parameters) }
            Some(_) => { self.optimize_alm(init_condition, immut_vars, mut_vars, parameters) }
        }
    }
    fn optimize_panoc(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters) -> OptimizerResult {
        let mut panoc_cache = PANOCCache::new(self.problem_size, 1e-5, 3);

        let mut_vars_mutex = Mutex::new(mut_vars);

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();
            let input = OptimaTensor::new_from_single_array(u);
            let res = self.cost_function.derivative(&input, immut_vars, *mut_vars, None).expect("error");
            let output = res.unwrap_tensor();
            let vectorized = output.vectorized_data();
            for (i, v) in vectorized.iter().enumerate() {
                grad[i] = *v;
            }
            Ok(())
        };
        let f = |u: &[f64], cost: &mut f64| -> Result<(), SolverError> {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();
            let input = OptimaTensor::new_from_single_array(u);
            let res = self.cost_function.call(&input, immut_vars, *mut_vars).expect("error");
            let output = res.unwrap_tensor();
            let val = output.unwrap_scalar();
            *cost = val;
            Ok(())
        };

        let bounds = constraints::Rectangle::new(Some(&self.bounds.0), Some(&self.bounds.1));

        let problem = Problem::new(&bounds, df, f);

        let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache);
        if let Some(a) = &parameters.max_time { panoc = panoc.with_max_duration(a.clone()); }
        if let Some(a) = &parameters.max_iterations { panoc = panoc.with_max_iter(a.clone()); }

        let mut u = init_condition.vectorized_data().to_vec();
        let status = panoc.solve(&mut u).unwrap();

        let open_result = OpEnResult {
            x_min: OptimaTensor::new_from_vector(DVector::from_vec(u)),
            exit_status: status.exit_status(),
            num_outer_iterations: 0,
            num_inner_iterations: status.iterations(),
            solve_time: status.solve_time(),
            cost: status.cost_value()
        };

        return OptimizerResult::OpEn(open_result);
    }
    fn optimize_alm(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters) -> OptimizerResult {
        let panoc_cache = PANOCCache::new(self.problem_size, 1e-5, 3);
        let mut alm_cache = AlmCache::new(panoc_cache, 0, match self.constraint_function {
            None => { 0 }
            Some(_) => { 1 }
        });

        let mut_vars_mutex = Mutex::new(mut_vars);

        let bounds = constraints::Rectangle::new(Some(&self.bounds.0), Some(&self.bounds.1));

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();
            let input = OptimaTensor::new_from_single_array(u);
            let res = self.cost_function.derivative(&input, immut_vars, *mut_vars, None).expect("error");
            let output = res.unwrap_tensor();
            let vectorized = output.vectorized_data();
            for (i, v) in vectorized.iter().enumerate() {
                grad[i] = *v;
            }
            Ok(())
        };
        let f = |u: &[f64], cost: &mut f64| -> Result<(), SolverError> {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();
            let input = OptimaTensor::new_from_single_array(u);
            let res = self.cost_function.call(&input, immut_vars, *mut_vars).expect("error");
            let output = res.unwrap_tensor();
            let val = output.unwrap_scalar();
            *cost = val;
            Ok(())
        };
        let f2 = | u: &[f64], f2u: &mut [f64] | -> Result<(), SolverError> {
            if let Some(constraint_function) = &self.constraint_function {
                let mut mut_vars = mut_vars_mutex.lock().unwrap();
                let input = OptimaTensor::new_from_single_array(u);
                let res = constraint_function.call(&input, immut_vars, *mut_vars).expect("error");
                let output = res.unwrap_tensor();
                let val = output.unwrap_scalar();
                f2u[0] = val;
            }
            Ok(())
        };
        let f2_jacobian_product = |u: &[f64], d: &[f64], res: &mut [f64]| -> Result<(), SolverError> {
            if let Some(constraint_function) = &self.constraint_function {
                let mut mut_vars = mut_vars_mutex.lock().unwrap();
                let input = OptimaTensor::new_from_single_array(u);
                let result = constraint_function.derivative(&input, immut_vars, *mut_vars, None).expect("error");
                let output = result.unwrap_tensor();
                let vectorized_data = output.vectorized_data();
                for (i, v) in vectorized_data.iter().enumerate() {
                    res[i] = *v * d[i];
                }
            }
            Ok(())
        };

        let factory = AlmFactory::new(
            f,
            df,
            NO_MAPPING,
            NO_JACOBIAN_MAPPING,
            match self.constraint_function {
                None => { None }
                Some(_) => { Some(f2) }
            },
            match self.constraint_function {
                None => { None }
                Some(_) => { Some(f2_jacobian_product) }
            },
            NO_SET,
            match self.constraint_function {
                None => { 0 }
                Some(_) => { 1 }
            },
        );

        let alm_problem = AlmProblem::new(
            bounds,
            NO_SET,
            NO_SET,
            |u: &[f64], xi: &[f64], cost: &mut f64| -> Result<(), SolverError> {
                factory.psi(u, xi, cost)
            },
            |u: &[f64], xi: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
                factory.d_psi(u, xi, grad)
            },
            NO_MAPPING,
            match self.constraint_function {
                None => { None }
                Some(_) => { Some(f2) }
            },
            0,
            match self.constraint_function {
                None => { 0 }
                Some(_) => { 1 }
            }
        );

        let mut alm_optimizer = AlmOptimizer::new(&mut alm_cache, alm_problem);
        if let Some(a) = &parameters.max_time { alm_optimizer = alm_optimizer.with_max_duration(a.clone()); }
        if let Some(a) = &parameters.max_iterations { alm_optimizer = alm_optimizer.with_max_inner_iterations(a.clone()); }
        if let Some(a) = &parameters.max_outer_iterations { alm_optimizer = alm_optimizer.with_max_outer_iterations(a.clone()); }

        let mut u = init_condition.vectorized_data().to_vec();
        let solver_result = alm_optimizer.solve(&mut u);
        let r = solver_result.unwrap();

        let open_result = OpEnResult {
            x_min: OptimaTensor::new_from_vector(DVector::from_vec(u)),
            exit_status: r.exit_status(),
            num_outer_iterations: r.num_outer_iterations(),
            num_inner_iterations: r.num_inner_iterations(),
            solve_time: r.solve_time(),
            cost: r.cost()
        };

        return OptimizerResult::OpEn(open_result);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct NLoptNonlinearOptimizer {
    algorithm: Algorithm,
    cost_function: Box<dyn OptimaTensorFunction>,
    equality_constraints: Vec<Box<dyn OptimaTensorFunction>>,
    inequality_constraints: Vec<Box<dyn OptimaTensorFunction>>,
    problem_size: usize,
    bounds: Option<(Vec<f64>, Vec<f64>)>
}
#[cfg(not(target_arch = "wasm32"))]
impl NLoptNonlinearOptimizer {
    pub fn new<F: OptimaTensorFunction + 'static>(cost: F, problem_size: usize, algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            cost_function: Box::new(cost),
            equality_constraints: vec![],
            inequality_constraints: vec![],
            problem_size,
            bounds: None
        }
    }
    pub fn new_slsqp<F: OptimaTensorFunction + 'static>(cost: F, problem_size: usize) -> Self {
        Self::new(cost, problem_size, Algorithm::Slsqp)
    }
    pub fn add_equality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        self.equality_constraints.push(Box::new(f));
    }
    pub fn add_less_than_zero_inequality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        self.inequality_constraints.push(Box::new(f));
    }
    pub fn set_bounds(&mut self, bounds: Vec<(f64, f64)>) {
        assert_eq!(self.problem_size, bounds.len());
        let mut lower_bounds = vec![];
        let mut upper_bounds = vec![];
        for b in bounds {
            lower_bounds.push(b.0);
            upper_bounds.push(b.1);
        }
        self.bounds = Some((lower_bounds, upper_bounds));
    }
    pub fn optimize(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters) -> OptimizerResult {
        let start = instant::Instant::now();
        let mut_vars_mutex = Mutex::new(mut_vars);

        let obj_f = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();

            let input = OptimaTensor::new_from_single_array(x);
            let res = self.cost_function.call(&input, immut_vars, *mut_vars).expect("error");

            let output = res.unwrap_tensor();
            let val = output.unwrap_scalar();

            if let Some(gradient) = _gradient {
                let res = self.cost_function.derivative(&input, immut_vars, *mut_vars, None).expect("error");
                let output = res.unwrap_tensor();
                let vectorized = output.vectorized_data();
                for (i, v) in vectorized.iter().enumerate() { gradient[i] = *v; }
            }

            return val;
        };
        let mut nlopt = Nlopt::new(self.algorithm.clone(), self.problem_size, obj_f, Target::Minimize, ());
        for c in &self.equality_constraints {
            let eq_con = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                let mut mut_vars = mut_vars_mutex.lock().unwrap();
                let input = OptimaTensor::new_from_single_array(x);
                let res = c.call(&input, immut_vars, *mut_vars).expect("error");
                let output = res.unwrap_tensor();
                let val = output.unwrap_scalar();

                if let Some(gradient) = _gradient {
                    let res = c.derivative(&input, immut_vars, *mut_vars, None).expect("error");
                    let output = res.unwrap_tensor();
                    let vectorized = output.vectorized_data();
                    for (i, v) in vectorized.iter().enumerate() { gradient[i] = *v; }
                }

                return val;
            };
            nlopt.add_equality_constraint(eq_con, (), 0.000001).expect("error");
        }
        for c in &self.inequality_constraints {
            let ineq_con = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                let mut mut_vars = mut_vars_mutex.lock().unwrap();
                let input = OptimaTensor::new_from_single_array(x);
                let res = c.call(&input, immut_vars, *mut_vars).expect("error");
                let output = res.unwrap_tensor();
                let val = output.unwrap_scalar();

                if let Some(gradient) = _gradient {
                    let res = c.derivative(&input, immut_vars, *mut_vars, None).expect("error");
                    let output = res.unwrap_tensor();
                    let vectorized = output.vectorized_data();
                    for (i, v) in vectorized.iter().enumerate() { gradient[i] = *v; }
                }

                return val;
            };
            nlopt.add_inequality_constraint(ineq_con, (), 0.000001).expect("error");
        }

        if let Some(bounds) = &self.bounds {
            nlopt.set_lower_bounds(&bounds.0).expect("error");
            nlopt.set_upper_bounds(&bounds.1).expect("error");
        }
        if let Some(a) = &parameters.max_time { nlopt.set_maxtime(a.as_secs_f64()).expect("error"); }
        if let Some(a) = &parameters.max_iterations { nlopt.set_maxeval(*a as u32).expect("error"); }

        nlopt.set_ftol_rel(0.0001).expect("error");
        nlopt.set_ftol_abs(0.0001).expect("error");
        nlopt.set_xtol_rel(0.0001).expect("error");

        let mut x = init_condition.vectorized_data().to_vec();
        let res = nlopt.optimize(&mut x);
        match res {
            Ok(r) => {
                let output = NloptResult {
                    x_min: OptimaTensor::new_from_vector(DVector::from_vec(x)),
                    solve_time: start.elapsed(),
                    cost: r.1
                };
                return OptimizerResult::Nlopt(output);
            }
            Err(e) => { panic!("Optimization failed: {:?}", e) }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub enum OptimizerResult {
    OpEn(OpEnResult),
    Nlopt(NloptResult)
}
impl OptimizerResult {
    pub fn unwrap_open_result(&self) -> &OpEnResult {
        match self {
            OptimizerResult::OpEn(o) => { return o; }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_nlopt_result(&self) -> &NloptResult {
        match self {
            OptimizerResult::Nlopt(o) => { return o; }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_x_min(&self) -> &OptimaTensor {
        return match self {
            OptimizerResult::OpEn(r) => { r.x_min() }
            OptimizerResult::Nlopt(r) => { r.x_min() }
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpEnResult {
    x_min: OptimaTensor,
    exit_status: ExitStatus,
    num_outer_iterations: usize,
    num_inner_iterations: usize,
    solve_time: Duration,
    cost: f64
}
impl OpEnResult {
    pub fn x_min(&self) -> &OptimaTensor {
        &self.x_min
    }
    pub fn exit_status(&self) -> ExitStatus {
        self.exit_status
    }
    pub fn num_outer_iterations(&self) -> usize {
        self.num_outer_iterations
    }
    pub fn num_inner_iterations(&self) -> usize {
        self.num_inner_iterations
    }
    pub fn solve_time(&self) -> Duration {
        self.solve_time
    }
    pub fn cost(&self) -> f64 {
        self.cost
    }
}

#[derive(Clone, Debug)]
pub struct NloptResult {
    x_min: OptimaTensor,
    solve_time: Duration,
    cost: f64
}
impl NloptResult {
    pub fn x_min(&self) -> &OptimaTensor {
        &self.x_min
    }
    pub fn solve_time(&self) -> Duration {
        self.solve_time
    }
    pub fn cost(&self) -> f64 {
        self.cost
    }
}

#[derive(Clone, Debug)]
pub struct OptimizerParameters {
    max_time: Option<Duration>,
    max_iterations: Option<usize>,
    max_outer_iterations: Option<usize>
}
impl OptimizerParameters {
    pub fn new_empty() -> Self {
        Self::default()
    }
    pub fn set_max_time(&mut self, max_time: Duration) {
        self.max_time = Some(max_time);
    }
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = Some(max_iterations);
    }
    pub fn set_max_outer_iterations(&mut self, max_outer_iterations: usize) {
        self.max_outer_iterations = Some(max_outer_iterations)
    }
}
impl Default for OptimizerParameters {
    fn default() -> Self {
        Self {
            max_time: None,
            max_iterations: None,
            max_outer_iterations: None
        }
    }
}