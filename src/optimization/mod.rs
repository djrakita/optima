use std::sync::Mutex;
use std::time::Duration;
use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use optimization_engine::{constraints, Optimizer, Problem, SolverError};
use optimization_engine::alm::{AlmCache, AlmFactory, AlmOptimizer, AlmProblem, NO_JACOBIAN_MAPPING, NO_MAPPING, NO_SET};
use optimization_engine::core::ExitStatus;
use optimization_engine::panoc::{PANOCCache, PANOCOptimizer};
use crate::optima_tensor_function::{OptimaTensor, OptimaTensorFunction, OTFImmutVars, OTFMutVars};
use crate::optima_tensor_function::standard_functions::{OTFAddScalar, OTFComposition, OTFMaxZero, OTFMultiplyByScalar, OTFWeightedSum};
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "nlopt_optimization")]
use nlopt::*;
use crate::utils::utils_console::OptimaDebug;

#[derive(Clone)]
pub enum NonlinearOptimizer {
    OpEn(OpEnNonlinearOptimizer),
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    Nlopt(NLoptNonlinearOptimizer)
}
impl NonlinearOptimizer {
    pub fn new(problem_size: usize, t: NonlinearOptimizerType) -> Self {
        return match t {
            NonlinearOptimizerType::OpEn => { Self::OpEn(OpEnNonlinearOptimizer::new(problem_size)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptGlobalNonderivativeDirect => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::DIRECT)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptGlobalNonderivativeDIRECTL => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::DIRECTL)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptGlobalNonderivativeCRS => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::CRS)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptGlobalDerivativeSTOGO => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::STOGO)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptGlobalDerivativeSTOGORAND => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::STOGORAND)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptGlobalNonderivativeISRES => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::ISRES)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptGlobalNonderivativeESCH => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::ESCH)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptLocalNonderivativeCOBYLA => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::COBYLA)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptLocalNonderivativeBOBYQA => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::BOBYQA)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptLocalDerivativeSLSQP => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::SLSQP)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptLocalDerivataiveMMA => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::MMA)) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizerType::NloptLocalDerivataiveCCSAQ => { Self::Nlopt(NLoptNonlinearOptimizer::new(problem_size, NloptAlgorithmWrapper::CCSAQ)) }
        }
    }
    pub fn add_cost_term<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F, weight: Option<f64>) {
        match self {
            NonlinearOptimizer::OpEn(n) => { n.add_cost_term(f, weight); }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizer::Nlopt(n) => { n.add_cost_term(f, weight); }
        }
    }
    pub fn add_equality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        match self {
            NonlinearOptimizer::OpEn(n) => { n.add_equality_constraint(f); }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizer::Nlopt(n) => { n.add_equality_constraint(f); }
        }
    }
    pub fn add_less_than_zero_inequality_constraint<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F) {
        match self {
            NonlinearOptimizer::OpEn(n) => { n.add_less_than_zero_inequality_constraint(f); }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizer::Nlopt(n) => { n.add_less_than_zero_inequality_constraint(f); }
        }
    }
    pub fn add_term_generic<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F, specification: &OptimizationTermSpecification) {
        match specification {
            OptimizationTermSpecification::DoNotInclude => {  }
            OptimizationTermSpecification::Include { optimization_assignment } => {
                match optimization_assignment {
                    OptimizationTermAssignment::Objective { weight } => {
                        self.add_cost_term(f, Some(*weight));
                    }
                    OptimizationTermAssignment::EqualityConstraint { must_equal } => {
                        let c = OTFComposition::new(OTFAddScalar { scalar: -*must_equal }, f);
                        self.add_equality_constraint(c);
                    }
                    OptimizationTermAssignment::LTInequalityConstraint { must_be_less_than } => {
                        let c = OTFComposition::new(OTFAddScalar { scalar: -*must_be_less_than }, f);
                        self.add_less_than_zero_inequality_constraint(c);
                    }
                    OptimizationTermAssignment::GTInequalityConstraint { must_be_greater_than } => {
                        let c = OTFComposition::new(OTFAddScalar { scalar: -*must_be_greater_than }, f);
                        let cc = OTFComposition::new(OTFMultiplyByScalar { scalar: -1.0 }, c);
                        self.add_less_than_zero_inequality_constraint(cc);
                    }
                }
            }
        }
    }
    pub fn set_bounds(&mut self, bounds: Vec<(f64, f64)>) {
        match self {
            NonlinearOptimizer::OpEn(n) => { n.set_bounds(bounds); }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizer::Nlopt(_) => {}
        }
    }
    pub fn optimize(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters, debug: OptimaDebug) -> OptimizerResult {
        return match self {
            NonlinearOptimizer::OpEn(n) => { n.optimize(init_condition, immut_vars, mut_vars, parameters, debug.clone()) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizer::Nlopt(n) => { n.optimize(init_condition, immut_vars, mut_vars, parameters, debug.clone()) }
        }
    }
    pub fn cost(&self) -> Box<&dyn OptimaTensorFunction> {
        return match self {
            NonlinearOptimizer::OpEn(o) => { Box::new(&o.cost_function) }
            #[cfg(not(target_arch = "wasm32"))]
            #[cfg(feature = "nlopt_optimization")]
            NonlinearOptimizer::Nlopt(o) => { Box::new(&o.cost_function) }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NonlinearOptimizerType {
    OpEn,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptGlobalNonderivativeDirect,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptGlobalNonderivativeDIRECTL,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptGlobalNonderivativeCRS,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptGlobalDerivativeSTOGO,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptGlobalDerivativeSTOGORAND,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptGlobalNonderivativeISRES,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptGlobalNonderivativeESCH,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptLocalNonderivativeCOBYLA,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptLocalNonderivativeBOBYQA,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptLocalDerivativeSLSQP,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptLocalDerivataiveMMA,
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "nlopt_optimization")]
    NloptLocalDerivataiveCCSAQ
}
impl Default for NonlinearOptimizerType {
    fn default() -> Self {
        Self::OpEn
    }
}

#[derive(Clone, Debug)]
#[allow(unused_must_use)]
#[cfg(not(target_arch = "wasm32"))]
pub enum NloptAlgorithmWrapper {
    /// Global, Non-derivative
    DIRECT,
    /// Global, Non-derivative
    DIRECTL,
    /// Global, Non-derivative
    CRS,
    /// Global, derivative
    STOGO,
    /// Global, derivative
    STOGORAND,
    /// Global, non-derivative
    ISRES,
    /// Global, non-derivative
    ESCH,
    /// Local, non-derivative
    COBYLA,
    /// Local, non-derivative
    BOBYQA,
    /// Local, derivative
    SLSQP,
    /// Local, derivative
    MMA,
    /// Local, derivative
    CCSAQ,
}
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "nlopt_optimization")]
impl NloptAlgorithmWrapper {
    fn map_to_algorithm(&self) -> Algorithm {
        match self {
            NloptAlgorithmWrapper::DIRECT => { Algorithm::Direct }
            NloptAlgorithmWrapper::DIRECTL => { Algorithm::DirectL }
            NloptAlgorithmWrapper::CRS => { Algorithm::Crs2Lm }
            NloptAlgorithmWrapper::STOGO => { Algorithm::StoGo }
            NloptAlgorithmWrapper::STOGORAND => { Algorithm::StoGoRand }
            NloptAlgorithmWrapper::ISRES => { Algorithm::Isres }
            NloptAlgorithmWrapper::ESCH => { Algorithm::Esch }
            NloptAlgorithmWrapper::COBYLA => { Algorithm::Cobyla }
            NloptAlgorithmWrapper::BOBYQA => { Algorithm::Bobyqa }
            NloptAlgorithmWrapper::SLSQP => { Algorithm::Slsqp }
            NloptAlgorithmWrapper::MMA => { Algorithm::Mma }
            NloptAlgorithmWrapper::CCSAQ => { Algorithm::Ccsaq }
        }
    }
    fn handles_equality_constraints(&self) -> bool {
        match self {
            NloptAlgorithmWrapper::DIRECT => { false  }
            NloptAlgorithmWrapper::DIRECTL => { false }
            NloptAlgorithmWrapper::CRS => { false }
            NloptAlgorithmWrapper::STOGO => { false }
            NloptAlgorithmWrapper::STOGORAND => { false }
            NloptAlgorithmWrapper::ISRES => { true }
            NloptAlgorithmWrapper::ESCH => { false }
            NloptAlgorithmWrapper::COBYLA => { false }
            NloptAlgorithmWrapper::BOBYQA => { false }
            NloptAlgorithmWrapper::SLSQP => { true }
            NloptAlgorithmWrapper::MMA => { false }
            NloptAlgorithmWrapper::CCSAQ => { false }
        }
    }
    fn handles_inequality_constraints(&self) -> bool {
        match self {
            NloptAlgorithmWrapper::DIRECT => { false  }
            NloptAlgorithmWrapper::DIRECTL => { false }
            NloptAlgorithmWrapper::CRS => { false }
            NloptAlgorithmWrapper::STOGO => { false }
            NloptAlgorithmWrapper::STOGORAND => { false }
            NloptAlgorithmWrapper::ISRES => { true }
            NloptAlgorithmWrapper::ESCH => { false }
            NloptAlgorithmWrapper::COBYLA => { true }
            NloptAlgorithmWrapper::BOBYQA => { false }
            NloptAlgorithmWrapper::SLSQP => { true }
            NloptAlgorithmWrapper::MMA => { true }
            NloptAlgorithmWrapper::CCSAQ => { true }
        }
    }
    fn is_global(&self) -> bool {
        match self {
            NloptAlgorithmWrapper::DIRECT => { true }
            NloptAlgorithmWrapper::DIRECTL => { true }
            NloptAlgorithmWrapper::CRS => { true }
            NloptAlgorithmWrapper::STOGO => { true }
            NloptAlgorithmWrapper::STOGORAND => { true }
            NloptAlgorithmWrapper::ISRES => { true }
            NloptAlgorithmWrapper::ESCH => { true }
            NloptAlgorithmWrapper::COBYLA => { false }
            NloptAlgorithmWrapper::BOBYQA => { false }
            NloptAlgorithmWrapper::SLSQP => { false }
            NloptAlgorithmWrapper::MMA => { false }
            NloptAlgorithmWrapper::CCSAQ => { false }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct OpEnNonlinearOptimizer {
    cost_function: OTFWeightedSum,
    constraint_function: Option<OTFWeightedSum>,
    problem_size: usize,
    bounds: (Vec<f64>, Vec<f64>),
}
impl OpEnNonlinearOptimizer {
    pub fn new(problem_size: usize) -> Self {
        let mut lower_bounds = vec![];
        let mut upper_bounds = vec![];
        for _ in 0..problem_size { lower_bounds.push(-f64::INFINITY); upper_bounds.push(f64::INFINITY); }
        Self {
            cost_function: OTFWeightedSum::new(),
            constraint_function: None,
            problem_size,
            bounds: (lower_bounds, upper_bounds)
        }
    }
    pub fn add_cost_term<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F, weight: Option<f64>) {
        self.cost_function.add_function(f, weight);
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
    pub fn optimize(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters, debug: OptimaDebug) -> OptimizerResult {
        return match self.constraint_function {
            None => { self.optimize_panoc(init_condition, immut_vars, mut_vars, parameters, debug.clone()) }
            Some(_) => { self.optimize_alm(init_condition, immut_vars, mut_vars, parameters, debug.clone()) }
        }
    }
    fn optimize_panoc(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters, debug: OptimaDebug) -> OptimizerResult {
        let mut panoc_cache = PANOCCache::new(self.problem_size, 1e-5, 3);

        let mut_vars_mutex = Mutex::new(mut_vars);

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();
            let input = OptimaTensor::new_from_single_array(u);
            let res = self.cost_function.derivative(&input, immut_vars, *mut_vars, None, debug.clone()).expect("error");
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
            let res = self.cost_function.call(&input, immut_vars, *mut_vars, debug.clone()).expect("error");
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
        panoc = panoc.with_tolerance(parameters.open_tolerance);

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
    fn optimize_alm(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters, debug: OptimaDebug) -> OptimizerResult {
        let panoc_cache = PANOCCache::new(self.problem_size, 1e-5, 3);
        let mut alm_cache = AlmCache::new(panoc_cache, 0, match self.constraint_function {
            None => { 0 }
            Some(_) => { self.problem_size }
        });

        let mut_vars_mutex = Mutex::new(mut_vars);

        let bounds = constraints::Rectangle::new(Some(&self.bounds.0), Some(&self.bounds.1));

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();
            let input = OptimaTensor::new_from_single_array(u);
            let res = self.cost_function.derivative(&input, immut_vars, *mut_vars, None, debug.clone()).expect("error");
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
            let res = self.cost_function.call(&input, immut_vars, *mut_vars, debug.clone()).expect("error");
            let output = res.unwrap_tensor();
            let val = output.unwrap_scalar();
            *cost = val;
            Ok(())
        };
        let f2 = | u: &[f64], f2u: &mut [f64] | -> Result<(), SolverError> {
            if let Some(constraint_function) = &self.constraint_function {
                let mut mut_vars = mut_vars_mutex.lock().unwrap();
                let input = OptimaTensor::new_from_single_array(u);
                let res = constraint_function.call(&input, immut_vars, *mut_vars, debug.clone()).expect("error");
                let output = res.unwrap_tensor();
                let val = output.unwrap_scalar();
                f2u[0] = val;
            }
            Ok(())
        };
        // d is output from f2
        let f2_jacobian_product = |u: &[f64], d: &[f64], res: &mut [f64]| -> Result<(), SolverError> {
            if let Some(constraint_function) = &self.constraint_function {
                let mut mut_vars = mut_vars_mutex.lock().unwrap();
                let input = OptimaTensor::new_from_single_array(u);
                let result = constraint_function.derivative(&input, immut_vars, *mut_vars, None, OptimaDebug::False).expect("error");
                let output = result.unwrap_tensor();
                let vectorized_data = output.vectorized_data();
                for (i, v) in vectorized_data.iter().enumerate() {
                    res[i] = *v * d[0];
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
#[cfg(feature = "nlopt_optimization")]
#[derive(Clone)]
pub struct NLoptNonlinearOptimizer {
    algorithm: Algorithm,
    algorithm_wrapper: NloptAlgorithmWrapper,
    cost_function: OTFWeightedSum,
    equality_constraints: Vec<Box<dyn OptimaTensorFunction>>,
    inequality_constraints: Vec<Box<dyn OptimaTensorFunction>>,
    problem_size: usize,
    bounds: Option<(Vec<f64>, Vec<f64>)>
}
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "nlopt_optimization")]
impl NLoptNonlinearOptimizer {
    pub fn new(problem_size: usize, algorithm: NloptAlgorithmWrapper) -> Self {
        Self {
            algorithm: algorithm.map_to_algorithm(),
            algorithm_wrapper: algorithm,
            cost_function: OTFWeightedSum::new(),
            equality_constraints: vec![],
            inequality_constraints: vec![],
            problem_size,
            bounds: None
        }
    }
    pub fn add_cost_term<F: OptimaTensorFunction + Clone + 'static>(&mut self, f: F, weight: Option<f64>) {
        self.cost_function.add_function(f, weight);
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
    pub fn optimize(&mut self, init_condition: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, parameters: &OptimizerParameters, debug: OptimaDebug) -> OptimizerResult {
        let start = instant::Instant::now();
        let mut_vars_mutex = Mutex::new(mut_vars);

        let obj_f = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let mut mut_vars = mut_vars_mutex.lock().unwrap();

            let input = OptimaTensor::new_from_single_array(x);
            let res = self.cost_function.call(&input, immut_vars, *mut_vars, debug.clone()).expect("error");

            let output = res.unwrap_tensor();
            let val = output.unwrap_scalar();

            if let Some(gradient) = _gradient {
                let res = self.cost_function.derivative(&input, immut_vars, *mut_vars, None, debug.clone()).expect("error");
                let output = res.unwrap_tensor();
                let vectorized = output.vectorized_data();
                for (i, v) in vectorized.iter().enumerate() { gradient[i] = *v; }
            }

            return val;
        };

        let mut used_outerloop = false;
        let algorithm = if !self.equality_constraints.is_empty() && !self.algorithm_wrapper.handles_equality_constraints() {
            used_outerloop = true;
            Algorithm::Auglag
        } else if !self.inequality_constraints.is_empty() && !self.algorithm_wrapper.handles_inequality_constraints()  && self.algorithm_wrapper.handles_equality_constraints() {
            used_outerloop = true;
            Algorithm::AuglagEq
        } else if !self.inequality_constraints.is_empty() && !self.algorithm_wrapper.handles_inequality_constraints() && !self.algorithm_wrapper.handles_equality_constraints() {
            used_outerloop = true;
            Algorithm::Auglag
        } else {
            self.algorithm.clone()
        };

        let mut nlopt = Nlopt::new(algorithm, self.problem_size, obj_f, Target::Minimize, ());

        for c in &self.equality_constraints {
            let eq_con = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                let mut mut_vars = mut_vars_mutex.lock().unwrap();
                let input = OptimaTensor::new_from_single_array(x);
                let res = c.call(&input, immut_vars, *mut_vars, OptimaDebug::False).expect("error");
                let output = res.unwrap_tensor();
                let val = output.unwrap_scalar();

                if let Some(gradient) = _gradient {
                    let res = c.derivative(&input, immut_vars, *mut_vars, None, OptimaDebug::False).expect("error");
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
                let res = c.call(&input, immut_vars, *mut_vars, OptimaDebug::False).expect("error");
                let output = res.unwrap_tensor();
                let val = output.unwrap_scalar();

                if let Some(gradient) = _gradient {
                    let res = c.derivative(&input, immut_vars, *mut_vars, None, OptimaDebug::False).expect("error");
                    let output = res.unwrap_tensor();
                    let vectorized = output.vectorized_data();
                    for (i, v) in vectorized.iter().enumerate() { gradient[i] = *v; }
                }

                return val;
            };
            nlopt.add_inequality_constraint(ineq_con, (), 0.000001).expect("error");
        }

        if used_outerloop {
            let mut l = nlopt.get_local_optimizer(self.algorithm.clone());
            l.set_ftol_rel(0.0001).expect("error");
            l.set_ftol_abs(0.0001).expect("error");
            l.set_xtol_rel(0.0001).expect("error");
            nlopt.set_local_optimizer(l).expect("error");
        }

        if let Some(bounds) = &self.bounds {
            nlopt.set_lower_bounds(&bounds.0).expect("error");
            nlopt.set_upper_bounds(&bounds.1).expect("error");
        }
        if self.bounds.is_none() && self.algorithm_wrapper.is_global() {
            nlopt.set_lower_bounds(&vec![-10000.0; self.problem_size]).expect("error");
            nlopt.set_upper_bounds(&vec![10000.0; self.problem_size]).expect("error");
        }
        if let Some(a) = &parameters.max_time { nlopt.set_maxtime(a.as_secs_f64()).expect("error"); }
        if let Some(a) = &parameters.max_iterations { nlopt.set_maxeval(*a as u32).expect("error"); }

        nlopt.set_ftol_rel(parameters.nlopt_ftol_rel).expect("error");
        nlopt.set_ftol_abs(parameters.nlopt_ftol_abs).expect("error");
        nlopt.set_xtol_rel(parameters.nlopt_xtol_rel).expect("error");

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
    max_outer_iterations: Option<usize>,
    nlopt_ftol_rel: f64,
    nlopt_ftol_abs: f64,
    nlopt_xtol_rel: f64,
    open_tolerance: f64
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
    pub fn set_ftol_rel(&mut self, val: f64) {
        self.nlopt_ftol_rel = val;
    }
    pub fn set_ftol_abs(&mut self, val: f64) {
        self.nlopt_ftol_abs = val;
    }
    pub fn set_xtol_rel(&mut self, val: f64) {
        self.nlopt_xtol_rel = val;
    }
    pub fn set_open_tolerance(&mut self, tolerance: f64) {
        self.open_tolerance = tolerance;
    }
}
impl Default for OptimizerParameters {
    fn default() -> Self {
        Self {
            max_time: None,
            max_iterations: None,
            max_outer_iterations: None,
            nlopt_ftol_rel: 0.0001,
            nlopt_ftol_abs: 0.0001,
            nlopt_xtol_rel: 0.0001,
            open_tolerance: 0.0001
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimizationTermSpecification {
    DoNotInclude,
    Include { optimization_assignment: OptimizationTermAssignment }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimizationTermAssignment {
    Objective { weight: f64 },
    EqualityConstraint { must_equal: f64 },
    LTInequalityConstraint { must_be_less_than: f64 },
    GTInequalityConstraint { must_be_greater_than: f64 }
}
