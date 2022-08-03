use std::vec;
#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::nonlinear_optimization::{NonlinearOptimizer, NonlinearOptimizerType, OptimizationTermAssignment, OptimizationTermSpecification, OptimizerParameters};
use crate::optima_tensor_function::{OptimaTensor, OptimaTensorFunction, OTFDerivativeMode, OTFImmutVars, OTFImmutVarsObject, OTFImmutVarsObjectType, OTFMutVars};
use crate::optima_tensor_function::robotics_functions::{OTFRobotCollisionProximityGeneric, OTFRobotJointStateDerivative, OTFRobotLinkTFSpec, RobotCollisionProximityGenericParams};
use crate::optima_tensor_function::standard_functions::{OTFComposition, OTFQuadraticNormalizer, OTFTensorPNorm};
use crate::robot_set_modules::robot_set_joint_state_module::{RobotSetJointState, RobotSetJointStateType};
use crate::scenes::robot_geometric_shape_scene::{RobotGeometricShapeScene, RobotGeometricShapeScenePy, RobotGeometricShapeSceneQuery};
use crate::utils::utils_console::{optima_print, OptimaDebug, PrintColor, PrintMode};
use crate::utils::utils_robot::robot_generic_structures::{GenericRobotJointState, TimedGenericRobotJointState, TimedGenericRobotJointStateWindowMemoryContainer};
use crate::utils::utils_robot::robot_set_link_specification::{RobotLinkTFGoalCollection, RobotLinkTFGoalErrorReportCollection, RobotLinkTFSpec, RobotLinkTFSpecAndAllowableErrorCollection, RobotLinkTFSpecAndAllowableErrorCollectionPy, RobotLinkTFSpecCollection, RobotLinkTFSpecCollectionPy};
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
use crate::utils::utils_shape_geometry::geometric_shape::{LogCondition, StopCondition};
use crate::utils::utils_traits::ToAndFromRonString;

#[derive(Clone)]
pub struct OptimaIK {
    n: NonlinearOptimizer,
    immut_vars: OTFImmutVars,
    mut_vars: OTFMutVars,
    problem_size: usize,
    mode: OptimaIKMode
}
impl OptimaIK {
    pub fn new_empty(robot_geometric_shape_scene: RobotGeometricShapeScene, nonlinear_optimizer_type: NonlinearOptimizerType) -> Self {
        let problem_size = robot_geometric_shape_scene.robot_set().robot_set_joint_state_module().num_dofs();
        let mut n = NonlinearOptimizer::new(problem_size, nonlinear_optimizer_type);
        let bounds = robot_geometric_shape_scene.robot_set().robot_set_joint_state_module().get_joint_state_bounds(&RobotSetJointStateType::DOF);
        n.set_bounds(bounds);
        let mut immut_vars = OTFImmutVars::new();
        immut_vars.insert_or_replace(OTFImmutVarsObject::RobotGeometricShapeScene(robot_geometric_shape_scene));
        let mut_vars = OTFMutVars::new();

        Self {
            n,
            immut_vars,
            mut_vars,
            problem_size,
            mode: OptimaIKMode::Custom
        }
    }
    pub fn new_motion_ik(robot_geometric_shape_scene: RobotGeometricShapeScene, nonlinear_optimizer_type: NonlinearOptimizerType, robot_collision_proximity_params: RobotCollisionProximityGenericParams, init_state: &RobotSetJointState, init_time: f64) -> Self {
        let mut out_self = Self::new_empty(robot_geometric_shape_scene, nonlinear_optimizer_type);
        out_self.add_robot_link_specification_term(OptimizationTermAssignment::Objective {weight: 1.0}, None);
        out_self.add_joint_velocity_term(init_state, init_time, OptimizationTermAssignment::Objective {weight: 1.0}, None, 1.0);
        out_self.add_joint_acceleration_term(init_state, init_time, OptimizationTermAssignment::Objective {weight: 1.0}, None, 20.0);
        out_self.add_joint_jerk_term(init_state, init_time, OptimizationTermAssignment::Objective {weight: 1.0}, None, 2000.0);
        match robot_collision_proximity_params {
            RobotCollisionProximityGenericParams::None => {}
            _ => {
                out_self.add_robot_collision_proximity_term(robot_collision_proximity_params, OptimizationTermAssignment::Objective {weight: 1.0});
            }
        }
        out_self.mode = OptimaIKMode::Motion;

        out_self
    }
    pub fn new_static_ik(robot_geometric_shape_scene: RobotGeometricShapeScene, nonlinear_optimizer_type: NonlinearOptimizerType, robot_collision_proximity_params: RobotCollisionProximityGenericParams) -> Self {
        let mut out_self = Self::new_empty(robot_geometric_shape_scene, nonlinear_optimizer_type);

        match robot_collision_proximity_params {
            RobotCollisionProximityGenericParams::None => {}
            _ => {
                out_self.add_robot_collision_proximity_term(robot_collision_proximity_params, OptimizationTermAssignment::LTInequalityConstraint { must_be_less_than: 1.0 });
            }
        }
        out_self.add_robot_link_specification_term(OptimizationTermAssignment::Objective { weight: 1.0 }, None);
        out_self.mode = OptimaIKMode::Static;

        out_self
    }

    pub fn add_robot_link_specification_term(&mut self, assignment: OptimizationTermAssignment, normalization_value: Option<f64>) {
        let normalization_value = match normalization_value {
            None => { Self::default_robot_link_specification_normalization_value() }
            Some(n) => {n}
        };
        let f = OTFComposition::new(OTFQuadraticNormalizer::new(normalization_value), OTFRobotLinkTFSpec);
        self.immut_vars.insert_or_replace(OTFImmutVarsObject::RobotLinkTFGoalCollection(RobotLinkTFGoalCollection::new()));

        self.n.add_term_generic(f, &OptimizationTermSpecification::Include { optimization_assignment: assignment });
    }
    pub fn add_joint_velocity_term(&mut self, init_state: &RobotSetJointState, init_time: f64, assignment: OptimizationTermAssignment, p: Option<f64>, normalization_value: f64) {
        self.add_generic_joint_derivative_term(1, init_state, init_time, assignment, p, normalization_value);
    }
    pub fn add_joint_acceleration_term(&mut self, init_state: &RobotSetJointState, init_time: f64, assignment: OptimizationTermAssignment, p: Option<f64>, normalization_value: f64) {
        self.add_generic_joint_derivative_term(2, init_state, init_time, assignment, p, normalization_value);
    }
    pub fn add_joint_jerk_term(&mut self, init_state: &RobotSetJointState, init_time: f64, assignment: OptimizationTermAssignment, p: Option<f64>, normalization_value: f64) {
        self.add_generic_joint_derivative_term(3, init_state, init_time, assignment, p, normalization_value);
    }
    pub fn add_robot_collision_proximity_term(&mut self, robot_collision_proximity_params: RobotCollisionProximityGenericParams, assignment: OptimizationTermAssignment) {
        let f = OTFComposition::new(OTFQuadraticNormalizer::new(1.0), OTFRobotCollisionProximityGeneric::new(robot_collision_proximity_params));
        self.n.add_term_generic(f, &OptimizationTermSpecification::Include { optimization_assignment: assignment });
    }

    pub fn add_generic_joint_derivative_term(&mut self, derivative_order: usize, init_state: &RobotSetJointState, init_time: f64, assignment: OptimizationTermAssignment, p: Option<f64>, normalization_value: f64) {
        let p = match p {
            None => { Self::default_joint_derivative_p() }
            Some(p) => {p}
        };

        let deriv = OTFRobotJointStateDerivative::new(self.problem_size, derivative_order, None);
        let comp1 = OTFComposition::new(OTFTensorPNorm { p }, deriv);
        let comp2 = OTFComposition::new(OTFQuadraticNormalizer::new(normalization_value), comp1);

        self.immut_vars.insert_or_replace(OTFImmutVarsObject::TimedGenericRobotJointStateWindowMemoryContainer(TimedGenericRobotJointStateWindowMemoryContainer::new(10, TimedGenericRobotJointState::new(init_state.clone(), init_time))));

        self.n.add_term_generic(comp2, &OptimizationTermSpecification::Include { optimization_assignment: assignment });
    }

    pub fn robot_geometric_shape_scene_mut_ref(&mut self) -> &mut RobotGeometricShapeScene {
        let obj = self.immut_vars.object_ref_mut(&OTFImmutVarsObjectType::RobotGeometricShapeScene).expect("error");
        obj.unwrap_robot_geometric_shape_scene_mut()
    }
    pub fn robot_geometric_shape_scene_ref(&self) -> &RobotGeometricShapeScene {
        let obj = self.immut_vars.object_ref(&OTFImmutVarsObjectType::RobotGeometricShapeScene).expect("error");
        obj.unwrap_robot_geometric_shape_scene()
    }
    pub fn robot_link_transform_goal_collection_mut_ref(&mut self) -> &mut RobotLinkTFGoalCollection {
        let obj = self.immut_vars.object_ref_mut(&OTFImmutVarsObjectType::RobotLinkTFGoalCollection).expect("error");
        obj.unwrap_robot_link_transform_specification_collection_mut()
    }
    pub fn robot_link_transform_goal_collection_ref(&self) -> &RobotLinkTFGoalCollection {
        let obj = self.immut_vars.object_ref(&OTFImmutVarsObjectType::RobotLinkTFGoalCollection).expect("error");
        obj.unwrap_robot_link_transform_specification_collection()
    }
    fn add_or_update_robot_link_transform_goal(&mut self, spec: RobotLinkTFSpec) {
        let robot_set = self.immut_vars.ref_robot_set();
        let goal = spec.recover_goal(robot_set);
        self.robot_link_transform_goal_collection_mut_ref().insert_or_replace(goal);
    }

    pub fn solve_motion_ik(&mut self, specs: RobotLinkTFSpecCollection, time: f64, params: &OptimizerParameters, debug: OptimaDebug) -> TimedGenericRobotJointState {
        assert_ne!(self.mode, OptimaIKMode::Static, "cannot solve motion ik with static solver.");

        for spec in specs.robot_set_link_transform_specifications() {
            self.add_or_update_robot_link_transform_goal(spec.clone());
        }

        self.immut_vars.insert_or_replace(OTFImmutVarsObject::GenericRobotJointStateCurrTime(time));

        let obj = self.immut_vars.object_ref(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).unwrap();
        let t = obj.unwrap_timed_generic_robot_joint_state_window_memory_container();
        let timed_generic_robot_joint_state = t.c.object_ref(0);

        let init_condition = OptimaTensor::new_from_vector(timed_generic_robot_joint_state.joint_state().clone());

        let result = self.n.optimize(&init_condition, &self.immut_vars, &mut self.mut_vars, params, debug.clone());
        let x_min = result.unwrap_x_min().unwrap_vector();

        let robot_geometric_shape_scene = self.immut_vars.ref_robot_geometric_shape_scene();
        let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(x_min.clone()).unwrap();

        let out = TimedGenericRobotJointState::new(robot_set_joint_state.clone(), time);

        let obj = self.immut_vars.object_ref_mut(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).unwrap();
        let t = obj.unwrap_timed_generic_robot_joint_state_window_memory_container_mut();
        t.c.update(out.clone());

        out
    }
    pub fn solve_static_ik(&mut self, spec_and_errors: RobotLinkTFSpecAndAllowableErrorCollection, params: &OptimizerParameters, initial_condition: Option<RobotSetJointState>, max_num_tries: Option<usize>, reject_states_in_collision: bool, debug: OptimaDebug) -> Option<RobotSetJointState> {
        assert_ne!(self.mode, OptimaIKMode::Motion, "cannot solve static ik with motion solver.");

        self.robot_link_transform_goal_collection_mut_ref().remove_all();
        for r in spec_and_errors.robot_set_link_transform_specification_and_errors() {
            let robot_set = self.immut_vars.ref_robot_set();
            let rr = r.spec().recover_goal(robot_set);
            self.robot_link_transform_goal_collection_mut_ref().insert_or_replace(rr);
        }

        return self.solve_static_ik_internal(&spec_and_errors, initial_condition, params, max_num_tries, reject_states_in_collision, 0, debug);
    }
    fn solve_static_ik_internal(&mut self, spec_and_errors: &RobotLinkTFSpecAndAllowableErrorCollection, initial_condition: Option<RobotSetJointState>, params: &OptimizerParameters, max_num_tries: Option<usize>, reject_states_in_collision: bool, curr_try_idx: usize, debug: OptimaDebug) -> Option<RobotSetJointState> {
        let max_num_tries_ = match max_num_tries {
            None => { 1000 }
            Some(m) => {m}
        };

        if curr_try_idx >= max_num_tries_ {
            optima_print(&format!("could not find solution to solve_static_ik in given number of tries.  Returning None."), PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
            return None;
        }

        let initial_condition_ = match &initial_condition {
            None => {
                let robot_set = self.immut_vars.ref_robot_set();
                robot_set.robot_set_joint_state_module().sample_set_joint_state(&RobotSetJointStateType::DOF)
            }
            Some(initial_condition) => { initial_condition.clone() }
        };

        let initial_condition_tensor = OptimaTensor::new_from_vector(initial_condition_.concatenated_state().clone());
        let solution = self.n.optimize(&initial_condition_tensor, &self.immut_vars, &mut self.mut_vars, params, debug.clone());
        let x_min = solution.unwrap_x_min();

        let robot_set = self.immut_vars.ref_robot_set();
        let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(x_min.unwrap_vector().clone()).expect("error");
        let robot_set_fk_res = robot_set.robot_set_kinematics_module().compute_fk(&robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");

        let goals = self.robot_link_transform_goal_collection_mut_ref().all_robot_set_link_transform_goal_refs();
        for (idx, g) in goals.iter().enumerate() {
            let allowable = g.is_error_allowable(&robot_set_fk_res, &spec_and_errors.robot_set_link_transform_specification_and_errors()[idx].allowable_error());
            if !allowable {
                optima_print(&format!("current solution to solve_static_ik had an error too high for the given goal {:?}.  Trying again.", g), PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
                return self.solve_static_ik_internal(spec_and_errors, None, params, max_num_tries, reject_states_in_collision, curr_try_idx+1, debug.clone());
            }
        }

        // collision check
        if reject_states_in_collision {
            let geometric_shape_scene = self.immut_vars.ref_robot_geometric_shape_scene();

            let input = RobotGeometricShapeSceneQuery::Contact {
                robot_set_joint_state: &robot_set_joint_state,
                env_obj_pose_constraint_group_input: None,
                prediction: 0.01,
                inclusion_list: &None
            };
            let res = geometric_shape_scene.shape_collection_query(&input, StopCondition::Intersection, LogCondition::Intersection, false).expect("error");
            let in_collision = res.intersection_found();
            if in_collision {
                optima_print(&format!("current solution was in collision.  Rejecting this solution, and trying again."), PrintMode::Println, PrintColor::Yellow, true, 0, None, vec![]);
                return self.solve_static_ik_internal(spec_and_errors, None, params, max_num_tries, reject_states_in_collision, curr_try_idx+1, debug.clone());
            }
        }

        return Some(robot_set_joint_state);
    }

    pub fn cost_and_vars(&mut self) -> (Box<&dyn OptimaTensorFunction>, &mut OTFImmutVars, &mut OTFMutVars) {
        (self.n.cost(), &mut self.immut_vars, &mut self.mut_vars)
    }
    pub fn cost_call(&mut self, input: &OptimaTensor, time: f64, debug: OptimaDebug) -> OptimaTensor {
        self.immut_vars.insert_or_replace(OTFImmutVarsObject::GenericRobotJointStateCurrTime(time));
        let res = self.n.cost().call(input, &self.immut_vars, &mut self.mut_vars, debug).expect("error");
        return res.unwrap_tensor().clone();
    }
    pub fn cost_derivative(&mut self, input: &OptimaTensor, time: f64, mode: Option<OTFDerivativeMode>, debug: OptimaDebug) -> OptimaTensor {
        self.immut_vars.insert_or_replace(OTFImmutVarsObject::GenericRobotJointStateCurrTime(time));
        let res = self.n.cost().derivative(input, &self.immut_vars, &mut self.mut_vars, mode, debug).expect("error");
        return res.unwrap_tensor().clone();
    }

    pub fn compute_robot_set_link_transform_goal_error_reports<T: GenericRobotJointState>(&self, joint_state: &T) -> RobotLinkTFGoalErrorReportCollection {
        let robot_set_joint_state = self.immut_vars.ref_robot_set().spawn_robot_set_joint_state(joint_state.joint_state().clone()).expect("error");
        let robot_set_fk_res = self.immut_vars.ref_robot_set().robot_set_kinematics_module().compute_fk(&robot_set_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");

        let mut out = RobotLinkTFGoalErrorReportCollection::new();
        let goals = self.robot_link_transform_goal_collection_ref().all_robot_set_link_transform_goal_refs();
        for g in goals { out.add(g.compute_error_report(&robot_set_fk_res)); }
        out
    }

    fn default_joint_derivative_p() -> f64 { 20.0 }
    fn default_robot_link_specification_normalization_value() -> f64 { 0.01 }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimaIKMode {
    Motion,
    Static,
    Custom
}

#[cfg(not(target_arch = "wasm32"))]
#[pyclass]
pub struct OptimaIKPy {
    optima_ik: OptimaIK
}
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl OptimaIKPy {
    #[cfg(not(target_arch = "wasm32"))]
    #[staticmethod]
    #[args(nonlinear_optimizer_type="\"OpEn\"", robot_collision_proximity_params="\"Proxima(r:0,a_max:0.2,d_max:0.2,budget:Accuracy(0.4),loss_function:Hinge,aggregator:PNorm(p:10),fd_mode:JacobianFiniteDifference(max_number_of_jacobians:15))\"")]
    pub fn new_motion_ik(robot_geometric_scene: RobotGeometricShapeScenePy, init_state: Vec<f64>, init_time: f64, nonlinear_optimizer_type: &str, robot_collision_proximity_params: &str) -> Self {
        let scene = robot_geometric_scene.robot_geometric_shape_scene().clone();
        let init_state = scene.robot_set().spawn_robot_set_joint_state(DVector::from_vec(init_state)).expect("error");
        let nonlinear_optimizer_type = NonlinearOptimizerType::from_ron_string(nonlinear_optimizer_type).unwrap();
        let robot_collision_proximity_params = RobotCollisionProximityGenericParams::from_ron_string(robot_collision_proximity_params).unwrap();

        let optima_ik = OptimaIK::new_motion_ik(scene, nonlinear_optimizer_type, robot_collision_proximity_params, &init_state, init_time);

        Self {
            optima_ik
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[staticmethod]
    #[args(nonlinear_optimizer_type="\"OpEn\"", robot_collision_proximity_params="\"None\"")]
    pub fn new_static_ik(robot_geometric_scene: RobotGeometricShapeScenePy, nonlinear_optimizer_type: &str, robot_collision_proximity_params: &str) -> Self {
        let scene = robot_geometric_scene.robot_geometric_shape_scene().clone();
        let nonlinear_optimizer_type = NonlinearOptimizerType::from_ron_string(nonlinear_optimizer_type).unwrap();
        let robot_collision_proximity_params = RobotCollisionProximityGenericParams::from_ron_string(robot_collision_proximity_params).unwrap();

        let optima_ik = OptimaIK::new_static_ik(scene, nonlinear_optimizer_type, robot_collision_proximity_params);

        Self {
            optima_ik
        }
    }

    pub fn solve_motion_ik(&mut self, specs: RobotLinkTFSpecCollectionPy, time: f64) -> Vec<f64> {
        let specs = specs.convert_to_robot_set_link_transform_specification_collection();
        let solution = self.optima_ik.solve_motion_ik(specs, time, &OptimizerParameters::default(), OptimaDebug::False);
        let state = solution.joint_state().data.as_slice().to_vec();
        return state;
    }

    #[args(reject_states_in_collision="false")]
    pub fn solve_static_ik(&mut self, spec_and_errors: RobotLinkTFSpecAndAllowableErrorCollectionPy, reject_states_in_collision: bool, initial_condition: Option<Vec<f64>>, max_num_tries: Option<usize>) -> Option<Vec<f64>> {
        let spec_and_errors = spec_and_errors.convert_to_robot_set_link_transform_specification_and_error_collection();
        let initial_condition = match &initial_condition {
            None => { None }
            Some(v) => {
                Some( self.optima_ik.immut_vars.ref_robot_set().spawn_robot_set_joint_state(DVector::from_vec(v.clone())).expect("error") )
            }
        };
        let solution = self.optima_ik.solve_static_ik(spec_and_errors, &OptimizerParameters::default(), initial_condition, max_num_tries, reject_states_in_collision, OptimaDebug::False);
        return match &solution {
            None => { None }
            Some(solution) => {
                let v = solution.joint_state().as_slice().to_vec();
                Some(v)
            }
        }
    }
}
