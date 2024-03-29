use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use crate::optima_tensor_function::{FD_PERTURBATION, OptimaTensor, OptimaTensorFunction,OTFImmutVars, OTFImmutVarsObjectType, OTFMutVars, OTFMutVarsObject, OTFMutVarsObjectType, OTFMutVarsParams, OTFMutVarsSessionKey, OTFResult, RecomputeVarIf};
use crate::optima_tensor_function::standard_functions::OTFZeroFunction;
use crate::robot_modules::robot_geometric_shape_module::RobotLinkShapeRepresentation;
use crate::robot_modules::robot_kinematics_module::{JacobianEndPoint, JacobianMode};
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::robot_set_modules::robot_set_kinematics_module::RobotSetFKResult;
use crate::scenes::robot_geometric_shape_scene::{RobotGeometricShapeScene, RobotGeometricShapeSceneQuery};
use crate::utils::utils_console::OptimaDebug;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_math::finite_difference::FiniteDifferenceUtils;
use crate::utils::utils_robot::robot_generic_structures::TimedGenericRobotJointState;
use crate::utils::utils_robot::robot_set_link_specification::RobotLinkTFGoal;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShapeSignature, LogCondition, StopCondition};
use crate::utils::utils_shape_geometry::shape_collection::{BVHVisit, ProximaBudget, ProximaEngine, ProximityOutputMode, ShapeCollectionQueryPairsList, SignedDistanceAggregator, SignedDistanceLossFunction, WitnessPoints, WitnessPointsCollection, WitnessPointsType};

#[derive(Clone)]
pub struct OTFRobotLinkTFSpec;
impl OTFRobotLinkTFSpec {
    fn internal_call(robot_set_fk_result: &RobotSetFKResult, specs: &Vec<RobotLinkTFGoal>) -> f64 {
        let mut out_error = 0.0;
        for s in specs {
            match s {
                RobotLinkTFGoal::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let se3_delta = goal.distance_function(pose, true).expect("error");
                    // let position = pose.translation();
                    // let r3_delta = (goal.translation() - &position).norm();
                    // let rotation = pose.rotation();
                    // let so3_delta = rotation.angle_between(&goal.rotation(), true).expect("error");
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_error += weight * se3_delta;
                }
                RobotLinkTFGoal::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let position = pose.translation();
                    let r3_delta = (goal - &position).norm();
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_error += weight * r3_delta;
                }
                RobotLinkTFGoal::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let rotation = pose.rotation();
                    let so3_delta = rotation.angle_between(goal, true).expect("error");
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_error += weight * so3_delta;
                }
            }
        }

        return out_error;
    }
}
impl OptimaTensorFunction for OTFRobotLinkTFSpec {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let recompute_var_ifs = vec![RecomputeVarIf::IsAnyNewInput];
        let signatures = vec![OTFMutVarsObjectType::RobotSetFKResult];
        let params = vec![OTFMutVarsParams::None];
        let vars = _mut_vars.get_vars(&signatures, &params, &recompute_var_ifs, input, _immut_vars, _session_key);
        let robot_set_fk_result = vars[0].unwrap_robot_set_fk_result();

        let spec_object = _immut_vars.object_ref(&OTFImmutVarsObjectType::RobotLinkTFGoalCollection).expect("must have RobotLinkSpecificationCollection");
        let spec = spec_object.unwrap_robot_link_transform_specification_collection();
        let specs = spec.robot_set_link_specification_refs();

        let out = Self::internal_call(robot_set_fk_result, specs);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)));
    }

    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let recompute_var_ifs = vec![RecomputeVarIf::IsAnyNewInput];
        let signatures = vec![OTFMutVarsObjectType::RobotSetFKDOFPerturbationsResult];
        let params = vec![OTFMutVarsParams::None];

        let vars = mut_vars.get_vars(&signatures, &params, &recompute_var_ifs, input, immut_vars, &session_key);
        let robot_set_fk_dof_perturbations_result = vars[0].unwrap_robot_set_fk_dof_perturbations_result();
        let perturbation = robot_set_fk_dof_perturbations_result.perturbation();

        let spec_object = immut_vars.object_ref(&OTFImmutVarsObjectType::RobotLinkTFGoalCollection).expect("must have RobotLinkSpecificationCollection");
        let spec = spec_object.unwrap_robot_link_transform_specification_collection();
        let specs = spec.robot_set_link_specification_refs();

        let mut grad = DVector::<f64>::zeros(input.vectorized_data().len());

        let x_0 = Self::internal_call(robot_set_fk_dof_perturbations_result.central_fk_result(), &specs);

        for (idx, robot_set_fk_result) in robot_set_fk_dof_perturbations_result.fk_dof_perturbation_results().iter().enumerate() {
            let x_h = Self::internal_call(robot_set_fk_result, &specs);
            grad[idx] = (x_h - x_0) / perturbation;
        }

        mut_vars.close_session(&session_key);

        return Ok(OTFResult::Complete(OptimaTensor::new_from_vector(grad)));
    }

    fn to_string(&self) -> String {
        "OTFRobotLinkTFSpec".to_string()
    }

    /*
    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let vars = mut_vars.get_vars(&vec![OTFMutVarsObjectType::RobotSetFKResult], &vec![RecomputeVarIf::IsAnyNewInput], input, immut_vars, session_key);
        let robot_set_fk_result = vars[0].unwrap_robot_set_fk_result();

        let robot_set_object = immut_vars.object_ref(&OTFImmutVarsObjectType::GetRobotSet).expect("error");
        let robot_set = robot_set_object.unwrap_get_robot_set().get_robot_set();
        let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");
        let num_dof = robot_set_joint_state.concatenated_state().len();
        let mut out_vec = DVector::zeros(num_dof);

        let specs_object = immut_vars.object_ref(&OTFImmutVarsObjectType::RobotLinkSpecificationCollection).expect("error");
        let specs = specs_object.unwrap_robot_link_specification_collection();

        for s in specs.robot_set_link_specification_refs() {
            match s {
                RobotSetLinkSpecification::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let pose_translation = pose.translation();
                    let pose_rotation = pose.rotation();
                    let goal_translation = goal.translation();
                    let goal_rotation = goal.rotation();
                    let disp_translation = &pose_translation - &goal_translation;
                    let disp_rotation = pose_rotation.displacement(&goal_rotation, true).expect("error").ln();
                    let disp_vector = Vector6::new(disp_translation[0], disp_translation[1], disp_translation[2], disp_rotation[0], disp_rotation[1], disp_rotation[2]);
                    let jacobian = robot_set.robot_set_kinematics_module().compute_jacobian(&robot_set_joint_state, *robot_idx_in_set, None, *link_idx_in_robot, &JacobianEndPoint::Link, None, JacobianMode::Full).expect("error");
                    let ps = jacobian.pseudo_inverse(0.0001).expect("error");
                    let delta_angle = ps * disp_vector;
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_vec += weight * delta_angle;
                }
                RobotSetLinkSpecification::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let translation = pose.translation();
                    let disp_translation = &translation - goal;
                    let jacobian = robot_set.robot_set_kinematics_module().compute_jacobian(&robot_set_joint_state, *robot_idx_in_set, None, *link_idx_in_robot, &JacobianEndPoint::Link, None, JacobianMode::Translational).expect("error");
                    let ps = jacobian.pseudo_inverse(0.0001).expect("error");
                    let delta_angle = ps * disp_translation;
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_vec += weight * delta_angle;
                }
                RobotSetLinkSpecification::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let rotation = pose.rotation();
                    let disp_rotation = goal.displacement(&rotation, true).expect("error").ln();
                    let jacobian = robot_set.robot_set_kinematics_module().compute_jacobian(&robot_set_joint_state, *robot_idx_in_set, None, *link_idx_in_robot, &JacobianEndPoint::Link, None, JacobianMode::Rotational).expect("error");
                    let ps = jacobian.pseudo_inverse(0.0001).expect("error");
                    let delta_angle = ps * disp_rotation;
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_vec += weight * delta_angle;
                }
            }
        }

        Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)))
    }
    */
}

#[derive(Clone)]
pub struct OTFRobotJointStateDerivative {
    num_dofs: usize,
    derivative_order: usize,
    window_size: usize
}
impl OTFRobotJointStateDerivative {
    pub fn new(num_dofs: usize, derivative_order: usize, window_size: Option<usize>) -> Self {
        assert!(derivative_order >= 1);
        if let Some(w) = window_size { assert!(w > 1); }

        Self {
            num_dofs,
            derivative_order,
            window_size: match window_size {
                None => { derivative_order + 1 }
                Some(w) => {w}
            }
        }
    }
}
impl OptimaTensorFunction for OTFRobotJointStateDerivative {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![self.num_dofs]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).expect("must have TimedGenericRobotJointStateWindowMemoryContainer");
        let window_memory_container = object.unwrap_timed_generic_robot_joint_state_window_memory_container();

        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::GenericRobotJointStateCurrTime).expect("must have GenericRobotJointStateCurrTime");
        let curr_time = object.unwrap_generic_robot_joint_state_curr_time();

        let robot_set = immut_vars.ref_robot_set();
        let joint_state = robot_set.spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");
        let curr_timed_generic_robot_joint_state = TimedGenericRobotJointState::new(joint_state, *curr_time);

        let mut timed_states = window_memory_container.c.previous_n_object_refs(self.window_size-1);
        timed_states.insert(0, &curr_timed_generic_robot_joint_state);

        let mut time_stencils = vec![];
        for timed_state in &timed_states {
            time_stencils.push(timed_state.time() - curr_time)
        }

        let coeffs = FiniteDifferenceUtils::get_fd_coefficients(&time_stencils, self.derivative_order);

        let mut out = DVector::zeros(self.num_dofs);
        for (coeff, timed_state) in coeffs.iter().zip(timed_states.iter()) {
            out += *coeff * timed_state.joint_state();
        }

        Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out)))
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).expect("must have TimedGenericRobotJointStateWindowMemoryContainer");
        let window_memory_container = object.unwrap_timed_generic_robot_joint_state_window_memory_container();

        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::GenericRobotJointStateCurrTime).expect("must have GenericRobotJointStateCurrTime");
        let curr_time = object.unwrap_generic_robot_joint_state_curr_time();

        let robot_set = immut_vars.ref_robot_set();
        let joint_state = robot_set.spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");
        let curr_timed_generic_robot_joint_state = TimedGenericRobotJointState::new(joint_state, *curr_time);

        let mut timed_states = window_memory_container.c.previous_n_object_refs(self.window_size-1);
        timed_states.insert(0, &curr_timed_generic_robot_joint_state);

        let mut time_stencils = vec![];
        for timed_state in &timed_states {
            time_stencils.push(timed_state.time() - curr_time)
        }
        let coeffs = FiniteDifferenceUtils::get_fd_coefficients(&time_stencils, self.derivative_order);

        let mut out = DMatrix::identity(self.num_dofs, self.num_dofs);
        out *= coeffs[0];

        Ok(OTFResult::Complete(OptimaTensor::new_from_matrix(out)))
    }

    fn derivative2_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(_input, 2));
        return Ok(OTFResult::Complete(out));
    }

    fn derivative3_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(_input, 3));
        return Ok(OTFResult::Complete(out));
    }

    fn derivative4_analytical_raw(&self, _input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(_input, 4));
        return Ok(OTFResult::Complete(out));
    }

    fn to_string(&self) -> String {
        format!("OTFRobotJointStateDerivative_{}", self.derivative_order)
    }
}

#[derive(Clone)]
struct OTFRobotCollisionProximityProxima {
    r: f64,
    a_max: f64,
    d_max: f64,
    budget: ProximaBudget,
    loss_function: SignedDistanceLossFunction,
    aggregator: SignedDistanceAggregator,
    fd_mode: RobotCollisionProximityGradientFDMode,
    robot_link_shape_representation: RobotLinkShapeRepresentation
}
impl OTFRobotCollisionProximityProxima {
    pub fn new(r: f64, a_max: f64, d_max: f64, budget: ProximaBudget, loss_function: SignedDistanceLossFunction, aggregator: SignedDistanceAggregator, fd_mode: RobotCollisionProximityGradientFDMode, robot_link_shape_representation: RobotLinkShapeRepresentation) -> Self {
        Self {
            r,
            a_max,
            d_max,
            budget,
            loss_function,
            aggregator,
            fd_mode,
            robot_link_shape_representation,
        }
    }
    fn call_raw_with_necessary_vars(&self,
                                    robot_set_joint_state: &RobotSetJointState,
                                    robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                    proxima_engine: &mut ProximaEngine,
                                    inclusion_list: Option<&ShapeCollectionQueryPairsList>,
                                    r_override: Option<f64>) -> Result<OTFResult, OptimaError> {
        let res = robot_geometric_shape_scene.proxima_proximity_query(&robot_set_joint_state,
                                                                      &self.robot_link_shape_representation,
                                                                      None,
                                                                      proxima_engine,
                                                                      self.d_max,
                                                                      self.a_max,
                                                                      self.loss_function.clone(),
                                                                      self.aggregator.clone(),
                                                                      match r_override {
                                                                          None => { self.r }
                                                                          Some(r) => { r }
                                                                      },
                                                                      self.budget.clone(),
                                                                      &inclusion_list).expect("error");
        let aggregated_output_value = res.aggregated_output_value();
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(aggregated_output_value)));
    }
    fn derivative_finite_difference_raw(&self,
                                        robot_set_joint_state: &RobotSetJointState,
                                        robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                        proxima_engine: &mut ProximaEngine) -> Result<OTFResult, OptimaError> {
        let num_dofs = robot_set_joint_state.concatenated_state().len();

        let x_0_res = self.call_raw_with_necessary_vars(robot_set_joint_state, robot_geometric_shape_scene, proxima_engine, None, Some(1.0)).expect("error");
        let x_0 = x_0_res.unwrap_tensor().unwrap_scalar();

        let mut out_vec = DVector::zeros(num_dofs);

        for i in 0..num_dofs {
            let mut robot_set_joint_state_h = robot_set_joint_state.clone();
            robot_set_joint_state_h[i] += FD_PERTURBATION;

            let x_h_res = self.call_raw_with_necessary_vars(&robot_set_joint_state_h, robot_geometric_shape_scene, proxima_engine, None, Some(1.0)).expect("error");
            let x_h = x_h_res.unwrap_tensor().unwrap_scalar();

            out_vec[i] = (x_h - x_0) / FD_PERTURBATION;
        }

        Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)))
    }
    fn derivative_finite_difference_pre_filtered(&self,
                                                 robot_set_joint_state: &RobotSetJointState,
                                                 robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                                 proxima_engine: &mut ProximaEngine) -> Result<OTFResult, OptimaError> {
        let num_dofs = robot_set_joint_state.concatenated_state().len();

        let filter_output = robot_geometric_shape_scene.proxima_scene_filter(robot_set_joint_state, &self.robot_link_shape_representation, None, proxima_engine, self.d_max, self.a_max, self.loss_function.clone(), 1.0, &None).expect("error");
        let inclusion_list = filter_output.query_pairs_list();

        let x_0_res = self.call_raw_with_necessary_vars(robot_set_joint_state, robot_geometric_shape_scene, proxima_engine, Some(inclusion_list), Some(1.0)).expect("error");
        let x_0 = x_0_res.unwrap_tensor().unwrap_scalar();

        let mut out_vec = DVector::zeros(num_dofs);

        for i in 0..num_dofs {
            let mut robot_set_joint_state_h = robot_set_joint_state.clone();
            robot_set_joint_state_h[i] += FD_PERTURBATION;

            let x_h_res = self.call_raw_with_necessary_vars(&robot_set_joint_state_h, robot_geometric_shape_scene, proxima_engine, Some(inclusion_list), Some(1.0)).expect("error");
            let x_h = x_h_res.unwrap_tensor().unwrap_scalar();

            out_vec[i] = (x_h - x_0) / FD_PERTURBATION;
        }

        Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)))
    }
    fn derivative_finite_difference_jacobian(&self,
                                             robot_set_joint_state: &RobotSetJointState,
                                             robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                             proxima_engine: &mut ProximaEngine,
                                             max_number_of_jacobians: usize) -> Result<OTFResult, OptimaError> {
        let res = robot_geometric_shape_scene.proxima_proximity_query(&robot_set_joint_state,
                                                                      &self.robot_link_shape_representation,
                                                                      None,
                                                                      proxima_engine,
                                                                      self.d_max,
                                                                      self.a_max,
                                                                      self.loss_function.clone(),
                                                                      self.aggregator.clone(),
                                                                      1.0,
                                                                      self.budget.clone(),
                                                                      &None).expect("error");
        let mut witness_points_collection = res.output_witness_points_collection();

        let out_vec = general_robot_jacobian_collision_proximity_gradient(&mut witness_points_collection, robot_set_joint_state, robot_geometric_shape_scene, &self.robot_link_shape_representation, self.a_max, &self.loss_function, &self.aggregator, max_number_of_jacobians);

        return Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)));
    }
}
impl OptimaTensorFunction for OTFRobotCollisionProximityProxima {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut vars = _mut_vars.get_vars(&vec![OTFMutVarsObjectType::ProximaEngine], &vec![OTFMutVarsParams::None], &vec![RecomputeVarIf::Never], input, _immut_vars, _session_key);
        let mut proxima_engine = vars[0].unwrap_proxima_engine_mut();

        let robot_geometric_shape_scene = _immut_vars.ref_robot_geometric_shape_scene();
        let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

        return self.call_raw_with_necessary_vars(&robot_set_joint_state, robot_geometric_shape_scene, &mut proxima_engine, None, None);
    }

    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let mut vars = mut_vars.get_vars(&vec![OTFMutVarsObjectType::ProximaEngine], &vec![OTFMutVarsParams::None], &vec![RecomputeVarIf::Never], input, immut_vars, &session_key);
        let proxima_engine = vars[0].unwrap_proxima_engine_mut();

        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();
        let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

        let ret = match &self.fd_mode {
            RobotCollisionProximityGradientFDMode::RawFiniteDifference => {
                self.derivative_finite_difference_raw(&robot_set_joint_state, robot_geometric_shape_scene, proxima_engine)
            }
            RobotCollisionProximityGradientFDMode::PreFilteredFiniteDifference => {
                self.derivative_finite_difference_pre_filtered(&robot_set_joint_state, robot_geometric_shape_scene, proxima_engine)
            }
            RobotCollisionProximityGradientFDMode::JacobianFiniteDifference { max_number_of_jacobians } => {
                self.derivative_finite_difference_jacobian(&robot_set_joint_state, robot_geometric_shape_scene, proxima_engine, *max_number_of_jacobians)
            }
        };
        mut_vars.close_session(&session_key);

        return ret;
    }

    fn to_string(&self) -> String {
        "OTFRobotCollisionProximityProxima".to_string()
    }
}

#[derive(Clone)]
struct OTFRobotCollisionProximityStandard {
    a_max: f64,
    d_max: f64,
    loss_function: SignedDistanceLossFunction,
    aggregator: SignedDistanceAggregator,
    fd_mode: RobotCollisionProximityGradientFDMode,
    bvh_mode: RobotCollisionProximityBVHMode,
    robot_link_shape_representation: RobotLinkShapeRepresentation
}
impl OTFRobotCollisionProximityStandard {
    pub fn new(a_max: f64, d_max: f64, loss_function: SignedDistanceLossFunction, aggregator: SignedDistanceAggregator, fd_mode: RobotCollisionProximityGradientFDMode, bvh_mode: RobotCollisionProximityBVHMode, robot_link_shape_representation: RobotLinkShapeRepresentation) -> Self {
        Self {
            a_max,
            d_max,
            loss_function,
            aggregator,
            fd_mode,
            bvh_mode,
            robot_link_shape_representation
        }
    }
    fn call_raw_with_necessary_vars(&self,
                                    robot_set_joint_state: &RobotSetJointState,
                                    robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                    inclusion_list: &Option<&ShapeCollectionQueryPairsList>) -> Result<OTFResult, OptimaError> {
        let input = RobotGeometricShapeSceneQuery::Contact {
            robot_set_joint_state: Some(&robot_set_joint_state),
            env_obj_pose_constraint_group_input: None,
            prediction: self.d_max,
            inclusion_list: &inclusion_list
        };

        let res = robot_geometric_shape_scene.shape_collection_query(&input, self.robot_link_shape_representation.clone(), StopCondition::None, LogCondition::LogAll, false).expect("error");
        let wp = res.output_witness_points_collection();

        let shape_collection = robot_geometric_shape_scene.robot_geometric_shape_scene_shape_collection(&self.robot_link_shape_representation).shape_collection();
        let output = wp.compute_proximity_output(&ProximityOutputMode::AverageSignedDistance {
            a_max: self.a_max,
            shape_collection
        }, &self.loss_function, &self.aggregator);

        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(output)));
    }
    fn derivative_finite_difference_standard(&self,
                                             robot_set_joint_state: &RobotSetJointState,
                                             robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                             inclusion_list: &Option<&ShapeCollectionQueryPairsList>) -> Result<OTFResult, OptimaError> {
        let num_dofs = robot_set_joint_state.concatenated_state().len();

        let x_0_res = self.call_raw_with_necessary_vars(robot_set_joint_state, robot_geometric_shape_scene, inclusion_list).expect("error");
        let x_0 = x_0_res.unwrap_tensor().unwrap_scalar();

        let mut out_vec = DVector::zeros(num_dofs);

        for i in 0..num_dofs {
            let mut robot_set_joint_state_h = robot_set_joint_state.clone();
            robot_set_joint_state_h[i] += FD_PERTURBATION;

            let x_h_res = self.call_raw_with_necessary_vars(&robot_set_joint_state_h, robot_geometric_shape_scene, inclusion_list).expect("error");
            let x_h = x_h_res.unwrap_tensor().unwrap_scalar();

            out_vec[i] = (x_h - x_0) / FD_PERTURBATION;
        }

        Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)))
    }
    fn derivative_finite_difference_jacobian(&self,
                                             robot_set_joint_state: &RobotSetJointState,
                                             robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                             inclusion_list: &Option<&ShapeCollectionQueryPairsList>,
                                             max_number_of_jacobians: usize) -> Result<OTFResult, OptimaError> {
        let input = RobotGeometricShapeSceneQuery::Contact {
            robot_set_joint_state: Some(&robot_set_joint_state),
            env_obj_pose_constraint_group_input: None,
            prediction: self.d_max,
            inclusion_list
        };

        let res = robot_geometric_shape_scene.shape_collection_query(&input, self.robot_link_shape_representation.clone(), StopCondition::None, LogCondition::LogAll, false).expect("error");
        let mut witness_points_collection = res.output_witness_points_collection();

        let out_vec = general_robot_jacobian_collision_proximity_gradient(&mut witness_points_collection, robot_set_joint_state, robot_geometric_shape_scene, &self.robot_link_shape_representation, self.a_max, &self.loss_function, &self.aggregator, max_number_of_jacobians);

        return Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)));
    }
}
impl OptimaTensorFunction for OTFRobotCollisionProximityStandard {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let mut vars = _mut_vars.get_vars(&vec![OTFMutVarsObjectType::AbstractBVH], &vec![OTFMutVarsParams::RobotCollisionProximityBVHMode(self.bvh_mode.clone())], &vec![RecomputeVarIf::Never], input, _immut_vars, _session_key);
        let abstract_bvh = &mut vars[0];

        let robot_geometric_shape_scene = _immut_vars.ref_robot_geometric_shape_scene();
        let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

        let bvh_filter = match abstract_bvh {
            OTFMutVarsObject::None => { None }
            OTFMutVarsObject::BVHAABB(bvh) => {
                Some(robot_geometric_shape_scene.bvh_scene_filter(bvh, &robot_set_joint_state, &self.robot_link_shape_representation, None, BVHVisit::Distance { margin: self.d_max }))
            }
            _ => { unreachable!("If you added another bvh type, you should add another arm to this match statement.") }
        };
        let inclusion_list = match &bvh_filter {
            None => { None }
            Some(bvh_filter) => { Some(bvh_filter.pairs_list()) }
        };

        self.call_raw_with_necessary_vars(&robot_set_joint_state, robot_geometric_shape_scene, &inclusion_list)
    }

    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let mut vars = mut_vars.get_vars(&vec![OTFMutVarsObjectType::AbstractBVH], &vec![OTFMutVarsParams::RobotCollisionProximityBVHMode(self.bvh_mode.clone())], &vec![RecomputeVarIf::Never], input, immut_vars, &session_key);
        let abstract_bvh = &mut vars[0];

        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();
        let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

        let out = match &self.fd_mode {
            RobotCollisionProximityGradientFDMode::RawFiniteDifference => {
                self.derivative_finite_difference_standard(&robot_set_joint_state, robot_geometric_shape_scene, &None)
            }
            RobotCollisionProximityGradientFDMode::PreFilteredFiniteDifference => {
                let bvh_filter = match abstract_bvh {
            OTFMutVarsObject::None => { None }
            OTFMutVarsObject::BVHAABB(bvh) => {
                Some(robot_geometric_shape_scene.bvh_scene_filter(bvh, &robot_set_joint_state, &self.robot_link_shape_representation, None, BVHVisit::Distance { margin: self.d_max }))
            }
            _ => { unreachable!("If you added another bvh type, you should add another arm to this match statement.") }
        };
                let inclusion_list = match &bvh_filter {
            None => { None }
            Some(bvh_filter) => { Some(bvh_filter.pairs_list()) }
        };
                self.derivative_finite_difference_standard(&robot_set_joint_state, robot_geometric_shape_scene, &inclusion_list)
            }
            RobotCollisionProximityGradientFDMode::JacobianFiniteDifference { max_number_of_jacobians } => {
                let bvh_filter = match abstract_bvh {
            OTFMutVarsObject::None => { None }
            OTFMutVarsObject::BVHAABB(bvh) => {
                Some(robot_geometric_shape_scene.bvh_scene_filter(bvh, &robot_set_joint_state, &self.robot_link_shape_representation, None, BVHVisit::Distance { margin: self.d_max }))
            }
            _ => { unreachable!("If you added another bvh type, you should add another arm to this match statement.") }
        };
                let inclusion_list = match &bvh_filter {
            None => { None }
            Some(bvh_filter) => { Some(bvh_filter.pairs_list()) }
        };

                self.derivative_finite_difference_jacobian(&robot_set_joint_state, robot_geometric_shape_scene, &inclusion_list, *max_number_of_jacobians)
            }
        };
        mut_vars.close_session(&session_key);

        return out;
    }

    fn to_string(&self) -> String {
        "OTFRobotCollisionProximityStandard".to_string()
    }
}

#[derive(Clone)]
pub struct OTFRobotCollisionProximityGeneric {
    f: Box<dyn OptimaTensorFunction>
}
impl OTFRobotCollisionProximityGeneric {
    pub fn new(params: RobotCollisionProximityGenericParams) -> Self {
        return match params {
            RobotCollisionProximityGenericParams::None => {
                Self {
                    f: Box::new(OTFZeroFunction)
                }
            }
            RobotCollisionProximityGenericParams::Proxima { r, a_max, d_max, budget, loss_function, aggregator, fd_mode, robot_link_shape_representation } => {
                let f = OTFRobotCollisionProximityProxima::new(r, a_max, d_max, budget, loss_function, aggregator, fd_mode, robot_link_shape_representation);
                Self {
                    f: Box::new(f)
                }
            }
            RobotCollisionProximityGenericParams::Standard { a_max, d_max, loss_function, aggregator, fd_mode, bvh_mode, robot_link_shape_representation } => {
                let f = OTFRobotCollisionProximityStandard::new(a_max, d_max, loss_function, aggregator, fd_mode, bvh_mode, robot_link_shape_representation);
                Self {
                    f: Box::new(f)
                }
            }
        }
    }
}
impl OptimaTensorFunction for OTFRobotCollisionProximityGeneric {
    fn output_dimensions(&self) -> Vec<usize> {
        self.f.output_dimensions()
    }

    fn call_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey, _debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        self.f.call_raw(input, _immut_vars, _mut_vars, _session_key, _debug)
    }

    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, debug: OptimaDebug) -> Result<OTFResult, OptimaError> {
        self.f.derivative_finite_difference(input, immut_vars, mut_vars, debug)
    }

    fn to_string(&self) -> String {
        self.f.to_string()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotCollisionProximityGradientFDMode {
    RawFiniteDifference,
    PreFilteredFiniteDifference,
    JacobianFiniteDifference { max_number_of_jacobians: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotCollisionProximityBVHMode {
    None,
    AABB
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RobotCollisionProximityGenericParams {
    None,
    Proxima { r: f64, a_max: f64, d_max: f64, budget: ProximaBudget, loss_function: SignedDistanceLossFunction, aggregator: SignedDistanceAggregator, fd_mode: RobotCollisionProximityGradientFDMode, robot_link_shape_representation: RobotLinkShapeRepresentation } ,
    Standard { a_max: f64, d_max: f64, loss_function: SignedDistanceLossFunction, aggregator: SignedDistanceAggregator, fd_mode: RobotCollisionProximityGradientFDMode, bvh_mode: RobotCollisionProximityBVHMode, robot_link_shape_representation: RobotLinkShapeRepresentation }
}

fn general_robot_jacobian_collision_proximity_gradient(witness_points_collection: &mut WitnessPointsCollection,
                                                       robot_set_joint_state: &RobotSetJointState,
                                                       robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                                       robot_link_shape_representation: &RobotLinkShapeRepresentation,
                                                       a_max: f64,
                                                       loss_function: &SignedDistanceLossFunction,
                                                       aggregator: &SignedDistanceAggregator,
                                                       max_number_of_jacobian_calculations: usize) -> DVector<f64> {
    let mut jacobian_option_pairs: Vec<[Option<DMatrix<f64>>; 2]> = Vec::new();

    let robot_set = robot_geometric_shape_scene.robot_set();

    let shape_collection = robot_geometric_shape_scene.robot_geometric_shape_scene_shape_collection(robot_link_shape_representation).shape_collection();
    witness_points_collection.sort_by_values_after_loss(ProximityOutputMode::AverageSignedDistance {
        a_max,
        shape_collection
    }, loss_function);
    witness_points_collection.cull_after_first_n(max_number_of_jacobian_calculations);

    // println!("{:?}", sorted_values_after_loss);

    let mut jacobian_computation_count = 0;
    // set up jacobian matrices for witness points
    for wp in witness_points_collection.collection() {
        if jacobian_computation_count >= max_number_of_jacobian_calculations { break; }
        let mut jacobian_option_pair = [None, None];

        let witness_points = &wp.witness_points();
        let signatures = wp.shape_signatures();

        match signatures.0 {
            GeometricShapeSignature::RobotSetLink { robot_idx_in_set, link_idx_in_robot, .. } => {
                let jacobian = robot_set.robot_set_kinematics_module().compute_jacobian(robot_set_joint_state, robot_idx_in_set, None, link_idx_in_robot, &JacobianEndPoint::Global(witness_points.0.clone()), None, JacobianMode::Translational).expect("error");
                jacobian_option_pair[0] = Some(jacobian);
            }
            _ => { }
        }
        match signatures.1 {
            GeometricShapeSignature::RobotSetLink { robot_idx_in_set, link_idx_in_robot, .. } => {
                let jacobian = robot_set.robot_set_kinematics_module().compute_jacobian(&robot_set_joint_state, robot_idx_in_set, None, link_idx_in_robot, &JacobianEndPoint::Global(witness_points.1.clone()), None, JacobianMode::Translational).expect("error");
                jacobian_option_pair[1] = Some(jacobian);
            }
            _ => { }
        }

        jacobian_option_pairs.push(jacobian_option_pair);
        jacobian_computation_count += 1;
    }

    let x_0 = witness_points_collection.compute_proximity_output(&ProximityOutputMode::AverageSignedDistance {
        a_max,
        shape_collection,
    }, loss_function, aggregator);

    let num_dofs = robot_set_joint_state.concatenated_state().len();
    let mut out_vec = DVector::zeros(num_dofs);
    for dof_idx in 0..num_dofs {
        let mut robot_set_joint_state_dvec = DVector::<f64>::zeros(num_dofs);
        robot_set_joint_state_dvec[dof_idx] += FD_PERTURBATION;

        let mut witness_points_collection_h = WitnessPointsCollection::new();

        let mut jacobian_count = 0;
        'j: for (wp_idx, wp) in witness_points_collection.collection().iter().enumerate() {
            if jacobian_count >= max_number_of_jacobian_calculations { break 'j; }
            let jacobian_option_pair = &jacobian_option_pairs[wp_idx];

            let jacobian_option0 = &jacobian_option_pair[0];
            let jacobian_option1 = &jacobian_option_pair[1];

            let adjusted_point0 = if let Some(jacobian0) = jacobian_option0 {
                let delta_x = jacobian0 * &robot_set_joint_state_dvec;
                &delta_x + &wp.witness_points().0
            } else {
                wp.witness_points().0.clone()
            };

            let adjusted_point1 = if let Some(jacobian1) = jacobian_option1 {
                let delta_x = jacobian1 * &robot_set_joint_state_dvec;
                &delta_x + &wp.witness_points().1
            } else {
                wp.witness_points().1.clone()
            };

            let mut adjusted_signed_distance = (&adjusted_point0 - &adjusted_point1).norm();
            if wp.signed_distance() < 0.0 { adjusted_signed_distance *= -1.0; }

            witness_points_collection_h.insert(WitnessPoints::new(adjusted_signed_distance, (adjusted_point0, adjusted_point1), wp.shape_signatures().clone(), WitnessPointsType::GroundTruth));
            jacobian_count += 1;
        }

        let x_h = witness_points_collection_h.compute_proximity_output(&ProximityOutputMode::AverageSignedDistance {
            a_max,
            shape_collection
        }, loss_function, aggregator);

        let f_h = (-x_0 + x_h) / FD_PERTURBATION;
        out_vec[dof_idx] = f_h;
    }

    out_vec
}



