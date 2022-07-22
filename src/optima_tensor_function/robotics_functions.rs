use itertools::izip;
use nalgebra::{DMatrix, DVector, Vector6};
use parry3d_f64::partitioning::QBVHDataGenerator;
use crate::optima_tensor_function::{FD_PERTURBATION, OptimaTensor, OptimaTensorFunction, OptimaTensorFunctionGenerics, OTFDimensions, OTFImmutVars, OTFImmutVarsObjectType, OTFMutVars, OTFMutVarsObjectType, OTFMutVarsParams, OTFMutVarsSessionKey, OTFResult, RecomputeVarIf};
use crate::robot_modules::robot_kinematics_module::{JacobianEndPoint, JacobianMode};
use crate::robot_set_modules::robot_set_joint_state_module::RobotSetJointState;
use crate::robot_set_modules::robot_set_kinematics_module::RobotSetFKResult;
use crate::scenes::robot_geometric_shape_scene::RobotGeometricShapeScene;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_generic_data_structures::AveragingFloat;
use crate::utils::utils_robot::robot_set_link_specification::RobotSetLinkTransformSpecification;
use crate::utils::utils_sampling::SimpleSamplers;
use crate::utils::utils_shape_geometry::geometric_shape::{BVHCombinableShapeAABB, GeometricShapeSignature};
use crate::utils::utils_shape_geometry::shape_collection::{BVH, ProximaBudget, ProximaEngine, ProximaFunctions, ProximaSceneFilterOutput, ProximityOutputSumMode, ShapeCollectionQueryPairsList, SignedDistanceAggregator, SignedDistanceLossFunction, WitnessPoints, WitnessPointsCollection, WitnessPointsType};

#[derive(Clone)]
pub struct OTFRobotLinkTransformSpecification;
impl OTFRobotLinkTransformSpecification {
    fn internal_call(robot_set_fk_result: &RobotSetFKResult, specs: &Vec<RobotSetLinkTransformSpecification>) -> f64 {
        let mut out_error = 0.0;
        for s in specs {
            match s {
                RobotSetLinkTransformSpecification::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let se3_delta = pose.distance_function(&goal, true).expect("error");
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_error += weight * se3_delta;
                }
                RobotSetLinkTransformSpecification::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let position = pose.translation();
                    let r3_delta = (goal - &position).norm();
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_error += weight * r3_delta;
                }
                RobotSetLinkTransformSpecification::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
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
impl OptimaTensorFunction for OTFRobotLinkTransformSpecification {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let recompute_var_ifs = vec![RecomputeVarIf::IsAnyNewInput];
        let signatures = vec![OTFMutVarsObjectType::RobotSetFKResult];
        let params = vec![OTFMutVarsParams::None];
        let vars = mut_vars.get_vars(&signatures, &params, &recompute_var_ifs, input, immut_vars, session_key);
        let robot_set_fk_result = vars[0].unwrap_robot_set_fk_result();

        let spec_object = immut_vars.object_ref(&OTFImmutVarsObjectType::RobotLinkTransformSpecificationCollection).expect("must have RobotLinkSpecificationCollection");
        let spec = spec_object.unwrap_robot_link_transform_specification_collection();
        let specs = spec.robot_set_link_specification_refs();

        let out = Self::internal_call(robot_set_fk_result, specs);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out)));
    }

    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let recompute_var_ifs = vec![RecomputeVarIf::IsAnyNewInput];
        let signatures = vec![OTFMutVarsObjectType::RobotSetFKDOFPerturbationsResult];
        let params = vec![OTFMutVarsParams::None];

        let vars = mut_vars.get_vars(&signatures, &params, &recompute_var_ifs, input, immut_vars, &session_key);
        let robot_set_fk_dof_perturbations_result = vars[0].unwrap_robot_set_fk_dof_perturbations_result();
        let perturbation = robot_set_fk_dof_perturbations_result.perturbation();

        let spec_object = immut_vars.object_ref(&OTFImmutVarsObjectType::RobotLinkTransformSpecificationCollection).expect("must have RobotLinkSpecificationCollection");
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

/*
#[derive(Clone)]
pub struct OTFRobotJointVelLinfNorm;
impl OptimaTensorFunction for OTFRobotJointVelLinfNorm {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).expect("must have TimedGenericRobotJointStateWindowMemoryContainer");
        let window_memory_container = object.unwrap_timed_generic_robot_joint_state_window_memory_container();

        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::GenericRobotJointStateCurrTime).expect("must have GenericRobotJointStateCurrTime");
        let curr_time = object.unwrap_generic_robot_joint_state_curr_time();

        let prev = window_memory_container.c.object_ref(0);

        let times = Some([*curr_time, prev.time()]);

        // These are assumed to be vectors as they are robot joint states.
        let curr_vector = input.unwrap_vector();
        let prev_vector = prev.joint_state();

        let res = RobotJointStateDerivativeUtils::joint_state_velocity_linfnorm(&curr_vector, &prev_vector, times);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(res)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).expect("must have TimedGenericRobotJointStateWindowMemoryContainer");
        let window_memory_container = object.unwrap_timed_generic_robot_joint_state_window_memory_container();

        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::GenericRobotJointStateCurrTime).expect("must have GenericRobotJointStateCurrTime");
        let curr_time = object.unwrap_generic_robot_joint_state_curr_time();

        let prev = window_memory_container.c.object_ref(0);

        let times = Some([*curr_time, prev.time()]);

        // These are assumed to be vectors as they are robot joint states.
        let curr_vector = input.unwrap_vector();
        let prev_vector = prev.joint_state();

        let res = RobotJointStateDerivativeUtils::joint_state_velocity_linfnorm_derivative(&curr_vector, &prev_vector, times);
        return Ok(OTFResult::Complete(OptimaTensor::new_from_vector(res)));

    }
}
*/

#[derive(Clone)]
pub struct OTFRobotJointStateDerivative {
    num_dofs: usize,
    derivative_order: usize
}
impl OTFRobotJointStateDerivative {
    pub fn new(num_dofs: usize, derivative_order: usize) -> Self {
        assert!(derivative_order == 1 || derivative_order == 2 || derivative_order == 3);

        Self {
            num_dofs,
            derivative_order
        }
    }
}
impl OptimaTensorFunction for OTFRobotJointStateDerivative {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![self.num_dofs]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).expect("must have TimedGenericRobotJointStateWindowMemoryContainer");
        let window_memory_container = object.unwrap_timed_generic_robot_joint_state_window_memory_container();

        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::GenericRobotJointStateCurrTime).expect("must have GenericRobotJointStateCurrTime");
        let curr_time = object.unwrap_generic_robot_joint_state_curr_time();

        let out_vec = if self.derivative_order == 1 {
            let prev0 = window_memory_container.c.object_ref(0);
            let times = Some([*curr_time, prev0.time()]);

            let curr_vector = input.unwrap_vector();
            let prev_vector = prev0.joint_state();

            RobotJointStateDerivativeUtils::joint_state_velocity(curr_vector, prev_vector, times)
        } else if self.derivative_order == 2 {
            let prev0 = window_memory_container.c.object_ref(0);
            let prev1 = window_memory_container.c.object_ref(1);
            let times = Some([*curr_time, prev0.time(), prev1.time()]);

            let curr_vector = input.unwrap_vector();
            let prev0_vector = prev0.joint_state();
            let prev1_vector = prev1.joint_state();

            RobotJointStateDerivativeUtils::joint_state_acceleration(curr_vector, prev0_vector, prev1_vector, times)
        } else if self.derivative_order == 3 {
            let prev0 = window_memory_container.c.object_ref(0);
            let prev1 = window_memory_container.c.object_ref(1);
            let prev2 = window_memory_container.c.object_ref(2);
            let times = Some([*curr_time, prev0.time(), prev1.time(), prev2.time()]);

            let curr_vector = input.unwrap_vector();
            let prev0_vector = prev0.joint_state();
            let prev1_vector = prev1.joint_state();
            let prev2_vector = prev2.joint_state();

            RobotJointStateDerivativeUtils::joint_state_jerk(curr_vector, prev0_vector, prev1_vector, prev2_vector, times)
        } else {
            unreachable!()
        };

        return Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)));
    }

    fn derivative_analytical_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut matrix = DMatrix::identity(self.num_dofs, self.num_dofs);

        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::TimedGenericRobotJointStateWindowMemoryContainer).expect("must have TimedGenericRobotJointStateWindowMemoryContainer");
        let window_memory_container = object.unwrap_timed_generic_robot_joint_state_window_memory_container();

        let object = immut_vars.c.object_ref(&OTFImmutVarsObjectType::GenericRobotJointStateCurrTime).expect("must have GenericRobotJointStateCurrTime");
        let curr_time = object.unwrap_generic_robot_joint_state_curr_time();

        if self.derivative_order == 1 {
            let prev0 = window_memory_container.c.object_ref(0);
            let times = [*curr_time, prev0.time()];
            let dt = RobotJointStateDerivativeUtils::joint_state_velocity_average_time(times);
            matrix *= 1.0/dt;
        } else if self.derivative_order == 2 {
            let prev0 = window_memory_container.c.object_ref(0);
            let prev1 = window_memory_container.c.object_ref(1);
            let times = [*curr_time, prev0.time(), prev1.time()];
            let dt = RobotJointStateDerivativeUtils::joint_state_acceleration_average_time(times);
            matrix *= 1.0/(dt*dt);
        } else if self.derivative_order == 3 {
            let prev0 = window_memory_container.c.object_ref(0);
            let prev1 = window_memory_container.c.object_ref(1);
            let prev2 = window_memory_container.c.object_ref(2);
            let times = [*curr_time, prev0.time(), prev1.time(), prev2.time()];
            let dt = RobotJointStateDerivativeUtils::joint_state_jerk_average_time(times);
            matrix *= 1.0/(dt*dt*dt);
        } else {
            unreachable!()
        };

        return Ok(OTFResult::Complete(OptimaTensor::new_from_matrix(matrix)));
    }

    fn derivative2_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(input, 2));
        return Ok(OTFResult::Complete(out));
    }

    fn derivative3_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(input, 3));
        return Ok(OTFResult::Complete(out));
    }

    fn derivative4_analytical_raw(&self, input: &OptimaTensor, _immut_vars: &OTFImmutVars, _mut_vars: &mut OTFMutVars, _session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let out = OptimaTensor::new_zeros(self.get_output_dimensions_from_derivative_order(input, 4));
        return Ok(OTFResult::Complete(out));
    }
}

pub struct RobotJointStateDerivativeUtils;
impl RobotJointStateDerivativeUtils {
    pub fn joint_state_velocity(curr_state: &DVector<f64>,
                                prev_state: &DVector<f64>,
                                times: Option<[f64; 2]>) -> DVector<f64> {
        let vel = Self::generic_joint_state_diff(curr_state, prev_state, match times {
            None => { None }
            Some(times) => { Some(Self::joint_state_velocity_average_time(times)) }
        });
        return vel;
    }
    pub fn joint_state_acceleration(curr_state: &DVector<f64>,
                                    prev_state0: &DVector<f64>,
                                    prev_state1: &DVector<f64>,
                                    times: Option<[f64; 3]>) -> DVector<f64> {
        let dt = match times {
            None => { 1.0 }
            Some(times) => {
                Self::joint_state_acceleration_average_time(times)
            }
        };

        let vel0 = Self::generic_joint_state_diff(curr_state, prev_state0, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        let vel1 = Self::generic_joint_state_diff(prev_state0, prev_state1, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        let accel = Self::generic_joint_state_diff(&vel0, &vel1, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });

        return accel;
    }
    pub fn joint_state_jerk(curr_state: &DVector<f64>,
                            prev_state0: &DVector<f64>,
                            prev_state1: &DVector<f64>,
                            prev_state2: &DVector<f64>,
                            times: Option<[f64; 4]>) -> DVector<f64> {
        let dt = match times {
            None => { 1.0 }
            Some(times) => {
                Self::joint_state_jerk_average_time(times)
            }
        };
        let vel0 = Self::generic_joint_state_diff(curr_state, prev_state0, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        let vel1 = Self::generic_joint_state_diff(prev_state0, prev_state1, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        let vel2 = Self::generic_joint_state_diff(prev_state1, prev_state2, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        let accel1 = Self::generic_joint_state_diff(&vel0, &vel1, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        let accel2 = Self::generic_joint_state_diff(&vel1, &vel2, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        let jerk = Self::generic_joint_state_diff(&accel1, &accel2, match times {
            None => { None }
            Some(times) => { Some(dt) }
        });
        return jerk;
    }

    pub fn joint_state_velocity_average_time(times: [f64; 2]) -> f64 {
        times[0] - times[1]
    }
    pub fn joint_state_acceleration_average_time(times: [f64; 3]) -> f64 {
        let mut sum = 0.0;
        for i in 0..2 {
            sum += times[i] - times[i+1]
        }
        sum / 2.0
    }
    pub fn joint_state_jerk_average_time(times: [f64; 4]) -> f64 {
        let mut sum = 0.0;
        for i in 0..3 {
            sum += times[i] - times[i+1]
        }
        sum / 3.0
    }

    /*
    pub fn joint_state_velocity_l2norm_squared(curr_state: &DVector<f64>,
                                               prev_state: &DVector<f64>,
                                               times: Option<[f64; 2]>) -> f64 {
        let vec = Self::joint_state_velocity(curr_state, prev_state, times);
        return vec.norm().powi(2);
    }
    pub fn joint_state_acceleration_l2norm_squared(curr_state: &DVector<f64>,
                                                   prev_state0: &DVector<f64>,
                                                   prev_state1: &DVector<f64>,
                                                   times: Option<[f64; 3]>) -> f64 {
        let vec = Self::joint_state_acceleration(curr_state, prev_state0, prev_state1, times);
        return vec.norm().powi(2);
    }
    pub fn joint_state_jerk_l2norm_squared(curr_state: &DVector<f64>,
                                           prev_state0: &DVector<f64>,
                                           prev_state1: &DVector<f64>,
                                           prev_state2: &DVector<f64>,
                                           times: Option<[f64; 4]>) -> f64 {
        let vec = Self::joint_state_jerk(curr_state, prev_state0, prev_state1, prev_state2, times);
        return vec.norm().powi(2);
    }

    pub fn joint_state_velocity_l2norm_squared_derivative(curr_state: &DVector<f64>,
                                                          prev_state: &DVector<f64>,
                                                          times: Option<[f64; 2]>) -> DVector<f64> {
        assert_eq!(curr_state.len(), prev_state.len());

        let delta_time = match times {
            None => { 1.0 }
            Some(times) => { times[0] - times[1] }
        };

        let delta_time_2 = 2.0 / (delta_time * delta_time);

        let mut out = DVector::zeros(curr_state.len());
        curr_state.iter().zip(prev_state.iter()).enumerate().for_each(|(idx, (x, y))| out[idx] = delta_time_2 * (*x - *y));

        return out;
    }
    pub fn joint_state_acceleration_l2norm_squared_derivative(curr_state: &DVector<f64>,
                                                              prev_state0: &DVector<f64>,
                                                              prev_state1: &DVector<f64>,
                                                              times: Option<[f64; 3]>) -> DVector<f64> {
        assert_eq!(curr_state.len(), prev_state0.len());
        assert_eq!(prev_state0.len(), prev_state1.len());

        let mut dt = match times {
            None => { 1.0 }
            Some(times) => {
                let mut sum = 0.0;
                for i in 0..2 {
                    sum += times[i] - times[i+1]
                }
                sum / 2.0
            }
        };
        let denom = dt.powi(4);
        /*
        let d0 = match times {
            None => { 1.0 }
            Some(times) => { times[0] - times[1] }
        };
        let d0_squared = d0 * d0;

        let d1 = match times {
            None => { 1.0 }
            Some(times) => { times[1] - times[2] }
        };

        let d2 = match times {
            None => { 1.0 }
            Some(times) => { times[0] - times[2] }
        };
        let d2_squared = d2 * d2;

        let denom = d0_squared * d1 * d2_squared;
        */

        let mut out = DVector::zeros(curr_state.len());
        izip!(curr_state, prev_state0, prev_state1)
            .enumerate()
            .for_each(|(idx, (x, y, z))|  out[idx] = (2.0*(*x - 2.0 * *y + *z)) / denom );

        out
    }
    pub fn joint_state_jerk_l2norm_squared_derivative(curr_state: &DVector<f64>,
                                                      prev_state0: &DVector<f64>,
                                                      prev_state1: &DVector<f64>,
                                                      prev_state2: &DVector<f64>,
                                                      times: Option<[f64; 4]>) -> DVector<f64> {
        assert_eq!(curr_state.len(), prev_state0.len());
        assert_eq!(prev_state0.len(), prev_state1.len());
        assert_eq!(prev_state1.len(), prev_state2.len());

        let mut dt = match times {
            None => { 1.0 }
            Some(times) => {
                let mut sum = 0.0;
                for i in 0..3 {
                    sum += times[i] - times[i+1]
                }
                sum / 3.0
            }
        };
        let denom = dt.powi(6);
        /*
        let d0 = match times {
            None => { 1.0 }
            Some(times) => { times[0] - times[1] }
        };
        let d0_squared = d0 * d0;

        let d1 = match times {
            None => { 1.0 }
            Some(times) => { times[1] - times[2] }
        };

        let d2 = match times {
            None => { 1.0 }
            Some(times) => { times[2] - times[3] }
        };

        let d3 = match times {
            None => { 1.0 }
            Some(times) => { ((times[0] - times[1]) + (times[1] - times[2])) * 0.5  }
        };
        let d3_squared = d3 * d3;

        let d4 = match times {
            None => { 1.0 }
            Some(times) => { ((times[1] - times[2]) + (times[2] - times[3])) * 0.5 }
        };

        let d5 = match times {
            None => { 1.0 }
            Some(times) => { (((times[0] - times[1]) + (times[1] - times[2])) * 0.5 + ((times[1] - times[2]) + (times[2] - times[3])) * 0.5) * 0.5 }
        };
        let d5_squared = d5*d5;

        let denom = -(d0_squared * d1 * d2 * d3_squared * d4 * d5_squared);
        */

        let mut out = DVector::zeros(curr_state.len());
        izip!(curr_state, prev_state0, prev_state1, prev_state2)
            .enumerate()
            .for_each(|(idx, (x, y, z, w))|  out[idx] = (2.0 * (-*w + *x - 3.0 * *y + 3.0 * *z)) / denom );

        out
    }

    pub fn joint_state_velocity_linfnorm(curr_state: &DVector<f64>,
                                         prev_state: &DVector<f64>,
                                         times: Option<[f64; 2]>) -> f64 {
        let vec = Self::joint_state_velocity(curr_state, prev_state, times);
        let mut max = -f64::INFINITY;
        for v in &vec {
            let a = v.abs();
            if a > max { max = a; }
        }
        return max;
    }
    pub fn joint_state_acceleration_linfnorm(curr_state: &DVector<f64>,
                                             prev_state0: &DVector<f64>,
                                             prev_state1: &DVector<f64>,
                                             times: Option<[f64; 3]>) -> f64 {
        let vec = Self::joint_state_acceleration(curr_state, prev_state0, prev_state1, times);
        let mut max = -f64::INFINITY;
        for v in &vec {
            let a = v.abs();
            if a > max { max = a; }
        }
        return max;
    }
    pub fn joint_state_jerk_linfnorm(curr_state: &DVector<f64>,
                                     prev_state0: &DVector<f64>,
                                     prev_state1: &DVector<f64>,
                                     prev_state2: &DVector<f64>,
                                     times: Option<[f64; 4]>) -> f64 {
        let vec = Self::joint_state_jerk(curr_state, prev_state0, prev_state1, prev_state2, times);
        let mut max = -f64::INFINITY;
        for v in &vec {
            let a = v.abs();
            if a > max { max = a; }
        }
        return max;
    }

    pub fn joint_state_velocity_linfnorm_derivative(curr_state: &DVector<f64>,
                                                    prev_state: &DVector<f64>,
                                                    times: Option<[f64; 2]>) -> DVector<f64> {
        let dt = match times {
            None => { 1.0 }
            Some(times) => { times[0] - times[1] }
        };
        let vec = Self::joint_state_velocity(curr_state, prev_state, times);
        let mut out = DVector::zeros(curr_state.len());
        let mut max = -f64::INFINITY;
        let mut max_idx = usize::MAX;
        for (i, v) in vec.iter().enumerate() {
            let a = v.abs();
            if a > max { max = a; max_idx = i; }
        }
        if vec[max_idx] > 0.0 { out[max_idx] = 1.0 / dt } else { out[max_idx] = -1.0 / dt };
        return out;
    }
    pub fn joint_state_acceleration_linfnorm_derivative(curr_state: &DVector<f64>,
                                                        prev_state0: &DVector<f64>,
                                                        prev_state1: &DVector<f64>,
                                                        times: Option<[f64; 3]>) -> DVector<f64> {
        let dt = match times {
            None => { 1.0 }
            Some(times) => {
                let mut sum = 0.0;
                for i in 0..2 {
                    sum += times[i] - times[i+1]
                }
                sum / 2.0
            }
        };
        let vec = Self::joint_state_acceleration(curr_state, prev_state0, prev_state1, times);
        let mut out = DVector::zeros(curr_state.len());
        let mut max = -f64::INFINITY;
        let mut max_idx = usize::MAX;
        for (i, v) in vec.iter().enumerate() {
            let a = v.abs();
            if a > max { max = a; max_idx = i; }
        }
        if vec[max_idx] > 0.0 { out[max_idx] = 1.0 / dt } else { out[max_idx] = -1.0 / dt };
        return out;
    }
    pub fn joint_state_jerk_linfnorm_derivative(curr_state: &DVector<f64>,
                                                prev_state0: &DVector<f64>,
                                                prev_state1: &DVector<f64>,
                                                prev_state2: &DVector<f64>,
                                                times: Option<[f64; 4]>) -> DVector<f64> {
        let dt = match times {
            None => { 1.0 }
            Some(times) => {
                let mut sum = 0.0;
                for i in 0..3 {
                    sum += times[i] - times[i+1]
                }
                sum / 3.0
            }
        };
        let vec = Self::joint_state_jerk(curr_state, prev_state0, prev_state1, prev_state2, times);
        let mut out = DVector::zeros(curr_state.len());
        let mut max = -f64::INFINITY;
        let mut max_idx = usize::MAX;
        for (i, v) in vec.iter().enumerate() {
            let a = v.abs();
            if a > max { max = a; max_idx = i; }
        }
        if vec[max_idx] > 0.0 { out[max_idx] = 1.0 / dt } else { out[max_idx] = -1.0 / dt };
        return out;
    }
    */

    fn generic_joint_state_diff(a: &DVector<f64>,
                                b: &DVector<f64>,
                                delta_time: Option<f64>) -> DVector<f64> {
        let delta_time = match delta_time {
            None => { 1.0 }
            Some(delta_time) => {
                delta_time
            }
        };

        let mut vel = (a - b) / delta_time;
        return vel;
    }
}

#[derive(Clone)]
pub struct OTFRobotCollisionProximityProxima {
    r: f64,
    a_max: f64,
    d_max: f64,
    budget: ProximaBudget,
    loss_function: SignedDistanceLossFunction,
    aggregator: SignedDistanceAggregator,
    fd_mode: RobotCollisionProximityGradientFDMode
}
impl OTFRobotCollisionProximityProxima {
    pub fn new(r: f64, a_max: f64, d_max: f64, budget: ProximaBudget, loss_function: SignedDistanceLossFunction, aggregator: SignedDistanceAggregator, fd_mode: RobotCollisionProximityGradientFDMode) -> Self {
        Self {
            r,
            a_max,
            d_max,
            budget,
            loss_function,
            aggregator,
            fd_mode
        }
    }
    fn call_raw_with_necessary_vars(&self,
                                    robot_set_joint_state: &RobotSetJointState,
                                    robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                    proxima_engine: &mut ProximaEngine,
                                    inclusion_list: Option<&ShapeCollectionQueryPairsList>,
                                    r_override: Option<f64>) -> Result<OTFResult, OptimaError> {
        let res = robot_geometric_shape_scene.proxima_proximity_query(&robot_set_joint_state,
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

        let filter_output = robot_geometric_shape_scene.proxima_scene_filter(robot_set_joint_state, None, proxima_engine, self.d_max, self.a_max, self.loss_function.clone(), 1.0, &None).expect("error");
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
                                             proxima_engine: &mut ProximaEngine) -> Result<OTFResult, OptimaError> {
        let res = robot_geometric_shape_scene.proxima_proximity_query(&robot_set_joint_state,
                                                                      None,
                                                                      proxima_engine,
                                                                      self.d_max,
                                                                      self.a_max,
                                                                      self.loss_function.clone(),
                                                                      self.aggregator.clone(),
                                                                      1.0,
                                                                      self.budget.clone(),
                                                                      &None).expect("error");
        let witness_points_collection = res.output_witness_points_collection();

        let out_vec = general_robot_jacobian_collision_proximity_gradient(&witness_points_collection, robot_set_joint_state, robot_geometric_shape_scene, self.a_max, &self.loss_function, &self.aggregator);

        return Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)));
    }
}
impl OptimaTensorFunction for OTFRobotCollisionProximityProxima {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let mut vars = mut_vars.get_vars(&vec![OTFMutVarsObjectType::ProximaEngine], &vec![OTFMutVarsParams::None], &vec![RecomputeVarIf::Never], input, immut_vars, session_key);
        let mut proxima_engine = vars[0].unwrap_proxima_engine_mut();

        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();
        let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

        return self.call_raw_with_necessary_vars(&robot_set_joint_state, robot_geometric_shape_scene, &mut proxima_engine, None, None);
    }

    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let mut vars = mut_vars.get_vars(&vec![OTFMutVarsObjectType::ProximaEngine], &vec![OTFMutVarsParams::None], &vec![RecomputeVarIf::Never], input, immut_vars, &session_key);
        let mut proxima_engine = vars[0].unwrap_proxima_engine_mut();

        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();
        let robot_set_joint_state = robot_geometric_shape_scene.robot_set().spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

        let ret = match &self.fd_mode {
            RobotCollisionProximityGradientFDMode::RawFiniteDifference => {
                self.derivative_finite_difference_raw(&robot_set_joint_state, robot_geometric_shape_scene, proxima_engine)
            }
            RobotCollisionProximityGradientFDMode::PreFilteredFiniteDifference => {
                self.derivative_finite_difference_pre_filtered(&robot_set_joint_state, robot_geometric_shape_scene, proxima_engine)
            }
            RobotCollisionProximityGradientFDMode::JacobianFiniteDifference => {
                self.derivative_finite_difference_jacobian(&robot_set_joint_state, robot_geometric_shape_scene, proxima_engine)
            }
        };

        mut_vars.close_session(&session_key);

        return ret;
    }
}

#[derive(Clone, Debug)]
pub enum RobotCollisionProximityGradientFDMode {
    RawFiniteDifference,
    PreFilteredFiniteDifference,
    JacobianFiniteDifference,
}

fn general_robot_jacobian_collision_proximity_gradient(witness_points_collection: &WitnessPointsCollection,
                                                       robot_set_joint_state: &RobotSetJointState,
                                                       robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                                       a_max: f64,
                                                       loss_function: &SignedDistanceLossFunction,
                                                       aggregator: &SignedDistanceAggregator) -> DVector<f64> {
    let mut jacobian_option_pairs: Vec<[Option<DMatrix<f64>>; 2]> = Vec::new();

    let robot_set = robot_geometric_shape_scene.robot_set();

    // set up jacobian matrices for witness points
    for wp in witness_points_collection.collection() {
            let mut jacobian_option_pair = [None, None];

            let witness_points = &wp.witness_points();
            let signatures = wp.shape_signatures();

            match signatures.0 {
                GeometricShapeSignature::RobotSetLink { robot_idx_in_set, link_idx_in_robot, .. } => {
                    let jacobian = robot_set.robot_set_kinematics_module().compute_jacobian(&robot_set_joint_state, robot_idx_in_set, None, link_idx_in_robot, &JacobianEndPoint::Global(witness_points.0.clone()), None, JacobianMode::Translational).expect("error");
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
        }

    let x_0 = witness_points_collection.compute_proximity_output_sum(&ProximityOutputSumMode::AverageSignedDistance {
        a_max,
        shape_collection: robot_geometric_shape_scene.shape_collection()
    }, loss_function, aggregator);

    let num_dofs = robot_set_joint_state.concatenated_state().len();
    let mut out_vec = DVector::zeros(num_dofs);
    for dof_idx in 0..num_dofs {
        let mut robot_set_joint_state_dvec = DVector::<f64>::zeros(num_dofs);
        robot_set_joint_state_dvec[dof_idx] += FD_PERTURBATION;

        let mut witness_points_collection_h = WitnessPointsCollection::new();

        for (wp_idx, wp) in witness_points_collection.collection().iter().enumerate() {
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
        }

        let x_h = witness_points_collection_h.compute_proximity_output_sum(&ProximityOutputSumMode::AverageSignedDistance {
            a_max,
            shape_collection: robot_geometric_shape_scene.shape_collection()
        }, loss_function, aggregator);

        let f_h = (-x_0 + x_h) / FD_PERTURBATION;
        out_vec[dof_idx] = f_h;
    }

    out_vec
}

/*
#[derive(Clone)]
pub struct OTFRobotSetCollisionProximityQuery {
    robot_set_collision_proximity_options: RobotSetCollisionProximityParams,
    robot_set_collision_proximity_gradient_fd_mode: RobotSetCollisionProximityGradientFDMode
}
impl OTFRobotSetCollisionProximityQuery {
    pub fn new(robot_set_collision_proximity_options: RobotSetCollisionProximityParams,
               robot_set_collision_proximity_gradient_mode: RobotSetCollisionProximityGradientFDMode) -> Self {
        Self {
            robot_set_collision_proximity_options,
            robot_set_collision_proximity_gradient_fd_mode: robot_set_collision_proximity_gradient_mode
        }
    }
    pub fn compare_fd_derivative_modes(&self, init_state: &OptimaTensor, num_calls: usize, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) {
        let num_dofs = init_state.unwrap_vector().len();

        let mut rand_inputs = vec![];
        rand_inputs.push(init_state.clone());
        for i in 0..num_calls {
            let rand_velocities = SimpleSamplers::uniform_samples(&vec![(-0.01, 0.01); num_dofs]);
            let rand_velocities_dvec = DVector::from_vec(rand_velocities);
            let new_state = rand_inputs[i].unwrap_vector() + &rand_velocities_dvec;
            rand_inputs.push(OptimaTensor::new_from_vector(new_state));
            // rand_inputs.push(OptimaTensor::new_random_sampling(OTFDimensions::Fixed(vec![num_dofs])));
        }

        let mut raw_times = AveragingFloat::new();
        let mut jacobian_times = AveragingFloat::new();

        for r in &rand_inputs {
            println!("input: {:?}", r);
            let start = instant::Instant::now();
            let res = self.raw_derivative_finite_difference(r, immut_vars, mut_vars).expect("error");
            let stop = start.elapsed().as_secs_f64();
            raw_times.add_new_value(stop);
            println!("{:?} raw fd ---> {:?}", instant::Duration::from_secs_f64(stop), res);
            let start = instant::Instant::now();
            let res = self.jacobian_derivative_finite_difference(r, immut_vars, mut_vars).expect("error");
            let stop = start.elapsed().as_secs_f64();
            jacobian_times.add_new_value(stop);
            println!("{:?} jacobian fd ---> {:?}", instant::Duration::from_secs_f64(stop), res);
            println!();
        }
    }
    fn call_raw_proxima_r_fixed_at_1(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);
        let new_proximity_options = match &self.robot_set_collision_proximity_options {
            RobotSetCollisionProximityParams::Proxima { budget, r: _, d_max, a_max, loss_function } => {
                RobotSetCollisionProximityParams::Proxima {
                    budget: budget.clone(),
                    r: 1.0,
                    d_max: *d_max,
                    a_max: *a_max,
                    loss_function: loss_function.clone()
                }
            }
            RobotSetCollisionProximityParams::BVHAABB { .. } => { self.robot_set_collision_proximity_options.clone() }
            RobotSetCollisionProximityParams::NaiveIteration { .. } => { self.robot_set_collision_proximity_options.clone() }
        };

        let signatures = vec![OTFMutVarsObjectType::WitnessPointsCollection];
        let params = vec![OTFMutVarsParams::RobotSetCollisionAvoidParams(&new_proximity_options), OTFMutVarsParams::None];
        let recompute_var_ifs = vec![RecomputeVarIf::Always];
        let vars = mut_vars.get_vars(&signatures, &params, &recompute_var_ifs, input, immut_vars, &session_key);
        let witness_points_collection = vars[0].unwrap_witness_points_collection();

        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();

        let out_sum = self.robot_set_collision_proximity_options.compute_robot_proximity_output_sum(witness_points_collection, robot_geometric_shape_scene);

        mut_vars.close_session(&session_key);

        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out_sum)));
    }
    fn raw_derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let x_0_res = self.call_raw_proxima_r_fixed_at_1(input, immut_vars, mut_vars).expect("error");
        let x_0 = x_0_res.unwrap_tensor().unwrap_scalar();

        let num_dofs = input.unwrap_vector().len();
        let mut out_vec = DVector::zeros(num_dofs);
        for dof_idx in 0..num_dofs {
            let mut input_h = input.clone();
            input_h.vectorized_data_mut()[dof_idx] += FD_PERTURBATION;
            let x_h_res = self.call_raw_proxima_r_fixed_at_1(&input_h, immut_vars, mut_vars).expect("error");
            let x_h = x_h_res.unwrap_tensor().unwrap_scalar();

            let f_h = (x_h - x_0) / FD_PERTURBATION;
            out_vec[dof_idx] = f_h;
        }
        return Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)));
    }
    fn jacobian_derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        let session_key = mut_vars.register_session(input);

        let new_proximity_options = match &self.robot_set_collision_proximity_options {
            RobotSetCollisionProximityParams::Proxima { budget, r: _, d_max, a_max, loss_function } => {
                RobotSetCollisionProximityParams::Proxima {
                    budget: budget.clone(),
                    r: 1.0,
                    d_max: *d_max,
                    a_max: *a_max,
                    loss_function: loss_function.clone()
                }
            }
            RobotSetCollisionProximityParams::BVHAABB { .. } => { self.robot_set_collision_proximity_options.clone() }
            RobotSetCollisionProximityParams::NaiveIteration { .. } => { self.robot_set_collision_proximity_options.clone() }
        };
        let signatures = vec![OTFMutVarsObjectType::WitnessPointsCollection];
        let params = vec![OTFMutVarsParams::RobotSetCollisionAvoidParams(&new_proximity_options)];
        let recompute_var_ifs = vec![RecomputeVarIf::Always];
        let vars = mut_vars.get_vars(&signatures, &params, &recompute_var_ifs, input, immut_vars, &session_key);
        let witness_points_collection = vars[0].unwrap_witness_points_collection();

        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();
        let robot_set = immut_vars.ref_robot_set();

        let x_0 = new_proximity_options.compute_robot_proximity_output_sum(&witness_points_collection, robot_geometric_shape_scene);

        let mut jacobian_option_pairs: Vec<[Option<DMatrix<f64>>; 2]> = Vec::new();

        let robot_set_joint_state = robot_set.spawn_robot_set_joint_state(input.unwrap_vector().clone()).expect("error");

        // set up jacobian matrices for witness points
        for wp in witness_points_collection.collection() {
            let mut jacobian_option_pair = [None, None];

            let witness_points = &wp.witness_points();
            let signatures = wp.shape_signatures();

            match signatures.0 {
                GeometricShapeSignature::RobotSetLink { robot_idx_in_set, link_idx_in_robot, .. } => {
                    let jacobian = robot_set.robot_set_kinematics_module().compute_jacobian(&robot_set_joint_state, robot_idx_in_set, None, link_idx_in_robot, &JacobianEndPoint::Global(witness_points.0.clone()), None, JacobianMode::Translational).expect("error");
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
        }

        let num_dofs = robot_set_joint_state.concatenated_state().len();
        let mut out_vec = DVector::zeros(num_dofs);
        for dof_idx in 0..num_dofs {
            let mut robot_set_joint_state_dvec = DVector::<f64>::zeros(num_dofs);
            robot_set_joint_state_dvec[dof_idx] += FD_PERTURBATION;

            let mut witness_points_collection_h = WitnessPointsCollection::new();

            for (wp_idx, wp) in witness_points_collection.collection().iter().enumerate() {
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
            }

            let x_h = new_proximity_options.compute_robot_proximity_output_sum(&witness_points_collection_h, robot_geometric_shape_scene);

            let f_h = (-x_0 + x_h) / FD_PERTURBATION;
            out_vec[dof_idx] = f_h;
        }
        mut_vars.close_session(&session_key);

        Ok(OTFResult::Complete(OptimaTensor::new_from_vector(out_vec)))
    }
}
impl OptimaTensorFunction for OTFRobotSetCollisionProximityQuery {
    fn output_dimensions(&self) -> Vec<usize> { vec![] }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let signatures = vec![OTFMutVarsObjectType::WitnessPointsCollection];
        let params = vec![OTFMutVarsParams::RobotSetCollisionAvoidParams(&self.robot_set_collision_proximity_options), OTFMutVarsParams::None];
        let recompute_var_ifs = vec![RecomputeVarIf::IsAnyNewInput];
        let vars = mut_vars.get_vars(&signatures, &params, &recompute_var_ifs, input, immut_vars, session_key);
        let witness_points_collection = vars[0].unwrap_witness_points_collection();

        let robot_geometric_shape_scene = immut_vars.ref_robot_geometric_shape_scene();

        let out_sum = self.robot_set_collision_proximity_options.compute_robot_proximity_output_sum(witness_points_collection, robot_geometric_shape_scene);

        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out_sum)));
    }

    fn derivative_finite_difference(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars) -> Result<OTFResult, OptimaError> {
        return match self.robot_set_collision_proximity_gradient_fd_mode {
            RobotSetCollisionProximityGradientFDMode::RawFiniteDifference => {
                self.raw_derivative_finite_difference(input, immut_vars, mut_vars)
            }
            RobotSetCollisionProximityGradientFDMode::JacobianFiniteDifference => {
                self.jacobian_derivative_finite_difference(input, immut_vars, mut_vars)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum RobotSetCollisionProximityParams {
    Proxima { budget: ProximaBudget, r: f64, d_max: f64, a_max: f64, loss_function: SignedDistanceLossFunction },
    BVHAABB { d_max: f64, a_max: f64, loss_function: SignedDistanceLossFunction },
    NaiveIteration { d_max: f64, a_max: f64, loss_function: SignedDistanceLossFunction }
}
impl RobotSetCollisionProximityParams {
    pub fn compute_robot_proximity_output_sum(&self, witness_points_collection: &WitnessPointsCollection, robot_geometric_shape_scene: &RobotGeometricShapeScene) -> f64 {
        let out_sum = match self {
            RobotSetCollisionProximityParams::Proxima {  a_max, loss_function, .. } => {
                witness_points_collection.compute_proximity_output_sum(&ProximityOutputSumMode::AverageSignedDistance { a_max: *a_max, shape_collection: robot_geometric_shape_scene.shape_collection() }, loss_function, &SignedDistanceAggregator::PNorm { p: 10.0 })
            }
            RobotSetCollisionProximityParams::BVHAABB { a_max, loss_function, .. } => {
                witness_points_collection.compute_proximity_output_sum(&ProximityOutputSumMode::AverageSignedDistance { a_max: *a_max, shape_collection: robot_geometric_shape_scene.shape_collection() }, loss_function, &SignedDistanceAggregator::PNorm { p: 10.0 })
            }
            RobotSetCollisionProximityParams::NaiveIteration { a_max, loss_function, .. } => {
                witness_points_collection.compute_proximity_output_sum(&ProximityOutputSumMode::AverageSignedDistance { a_max: *a_max, shape_collection: robot_geometric_shape_scene.shape_collection() }, loss_function, &SignedDistanceAggregator::PNorm { p: 10.0 })
            }
        };
        out_sum
    }
}

#[derive(Clone, Debug)]
pub enum RobotSetCollisionProximityGradientFDMode {
    RawFiniteDifference,
    JacobianFiniteDifference,
}

#[derive(Clone, Debug)]
pub enum RobotSetCollisionProximityAggregationMode {
    SimpleSum,
    PNorm { p: f64 }
}
impl RobotSetCollisionProximityAggregationMode {
    pub fn compute_robot_proximity_output_value(&self,
                                                witness_points_collection: &WitnessPointsCollection,
                                                robot_geometric_shape_scene: &RobotGeometricShapeScene,
                                                a_max: f64,
                                                loss_function: &SignedDistanceLossFunction) -> f64 {
        todo!()
    }
}
*/


