use nalgebra::{DVector, Vector6};
use crate::optima_tensor_function::{OptimaTensor, OptimaTensorFunction, OptimaTensorFunctionClone, OTFImmutVars, OTFImmutVarsObject, OTFImmutVarsObjectType, OTFMutVars, OTFMutVarsObjectType, OTFMutVarsSessionKey, OTFResult, RecomputeVarIf};
use crate::robot_modules::robot_kinematics_module::{JacobianEndPoint, JacobianMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_robot::robot_set_link_specification::RobotSetLinkSpecification;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;

#[derive(Clone)]
pub struct OTFRobotSetLinkSpecification;
impl OptimaTensorFunction for OTFRobotSetLinkSpecification {
    fn output_dimensions(&self) -> Vec<usize> {
        vec![]
    }

    fn call_raw(&self, input: &OptimaTensor, immut_vars: &OTFImmutVars, mut_vars: &mut OTFMutVars, session_key: &OTFMutVarsSessionKey) -> Result<OTFResult, OptimaError> {
        let recompute_var_ifs = vec![RecomputeVarIf::IsAnyNewInput];
        let signatures = vec![OTFMutVarsObjectType::RobotSetFKResult];
        let vars = mut_vars.get_vars(&signatures, &recompute_var_ifs, input, immut_vars, session_key);
        let robot_set_fk_result = vars[0].unwrap_robot_set_fk_result();

        let spec_object = immut_vars.object_ref(&OTFImmutVarsObjectType::RobotLinkSpecificationCollection).expect("error");
        let spec = spec_object.unwrap_robot_link_specification_collection();
        let specs = spec.robot_set_link_specification_refs();

        let mut out_error = 0.0;
        for s in specs {
            match s {
                RobotSetLinkSpecification::LinkSE3PoseGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => unsafe {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let se3_delta = pose.distance_function(&goal, true).expect("error");
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_error += weight * se3_delta;
                }
                RobotSetLinkSpecification::LinkPositionGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
                    let pose = robot_set_fk_result.get_pose_from_idxs(*robot_idx_in_set, *link_idx_in_robot);
                    let position = pose.translation();
                    let r3_delta = (goal - &position).norm();
                    let weight = match weight {
                        None => { 1.0 }
                        Some(weight) => { *weight }
                    };
                    out_error += weight * r3_delta;
                }
                RobotSetLinkSpecification::LinkRotationGoal { robot_idx_in_set, link_idx_in_robot, goal, weight } => {
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

        return Ok(OTFResult::Complete(OptimaTensor::new_from_scalar(out_error)));
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
                    let disp_translation = (&pose_translation - &goal_translation);
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
                    let disp_translation = (&translation - goal);
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