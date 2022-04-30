#[cfg(not(target_arch = "wasm32"))]
use pyo3::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use serde::{Serialize, Deserialize};
use nalgebra::{DMatrix, Vector3};
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::robot_modules::robot_joint_state_module::{RobotJointState, RobotJointStateModule, RobotJointStateType};
use crate::utils::utils_console::{optima_print, PrintColor, PrintMode};
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{load_object_from_json_string, OptimaStemCellPath};
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_robot::joint::JointAxisPrimitiveType;
use crate::utils::utils_robot::robot_module_utils::RobotNames;
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3Pose, OptimaSE3PoseType};
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3PosePy};
#[cfg(target_arch = "wasm32")]
use crate::utils::utils_se3::optima_se3_pose::{OptimaSE3PoseWASM};
use crate::utils::utils_traits::{SaveAndLoadable, ToAndFromRonString};
#[cfg(target_arch = "wasm32")]
use crate::utils::utils_wasm::JsMatrix;

/// The `RobotKinematicsModule` performs operations related to a robot's kinematics.
/// For instance, one of the main subroutines afforded by this module is forward kinematics which
/// takes as input a robot joint_state and outputs the SE(3) poses of all links on the robot.
///
/// # Example
/// ```
/// use nalgebra::DVector;
/// use optima::robot_modules::robot_kinematics_module::RobotKinematicsModule;
/// use optima::robot_modules::robot_joint_state_module::RobotJointStateModule;
/// use optima::utils::utils_robot::robot_module_utils::RobotNames;
/// use optima::utils::utils_se3::optima_se3_pose::OptimaSE3PoseType;
///
/// let robot_joint_state_module = RobotJointStateModule::new_from_names(RobotNames::new_base("ur5")).expect("error");
/// let robot_joint_state = robot_joint_state_module.spawn_robot_joint_state_try_auto_type(DVector::zeros(6)).expect("error");
///
/// let robot_kinematics_module = RobotKinematicsModule::new_from_names(RobotNames::new_base("ur5")).expect("error");
/// let fk_res = robot_kinematics_module.compute_fk(&robot_joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion).expect("error");
/// fk_res.print_summary();
///
/// // Output:
/// // Link 0 base_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.0, 0.0]], is_identity: true, rot_is_identity: true, translation_is_zeros: true }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.0, 0.0]]
/// // Link 1 shoulder_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.0, 0.089159]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.0, 0.089159]]
/// // Link 2 upper_arm_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.13585, 0.089159]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.13585, 0.089159]]
/// // Link 3 forearm_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.016149999999999998, 0.514159]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.016149999999999998, 0.514159]]
/// // Link 4 wrist_1_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.016149999999999998, 0.906409]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.016149999999999998, 0.906409]]
/// // Link 5 wrist_2_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.10915, 0.906409]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.10915, 0.906409]]
/// // Link 6 wrist_3_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.10915, 1.001059]], is_identity: false, rot_is_identity: true, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.10915, 1.001059]]
/// // Link 7 ee_link --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.7071067805519557, 0.7071067818211392], translation: [[0.0, 0.19145, 1.001059]], is_identity: false, rot_is_identity: false, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 1.5707963250000003]]
/// //    > Pose Translation: [[0.0, 0.19145, 1.001059]]
/// // Link 8 base --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, -1.0, 1.7948965149208059e-9], translation: [[0.0, 0.0, 0.0]], is_identity: false, rot_is_identity: false, translation_is_zeros: true }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, 0.0, -3.14159265]]
/// //    > Pose Translation: [[0.0, 0.0, 0.0]]
/// // Link 9 tool0 --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [-0.7071067805519557, 0.0, 0.0, 0.7071067818211392], translation: [[0.0, 0.19145, 1.001059]], is_identity: false, rot_is_identity: false, translation_is_zeros: false }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[-1.5707963250000003, 0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.19145, 1.001059]]
/// // Link 10 world --->
/// //    > Pose: Some(ImplicitDualQuaternion { data: ImplicitDualQuaternion { rotation: [0.0, 0.0, 0.0, 1.0], translation: [[0.0, 0.0, 0.0]], is_identity: true, rot_is_identity: true, translation_is_zeros: true }, pose_type: ImplicitDualQuaternion })
/// //    > Pose Euler Angles: [[0.0, -0.0, 0.0]]
/// //    > Pose Translation: [[0.0, 0.0, 0.0]]
/// //
/// ```
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotKinematicsModule {
    robot_configuration_module: RobotConfigurationModule,
    robot_joint_state_module: RobotJointStateModule,
    starter_result: RobotFKResult
}
impl RobotKinematicsModule {
    pub fn new(robot_configuration_module: RobotConfigurationModule) -> Self {
        let robot_joint_state_module = RobotJointStateModule::new(robot_configuration_module.clone());

        let mut starter_result = RobotFKResult { link_entries: vec![] };
        let links = robot_configuration_module.robot_model_module().links();
        for (i, link) in links.iter().enumerate() {
            starter_result.link_entries.push( RobotFKResultLinkEntry {
                link_idx: i,
                link_name: link.name().to_string(),
                pose: None
            } )
        }

        Self {
            robot_configuration_module,
            robot_joint_state_module,
            starter_result
        }
    }
    pub fn new_from_names(robot_names: RobotNames) -> Result<Self, OptimaError> {
        let robot_configuration_module = RobotConfigurationModule::new_from_names(robot_names)?;
        return Ok(Self::new(robot_configuration_module));
    }
    pub fn compute_fk(&self, joint_state: &RobotJointState, t: &OptimaSE3PoseType) -> Result<RobotFKResult, OptimaError> {
        let joint_state = self.robot_joint_state_module.convert_joint_state_to_full_state(joint_state)?;
        let mut output = self.starter_result.clone();

        let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();

        let links = self.robot_configuration_module.robot_model_module().links();

        for link_tree_traversal_layer in link_tree_traversal_layers {
            for link_idx in link_tree_traversal_layer {
                if links[*link_idx].present() {
                    self.compute_fk_on_single_link(&joint_state, *link_idx, t, &mut output)?;
                }
            }
        }

        return Ok(output);
    }
    /// This function computes the forward kinematics for some part of the whole robot configuration.
    /// It provides three primary arguments over the standard `compute_fk` function:
    /// - start_link_idx: An optional link index that will serve as the beginning of the partial
    /// floating chain.  If this is None, the start of the chain will be the default world base link.
    /// - end_link_idx: An optional link index that will serve as the end of the partial floating chain.
    /// If this is None, the whole forward kinematics process will play out from the start_link_idx on.
    /// - start_link_pose: An optional SE(3) pose used to situate the beginning link (start_link_idx) in the chain.
    /// If this is None, this pose will be the default pose just based on the given `RobotState`.
    pub fn compute_fk_floating_chain(&self, joint_state: &RobotJointState, t: &OptimaSE3PoseType, floating_link_input: &FloatingLinkInput) -> Result<RobotFKResult, OptimaError> {
        let start_link_idx = floating_link_input.start_link_idx;
        let end_link_idx = floating_link_input.end_link_idx;
        let start_link_pose = floating_link_input.start_link_pose.clone();

        let num_links = self.robot_configuration_module.robot_model_module().links().len();
        if let Some(start_link_idx) = start_link_idx { OptimaError::new_check_for_idx_out_of_bound_error(start_link_idx, num_links, file!(), line!())?; }
        if let Some(end_link_idx) = end_link_idx { OptimaError::new_check_for_idx_out_of_bound_error(end_link_idx, num_links, file!(), line!())?;}

        let joint_state = self.robot_joint_state_module.convert_joint_state_to_full_state(joint_state)?;
        let mut output = self.starter_result.clone();

        let start_link_idx = match start_link_idx {
            None => { self.robot_configuration_module.robot_model_module().robot_base_link_idx() }
            Some(start_link_idx) => { start_link_idx }
        };

        match &start_link_pose {
            None => {
                let mut tmp_output = self.starter_result.clone();
                let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();

                for link_tree_traversal_layer in link_tree_traversal_layers {
                    for link_idx in link_tree_traversal_layer {
                        self.compute_fk_on_single_link(&joint_state, *link_idx, t, &mut tmp_output)?;
                        if *link_idx == start_link_idx {
                            output.link_entries[start_link_idx].pose = tmp_output.link_entries[start_link_idx].pose.clone();
                            break;
                        }
                    }
                }
            }
            Some(s) => {
                if s.map_to_pose_type() != t {
                    return Err(OptimaError::new_generic_error_str(&format!("Given start link pose was not the correct type ({:?} instead of {:?})", s.map_to_pose_type(), t), file!(), line!()));
                }
                output.link_entries[start_link_idx].pose = Some(s.clone());
            }
        }

        let links = self.robot_configuration_module.robot_model_module().links();

        match end_link_idx {
            None => {
                let link_tree_traversal_layers = self.robot_configuration_module.robot_model_module().link_tree_traversal_layers();
                for link_tree_traversal_layer in link_tree_traversal_layers {
                    for link_idx in link_tree_traversal_layer {
                        if links[*link_idx].present() {
                            let predecessor_link_idx_option = links[*link_idx].preceding_link_idx();
                            if predecessor_link_idx_option.is_none() { continue; }
                            let predecessor_link_idx = predecessor_link_idx_option.unwrap();
                            if output.link_entries[predecessor_link_idx].pose.is_some() {
                                self.compute_fk_on_single_link(&joint_state, *link_idx, t, &mut output)?;
                            }
                        }
                    }
                }
            }
            Some(end_link_idx) => {
                let chain = self.robot_configuration_module.robot_model_module().get_link_chain(start_link_idx, end_link_idx)?;
                if chain.is_none() {
                    return Err(OptimaError::new_generic_error_str(&format!("No link chain exists between link {} and link {}", start_link_idx, end_link_idx), file!(), line!()));
                }
                let chain = chain.unwrap();
                for link_idx in chain {
                    if links[*link_idx].present() {
                        if output.link_entries[*link_idx].pose.is_some() { continue; }
                        let predecessor_link_idx_option = links[*link_idx].preceding_link_idx();
                        if predecessor_link_idx_option.is_none() { continue; }
                        let predecessor_link_idx = predecessor_link_idx_option.unwrap();
                        if output.link_entries[predecessor_link_idx].pose.is_some() {
                            self.compute_fk_on_single_link(&joint_state, *link_idx, t, &mut output)?;
                        }
                    }
                }
            }
        }

        return Ok(output);
    }
    pub fn compute_fk_dof_perturbations(&self, joint_state: &RobotJointState, t: &OptimaSE3PoseType, perturbation: Option<f64>) -> Result<RobotFKDOFPerturbationsResult, OptimaError> {
        let perturbation = match perturbation {
            None => { 0.00001 }
            Some(p) => { p }
        };

        let dof_joint_state = self.robot_joint_state_module.convert_joint_state_to_dof_state(joint_state)?;

        let central_fk_result = self.compute_fk(&dof_joint_state, t)?;

        let mut fk_dof_perturbation_results = vec![];

        let len = dof_joint_state.joint_state().len();
        for i in 0..len {
            let mut dof_joint_state_copy = dof_joint_state.clone();
            dof_joint_state_copy[i] += perturbation;
            let fk_res = self.compute_fk(&dof_joint_state_copy, t)?;
            fk_dof_perturbation_results.push(fk_res);
        }

        Ok(RobotFKDOFPerturbationsResult {
            perturbation,
            central_fk_result,
            fk_dof_perturbation_results
        })
    }
    pub fn compute_jacobian(&self,
                            joint_state: &RobotJointState,
                            start_link_idx: Option<usize>,
                            end_link_idx: usize,
                            robot_jacobian_end_point: &JacobianEndPoint,
                            start_link_pose: Option<OptimaSE3Pose>,
                            jacobian_mode: JacobianMode) -> Result<DMatrix<f64>, OptimaError> {
        let start_idx = match start_link_idx {
            None => { self.robot_configuration_module.robot_model_module().world_link_idx() }
            Some(s) => {s}
        };
        let chain = self.robot_configuration_module.robot_model_module().get_link_chain(start_idx, end_link_idx)?;
        if chain.is_none() {
            let s = format!("Link chain does not exist between link {} and {}.  Cannot perform jacobian calculation.", start_idx, end_link_idx);
            return Err(OptimaError::new_generic_error_str(&s, file!(), line!()));
        }
        let chain = chain.unwrap();
        for c in chain {
            if !self.robot_configuration_module.robot_model_module().links()[*c].present() {
                let s = format!("Valid link chain does not exist between link {} and {} because link {} is not present.  Cannot perform jacobian calculation.", start_idx, end_link_idx, *c);
                return Err(OptimaError::new_generic_error_str(&s, file!(), line!()));
            }
        }

        let num_dofs = self.robot_joint_state_module.num_dofs();
        let num_rows = match jacobian_mode {
            JacobianMode::Full => { 6 }
            JacobianMode::Translational => { 3 }
            JacobianMode::Rotational => { 3 }
        };
        let mut jacobian = DMatrix::zeros(num_rows, num_dofs);

        let floating_link_input = FloatingLinkInput {
            start_link_idx,
            end_link_idx: Some(end_link_idx),
            start_link_pose
        };

        let fk_res = self.compute_fk_floating_chain(joint_state, &OptimaSE3PoseType::ImplicitDualQuaternion, &floating_link_input)?;
        let link_entries = &fk_res.link_entries;

        let end_pose = link_entries.get(end_link_idx).unwrap().pose.as_ref().unwrap().unwrap_implicit_dual_quaternion()?;

        let end_point = match robot_jacobian_end_point {
            JacobianEndPoint::Link => { end_pose.translation().clone() }
            JacobianEndPoint::Local(p) => { end_pose.multiply_by_point(p) }
            JacobianEndPoint::Global(p) => { p.clone() }
            JacobianEndPoint::InertialOrigin => {
                let link = self.robot_configuration_module.robot_model_module().links().get(end_link_idx).unwrap();
                let inertial_center = link.urdf_link().inertial_origin_xyz();
                end_pose.multiply_by_point(&inertial_center)
            }
        };

        for link_entry in link_entries {
            if let Some(pose) = &link_entry.pose {
                let link_idx = link_entry.link_idx;
                let joint_idx = self.robot_configuration_module.robot_model_module().links().get(link_idx).unwrap().preceding_joint_idx();
                if let Some(joint_idx) = joint_idx {
                    let joint_state_idxs = self.robot_joint_state_module.map_joint_idx_to_joint_state_idxs(joint_idx, &RobotJointStateType::DOF)?;
                    for joint_state_idx in joint_state_idxs {
                        let pose_as_idq = pose.unwrap_implicit_dual_quaternion()?;
                        let joint_axis = self.robot_joint_state_module.ordered_dof_joint_axes().get(*joint_state_idx).unwrap();
                        let axis = joint_axis.axis();

                        match joint_axis.axis_primitive_type() {
                            JointAxisPrimitiveType::Rotation => {
                                match jacobian_mode {
                                    JacobianMode::Full => {
                                        let rotated_axis = pose_as_idq.rotation() * &axis;
                                        let connector_vec = &end_point - pose_as_idq.translation();
                                        let cross_vec = rotated_axis.cross(&connector_vec);

                                        jacobian[(0, *joint_state_idx)] = cross_vec.x; jacobian[(1, *joint_state_idx)] = cross_vec.y; jacobian[(2, *joint_state_idx)] = cross_vec.z;
                                        jacobian[(3, *joint_state_idx)] = rotated_axis.x; jacobian[(4, *joint_state_idx)] = rotated_axis.y; jacobian[(5, *joint_state_idx)] = rotated_axis.z;
                                    }
                                    JacobianMode::Translational => {
                                        let rotated_axis = pose_as_idq.rotation() * &axis;
                                        let connector_vec = &end_point - pose_as_idq.translation();
                                        let cross_vec = rotated_axis.cross(&connector_vec);

                                        jacobian[(0, *joint_state_idx)] = cross_vec.x; jacobian[(1, *joint_state_idx)] = cross_vec.y; jacobian[(2, *joint_state_idx)] = cross_vec.z;
                                    }
                                    JacobianMode::Rotational => {
                                        let rotated_axis = pose_as_idq.rotation() * &axis;

                                        jacobian[(0, *joint_state_idx)] = rotated_axis.x; jacobian[(1, *joint_state_idx)] = rotated_axis.y; jacobian[(2, *joint_state_idx)] = rotated_axis.z;
                                    }
                                }
                            }
                            JointAxisPrimitiveType::Translation => {
                                match jacobian_mode {
                                    JacobianMode::Full => {
                                        let rotated_axis = pose_as_idq.rotation() * &axis;

                                        jacobian[(0, *joint_state_idx)] = rotated_axis.x; jacobian[(1, *joint_state_idx)] = rotated_axis.y; jacobian[(2, *joint_state_idx)] = rotated_axis.z;
                                        jacobian[(3, *joint_state_idx)] = 0.0; jacobian[(4, *joint_state_idx)] = 0.0; jacobian[(5, *joint_state_idx)] = 0.0;
                                    }
                                    JacobianMode::Translational => {
                                        let rotated_axis = pose_as_idq.rotation() * &axis;

                                        jacobian[(0, *joint_state_idx)] = rotated_axis.x; jacobian[(1, *joint_state_idx)] = rotated_axis.y; jacobian[(2, *joint_state_idx)] = rotated_axis.z;
                                    }
                                    JacobianMode::Rotational => {
                                        jacobian[(0, *joint_state_idx)] = 0.0; jacobian[(1, *joint_state_idx)] = 0.0; jacobian[(2, *joint_state_idx)] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(jacobian)
    }
    pub fn robot_name(&self) -> &str {
        return self.robot_configuration_module.robot_model_module().robot_name()
    }
    fn compute_fk_on_single_link(&self, joint_state: &RobotJointState, link_idx: usize, t: &OptimaSE3PoseType, output: &mut RobotFKResult) -> Result<(), OptimaError> {
        let link = self.robot_configuration_module.robot_model_module().get_link_by_idx(link_idx)?;
        let is_chain_base_link = link.is_chain_base_link();

        let preceding_link_option = link.preceding_link_idx();
        if preceding_link_option.is_none() {
            output.link_entries[link_idx].pose = Some(self.robot_configuration_module.robot_configuration_info().base_offset().get_pose_by_type(t).clone());
            return Ok(());
        }

        let preceding_link_idx = preceding_link_option.unwrap();

        let preceding_joint_option = link.preceding_joint_idx();
        if preceding_joint_option.is_none() {
            output.link_entries[link_idx].pose = output.link_entries[preceding_link_idx].pose.clone();
            return Ok(());
        }

        let preceding_joint_idx = preceding_joint_option.unwrap();
        let preceding_joint = &self.robot_configuration_module.robot_model_module().joints()[preceding_joint_idx];

        let full_state_idxs = self.robot_joint_state_module.map_joint_idx_to_joint_state_idxs(preceding_joint_idx, &RobotJointStateType::Full)?;

        let out_pose = output.link_entries[preceding_link_idx].pose.clone();
        if out_pose.is_none() { return Ok(()); }
        let mut out_pose = out_pose.unwrap();

        let offset_pose_all = preceding_joint.origin_offset_pose();
        out_pose = out_pose.multiply(offset_pose_all.get_pose_by_type(t), false)?;

        let joint_axes = preceding_joint.joint_axes();

        // On a chain base link, we can just use Euler angles.  On other links, we cannot guarantee
        // that the axis will be [1,0,0], [0,1,0], or [0,0,1], so we must do a sequence of multiplications
        // on axis angle representations.
        if is_chain_base_link {
            if full_state_idxs.len() > 0 {
                let mut tt = vec![0., 0., 0.];
                let mut rr = vec![0., 0., 0.];

                for (i, full_state_idx) in full_state_idxs.iter().enumerate() {
                    let joint_axis = &joint_axes[i];
                    let joint_value = joint_state[*full_state_idx];

                    let axis = joint_axis.axis();
                    match joint_axis.axis_primitive_type() {
                        JointAxisPrimitiveType::Rotation => {
                            if axis[0] == 1.0 { rr[0] = joint_value }
                            else if axis[1] == 1.0 { rr[1] = joint_value }
                            else if axis[2] == 1.0 { rr[2] = joint_value }
                        }
                        JointAxisPrimitiveType::Translation => {
                            if axis[0] == 1.0 { tt[0] = joint_value }
                            else if axis[1] == 1.0 { tt[1] = joint_value }
                            else if axis[2] == 1.0 { tt[2] = joint_value }
                        }
                    }
                }

                let euler_pose = OptimaSE3Pose::new_from_euler_angles(rr[0], rr[1], rr[2], tt[0], tt[1], tt[2], t);
                out_pose = out_pose.multiply(&euler_pose, false)?;
            }
        } else {
            for (i, full_state_idx) in full_state_idxs.iter().enumerate() {
                let joint_axis = &joint_axes[i];
                let joint_value = joint_state[*full_state_idx];

                let axis_pose = match joint_axis.axis_primitive_type() {
                    JointAxisPrimitiveType::Rotation => {
                        let axis = &joint_axis.axis_as_unit();
                        OptimaSE3Pose::new_from_axis_angle(axis, joint_value, 0.,0.,0., t)
                    }
                    JointAxisPrimitiveType::Translation => {
                        let axis = joint_value * &joint_axis.axis();
                        OptimaSE3Pose::new_from_euler_angles(0.,0.,0., axis[0], axis[1], axis[2], t)
                    }
                };

                out_pose = out_pose.multiply(&axis_pose, false)?;
            }
        }

        output.link_entries[link_idx].pose = Some(out_pose);

        Ok(())
    }
    pub fn robot_configuration_module(&self) -> &RobotConfigurationModule {
        &self.robot_configuration_module
    }
    pub fn robot_joint_state_module(&self) -> &RobotJointStateModule {
        &self.robot_joint_state_module
    }
}
impl SaveAndLoadable for RobotKinematicsModule {
    type SaveType = RobotConfigurationModule;

    fn get_save_serialization_object(&self) -> Self::SaveType {
        self.robot_configuration_module.clone()
    }

    fn load_from_path(path: &OptimaStemCellPath) -> Result<Self, OptimaError> where Self: Sized {
        let r: Self::SaveType = path.load_object_from_json_file()?;
        return Ok(RobotKinematicsModule::new(r));
    }

    fn load_from_json_string(json_str: &str) -> Result<Self, OptimaError> where Self: Sized {
        let r: Self::SaveType = load_object_from_json_string(json_str)?;
        return Ok(RobotKinematicsModule::new(r));
    }
}

#[derive(Clone, Debug)]
pub struct FloatingLinkInput {
    start_link_idx: Option<usize>,
    end_link_idx: Option<usize>,
    start_link_pose: Option<OptimaSE3Pose>
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotKinematicsModule {
    #[new]
    pub fn new_py(robot_name: &str, configuration_name: Option<&str>) -> RobotKinematicsModule {
        return Self::new_from_names(RobotNames::new(robot_name, configuration_name)).expect("error");
    }
    #[args(pose_type = "\"ImplicitDualQuaternion\"")]
    pub fn compute_fk_py(&self, joint_state: Vec<f64>, pose_type: &str) -> RobotFKResult {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        return self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::from_ron_string(pose_type).expect("error")).expect("error");
    }
    #[args(pose_type = "\"ImplicitDualQuaternion\"")]
    pub fn compute_fk_floating_chain_py(&self, joint_state: Vec<f64>, pose_type: &str, start_link_idx: Option<usize>, end_link_idx: Option<usize>, start_link_pose: Option<OptimaSE3PosePy>) -> RobotFKResult {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let floating_link_input = FloatingLinkInput {
            start_link_idx,
            end_link_idx,
            start_link_pose: match start_link_pose {
                None => { None }
                Some(start_link_pose) => { Some(start_link_pose.pose().clone()) }
            }
        };

        return self.compute_fk_floating_chain(&robot_joint_state, &OptimaSE3PoseType::from_ron_string(pose_type).expect("error"), &floating_link_input).expect("error");
    }
    #[args(robot_jacobian_end_point = "\"Link\"", jacobian_mode = "\"Full\"")]
    pub fn compute_jacobian_py(&self, joint_state: Vec<f64>, end_link_idx: usize, start_link_idx: Option<usize>, start_link_pose: Option<OptimaSE3PosePy>, robot_jacobian_end_point: &str, jacobian_mode: &str) -> Vec<Vec<f64>> {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let start_link_pose = match start_link_pose {
            None => { None }
            Some(p) => { Some(p.pose().clone()) }
        };
        let jac = self.compute_jacobian(&robot_joint_state,
                                        start_link_idx,
                                        end_link_idx,
                                        &JacobianEndPoint::from_ron_string(robot_jacobian_end_point).expect("error"),
                                        start_link_pose,
                                        JacobianMode::from_ron_string(jacobian_mode).expect("error")).expect("error");

        let jac_vecs = NalgebraConversions::dmatrix_to_vecs(&jac);
        return jac_vecs;
    }
}

/// WASM implementations.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl RobotKinematicsModule {
    #[wasm_bindgen(constructor)]
    pub fn new_wasm(robot_name: String, configuration_name: Option<String>) -> RobotKinematicsModule {
        let robot_names = match &configuration_name {
            None => { RobotNames::new_base(&robot_name) }
            Some(configuration_name) => { RobotNames::new(&robot_name, Some(configuration_name)) }
        };
        return RobotKinematicsModule::new_from_names(robot_names).expect("error");
    }
    pub fn compute_fk_wasm(&self, joint_state: Vec<f64>, pose_type: &str) -> JsValue {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let res = self.compute_fk(&robot_joint_state, &OptimaSE3PoseType::from_ron_string(pose_type).expect("error")).expect("error");
       return  JsValue::from_serde(&res).unwrap();
    }
    pub fn compute_fk_floating_chain_py(&self, joint_state: Vec<f64>, pose_type: &str, start_link_idx: Option<usize>, end_link_idx: Option<usize>, start_link_pose: Option<OptimaSE3PoseWASM>) -> JsValue {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let floating_link_input = FloatingLinkInput {
            start_link_idx,
            end_link_idx,
            start_link_pose: match start_link_pose {
                None => { None }
                Some(start_link_pose) => { Some(start_link_pose.pose().clone()) }
            }
        };

        let res = self.compute_fk_floating_chain(&robot_joint_state, &OptimaSE3PoseType::from_ron_string(pose_type).expect("error"), &floating_link_input).expect("error");
        return JsValue::from_serde(&res).unwrap();
    }
    pub fn compute_jacobian_wasm(&self, joint_state: Vec<f64>, end_link_idx: usize, start_link_idx: Option<usize>, start_link_pose: Option<OptimaSE3PoseWASM>, robot_jacobian_end_point: &str, jacobian_mode: &str) -> JsValue {
        let robot_joint_state = self.robot_joint_state_module.spawn_robot_joint_state_try_auto_type(NalgebraConversions::vec_to_dvector(&joint_state)).expect("error");
        let start_link_pose = match start_link_pose {
            None => { None }
            Some(p) => { Some(p.pose().clone()) }
        };
        let jac = self.compute_jacobian(&robot_joint_state,
                                        start_link_idx,
                                        end_link_idx,
                                        &JacobianEndPoint::from_ron_string(robot_jacobian_end_point).expect("error"),
                                        start_link_pose,
                                        JacobianMode::from_ron_string(jacobian_mode).expect("error")).expect("error");

        let jac_vecs = NalgebraConversions::dmatrix_to_vecs(&jac);
        let jac_vecs_js = JsMatrix::new(jac_vecs);
        return JsValue::from_serde(&jac_vecs_js).unwrap();
    }
}

/// The output of a forward kinematics computation.
/// The primary field in this object is `link_entries`.  This is a list of `RobotFKResultLinkEntry`
/// objects.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFKResult {
    link_entries: Vec<RobotFKResultLinkEntry>
}
impl RobotFKResult {
    pub fn new_empty(robot_kinematics_module: &RobotKinematicsModule) -> Self {
        return robot_kinematics_module.starter_result.clone();
    }
    /// Returns a reference to the results link entries.
    pub fn link_entries(&self) -> &Vec<RobotFKResultLinkEntry> {
        &self.link_entries
    }
    /// Prints a summary of the forward kinematics result.
    pub fn print_summary(&self) {
        for e in self.link_entries() {
            optima_print(&format!("Link {} {} ---> ", e.link_idx, e.link_name), PrintMode::Println, PrintColor::Blue, true);
            optima_print(&format!("   > Pose: {:?}", e.pose), PrintMode::Println, PrintColor::None, false);
            if e.pose.is_some() {
                let euler_angles = e.pose.as_ref().unwrap().to_euler_angles_and_translation();
                optima_print(&format!("   > Pose Euler Angles: {:?}", euler_angles.0), PrintMode::Println, PrintColor::None, false);
                optima_print(&format!("   > Pose Translation: {:?}", euler_angles.1), PrintMode::Println, PrintColor::None, false);
            }
        }
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotFKResult {
    pub fn print_summary_py(&self) {
        self.print_summary();
    }
    pub fn get_robot_fk_result_link_entry(&self, link_idx: usize) -> RobotFKResultLinkEntry {
        return self.link_entries[link_idx].clone();
    }
    pub fn get_link_pose(&self, link_idx: usize) -> Option<Vec<Vec<f64>>> {
        let e = &self.link_entries[link_idx];
        return e.pose_py();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotFKDOFPerturbationsResult {
    perturbation: f64,
    central_fk_result: RobotFKResult,
    fk_dof_perturbation_results: Vec<RobotFKResult>
}

/// A `RobotFKResultLinkEntry` specifies information about one particular link in the forward kinematics
/// process.  It provides the link index, the link's name, and the pose of the link.
/// If the link is NOT included in the FK computation (the link is not present in the model, etc)
/// the pose will be None.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen, derive(Clone, Debug, Serialize, Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), pyclass, derive(Clone, Debug, Serialize, Deserialize))]
pub struct RobotFKResultLinkEntry {
    link_idx: usize,
    link_name: String,
    pose: Option<OptimaSE3Pose>
}
impl RobotFKResultLinkEntry {
    pub fn link_idx(&self) -> usize {
        self.link_idx
    }
    pub fn link_name(&self) -> &str {
        &self.link_name
    }
    pub fn pose(&self) -> &Option<OptimaSE3Pose> {
        &self.pose
    }
}

/// Python implementations.
#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl RobotFKResultLinkEntry {
    pub fn link_idx_py(&self) -> usize { self.link_idx }
    pub fn link_name_py(&self) -> String { self.link_name.clone() }
    pub fn pose_py(&self) -> Option<Vec<Vec<f64>>> {
        return match &self.pose {
            None => {
                None
            }
            Some(pose) => {
                Some(pose.to_vec_representation())
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JacobianMode {
    Full, Translational, Rotational
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JacobianEndPoint {
    Link,
    Local(Vector3<f64>),
    Global(Vector3<f64>),
    InertialOrigin
}
