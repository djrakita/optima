use bevy::asset::Assets;
use bevy::math::Vec3;
use bevy::prelude::{Commands, Mesh, Res, ResMut, StandardMaterial, PbrBundle, Transform, default, Color, Query};
use bevy::prelude::shape::Icosphere;
use bevy_mod_picking::PickableBundle;
use bevy_transform_gizmo::GizmoTransformable;
use bevy::ecs::component::Component;
use bevy_prototype_debug_lines::DebugLines;
use nalgebra::DVector;
use crate::optima_bevy::optima_bevy_utils::transform::TransformUtils;
use crate::optima_bevy::optima_bevy_utils::viewport_visuals::ViewportVisualsActions;
use crate::utils::utils_splines::{BSpline, InterpolatingSpline, InterpolatingSplineType};

pub struct SplineActions;
impl SplineActions {
    pub fn update_spline_control_points(query: &mut Query<(&SplineControlPoint, &mut Transform)>,
                                        control_point_locations: Vec<Vec3>) {
        assert_eq!(control_point_locations.len(), query.iter().len());

        for mut q in query.iter_mut() {
            let spline_control_point: &SplineControlPoint = &q.0;
            let transform: &mut Transform = &mut q.1;

            transform.translation = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(control_point_locations[spline_control_point.idx]);
        }
    }
}

pub struct SplineSystems;
impl SplineSystems {
    pub fn interpolating_spline_startup_system(mut commands: Commands,
                                               interpolating_spline: Res<InterpolatingSpline>,
                                               mut meshes: ResMut<Assets<Mesh>>,
                                               mut materials: ResMut<Assets<StandardMaterial>>) {
        let control_points = interpolating_spline.control_points();
        let spline_type = interpolating_spline.spline_type();

        for (control_point_idx, control_point) in control_points.iter().enumerate() {
            assert_eq!(control_point.len(), 3);

            let p = match &spline_type {
                InterpolatingSplineType::Linear => {
                    control_point.clone()
                }
                InterpolatingSplineType::Quadratic => {
                    control_point.clone()
                }
                InterpolatingSplineType::HermiteCubic => {
                    if control_point_idx % 2 == 1 {
                        control_point + &control_points[control_point_idx - 1]
                    } else {
                        control_point.clone()
                    }
                }
                InterpolatingSplineType::NaturalCubic => {
                    if control_point_idx % 3 == 1 {
                        control_point + &control_points[control_point_idx - 1]
                    } else if control_point_idx % 3 == 2 {
                        control_point + &control_points[control_point_idx - 2]
                    }
                    else {
                        control_point.clone()
                    }
                }
                InterpolatingSplineType::CardinalCubic { w } => {
                    let control_point_idx_in_segment = interpolating_spline.map_control_point_idx_to_idx_in_spline_segments(control_point_idx)[0].1;

                    if control_point_idx_in_segment == 1 {
                        let d_p1_d_t = control_point.clone();
                        let p2 = &control_points[control_point_idx + 1];
                        let p0 = p2 - (2./1.0-*w) * d_p1_d_t;
                        p0
                    } else if control_point_idx_in_segment == 3 {
                        let d_p2_d_t = control_point.clone();
                        let p1 = &control_points[control_point_idx - 3];
                        let p3 = p1 + (2./1.0-*w) * d_p2_d_t;
                        p3
                    } else {
                        control_point.clone()
                    }
                }
                InterpolatingSplineType::BezierCubic => {
                    let control_point_idx_in_segment = interpolating_spline.map_control_point_idx_to_idx_in_spline_segments(control_point_idx)[0].1;

                    if control_point_idx_in_segment == 1 {
                        let d_p0_d_t = control_point.clone();
                        let p0 = &control_points[control_point_idx - 1];
                        let p1 = (1./3.)*d_p0_d_t + p0;
                        p1
                    } else if control_point_idx_in_segment == 3 {
                        let d_p3_d_t = control_point.clone();
                        let p3 = &control_points[control_point_idx - 1];
                        let p2 = p3 -(1./3.)*d_p3_d_t;
                        p2
                    } else {
                        control_point.clone()
                    }
                }
            };

            let translation = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32));
            let transform = Transform::from_translation(translation);

            commands.spawn_bundle(PbrBundle {
                mesh: meshes.add(Icosphere {
                    radius: 0.044,
                    subdivisions: 3
                }.into()),
                material: materials.add(Color::rgba(0.2,0.3,1.0,1.0).into()),
                transform,
                ..default()
            }).insert(SplineControlPoint {
                idx: control_point_idx
            })
                .insert_bundle(PickableBundle::default())
                .insert(GizmoTransformable);
        }
    }

    pub fn interpolating_spline_system(mut interpolating_spline: ResMut<InterpolatingSpline>,
                                       mut lines: ResMut<DebugLines>,
                                       mut query: Query<(&SplineControlPoint, &mut Transform)>) {

        let spline_type = interpolating_spline.spline_type();

        /*
        for q in query.iter() {
            let spline_control_point: &SplineControlPoint = &q.0;
            let transform: &Transform = &q.1;

            let translation = match spline_type {
                InterpolatingSplineType::Linear => {
                    TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(transform.translation)
                }
                InterpolatingSplineType::Quadratic => {
                    TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(transform.translation)
                }
                InterpolatingSplineType::HermiteCubic => {
                    if spline_control_point.idx % 2 == 1 {
                        let control_points = interpolating_spline.control_points();
                        let curr_location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(transform.translation);
                        let p = &control_points[spline_control_point.idx-1];
                        let preceding_control_point_location = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32);

                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, curr_location, preceding_control_point_location, Color::rgb(0.0, 0.7, 0.2), 3.0, 5, 2, 0.0);

                        let vel = curr_location - preceding_control_point_location;

                        vel
                    } else {
                        TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(transform.translation)
                    }
                }
            };

            let dvec = DVector::from_vec(vec![translation.x as f64, translation.y as f64, translation.z as f64]);

            interpolating_spline.update_control_point(spline_control_point.idx, dvec);
        }
        */

        let mut control_point_locations = vec![];
        for q in query.iter() {
            let spline_control_point: &SplineControlPoint = &q.0;
            let transform: &Transform = &q.1;
            control_point_locations.push((spline_control_point.idx, transform.translation))
        }
        control_point_locations.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        let mut out_control_point_locations = vec![];
        for control_point_location in control_point_locations {
            match spline_type {
                InterpolatingSplineType::Linear => {
                    let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                    let dvec = DVector::from_vec(vec![location.x as f64, location.y as f64, location.z as f64]);
                    interpolating_spline.update_control_point(control_point_location.0, dvec);
                    out_control_point_locations.push(location);
                }
                InterpolatingSplineType::Quadratic => {
                    let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                    let dvec = DVector::from_vec(vec![location.x as f64, location.y as f64, location.z as f64]);
                    interpolating_spline.update_control_point(control_point_location.0, dvec);
                    out_control_point_locations.push(location);
                }
                InterpolatingSplineType::HermiteCubic => {
                    if control_point_location.0 % 2 == 1 {
                        let control_points = interpolating_spline.control_points();
                        let curr_control_point_location_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let preceding_control_point_location_as_dvec = &control_points[control_point_location.0 - 1];
                        let preceding_control_point_location_as_vec3 = Vec3::new(preceding_control_point_location_as_dvec[0] as f32, preceding_control_point_location_as_dvec[1] as f32, preceding_control_point_location_as_dvec[2] as f32);

                        let vel = curr_control_point_location_as_vec3 - preceding_control_point_location_as_vec3;

                        let dvec = DVector::from_vec(vec![vel.x as f64, vel.y as f64, vel.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, dvec);

                        out_control_point_locations.push(preceding_control_point_location_as_vec3 + vel);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, preceding_control_point_location_as_vec3, preceding_control_point_location_as_vec3 + vel, Color::rgb(0.0, 0.7, 0.2), 3.0, 5, 2, 0.0);
                    } else {
                        let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let dvec = DVector::from_vec(vec![location.x as f64, location.y as f64, location.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, dvec);
                        out_control_point_locations.push(location);
                    }
                }
                InterpolatingSplineType::NaturalCubic => {
                    let control_point_idx_in_segment = interpolating_spline.map_control_point_idx_to_idx_in_spline_segments(control_point_location.0)[0].1;
                    if control_point_idx_in_segment % 3 == 1 {
                        let control_points = interpolating_spline.control_points();
                        let curr_control_point_location_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let preceding_control_point_location_as_dvec = &control_points[control_point_location.0 - 1];
                        let preceding_control_point_location_as_vec3 = Vec3::new(preceding_control_point_location_as_dvec[0] as f32, preceding_control_point_location_as_dvec[1] as f32, preceding_control_point_location_as_dvec[2] as f32);

                        let vel = curr_control_point_location_as_vec3 - preceding_control_point_location_as_vec3;

                        let dvec = DVector::from_vec(vec![vel.x as f64, vel.y as f64, vel.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, dvec);

                        out_control_point_locations.push(preceding_control_point_location_as_vec3 + vel);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, preceding_control_point_location_as_vec3, preceding_control_point_location_as_vec3 + vel, Color::rgb(0.0, 0.7, 0.2), 3.0, 5, 2, 0.0);
                    } else if control_point_idx_in_segment % 3 == 2 {
                        let control_points = interpolating_spline.control_points();
                        let curr_control_point_location_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let preceding_control_point_location_as_dvec = &control_points[control_point_location.0 - 2];
                        let preceding_control_point_location_as_vec3 = Vec3::new(preceding_control_point_location_as_dvec[0] as f32, preceding_control_point_location_as_dvec[1] as f32, preceding_control_point_location_as_dvec[2] as f32);

                        let acc = curr_control_point_location_as_vec3 - preceding_control_point_location_as_vec3;

                        let dvec = DVector::from_vec(vec![acc.x as f64, acc.y as f64, acc.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, dvec);

                        out_control_point_locations.push(preceding_control_point_location_as_vec3 + acc);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, preceding_control_point_location_as_vec3, preceding_control_point_location_as_vec3 + acc, Color::rgb(0.9, 0.3, 0.5), 3.0, 5, 2, 0.0);
                    } else {
                        let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let dvec = DVector::from_vec(vec![location.x as f64, location.y as f64, location.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, dvec);
                        out_control_point_locations.push(location);
                    }
                }
                InterpolatingSplineType::CardinalCubic { w } => {
                    let control_point_idx_in_segment = interpolating_spline.map_control_point_idx_to_idx_in_spline_segments(control_point_location.0)[0].1;
                    if control_point_idx_in_segment == 1 {
                        let control_points = interpolating_spline.control_points();

                        let p0_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let p2 = &control_points[control_point_location.0 + 1];
                        let p2_as_vec3 = Vec3::new(p2[0] as f32, p2[1] as f32, p2[2] as f32);
                        let diff = p2_as_vec3 - p0_as_vec3;
                        let d_p1_d_t_as_vec3 = 0.5*(1.0-w as f32)*diff;
                        let d_p1_d_t = DVector::from_vec(vec![d_p1_d_t_as_vec3.x as f64, d_p1_d_t_as_vec3.y as f64, d_p1_d_t_as_vec3.z as f64]);

                        let p1 = &control_points[control_point_location.0 - 1];
                        let p1_as_vec3 = Vec3::new(p1[0] as f32, p1[1] as f32, p1[2] as f32);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p1_as_vec3, p1_as_vec3 + d_p1_d_t_as_vec3, Color::rgb(0.0, 0.7, 0.2), 3.0, 5, 2, 0.0);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p0_as_vec3, p0_as_vec3 + diff, Color::rgb(0.2, 0.2, 0.2), 1.0, 5, 1, 0.0);

                        interpolating_spline.update_control_point(control_point_location.0, d_p1_d_t);

                        out_control_point_locations.push(p0_as_vec3);
                    } else if control_point_idx_in_segment == 3 {
                        let control_points = interpolating_spline.control_points();

                        let p3_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let p1 = &control_points[control_point_location.0 - 3];
                        let p1_as_vec3 = Vec3::new(p1[0] as f32, p1[1] as f32, p1[2] as f32);
                        let diff = p3_as_vec3 - p1_as_vec3;
                        let d_p2_d_t_as_vec3 = 0.5*(1.0-w as f32)*diff;
                        let d_p2_d_t = DVector::from_vec(vec![d_p2_d_t_as_vec3.x as f64, d_p2_d_t_as_vec3.y as f64, d_p2_d_t_as_vec3.z as f64]);

                        let p2 = &control_points[control_point_location.0 - 1];
                        let p2_as_vec3 = Vec3::new(p2[0] as f32, p2[1] as f32, p2[2] as f32);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p2_as_vec3, p2_as_vec3 + d_p2_d_t_as_vec3, Color::rgb(0.0, 0.7, 0.2), 3.0, 5, 2, 0.0);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p1_as_vec3, p1_as_vec3 + diff, Color::rgb(0.2, 0.2, 0.2), 1.0, 5, 1, 0.0);

                        interpolating_spline.update_control_point(control_point_location.0, d_p2_d_t);

                        out_control_point_locations.push(p3_as_vec3);
                    } else {
                        let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let dvec = DVector::from_vec(vec![location.x as f64, location.y as f64, location.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, dvec);
                        out_control_point_locations.push(location);
                    }

                    /*
                    if control_point_idx_in_segment == 1 || control_point_idx_in_segment == 3 {
                        let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let control_points = interpolating_spline.control_points();
                        let curr_control_point_location_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let preceding_control_point_location_as_dvec = &control_points[control_point_location.0 - 1];
                        let preceding_control_point_location_as_vec3 = Vec3::new(preceding_control_point_location_as_dvec[0] as f32, preceding_control_point_location_as_dvec[1] as f32, preceding_control_point_location_as_dvec[2] as f32);
                        let vel = preceding_control_point_location_as_vec3 - curr_control_point_location_as_vec3;
                        let d = 0.5_f32*(1.-w as f32)*vel;
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, preceding_control_point_location_as_vec3, preceding_control_point_location_as_vec3 + d, Color::rgb(0.0, 0.7, 0.2), 3.0, 5, 2, 0.0);
                        interpolating_spline.update_control_point(control_point_location.0, DVector::from_vec(vec![d.x as f64, d.y as f64, d.z as f64]));
                        out_control_point_locations.push(location);
                    } else {

                    } */
                }
                InterpolatingSplineType::BezierCubic => {
                    let control_point_idx_in_segment = interpolating_spline.map_control_point_idx_to_idx_in_spline_segments(control_point_location.0)[0].1;

                    if control_point_idx_in_segment == 1 {
                        let control_points = interpolating_spline.control_points();

                        let p1_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let p0 = &control_points[control_point_location.0 - 1];
                        let p0_as_vec3 = Vec3::new(p0[0] as f32, p0[1] as f32, p0[2] as f32);
                        let diff = p1_as_vec3 - p0_as_vec3;
                        let d_p0_d_t_as_vec3 = 3.0 * diff;
                        let d_p0_d_t = DVector::from_vec(vec![d_p0_d_t_as_vec3.x as f64, d_p0_d_t_as_vec3.y as f64, d_p0_d_t_as_vec3.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, d_p0_d_t);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p1_as_vec3, p0_as_vec3, Color::rgb(0.3, 0.3, 0.3), 3.0, 5, 2, 0.0);
                        out_control_point_locations.push(p1_as_vec3);
                    } else if control_point_idx_in_segment == 3 {
                        let control_points = interpolating_spline.control_points();

                        let p2_as_vec3 = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let p3 = &control_points[control_point_location.0 - 1];
                        let p3_as_vec3 = Vec3::new(p3[0] as f32, p3[1] as f32, p3[2] as f32);
                        let diff = p3_as_vec3 - p2_as_vec3;
                        let d_p3_d_t_as_vec3 = 3.0 * diff;
                        let d_p3_d_t = DVector::from_vec(vec![d_p3_d_t_as_vec3.x as f64, d_p3_d_t_as_vec3.y as f64, d_p3_d_t_as_vec3.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, d_p3_d_t);
                        ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p3_as_vec3, p2_as_vec3, Color::rgb(0.3, 0.3, 0.3), 3.0, 5, 2, 0.0);
                        out_control_point_locations.push(p2_as_vec3);
                    } else {
                        let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(control_point_location.1);
                        let dvec = DVector::from_vec(vec![location.x as f64, location.y as f64, location.z as f64]);
                        interpolating_spline.update_control_point(control_point_location.0, dvec);
                        out_control_point_locations.push(location);
                    }
                }
            }
        }

        SplineActions::update_spline_control_points(&mut query, out_control_point_locations);

        let num_spline_segments = interpolating_spline.num_spline_segments() as f64;

        let stride = 0.005;
        let mut t = 0.0 + stride;

        while t < num_spline_segments {
            let p0 = interpolating_spline.interpolate(t - stride);
            let p1 = interpolating_spline.interpolate(t);

            let p0_v = Vec3::new(p0[0] as f32, p0[1] as f32, p0[2] as f32);
            let p1_v = Vec3::new(p1[0] as f32, p1[1] as f32, p1[2] as f32);

            ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p0_v, p1_v, Color::rgb(0.0, 0.7, 0.8), 11.0, 5, 3, 0.0);

            t += stride;
        }
    }

    pub fn bspline_startup_system(mut commands: Commands,
                                  control_points: Res<Vec<DVector<f64>>>,
                                  mut meshes: ResMut<Assets<Mesh>>,
                                  mut materials: ResMut<Assets<StandardMaterial>>) {

        for (control_point_idx, control_point) in control_points.iter().enumerate() {
            let translation = TransformUtils::util_convert_z_up_vec3_to_y_up_bevy_vec3(Vec3::new(control_point[0] as f32, control_point[1] as f32, control_point[2] as f32));
            let transform = Transform::from_translation(translation);

            commands.spawn_bundle(PbrBundle {
                mesh: meshes.add(Icosphere {
                    radius: 0.044,
                    subdivisions: 3
                }.into()),
                material: materials.add(Color::rgba(0.2,0.3,1.0,1.0).into()),
                transform,
                ..default()
            }).insert(SplineControlPoint {
                idx: control_point_idx
            })
                .insert_bundle(PickableBundle::default())
                .insert(GizmoTransformable);
        }
    }

    pub fn bspline_system(mut lines: ResMut<DebugLines>,
                          mut control_points: ResMut<Vec<DVector<f64>>>,
                          query: Query<(&SplineControlPoint, &Transform)>) {

        for q in query.iter() {
            let spline_control_point: &SplineControlPoint = &q.0;
            let transform: &Transform = &q.1;

            let location = TransformUtils::util_convert_bevy_y_up_vec3_to_z_up_vec3(transform.translation);
            control_points[spline_control_point.idx] = DVector::from_vec(vec![location.x as f64, location.y as f64, location.z as f64])
        }

        let mut splines = vec![];
        splines.push(BSpline::new(control_points.clone(), 2));
        splines.push(BSpline::new(control_points.clone(), 3));
        splines.push(BSpline::new(control_points.clone(), 4));
        splines.push(BSpline::new(control_points.clone(), 5));
        splines.push(BSpline::new(control_points.clone(), 6));

        let colors = vec![
            Color::rgb(1.0, 0.47, 0.0),
            Color::rgb(1.0, 0.7, 0.0),
            Color::rgb(1.0, 0.98, 0.0),
            Color::rgb(0.75, 1.0,0.0),
            Color::rgb(0.0, 1.0, 0.32)
        ];

        for (idx, spline) in splines.iter().enumerate() {
            let stride = 0.01;
            let mut t = 0.0 + stride;

            while t < (control_points.len() - 1) as f64 {
                let p0 = spline.interpolate(t - stride);
                let p1 = spline.interpolate(t);

                let p0_v = Vec3::new(p0[0] as f32, p0[1] as f32, p0[2] as f32);
                let p1_v = Vec3::new(p1[0] as f32, p1[1] as f32, p1[2] as f32);

                ViewportVisualsActions::action_draw_gpu_line_optima_space(&mut lines, p0_v, p1_v, colors[idx], 11.0, 3, 1, 0.0);

                t += stride;
            }
        }

    }
}

#[derive(Component)]
pub struct SplineControlPoint {
    pub idx: usize
}