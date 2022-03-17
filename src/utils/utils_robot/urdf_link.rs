use urdf_rs::*;
use nalgebra::{Vector3, Matrix3};
use serde::{Serialize, Deserialize};

/// This struct holds all information provided by a URDF file on a Link when parsed by urdf_rs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct URDFLink {
    name: String,
    inertial_origin_xyz: Vector3<f64>,
    intertial_origin_rpy: Vector3<f64>,
    inertial_matrix: Matrix3<f64>,
    intertial_mass: f64,
    visual_origin_xyz: Option<Vector3<f64>>,
    visual_origin_rpy: Option<Vector3<f64>>,
    visual_mesh_filename: Option<String>,
    visual_mesh_scale: Option<Vector3<f64>>,
    collision_origin_xyz: Option<Vector3<f64>>,
    collision_origin_rpy: Option<Vector3<f64>>,
    collision_mesh_filename: Option<String>,
    collision_mesh_scale: Option<Vector3<f64>>,
}
impl URDFLink {
    pub fn new_from_urdf_link(link: &Link) -> Self {
        /*
        match link.visual[0].geometry {
            Geometry::Box { .. } => {}
            Geometry::Cylinder { .. } => {}
            Geometry::Capsule { .. } => {}
            Geometry::Sphere { .. } => {}
            Geometry::Mesh { filename, scale } => {}
        }
        */
        let visual_mesh_filename = if link.visual.len() > 0 {
            match &link.visual[0].geometry {
                Geometry::Mesh { filename, scale } => { Some(filename.clone()) }
                _ => { None }
            }
        } else {
            None
        };
        let visual_mesh_scale: Option<Vector3<f64>> = if link.visual.len() > 0 {
            match &link.visual[0].geometry {
                Geometry::Mesh { filename, scale } => {
                    match scale {
                        None => { None }
                        Some(v) => { Some(Vector3::new(v[0], v[1], v[2])) }
                    }
                }
                _ => { None }
            }
        } else {
            None
        };

        let collision_mesh_filename = if link.collision.len() > 0 {
            match &link.collision[0].geometry {
                Geometry::Mesh { filename, scale } => { Some(filename.clone()) }
                _ => { None }
            }
        } else {
            None
        };
        let collision_mesh_scale: Option<Vector3<f64>> = if link.collision.len() > 0 {
            match &link.collision[0].geometry {
                Geometry::Mesh { filename, scale } => {
                    match scale {
                        None => { None }
                        Some(v) => { Some(Vector3::new(v[0], v[1], v[2])) }
                    }
                }
                _ => { None }
            }
        } else {
            None
        };

        Self {
            name: link.name.clone(),
            inertial_origin_xyz: Vector3::new(link.inertial.origin.xyz[0], link.inertial.origin.xyz[1], link.inertial.origin.xyz[2]),
            intertial_origin_rpy: Vector3::new(link.inertial.origin.rpy[0], link.inertial.origin.rpy[1], link.inertial.origin.rpy[2]),
            inertial_matrix: Matrix3::new(link.inertial.inertia.ixx, link.inertial.inertia.ixy, link.inertial.inertia.ixz, link.inertial.inertia.ixy, link.inertial.inertia.iyy, link.inertial.inertia.iyz, link.inertial.inertia.ixz, link.inertial.inertia.iyz, link.inertial.inertia.izz),
            intertial_mass: link.inertial.mass.value,
            visual_origin_xyz: if link.visual.len() > 0 { Some( Vector3::new(link.visual[0].origin.xyz[0], link.visual[0].origin.xyz[1], link.visual[0].origin.xyz[2])) } else { None } ,
            visual_origin_rpy: if link.visual.len() > 0 { Some( Vector3::new(link.visual[0].origin.rpy[0], link.visual[0].origin.rpy[1], link.visual[0].origin.rpy[2])) } else { None },
            visual_mesh_filename,
            visual_mesh_scale,
            collision_origin_xyz: if link.collision.len() > 0 { Some( Vector3::new(link.collision[0].origin.xyz[0], link.collision[0].origin.xyz[1], link.collision[0].origin.xyz[2])) } else { None },
            collision_origin_rpy: if link.collision.len() > 0 { Some( Vector3::new(link.collision[0].origin.rpy[0], link.collision[0].origin.rpy[1], link.collision[0].origin.rpy[2])) } else { None },
            collision_mesh_filename,
            collision_mesh_scale
        }
    }
    pub fn new_empty() -> Self {
        Self {
            name: "".to_string(),
            inertial_origin_xyz: Default::default(),
            intertial_origin_rpy: Default::default(),
            inertial_matrix: Default::default(),
            intertial_mass: 0.0,
            visual_origin_xyz: None,
            visual_origin_rpy: None,
            visual_mesh_filename: None,
            visual_mesh_scale: None,
            collision_origin_xyz: None,
            collision_origin_rpy: None,
            collision_mesh_filename: None,
            collision_mesh_scale: None
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn inertial_origin_xyz(&self) -> Vector3<f64> {
        self.inertial_origin_xyz
    }
    pub fn intertial_origin_rpy(&self) -> Vector3<f64> {
        self.intertial_origin_rpy
    }
    pub fn inertial_matrix(&self) -> Matrix3<f64> {
        self.inertial_matrix
    }
    pub fn intertial_mass(&self) -> f64 {
        self.intertial_mass
    }
    pub fn visual_origin_xyz(&self) -> Option<Vector3<f64>> {
        self.visual_origin_xyz
    }
    pub fn visual_origin_rpy(&self) -> Option<Vector3<f64>> {
        self.visual_origin_rpy
    }
    pub fn visual_mesh_filename(&self) -> &Option<String> {
        &self.visual_mesh_filename
    }
    pub fn visual_mesh_scale(&self) -> Option<Vector3<f64>> {
        self.visual_mesh_scale
    }
    pub fn collision_origin_xyz(&self) -> Option<Vector3<f64>> {
        self.collision_origin_xyz
    }
    pub fn collision_origin_rpy(&self) -> Option<Vector3<f64>> {
        self.collision_origin_rpy
    }
    pub fn collision_mesh_filename(&self) -> &Option<String> {
        &self.collision_mesh_filename
    }
    pub fn collision_mesh_scale(&self) -> Option<Vector3<f64>> {
        self.collision_mesh_scale
    }
}