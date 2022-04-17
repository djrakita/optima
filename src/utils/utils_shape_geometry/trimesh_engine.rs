use std::collections::HashMap;
use std::fs::File;
use std::str::FromStr;
use collada::PrimitiveElement;
use serde::{Serialize, Deserialize};
use collada::document::ColladaDocument;
use dae_parser::{Document, Transform};
use nalgebra::{Matrix4, Point3, Unit, UnitQuaternion, Vector3};
use parry3d_f64::transformation::convex_hull;
use parry3d_f64::transformation::vhacd::{VHACD, VHACDParameters};
use stl_io::IndexedMesh;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_files::optima_path::{OptimaPath, OptimaStemCellPath};
use crate::utils::utils_nalgebra::conversions::NalgebraConversions;
use crate::utils::utils_se3::homogeneous_matrix::HomogeneousMatrix;
use crate::utils::utils_se3::optima_rotation::OptimaRotation;
use crate::utils::utils_se3::optima_se3_pose::OptimaSE3Pose;
use crate::utils::utils_se3::rotation_and_translation::RotationAndTranslation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrimeshEngine {
    vertices: Vec<Vector3<f64>>,
    indices: Vec<[usize; 3]>,
    normals: Option<Vec<Vector3<f64>>>
}
impl TrimeshEngine {
    pub fn new_from_vertices_and_indices(vertices: Vec<Vector3<f64>>, indices: Vec<[usize; 3]>, normals: Option<Vec<Vector3<f64>>>) -> Self {
        Self {
            vertices,
            indices,
            normals
        }
    }
    pub fn compute_convex_decomposition(&self) -> Vec<TrimeshEngine> {
        let points: Vec<Point3<f64>> = self.vertices.iter().map(|v| NalgebraConversions::vector3_to_point3(v)).collect();
        let indices: Vec<[u32; 3]> = self.indices.iter().map(|i| [i[0] as u32, i[1] as u32, i[2] as u32] ).collect();

        let params = VHACDParameters {
            ..Default::default()
        };
        let v = VHACD::decompose(&params, points.as_slice(), indices.as_slice(), true);
        let res_vec = v.compute_convex_hulls(5);

        let mut out_vec = vec![];
        for res in &res_vec {
            let vertices: Vec<Vector3<f64>> = res.0.iter().map(|p| NalgebraConversions::point3_to_vector3(p) ).collect();
            let indices: Vec<[usize; 3]> = res.1.iter().map(|i| [i[0] as usize, i[1] as usize, i[2] as usize] ).collect();
            out_vec.push(TrimeshEngine::new_from_vertices_and_indices(vertices, indices, None));
        }

        return out_vec;
    }
    pub fn compute_convex_hull(&self) -> TrimeshEngine {
        let points: Vec<Point3<f64>> = self.vertices.iter().map(|v| NalgebraConversions::vector3_to_point3(v)).collect();

        let res = convex_hull(&points);

        let vertices: Vec<Vector3<f64>> = res.0.iter().map(|p| NalgebraConversions::point3_to_vector3(p) ).collect();
        let indices: Vec<[usize; 3]> = res.1.iter().map(|i| [i[0] as usize, i[1] as usize, i[2] as usize] ).collect();

        return TrimeshEngine::new_from_vertices_and_indices(vertices, indices, None);
    }
    pub fn vertices(&self) -> &Vec<Vector3<f64>> {
        &self.vertices
    }
    pub fn indices(&self) -> &Vec<[usize; 3]> {
        &self.indices
    }
    pub fn normals(&self) -> &Option<Vec<Vector3<f64>>> {
        &self.normals
    }
}

/// Implementations for TrimeshEngine.
impl OptimaStemCellPath {
    pub fn load_file_to_trimesh_engine(&self) -> Result<TrimeshEngine, OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::load_file_to_trimesh_engine, "load_file_to_trimesh_engine")
    }
    pub fn load_stl_to_trimesh_engine(&self) -> Result<TrimeshEngine, OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::load_stl_to_trimesh_engine, "load_stl_to_trimesh_engine")
    }
    pub fn load_dae_to_trimesh_engine(&self) -> Result<TrimeshEngine, OptimaError> {
        self.try_function_on_all_optima_file_paths(OptimaPath::load_dae_to_trimesh_engine, "load_dae_to_trimesh_engine")
    }
    pub fn load_stl(&self) -> Result<IndexedMesh, OptimaError> {
        return self.try_function_on_all_optima_file_paths(OptimaPath::load_stl, "load_stl");
    }
    pub fn load_dae(&self) -> Result<Document, OptimaError> {
        return self.try_function_on_all_optima_file_paths(OptimaPath::load_dae, "load_dae");
    }
    pub fn load_collada_dae(&self) -> Result<ColladaDocument, OptimaError> {
        return self.try_function_on_all_optima_file_paths(OptimaPath::load_collada_dae, "load_collada_dae");
    }
    pub fn save_trimesh_engine_to_stl(&self, trimesh_engine: &TrimeshEngine) -> Result<(), OptimaError> {
        for p in self.optima_file_paths() {
            let res = p.save_trimesh_engine_to_stl(trimesh_engine);
            if res.is_ok() { return Ok(()) }
        }

        return Err(OptimaError::new_generic_error_str("No valid optima_path in function save_object_to_file_as_json()", file!(), line!()));
    }
}

/// Implementations for TrimeshEngine.
impl OptimaPath {
    pub fn load_file_to_trimesh_engine(&self) -> Result<TrimeshEngine, OptimaError> {
        let extension_option = self.extension();
        return match &extension_option {
            None => {
                Err(OptimaError::new_generic_error_str("Could not load file {:?} as TrimeshEngine", file!(), line!()))
            }
            Some(extension) => {
                if extension == "stl" || extension == "STL" {
                    self.load_stl_to_trimesh_engine()
                } else if extension == "dae" || extension == "DAE" {
                    self.load_dae_to_trimesh_engine()
                } else {
                    Err(OptimaError::new_generic_error_str("Could not load file {:?} as TrimeshEngine", file!(), line!()))
                }
            }
        }
    }
    pub fn load_stl_to_trimesh_engine(&self) -> Result<TrimeshEngine, OptimaError> {
        let indexed_mesh = self.load_stl()?;

        let mut vertices = vec![];
        let mut indices = vec![];

        for v in &indexed_mesh.vertices {
            vertices.push(Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64));
        }
        for f in &indexed_mesh.faces {
            indices.push(f.vertices.clone());
        }

        return Ok(TrimeshEngine::new_from_vertices_and_indices(vertices, indices, None));
    }
    pub fn load_dae_to_trimesh_engine(&self) -> Result<TrimeshEngine, OptimaError> {
        let collada_dae = self.load_collada_dae()?;
        let dae = self.load_dae()?;

        let mut hash = HashMap::new();

        let visual_scene_option = dae.get_visual_scene();
        match visual_scene_option {
            Some(visual_scene) => {
                for node in &visual_scene.nodes {
                    let instance_geometry_vec = &node.instance_geometry;
                    if instance_geometry_vec.len() > 0 {
                        let url = instance_geometry_vec[0].url.val.to_string();
                        let url_without_hashtag = url.strip_prefix("#").unwrap().to_string();

                        let transforms_vec = &node.transforms;
                        let mut total_transform = OptimaSE3Pose::new_homogeneous_matrix_from_euler_angles(0.,0.,0.,0.,0.,0.);
                        for transform in transforms_vec {
                            let pose = match transform {
                                Transform::LookAt(t) => {
                                    let mut mat = Matrix4::zeros();
                                    mat[(0,0)] = t.0[0] as f64;
                                    mat[(1,0)] = t.0[1] as f64;
                                    mat[(2,0)] = t.0[2] as f64;

                                    mat[(0,1)] = t.0[3] as f64;
                                    mat[(1,1)] = t.0[4] as f64;
                                    mat[(2,1)] = t.0[5] as f64;

                                    mat[(0,2)] = t.0[6] as f64;
                                    mat[(1,2)] = t.0[7] as f64;
                                    mat[(2,2)] = t.0[8] as f64;

                                    mat[(3,3)] = 1.0;

                                    OptimaSE3Pose::new_homogeneous_matrix(HomogeneousMatrix::new(mat))
                                }
                                Transform::Matrix(t) => {
                                    let mut mat = Matrix4::zeros();
                                    mat[(0,0)] = t.0[0] as f64;
                                    mat[(1,0)] = t.0[1] as f64;
                                    mat[(2,0)] = t.0[2] as f64;
                                    mat[(3,0)] = t.0[3] as f64;

                                    mat[(0,1)] = t.0[4] as f64;
                                    mat[(1,1)] = t.0[5] as f64;
                                    mat[(2,1)] = t.0[6] as f64;
                                    mat[(3,1)] = t.0[7] as f64;

                                    mat[(0,2)] = t.0[8] as f64;
                                    mat[(1,2)] = t.0[9] as f64;
                                    mat[(2,2)] = t.0[10] as f64;
                                    mat[(3,2)] = t.0[11] as f64;

                                    mat[(0,3)] = t.0[12] as f64;
                                    mat[(1,3)] = t.0[13] as f64;
                                    mat[(2,3)] = t.0[14] as f64;
                                    mat[(3,3)] = t.0[15] as f64;

                                    OptimaSE3Pose::new_homogeneous_matrix(HomogeneousMatrix::new(mat))
                                }
                                Transform::Rotate(t) => {
                                    let axis = Vector3::new(t.0[0] as f64, t.0[1] as f64, t.0[2] as f64);
                                    let angle = (t.0[3] as f64).to_degrees();
                                    let quat = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle);
                                    let r = RotationAndTranslation::new(OptimaRotation::new_unit_quaternion(quat), Vector3::zeros());

                                    OptimaSE3Pose::new_rotation_and_translation(r)
                                }
                                Transform::Scale(t) => {
                                    let mut mat = Matrix4::zeros();
                                    mat[(0,0)] = t.0[0] as f64;
                                    mat[(1,1)] = t.0[1] as f64;
                                    mat[(2,2)] = t.0[2] as f64;

                                    mat[(3,3)] = 1.0;

                                    OptimaSE3Pose::new_homogeneous_matrix(HomogeneousMatrix::new(mat))
                                }
                                Transform::Skew(_) => {
                                    unimplemented!()
                                }
                                Transform::Translate(t) => {
                                    let mut mat = Matrix4::zeros();
                                    mat[(0,3)] = t.0[0] as f64;
                                    mat[(1,3)] = t.0[1] as f64;
                                    mat[(2,3)] = t.0[2] as f64;

                                    mat[(3,3)] = 1.0;

                                    OptimaSE3Pose::new_homogeneous_matrix(HomogeneousMatrix::new(mat))
                                }
                            };
                            total_transform = total_transform.multiply(&pose, true).expect("error");
                        }

                        hash.insert(url_without_hashtag, total_transform);
                    }
                }
            }
            _ => { }
        }

        let mut vertices = vec![];
        let mut indices = vec![];

        let obj_set_option = collada_dae.get_obj_set();
        match &obj_set_option {
            None => { return Err(OptimaError::new_generic_error_str(&format!("dae file {:?}, has no object set.", self), file!(), line!())) }
            Some(obj_set) => {
                let objects = &obj_set.objects;
                for obj in objects {
                    let transform_option = hash.get(&obj.id);

                    let vertices_len = vertices.len();

                    for verts in &obj.vertices {
                        match transform_option {
                            None => {
                                vertices.push(Vector3::new(verts.x, verts.y, verts.z));
                            }
                            Some(transform) => {
                                vertices.push(transform.multiply_by_point(&Vector3::new(verts.x, verts.y, verts.z)));
                            }
                        }
                    }

                    for geom in &obj.geometry {
                        let primitive_elements = &geom.mesh;
                        for primitive_element in primitive_elements {
                            match primitive_element {
                                PrimitiveElement::Polylist(_) => {
                                    // optima_print("WARNING: DAE parsing in Optima does not support Polylist elements.", PrintMode::Println, PrintColor::Yellow, true);
                                }
                                PrimitiveElement::Triangles(triangles) => {
                                    for triangle in &triangles.vertices {
                                        indices.push([triangle.0 as usize + vertices_len, triangle.1 as usize + vertices_len, triangle.2 as usize + vertices_len]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(TrimeshEngine::new_from_vertices_and_indices(vertices, indices, None));
    }
    pub fn load_stl(&self) -> Result<IndexedMesh, OptimaError> {
        self.verify_extension(&vec!["stl", "STL"])?;
        return match self {
            OptimaPath::Path(p) => {
                let mut file = File::open(p);
                match &mut file {
                    Ok(f) => {
                        let read_res = stl_io::read_stl(f);
                        match read_res {
                            Ok(read) => { Ok(read) }
                            Err(_) => {
                                Err(OptimaError::new_generic_error_str(&format!("File with path {:?} could not be read as an stl file.", p), file!(), line!()))
                            }
                        }
                    }
                    Err(_) => {
                        Err(OptimaError::new_generic_error_str(&format!("File with path {:?} could not be opened.", p), file!(), line!()))
                    }
                }
            }
            OptimaPath::VfsPath(p) => {
                let mut file = p.open_file();
                match &mut file {
                    Ok(f) => {
                        let read_res = stl_io::read_stl(f);
                        match read_res {
                            Ok(read) => { Ok(read) }
                            Err(_) => {
                                Err(OptimaError::new_generic_error_str(&format!("File with path {:?} could not be read as an stl file.", p), file!(), line!()))
                            }
                        }
                    }
                    Err(_) => {
                        Err(OptimaError::new_generic_error_str(&format!("File with path {:?} could not be opened.", p), file!(), line!()))
                    }
                }
            }
        }
    }
    pub fn load_dae(&self) -> Result<Document, OptimaError> {
        self.verify_extension(&vec!["dae", "DAE"])?;
        let string = self.read_file_contents_to_string()?;
        let dae_result = Document::from_str(&string);
        // println!("{:?}", dae_result);
        return match dae_result {
            Ok(dae) => {
                Ok(dae)
            }
            Err(e) => {
                Err(OptimaError::new_generic_error_str(&format!("Could not parse dae file at path {:?}.  The error was {:?}.", self, e), file!(), line!()))
            }
        }
    }
    pub fn load_collada_dae(&self) -> Result<ColladaDocument, OptimaError> {
        self.verify_extension(&vec!["dae", "DAE"])?;
        let string = self.read_file_contents_to_string()?;
        let collada_result = ColladaDocument::from_str(&string);
        return match collada_result {
            Ok(dae) => {
                Ok(dae)
            }
            Err(_) => {
                Err(OptimaError::new_generic_error_str(&format!("Could not parse dae file at path {:?}", self), file!(), line!()))
            }
        }
    }
    pub fn save_trimesh_engine_to_stl(&self, trimesh_engine: &TrimeshEngine) -> Result<(), OptimaError> {
        self.verify_extension(&vec!["stl", "STL"])?;

        let normals_option = &trimesh_engine.normals;

        let mut mesh = vec![];
        let l = trimesh_engine.indices.len();
        for i in 0..l {
            let _v1 = trimesh_engine.vertices[ trimesh_engine.indices[i][0] ].clone();
            let _v2 = trimesh_engine.vertices[ trimesh_engine.indices[i][1] ].clone();
            let _v3 = trimesh_engine.vertices[ trimesh_engine.indices[i][2] ].clone();

            let v1 = stl_io::Vertex::new( [_v1[0] as f32, _v1[1] as f32, _v1[2] as f32]  );
            let v2 = stl_io::Vertex::new( [_v2[0] as f32, _v2[1] as f32, _v2[2] as f32]  );
            let v3 = stl_io::Vertex::new( [_v3[0] as f32, _v3[1] as f32, _v3[2] as f32]  );

            let normal = match normals_option {
                None => {
                    let a = &_v2 - &_v1;
                    let b = &_v3 - &_v2;
                    let n = a.cross(&b);

                    stl_io::Normal::new([n[0] as f32, n[1] as f32, n[2] as f32])
                }
                Some(normals) => {
                    let n = normals[i].clone();
                    stl_io::Normal::new([n[0] as f32, n[1] as f32, n[2] as f32])
                }
            };

            let triangle = stl_io::Triangle{ normal, vertices: [v1, v2, v3] };
            mesh.push(triangle);
        }

        let mut file_for_writing = self.get_file_for_writing()?;
        let res = stl_io::write_stl(&mut file_for_writing, mesh.iter());
        if res.is_err() {
            return Err(OptimaError::new_generic_error_str(&format!("TrimeshEngine could not be written to stl at path {:?}", self), file!(), line!()));
        }
        Ok(())
    }
}

