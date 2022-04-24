use crate::utils::utils_shape_geometry::geometric_shape::GeometricShape;

pub struct SceneDescriptor {

}
impl SceneDescriptor {

}


pub struct SceneObject {
    asset_folder_name: String,
    scale: f64,
    shape_representation: SceneObjectShapeRepresentation
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Serialize, Deserialize)]
pub enum SceneObjectShapeRepresentation {
    Cubes,
    ConvexShapes,
    SphereSubcomponents,
    CubeSubcomponents,
    ConvexShapeSubcomponents,
    TriangleMeshes
}