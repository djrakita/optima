use std::collections::HashMap;
use std::fmt::Debug;
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use crate::utils::utils_errors::OptimaError;
use crate::utils::utils_shape_geometry::geometric_shape::{GeometricShape, GeometricShapeSignature};

static mut GEOMETRIC_SHAPE_LIST_GENERATOR: GeometricShapeListGenerator = GeometricShapeListGenerator { bindings: vec![] };

#[derive(Clone, Debug)]
pub struct GeometricShapeListGenerator {
    bindings: Vec<(GeometricShapeListIdentifier, Vec<GeometricShapeSignature>)>
}
impl GeometricShapeListGenerator {
    fn generate_shape_list_global(&mut self, identifier: GeometricShapeListIdentifier, geometric_shapes: Vec<GeometricShape>) -> Result<GeometricShapeList, OptimaError> {
        let mut signature_to_idx_mapper = HashMap::new();
        for (i, shape) in geometric_shapes.iter().enumerate() {
            signature_to_idx_mapper.insert(shape.signature().clone(), i);
        }

        for binding in &self.bindings {
            if &binding.0 == &identifier {
                let signatures = &binding.1;
                if signatures.len() != geometric_shapes.len() {
                    return Err(OptimaError::new_generic_error_str(&format!("The given shapes do not match the required signatures for geometric shape list identifier {:?}.", identifier), file!(), line!()));
                }

                for (i, signature) in signatures.iter().enumerate() {
                    if signature != geometric_shapes[i].signature() {
                        return Err(OptimaError::new_generic_error_str(&format!("The given shapes do not match the required signatures for geometric shape list identifier {:?}.", identifier), file!(), line!()));
                    }
                }

                return Ok(GeometricShapeList { identifier, geometric_shapes, signature_to_idx_mapper });
            }
        }

        let mut signatures = vec![];

        for shape in &geometric_shapes {
            signatures.push(shape.signature().clone());
        }

        self.bindings.push( (identifier.clone(), signatures) );

        return Ok(GeometricShapeList { identifier, geometric_shapes, signature_to_idx_mapper });
    }
    pub unsafe fn generate_shape_list(identifier: GeometricShapeListIdentifier, geometric_shapes: Vec<GeometricShape>) -> Result<GeometricShapeList, OptimaError> {
        return GEOMETRIC_SHAPE_LIST_GENERATOR.generate_shape_list_global(identifier, geometric_shapes);
    }
}

#[derive(Clone)]
pub struct GeometricShapeList {
    identifier: GeometricShapeListIdentifier,
    geometric_shapes: Vec<GeometricShape>,
    signature_to_idx_mapper: HashMap<GeometricShapeSignature, usize>
}
impl GeometricShapeList {
    pub fn geometric_shapes(&self) -> &Vec<GeometricShape> {
        &self.geometric_shapes
    }
    pub fn identifier(&self) -> &GeometricShapeListIdentifier {
        &self.identifier
    }
    pub fn signature_to_idx_mapper(&self) -> &HashMap<GeometricShapeSignature, usize> {
        &self.signature_to_idx_mapper
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricShapeListCertificate {
    identifier: GeometricShapeListIdentifier,
    signatures: Vec<GeometricShapeSignature>
}
impl GeometricShapeListCertificate {
    pub fn new(geometric_shape_list: &GeometricShapeList) -> Self {
        let mut signatures = vec![];

        for shape in &geometric_shape_list.geometric_shapes { signatures.push(shape.signature().clone()); }

        Self {
            identifier: geometric_shape_list.identifier.clone(),
            signatures
        }
    }
    pub fn verify(&self, geometric_shape_list: &GeometricShapeList) -> bool {
        let mut signatures = vec![];

        for shape in &geometric_shape_list.geometric_shapes { signatures.push(shape.signature().clone()); }

        return self.verify_geometric_shapes_from_signatures(geometric_shape_list.identifier.clone(), &signatures);
    }
    fn verify_geometric_shapes_from_signatures(&self, identifier: GeometricShapeListIdentifier, signatures: &Vec<GeometricShapeSignature>) -> bool {
        if identifier != self.identifier { return false; }
        if signatures.len() != self.signatures.len() { return false; }
        if signatures != &self.signatures { return false; }

        return true;
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GeometricShapeListIdentifier {
    Test
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricShapeListQueryOrdering {
    order: Vec<(usize, usize)>
}
impl GeometricShapeListQueryOrdering {
    pub fn new_default(geometric_shape_list: &GeometricShapeList) -> Self {
        let mut order = vec![];
        let num_geometric_shapes = geometric_shape_list.geometric_shapes.len();
        for i in 0..num_geometric_shapes {
            for j in 0..num_geometric_shapes {
                if i <= j {
                    order.push((i,j));
                }
            }
        }

        Self {
            order
        }
    }
    pub fn set_order(&mut self, order: Vec<(usize, usize)>) {
        self.order = order;
    }
}

/// Bundles an object with a certificate.  This is useful as, when an object must be saved to a file, it
/// can be saved along with its certificate and can be unlocked via a `ObjectWithGeometricShapeListCertificate`
/// when loaded to guarantee that the `GeometricShapeList` corresponds to the right objects before and
/// after loading.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectWithGeometricShapeListCertificate<T> where T: Clone + Debug + Serialize + DeserializeOwned {
    #[serde(deserialize_with = "T::deserialize")]
    data: T,
    certificate: GeometricShapeListCertificate
}
impl <T> ObjectWithGeometricShapeListCertificate<T> where T: Clone + Debug + Serialize + DeserializeOwned {
    pub fn new(data: T, geometric_shape_list: &GeometricShapeList) -> Self {
        let certificate = GeometricShapeListCertificate::new(geometric_shape_list);
        Self {
            data,
            certificate
        }
    }
}

/// Ensures that an `ObjectWithGeometricShapeListCertificate` that is saved to a file must be
/// unlocked with a valid `GeometricShapeList` prior to accessing or modifying the underlying data.
#[derive(Clone, Debug)]
pub struct ObjectWithGeometricShapeListLock<T> where T: Clone + Debug + Serialize + DeserializeOwned {
    object_with_certificate: ObjectWithGeometricShapeListCertificate<T>,
    is_locked: bool
}
impl <T> ObjectWithGeometricShapeListLock<T> where T: Clone + Debug + Serialize + DeserializeOwned {
    pub fn new_locked(object_with_certificate: ObjectWithGeometricShapeListCertificate<T>) -> Self {
        Self {
            object_with_certificate,
            is_locked: true
        }
    }
    pub fn new_and_try_unlock(object_with_certificate: ObjectWithGeometricShapeListCertificate<T>, geometric_shape_list: &GeometricShapeList) -> Result<Self, OptimaError> {
        let mut out_self = Self::new_locked(object_with_certificate);
        out_self.try_unlock(geometric_shape_list)?;
        return Ok(out_self);
    }
    pub fn try_unlock(&mut self, geometric_shape_list: &GeometricShapeList) -> Result<(), OptimaError> {
        return if self.object_with_certificate.certificate.verify(geometric_shape_list) {
            self.is_locked = false;
            Ok(())
        } else {
            Err(OptimaError::new_generic_error_str("Could not unlock ObjectWithGeometricShapeListLock.", file!(), line!()))
        }
    }
    pub fn is_locked(&self) -> bool { return self.is_locked; }
    pub fn object(&self) -> Result<&T, OptimaError> {
        return if self.is_locked {
            Err(OptimaError::new_generic_error_str("Cannot access object as the container is still locked.", file!(), line!()))
        } else {
            Ok(&self.object_with_certificate.data)
        }
    }
    pub fn object_mut(&mut self) -> Result<&mut T, OptimaError> {
        return if self.is_locked {
            Err(OptimaError::new_generic_error_str("Cannot access object as the container is still locked.", file!(), line!()))
        } else {
            Ok(&mut self.object_with_certificate.data)
        }
    }
}





