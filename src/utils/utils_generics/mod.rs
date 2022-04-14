// use std::fmt::Debug;

/*
pub trait SignatureObject where Self: Clone + Debug + PartialEq + PartialOrd {
    type SignatureObjectType: Clone + Debug + PartialEq + PartialOrd;
    fn map_to_type() -> Self::SignatureObjectType;
}

pub struct SignatureObjectContainer<T: SignatureObject> {
    objects: Vec<T>
}
impl <T> SignatureObjectContainer<T> where T: SignatureObject {
    pub fn objects(&self) -> &Vec<T> {
        &self.objects
    }
    pub fn insert(&mut self, o: T) {
        let binary_search_res = self.objects.binary_search_by(|x| x.partial_cmp(&o).unwrap());
        let idx = match binary_search_res { Ok(i) => {i} Err(i) => {i} };
        self.objects.insert(idx, o);
    }
    pub fn get(&self, )
}
*/

