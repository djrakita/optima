use strum::IntoEnumIterator;
use crate::utils::utils_traits::ToAndFromRonString;

pub struct EnumUtils;
impl EnumUtils {
    pub fn get_all_variants_of_enum<T: IntoEnumIterator>() -> Vec<T> {
        let out: Vec<T> = T::iter().collect();
        out
    }
    pub fn convert_all_variants_of_enum_into_ron_strings<T: IntoEnumIterator + ToAndFromRonString>() -> Vec<String> {
        let mut out = vec![];

        let variants = Self::get_all_variants_of_enum::<T>();
        for v in &variants { out.push(v.to_ron_string()); }

        out
    }
}
