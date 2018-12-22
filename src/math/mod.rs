pub type Vec3 = [f32;3];

pub type Mat4 =  [[f32;4]; 4];

pub mod vec3_util;
pub mod mat4_impl;

pub use self::vec3_util::*;
pub use self::mat4_impl::*;