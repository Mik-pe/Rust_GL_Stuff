#[derive(Clone, Copy, Default, Debug)]
pub struct Vec3
{
  pub x : f32,
  pub y : f32, 
  pub z : f32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Mat4
{
  pub c : [[f32;4]; 4],
}

pub mod vec3_util;
pub mod mat4_impl;