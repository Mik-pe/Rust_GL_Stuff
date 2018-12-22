#[derive(Debug)]
pub struct Vec3(pub [f32;3]);
#[derive(Debug)]
pub struct Quat(pub [f32;4]);
#[derive(Debug)]
pub struct Vec4(pub [f32;4]);
#[derive(Debug)]
pub struct Mat4(pub [Vec4; 4]);


pub mod vec3_util;
pub mod vec4_util;
pub mod mat4_impl;

pub use self::vec3_util::*;
pub use self::vec4_util::*;
pub use self::mat4_impl::*;

pub fn quat_new() -> Quat
{
  Quat([0.0, 0.0, 0.0, 1.0])
}
//Assume lower row is 0_0_0_1
pub fn mat4_mul_vec3(a : &Mat4, b : &Vec3) -> Vec3
{
  let row0 = extract_row(&a, 0);
  let row1 = extract_row(&a, 1);
  let row2 = extract_row(&a, 2);
  //TODO: Don't create a new object here:
  Vec3([
    vec3_dot(&Vec3([row0[0], row0[1], row0[2]]), &b) + row0[3],
    vec3_dot(&Vec3([row1[0], row1[1], row1[2]]), &b) + row1[3],
    vec3_dot(&Vec3([row2[0], row2[1], row2[2]]), &b) + row2[3],
  ])
}

pub fn mat4_mul_vec4(a : &Mat4, b : &Vec4) -> Vec4
{
  let row0 = extract_row(&a, 0);
  let row1 = extract_row(&a, 1);
  let row2 = extract_row(&a, 2);
  let row3 = extract_row(&a, 3);
  Vec4([
    vec4_dot(&row0, &b),
    vec4_dot(&row1, &b),
    vec4_dot(&row2, &b),
    vec4_dot(&row3, &b),
  ])
}