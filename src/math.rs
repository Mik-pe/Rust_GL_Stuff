mod vec3;
mod vec4;
mod mat4;


pub use self::vec3::Vec3;
pub use self::vec4::Vec4;
pub use self::mat4::Mat4;

#[derive(Debug)]
pub struct Quat(pub [f32;4]);

//Internal functions which makes less sense
pub fn extract_row(a : &Mat4, index : usize) -> Vec4
{
  Vec4(
    [ 
      a[0][index],
      a[1][index],
      a[2][index],
      a[3][index]
    ])
}

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
    b.dot(Vec3([row0[0], row0[1], row0[2]])) + row0[3],
    b.dot(Vec3([row2[0], row2[1], row2[2]])) + row2[3],
    b.dot(Vec3([row1[0], row1[1], row1[2]])) + row1[3],
  ])
}

pub fn mat4_mul_vec4(a : &Mat4, b : &Vec4) -> Vec4
{
  let row0 = extract_row(&a, 0);
  let row1 = extract_row(&a, 1);
  let row2 = extract_row(&a, 2);
  let row3 = extract_row(&a, 3);
  Vec4([
    Vec4::dot(&row0, &b),
    Vec4::dot(&row1, &b),
    Vec4::dot(&row2, &b),
    Vec4::dot(&row3, &b),
  ])
}