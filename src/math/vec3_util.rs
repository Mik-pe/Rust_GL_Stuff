use super::Vec3;

impl std::ops::Index<usize> for Vec3 {
  type Output = f32;
  fn index(&self, index: usize) -> &f32 {
      match index {
          0 => &self.0[0],
          1 => &self.0[1],
          2 => &self.0[2],
          _ => panic!("INDEXING OUT_OF_BOUNDS in Vec3")
      }
  }
}

#[inline(always)]
pub fn vec3_new(x : f32, y : f32, z : f32) -> Vec3
{
  Vec3([
    x, 
    y,
    z,
  ])
}

#[inline(always)]
pub fn vec3_add(_lhs : Vec3, _rhs : Vec3) -> Vec3
{
  Vec3([
      _lhs[0] + _rhs[0], 
      _lhs[1] + _rhs[1],
      _lhs[2] + _rhs[2],
    ])
}


#[inline(always)]
pub fn vec3_dot(a : &Vec3, b : &Vec3) -> f32
{
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}