use super::Vec3;
// impl Vec3{
  #[inline(always)]
  pub fn vec3_new(x : f32, y : f32, z : f32) -> Vec3
  {
    [
      x, 
      y,
      z,
    ]
  }

  #[inline(always)]
  pub fn vec3_add(_lhs : Vec3, _rhs : Vec3) -> Vec3
  {
    [
      _lhs[0] + _rhs[0], 
      _lhs[1] + _rhs[1],
      _lhs[2] + _rhs[2],
    ]
  }
// }