use super::Vec4;

impl std::ops::Index<usize> for Vec4 {
  type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.0[0],
            1 => &self.0[1],
            2 => &self.0[2],
            3 => &self.0[3],
            _ => panic!("INDEXING OUT_OF_BOUNDS in Vec4")
        }
    }
}

#[inline(always)]
pub fn vec4_new(x : f32, y : f32, z : f32) -> Vec4
{
  Vec4([
    x, 
    y,
    z,
    1.0,
  ])
}

#[inline(always)]
pub fn vec4_add(_lhs : Vec4, _rhs : Vec4) -> Vec4
{
  Vec4([
    _lhs[0] + _rhs[0], 
    _lhs[1] + _rhs[1],
    _lhs[2] + _rhs[2],
    _lhs[3] + _rhs[3],
  ])
}


#[inline(always)]
pub fn vec4_dot(a : &Vec4, b : &Vec4) -> f32
{
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}