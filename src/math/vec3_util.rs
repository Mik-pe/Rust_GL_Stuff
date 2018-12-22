use super::Vec3;
impl Vec3{
  #[inline(always)]
  pub fn add(self, _rhs : Vec3) -> Vec3
  {
    Vec3{
      x : self.x + _rhs.x, 
      y : self.y + _rhs.y,
      z : self.z + _rhs.z}
  }
}