#[derive(Debug)]
pub struct Vec3(pub [f32; 3]);

impl std::ops::Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.0[0],
            1 => &self.0[1],
            2 => &self.0[2],
            _ => panic!("INDEXING OUT_OF_BOUNDS in Vec3"),
        }
    }
}
impl Vec3 {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3([x, y, z])
    }

    #[inline]
    pub fn add(self, _rhs: Vec3) -> Vec3 {
        Vec3([self[0] + _rhs[0], self[1] + _rhs[1], self[2] + _rhs[2]])
    }

    #[inline]
    pub fn dot(&self, b: Vec3) -> f32 {
        self[0] * b[0] + self[1] * b[1] + self[2] * b[2]
    }
}
