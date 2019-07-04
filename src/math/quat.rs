#[derive(Debug)]
#[allow(dead_code)]
pub struct Quat(pub [f32; 4]);

impl std::ops::Index<usize> for Quat {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.0[0],
            1 => &self.0[1],
            2 => &self.0[2],
            3 => &self.0[3],
            _ => panic!("INDEXING OUT_OF_BOUNDS in Vec3"),
        }
    }
}
impl Quat {
    #[inline]
    #[allow(dead_code)]
    pub fn new() -> Quat {
        Quat([0.0, 0.0, 0.0, 1.0])
    }
}
