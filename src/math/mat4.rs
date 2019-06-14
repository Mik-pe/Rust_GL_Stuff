use super::Vec4;
use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct Mat4(pub [Vec4; 4]);

// macro_rules! index_operators {
//     ($MatrixN:ident, $n:expr, $Output:ty, $I:ty) => {
//         impl Index<$I> for $MatrixN {
//             type Output = $Output;

//             #[inline]
//             fn index<'a>(&'a self, i: $I) -> &'a $Output {
//                 let v: &[[f32; $n]; $n] = self.as_ref();
//                 From::from(&v[i])
//             }
//         }

//         impl IndexMut<$I> for $MatrixN {
//             #[inline]
//             fn index_mut<'a>(&'a mut self, i: $I) -> &'a mut $Output {
//                 let v: &mut [[f32; $n]; $n] = self.as_mut();
//                 From::from(&mut v[i])
//             }
//         }
//     }
// }

// index_operators!(Mat4, 4, Vec4, usize);

impl Index<usize> for Mat4 {
    type Output = Vec4;

    fn index(&self, index: usize) -> &Vec4 {
        match index {
            0 => &self.0[0],
            1 => &self.0[1],
            2 => &self.0[2],
            3 => &self.0[3],
            _ => panic!("INDEXING OUT_OF_BOUNDS in Mat4"),
        }
    }
}

//Mat4 is considered a column-major matrix
impl Mat4 {
    pub fn new() -> Mat4 {
        Mat4([
            Vec4([1.0, 0.0, 0.0, 0.0]),
            Vec4([0.0, 1.0, 0.0, 0.0]),
            Vec4([0.0, 0.0, 1.0, 0.0]),
            Vec4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    pub fn from_translation(pos: [f32; 3]) -> Mat4 {
        Mat4([
            Vec4([1.0, 0.0, 0.0, 0.0]),
            Vec4([0.0, 1.0, 0.0, 0.0]),
            Vec4([0.0, 0.0, 1.0, 0.0]),
            Vec4([pos[0], pos[1], pos[2], 1.0]),
        ])
    }

    //Internal functions which makes less sense
    pub fn extract_row(&self, index: usize) -> Vec4 {
        Vec4([
            self[0][index],
            self[1][index],
            self[2][index],
            self[3][index],
        ])
    }

    pub fn from_rotaxis(angle: &f32, axis: [f32; 3]) -> Mat4 {
        let cos_part = angle.cos();
        let sin_part = angle.sin();
        let one_sub_cos = 1.0 - cos_part;
        Mat4([
            Vec4([
                one_sub_cos * axis[0] * axis[0] + cos_part,
                one_sub_cos * axis[0] * axis[1] + sin_part * axis[2],
                one_sub_cos * axis[0] * axis[2] - sin_part * axis[1],
                0.0,
            ]),
            Vec4([
                one_sub_cos * axis[0] * axis[1] - sin_part * axis[2],
                one_sub_cos * axis[1] * axis[1] + cos_part,
                one_sub_cos * axis[1] * axis[2] + sin_part * axis[0],
                0.0,
            ]),
            Vec4([
                one_sub_cos * axis[0] * axis[2] + sin_part * axis[1],
                one_sub_cos * axis[1] * axis[2] - sin_part * axis[0],
                one_sub_cos * axis[2] * axis[2] + cos_part,
                0.0,
            ]),
            Vec4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    pub fn identity() -> Mat4 {
        Mat4([
            Vec4([1.0, 0.0, 0.0, 0.0]),
            Vec4([0.0, 1.0, 0.0, 0.0]),
            Vec4([0.0, 0.0, 1.0, 0.0]),
            Vec4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    pub fn mul(self, _rhs: &Mat4) -> Mat4 {
        let row0 = self.extract_row(0);
        let row1 = self.extract_row(1);
        let row2 = self.extract_row(2);
        let row3 = self.extract_row(3);

        Mat4([
            Vec4([
                Vec4::dot(&row0, &_rhs[0]),
                Vec4::dot(&row1, &_rhs[0]),
                Vec4::dot(&row2, &_rhs[0]),
                Vec4::dot(&row3, &_rhs[0]),
            ]),
            Vec4([
                Vec4::dot(&row0, &_rhs[1]),
                Vec4::dot(&row1, &_rhs[1]),
                Vec4::dot(&row2, &_rhs[1]),
                Vec4::dot(&row3, &_rhs[1]),
            ]),
            Vec4([
                Vec4::dot(&row0, &_rhs[2]),
                Vec4::dot(&row1, &_rhs[2]),
                Vec4::dot(&row2, &_rhs[2]),
                Vec4::dot(&row3, &_rhs[2]),
            ]),
            Vec4([
                Vec4::dot(&row0, &_rhs[3]),
                Vec4::dot(&row1, &_rhs[3]),
                Vec4::dot(&row2, &_rhs[3]),
                Vec4::dot(&row3, &_rhs[3]),
            ]),
        ])
    }

    pub fn create_ortho(bottom: f32, top: f32, left: f32, right: f32, near: f32, far: f32) -> Mat4 {
        Mat4([
            Vec4([2.0 / (right - left), 0.0, 0.0, 0.0]),
            Vec4([0.0, 2.0 / (top - bottom), 0.0, 0.0]),
            Vec4([0.0, 0.0, -2.0 / (far - near), 0.0]), // <-- Revise negativity
            Vec4([
                -(right + left) / (right - left),
                -(top + bottom) / (top - bottom),
                -(far + near) / (far - near),
                1.0,
            ]),
        ])
    }
}
