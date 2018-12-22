use std::ops::Index;
use super::{Mat4, Vec4};


impl Index<usize> for Mat4 {
  type Output = Vec4;

    fn index(&self, index: usize) -> &Vec4 {
        match index {
            0 => &self.0[0],
            1 => &self.0[1],
            2 => &self.0[2],
            3 => &self.0[3],
            _ => panic!("INDEXING OUT_OF_BOUNDS in Mat4")
        }
    }
}

//Mat4 is considered a column-major matrix

  pub fn mat4_translation(pos : [f32;3]) -> Mat4
  { 
    Mat4([
      Vec4([1.0, 0.0, 0.0, 0.0]),
      Vec4([0.0, 1.0, 0.0, 0.0]),
      Vec4([0.0, 0.0, 1.0, 0.0]),
      Vec4([pos[0], pos[1], pos[2], 1.0]),
    ])
  }

  pub fn mat4_rotation(angle : f32, axis : [f32;3]) -> Mat4
  { 
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
      Vec4([
        0.0, 0.0, 0.0, 1.0,
      ])
    ])
  }

  pub fn mat4_identity() -> Mat4
  {
    Mat4([
      Vec4([1.0, 0.0, 0.0, 0.0]),
      Vec4([0.0, 1.0, 0.0, 0.0]),
      Vec4([0.0, 0.0, 1.0, 0.0]),
      Vec4([0.0, 0.0, 0.0, 1.0]),
    ])
  }



  pub fn mat4_mul(a : &Mat4, _rhs : &Mat4) -> Mat4
  {
    let row0 = extract_row(&a, 0);
    let row1 = extract_row(&a, 1);
    let row2 = extract_row(&a, 2);
    let row3 = extract_row(&a, 3);
    
    Mat4([
     Vec4([
        dot_product(&row0, &_rhs[0]),
        dot_product(&row1, &_rhs[0]),
        dot_product(&row2, &_rhs[0]),
        dot_product(&row3, &_rhs[0]),
      ]),
      Vec4([
        dot_product(&row0, &_rhs[1]),
        dot_product(&row1, &_rhs[1]),
        dot_product(&row2, &_rhs[1]),
        dot_product(&row3, &_rhs[1]),
      ]),
      Vec4([ 
        dot_product(&row0, &_rhs[2]),
        dot_product(&row1, &_rhs[2]),
        dot_product(&row2, &_rhs[2]),
        dot_product(&row3, &_rhs[2]),
      ]),
      Vec4([
        dot_product(&row0, &_rhs[3]),
        dot_product(&row1, &_rhs[3]),
        dot_product(&row2, &_rhs[3]),
        dot_product(&row3, &_rhs[3]),
      ]),
    ])
  }

//Internal functions which makes less sense
  pub(in super) fn extract_row(a : &Mat4, index : usize) -> Vec4
  {
    Vec4(
      [ 
        a[0][index],
        a[1][index],
        a[2][index],
        a[3][index]
      ])
  }

  fn dot_product(a : &Vec4, b : &Vec4) -> f32
  {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
  }
