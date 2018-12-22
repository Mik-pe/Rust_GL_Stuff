use super::Mat4;


  fn dot_product(a : [f32;4], b : [f32;4]) -> f32
  {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
  }

//Mat4 is considered a column-major matrix
impl Mat4
{
  pub fn identity() -> Mat4
  {
    Mat4{
      c : 
      [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ]
    }
  }
  fn extract_row(self, index : usize) -> [f32;4]
  {
    [ self.c[0][index],
      self.c[1][index],
      self.c[2][index],
      self.c[3][index]]
  }


  pub fn mul(self, _rhs : Mat4) -> Mat4
  {
    let row0 = self.extract_row(0);
    let row1 = self.extract_row(1);
    let row2 = self.extract_row(2);
    let row3 = self.extract_row(3);
    
    Mat4{c: [
        [
          dot_product(row0, _rhs.c[0]),
          dot_product(row1, _rhs.c[0]),
          dot_product(row2, _rhs.c[0]),
          dot_product(row3, _rhs.c[0]),
        ],
        [
          dot_product(row0, _rhs.c[1]),
          dot_product(row1, _rhs.c[1]),
          dot_product(row2, _rhs.c[1]),
          dot_product(row3, _rhs.c[1]),
        ],
        [ 
          dot_product(row0, _rhs.c[2]),
          dot_product(row1, _rhs.c[2]),
          dot_product(row2, _rhs.c[2]),
          dot_product(row3, _rhs.c[2]),
        ],
        [
          dot_product(row0, _rhs.c[3]),
          dot_product(row1, _rhs.c[3]),
          dot_product(row2, _rhs.c[3]),
          dot_product(row3, _rhs.c[3]),
        ],
      ]
    }
  }
}