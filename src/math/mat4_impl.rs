use super::Mat4;



//Mat4 is considered a column-major matrix

  pub fn mat4_translation(pos : [f32;3]) -> Mat4
  { 
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [pos[0], pos[1], pos[2], 1.0],
    ]
  }

  pub fn mat4_identity() -> Mat4
  {
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ]
  }



  pub fn mat4_mul(a : Mat4, _rhs : Mat4) -> Mat4
  {
    let row0 = extract_row(a, 0);
    let row1 = extract_row(a, 1);
    let row2 = extract_row(a, 2);
    let row3 = extract_row(a, 3);
    
    [
      [
        dot_product(row0, _rhs[0]),
        dot_product(row1, _rhs[0]),
        dot_product(row2, _rhs[0]),
        dot_product(row3, _rhs[0]),
      ],
      [
        dot_product(row0, _rhs[1]),
        dot_product(row1, _rhs[1]),
        dot_product(row2, _rhs[1]),
        dot_product(row3, _rhs[1]),
      ],
      [ 
        dot_product(row0, _rhs[2]),
        dot_product(row1, _rhs[2]),
        dot_product(row2, _rhs[2]),
        dot_product(row3, _rhs[2]),
      ],
      [
        dot_product(row0, _rhs[3]),
        dot_product(row1, _rhs[3]),
        dot_product(row2, _rhs[3]),
        dot_product(row3, _rhs[3]),
      ],
    ]
  }

//Internal functions which makes less sense
  fn extract_row(a : Mat4, index : usize) -> [f32;4]
  {
    [ 
      a[0][index],
      a[1][index],
      a[2][index],
      a[3][index]
    ]
  }
  
  fn dot_product(a : [f32;4], b : [f32;4]) -> f32
  {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
  }
