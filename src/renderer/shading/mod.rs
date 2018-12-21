use std::fs::File;
use std::io::prelude::*;

pub enum ShaderType
{
  Vertex,
  Fragment,
}

pub struct Shader
{
  pub code: String,
  pub shadertype: ShaderType,
}

impl Shader
{
  pub fn new(code: String, shadertype: ShaderType) -> Shader
  {
    Shader{
      code: code,
      shadertype: shadertype,
    }
  }

  pub fn read(path : std::path::PathBuf) -> Shader
  {
    let mut f = File::open(&path).expect("File not found!");
    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .expect("something went wrong reading the file");
    match path.extension() {
      Some(ext) => {
        match ext.to_str().unwrap() {
          "frag" => {
            Shader::new(
              contents, 
              ShaderType::Fragment)
          },
          "vert" => {
            Shader::new(
              contents, 
              ShaderType::Vertex)
          },
          _ => {
            Shader::new(
              "not_a_shader".to_owned(), 
              ShaderType::Vertex)
          }
        }
      },
      None => {
        Shader::new(
          "not_a_shader".to_owned(), 
          ShaderType::Fragment)
      }
    }    
  }
}