use glsl_to_spirv::ShaderType;
use std::fs::File;
use std::io::prelude::*;

pub struct Shader {
    pub code: String,
    pub shader_type: ShaderType,
}

impl Shader {
    pub fn new(code: String, shader_type: ShaderType) -> Shader {
        Shader {
            code: code,
            shader_type: shader_type,
        }
    }
    pub fn compile(self) -> Vec<u8> {
        glsl_to_spirv::compile(&self.code, self.shader_type)
            .unwrap()
            .bytes()
            .map(|b| b.unwrap())
            .collect()
    }

    pub fn read(path: std::path::PathBuf) -> Shader {
        let mut f = File::open(&path).expect("File not found!");
        let mut contents = String::new();
        f.read_to_string(&mut contents)
            .expect("something went wrong reading the file");
        match path.extension() {
            Some(ext) => match ext.to_str().unwrap() {
                "frag" => Shader::new(contents, ShaderType::Fragment),
                "vert" => Shader::new(contents, ShaderType::Vertex),
                _ => Shader::new("not_a_shader".to_owned(), ShaderType::Vertex),
            },
            None => Shader::new("not_a_shader".to_owned(), ShaderType::Fragment),
        }
    }
}
