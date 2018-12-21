#[macro_use]
extern crate glium;
extern crate notify;

mod renderer;
use glium::index::PrimitiveType;
use glium::{glutin, Surface};
use std::env;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, SystemTime};

fn watch(path: std::path::PathBuf, sender: std::sync::mpsc::Sender<notify::DebouncedEvent>) {
  let (tx, rx) = channel();

  let mut shader_watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_millis(50)).unwrap();
  loop {
    shader_watcher
      .watch(path.clone(), RecursiveMode::Recursive)
      .unwrap();
    let event = rx.recv().unwrap();
    sender.send(event).unwrap();
  }
}

fn main() {
  let mut events_loop = glutin::EventsLoop::new();
  let window = glutin::WindowBuilder::new()
    .with_title("mikpe_demo")
    .with_dimensions(glutin::dpi::LogicalSize::new(1024.0, 768.0));
  let context = glutin::ContextBuilder::new().with_vsync(true);
  let display = glium::Display::new(window, context, &events_loop).unwrap();

  let mut running = true;
  let (tx, rx) = channel();
  let mut current_path = env::current_exe().unwrap();

  current_path.pop();
  let path = current_path.join("../../shaders");
  let mut shader_list: Vec<renderer::shading::Shader> = Vec::new();
  if path.is_dir() {
    for entry in path.read_dir().expect("Read path failed!") {
      if let Ok(entry) = entry {
        let some_shader = renderer::shading::Shader::read(entry.path().clone());
        shader_list.push(some_shader);
      }
    }
  }

  let child = thread::spawn(move || {
    watch(path, tx.clone());
  });

  //TODO: Make shaders hot-reloadable!
  //TODO: make a pass out of this:
  let vertex_buffer = {
    #[derive(Copy, Clone)]
    struct Vertex {
      position: [f32; 2],
    }
    implement_vertex!(Vertex, position);
    glium::VertexBuffer::new(
      &display,
      &[
        Vertex {
          position: [0.0, 0.0],
        },
        Vertex {
          position: [0.0, 0.0],
        },
        Vertex {
          position: [0.0, 0.0],
        },
      ],
    ).unwrap()
  };

  let index_buffer =
    glium::IndexBuffer::new(&display, PrimitiveType::TriangleStrip, &[0 as u16, 1, 2]).unwrap();

  let mut vert_shader = String::new();
  let mut frag_shader = String::new();
  for shader in shader_list {
    match shader.shadertype {
      renderer::shading::ShaderType::Fragment => frag_shader = shader.code.clone(),
      renderer::shading::ShaderType::Vertex => vert_shader = shader.code.clone(),
    }
  }
  let mut program = glium::Program::from_source(&display, &vert_shader, &frag_shader, None).unwrap();

  

  while running {
    
    let begin = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().subsec_millis() as f32;
    let mut target = display.draw();
    target
      .draw(
        &vertex_buffer,
        &index_buffer,
        &program,
        &uniform!{
          time: begin,
        },
        &Default::default(),
      ).unwrap();
    target.finish().unwrap();

    match rx.try_recv() {
      Ok(event) => match event {
        notify::DebouncedEvent::NoticeWrite(path) => println!("NoticeWrite to path: {:?}", path),
        notify::DebouncedEvent::Write(path) => {
          println!("Wrote to path: {:?}", path);
          let some_shader = renderer::shading::Shader::read(path);
          match some_shader.shadertype {
            renderer::shading::ShaderType::Fragment => frag_shader = some_shader.code.clone(),
            renderer::shading::ShaderType::Vertex => vert_shader = some_shader.code.clone(),
          }
          program = glium::Program::from_source(&display, &vert_shader, &frag_shader, None).unwrap();
        },
        notify::DebouncedEvent::Remove(path) => println!("Removed path: {:?}", path),
        notify::DebouncedEvent::Create(path) => println!("Created to path: {:?}", path),
        _ => (),
      },
      _ => (),
    }

    events_loop.poll_events(|event| match event {
      glutin::Event::WindowEvent { event, .. } => match event {
        glutin::WindowEvent::CloseRequested => running = false,
        glutin::WindowEvent::Resized(logical_size) => {
          println!(
            "Got some resize event w{}:h{}",
            logical_size.width, logical_size.height
          );
        }
        _ => (),
      },
      _ => (),
    })
  }
  child.join().unwrap();
}
