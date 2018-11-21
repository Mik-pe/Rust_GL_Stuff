#[macro_use]
extern crate glium;
extern crate notify;

mod shading;
use glium::{glutin, Surface};
use glium::index::PrimitiveType;
use std::env;

// use Shader;

enum PathEvent
{
  Write(std::path::PathBuf),
  SomeRandomEvent(notify::DebouncedEvent),
}

use notify::{RecommendedWatcher, Watcher, RecursiveMode};
use std::sync::mpsc::channel;
use std::time::{Duration, SystemTime};
use std::thread;

fn watch(path : std::path::PathBuf, sender : std::sync::mpsc::Sender<notify::DebouncedEvent>)
{
  let (tx, rx) = channel();

  let mut shader_watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_secs(2)).unwrap();
  loop
  {
    shader_watcher.watch(path.clone(), RecursiveMode::Recursive).unwrap();
    let event = rx.recv().unwrap();
    sender.send(event).unwrap();
  }
}


fn main() {

  let mut events_loop = glutin::EventsLoop::new();
  let window = glutin::WindowBuilder::new()
    .with_title("OpieOp")
    .with_dimensions(glutin::dpi::LogicalSize::new(1024.0, 768.0));
  let context = glutin::ContextBuilder::new()
    .with_vsync(true);
  let display = glium::Display::new(window, context, &events_loop).unwrap();

  let mut running = true;
  let (tx, rx) = channel();
  let mut current_path = env::current_exe()
  .unwrap();

  current_path.pop();
  let path = current_path.join("../../shaders");
  let mut currentListing : Vec<std::path::PathBuf> = Vec::new();
  let mut shaderList : Vec<shading::Shader> = Vec::new();
  if path.is_dir()
  {
    for entry in path.read_dir().expect("Read path failed!")
    {   
      if let Ok(entry) = entry {
        let some_shader = shading::Shader::read(entry.path().clone());
        println!("Got shader: {}", some_shader.code);
        shaderList.push(some_shader);
        println!("These are the files within the folder: {:?}", entry.path().file_name().unwrap());
      }   
    } 
  }

  let child = thread::spawn(move || {
    watch(path, tx.clone());
  });
  
  let vertex_buffer = {
    #[derive(Copy, Clone)]
    struct Vertex {
      position :  [f32;2],
    }
    implement_vertex!(Vertex, position);
    glium::VertexBuffer::new(&display, &[Vertex{ position: [0.0, 0.0]},
    Vertex{ position: [0.0, 0.0]},
    Vertex{ position: [0.0, 0.0]}]).unwrap()
  };

  let index_buffer = glium::IndexBuffer::new(&display, PrimitiveType::TriangleStrip,
&[0 as u16, 1, 2]).unwrap();

  let mut vert_shader = String::new();
  let mut frag_shader = String::new();
  for shader in shaderList {
    match shader.shadertype {
      shading::ShaderType::Fragment => frag_shader = shader.code.clone(),
      shading::ShaderType::Vertex => vert_shader = shader.code.clone(),
    }
  }
  let program = glium::Program::from_source(&display, &vert_shader, &frag_shader, None).unwrap();

  // let begin = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);


  while running {
    let mut target = display.draw();
    target.draw(&vertex_buffer, &index_buffer, &program, &uniform!{}, &Default::default()).unwrap();
    target.finish().unwrap();

    match rx.try_recv() 
    {
      Ok(event) => match event
      {
        notify::DebouncedEvent::NoticeWrite(path) => println!("NoticeWrite to path: {:?}", path),     notify::DebouncedEvent::Write(path) => println!("Wrote to path: {:?}", path),
        notify::DebouncedEvent::Remove(path) => println!("Removed path: {:?}", path),        notify::DebouncedEvent::Create(path) => println!("Created to path: {:?}", path),
        _ => (),
      },
      _ => (),
    }


    events_loop.poll_events(|event| {
      match event {
        glutin::Event::WindowEvent { event, .. } => match event {
            glutin::WindowEvent::CloseRequested => running = false,
            glutin::WindowEvent::Resized(logical_size) => {
              println!("Got some resize event, whatever that means..!???!");
            },
            _ => ()
        },
        _ => () 
      }
    })
  }
  child.join().unwrap();
}
