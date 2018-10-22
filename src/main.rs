extern crate glutin;

use glutin::dpi::*;
use glutin::GlContext;

fn main() {
  let mut events_loop = glutin::EventsLoop::new();
  let window = glutin::WindowBuilder::new()
    .with_title("Hello, rust")
    .with_dimensions(LogicalSize::new(1024.0, 768.0));
  let context = glutin::ContextBuilder::new()
    .with_vsync(true);
  let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

  let mut running = true;

  while running {
    events_loop.poll_events(|event| {
      match event {
        glutin::Event::WindowEvent { event, .. } => match event {
            glutin::WindowEvent::CloseRequested => running = false,
            glutin::WindowEvent::Resized(logical_size) => {
                let dpi_factor = gl_window.get_hidpi_factor();
                gl_window.resize(logical_size.to_physical(dpi_factor));
            },
            _ => ()
        },
        _ => () 
      }
    })
  }

  gl_window.swap_buffers().unwrap();
}
