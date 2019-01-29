mod math;
mod particles;
mod renderer;


#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as backend;
#[cfg(feature = "gl")]
use gfx_backend_gl as backend;


// There are a lot of imports - best to just accept it.
use gfx_hal::{
    command::{ClearColor, ClearValue},
    format::{Aspects, ChannelType, AsFormat, Rgba8Srgb as ColorFormat, Swizzle},
    image::{Access, Layout, SubresourceRange, ViewKind},
    pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
        SubpassDesc, SubpassRef,
    },
    pool::CommandPoolCreateFlags,
    pso::{
        BlendState, ColorBlendDesc, ColorMask, EntryPoint, GraphicsPipelineDesc, GraphicsShaderSet,
        PipelineStage, Rasterizer, Rect, Viewport,
    },
    queue::Submission,
    Backbuffer, Device, FrameSync, Graphics, Instance, Primitive, Surface, PhysicalDevice, SwapImageIndex,
    Swapchain, SwapchainConfig,
};
use winit::{Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder, WindowEvent};


// use glium::index::PrimitiveType;
// use glium::{glutin, implement_vertex, uniform, Surface};
use std::env;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, SystemTime};

fn watch(path: std::path::PathBuf, sender: std::sync::mpsc::Sender<notify::DebouncedEvent>) {
    let (tx, rx) = channel();

    let mut shader_watcher: RecommendedWatcher =
        Watcher::new(tx, Duration::from_millis(50)).unwrap();
    loop {
        shader_watcher
            .watch(path.clone(), RecursiveMode::Recursive)
            .unwrap();
        let event = rx.recv().unwrap();
        sender.send(event).unwrap();
    }
}

fn runShaderCode(path : &std::path::PathBuf) -> Vec<renderer::Shader>
{
    let mut shader_list : Vec<renderer::Shader> = Vec::new();
    if path.is_dir() {
        for entry in path.read_dir().expect("Read path failed!") {
            if let Ok(entry) = entry {
                let some_shader = renderer::Shader::read(entry.path().clone());
                shader_list.push(some_shader);
            }
        }
    }
    shader_list
}

fn main() {
    //TODO MATH STUFF:
    let mut my_emitter = particles::Emitter::new(20);
    // dbg!(&my_emitter);
    let some_vec = math::Vec3::new(0.0, 1.0, 2.0);
    let some_other_vec = math::Vec3::new(0.0, 1.0, 2.0);
    let some_third_vec = some_vec.add(&some_other_vec);
    let some_vec4 = math::Vec4::from_xyz(0.0, 1.0, 5.0);
    let some_mat4 = math::Mat4::from_translation([10.0, 5.0, 5.0]);
    let some_other_mat4 = math::Mat4::from_translation([10.0, 0.0, 5.0]);
    let third_mat4 = some_mat4.mul(&some_other_mat4);
    let multiplied_vec = math::mat4_mul_vec3(&third_mat4, &some_third_vec);
    let multiplied_vec4 = math::mat4_mul_vec4(&third_mat4, &some_vec4);
    //TODO MATH STUFF
    

    let mut running = true;
    let (tx, rx) = channel();

    let mut current_path = env::current_exe().unwrap();
    current_path.pop();
    let path = current_path.join("../../assets/shaders");
    let mut shader_list = runShaderCode(&path);
    let mut vert_shader = String::new();
    let mut frag_shader = String::new();
    for shader in shader_list {
        match shader.shadertype {
            renderer::ShaderType::Fragment => frag_shader = shader.code.clone(),
            renderer::ShaderType::Vertex => vert_shader = shader.code.clone(),
        }
    }

    let child = thread::spawn(move || {
        watch(path, tx.clone());
    });

    // Create a window with winit.
    let mut events_loop = EventsLoop::new();

    let wb = WindowBuilder::new();

    #[cfg(not(feature = "gl"))]
    let (_window, _instance, mut adapters, mut surface) = {
        let window = wb.with_title("Particle Test (Vulkan)").with_dimensions((640, 480).into()).build(&events_loop).unwrap();
        let instance = backend::Instance::create("Particle instance", 1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();
        (window, instance, adapters, surface)
    };

    #[cfg(feature = "gl")]
    let (mut adapters, mut surface) = {
        let window = {
            let builder =
                backend::config_context(backend::glutin::ContextBuilder::new(), ColorFormat::SELF, None)
                    .with_vsync(true);
            backend::glutin::GlWindow::new(wb.with_title("Particle Test (OpenGL)").with_dimensions((640, 480).into()), builder, &events_loop).unwrap()
        };

        let surface = backend::Surface::from_window(window);
        let adapters = surface.enumerate_adapters();
        (adapters, surface)
    };


    for adapter in &adapters {
        dbg!(&adapter.info);
    }
    
    //First gpu will probably do!
    let mut adapter = adapters.remove(0);
    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let limits = adapter.physical_device.limits();
    

    // //TODO: make a pass out of this:
    // let vertex_buffer = {
    //     #[derive(Copy, Clone)]
    //     struct Vertex {
    //         position: [f32; 2],
    //     }
    //     glium::implement_vertex!(Vertex, position);
    //     glium::VertexBuffer::new(
    //         &display,
    //         &[
    //             Vertex {
    //                 position: [0.0, 0.0],
    //             },
    //             Vertex {
    //                 position: [0.0, 0.0],
    //             },
    //             Vertex {
    //                 position: [0.0, 0.0],
    //             },
    //         ],
    //     )
    //     .unwrap()
    // };

    // let index_buffer =
    //     glium::IndexBuffer::new(&display, PrimitiveType::TriangleStrip, &[0 as u16, 1, 2]).unwrap();


    // let mut program =
    //     glium::Program::from_source(&display, &vert_shader, &frag_shader, None).unwrap();
    // //TODO: structure code in a more sensical manner
    // //TODO: Create better renderer-functions to wrap glutin
    // //TODO: Create some material-structuringish
    // let mut delta_time;
    // let mut last_frame_time = SystemTime::now();
    // while running {
    //     //SETUP FRAME:
    //     let this_frame_time = SystemTime::now();
    //     let sec_part = this_frame_time.duration_since(last_frame_time).unwrap().as_secs() as f32;
    //     let microsec_part = this_frame_time.duration_since(last_frame_time).unwrap().subsec_micros() as f32;
    //     delta_time =  sec_part * 1000.0 + microsec_part / 1000.0;
    //     last_frame_time = this_frame_time;

    //     //UPDATE FRAME:
    //     my_emitter.tick(delta_time);

    //     //DRAW FRAME:
    //     let mut target = display.draw();
    //     target
    //         .draw(
    //             &vertex_buffer,
    //             &index_buffer,
    //             &program,
    //             &glium::uniform! {
    //               time: delta_time,
    //             },
    //             &Default::default(),
    //         )
    //         .unwrap();
    //     target.finish().unwrap();

    //     match rx.try_recv() {
    //         Ok(event) => match event {
    //             notify::DebouncedEvent::Write(path) => {
    //                 let some_shader = renderer::Shader::read(path);
    //                 match some_shader.shadertype {
    //                     renderer::ShaderType::Fragment => frag_shader = some_shader.code.clone(),
    //                     renderer::ShaderType::Vertex => vert_shader = some_shader.code.clone(),
    //                 }
    //                 program =
    //                     glium::Program::from_source(&display, &vert_shader, &frag_shader, None)
    //                         .unwrap();
    //             }
    //             notify::DebouncedEvent::Remove(path) => println!("Removed path: {:?}", path),
    //             notify::DebouncedEvent::Create(path) => println!("Created to path: {:?}", path),
    //             _ => (),
    //         },
    //         _ => (),
    //     }

    //     events_loop.poll_events(|event| match event {
    //         glutin::Event::WindowEvent { event, .. } => match event {
    //             glutin::WindowEvent::Touch(touch) => (),
    //             glutin::WindowEvent::CloseRequested => running = false,
    //             glutin::WindowEvent::Resized(logical_size) => {
    //                 dbg!(logical_size);
    //             }
    //             _ => (),
    //         },
    //         _ => (),
    //     })
    // }
    child.join().unwrap();
}
