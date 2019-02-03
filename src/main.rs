mod math;
mod particles;
mod renderer;

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as backend;
#[cfg(feature = "gl")]
use gfx_backend_gl as backend;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as backend;

use gfx_hal::{
    command::{ClearColor, ClearValue, Primary},
    format::{AsFormat, Aspects, ChannelType, Format, Rgba8Srgb as ColorFormat, Swizzle},
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
    Backbuffer, Backend, CommandQueue, Device, FrameSync, Graphics, Instance, PhysicalDevice,
    Primitive, Surface, SwapImageIndex, Swapchain, SwapchainConfig,
};
use glsl_to_spirv::ShaderType;
use winit::{Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder, WindowEvent};

// use glium::index::PrimitiveType;
// use glium::{glutin, implement_vertex, uniform, Surface};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::cell::RefCell;
use std::env;
use std::rc::Rc;
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

fn run_shader_code(path: &std::path::PathBuf) -> (Vec<u8>, Vec<u8>) {
    let mut shader_list: Vec<renderer::Shader> = Vec::new();
    if path.is_dir() {
        for entry in path.read_dir().expect("Read path failed!") {
            if let Ok(entry) = entry {
                let some_shader = renderer::Shader::read(entry.path().clone());
                shader_list.push(some_shader);
            }
        }
    }
    let mut vert_shader = renderer::Shader::new("".to_owned(), ShaderType::Vertex);
    let mut frag_shader = renderer::Shader::new("".to_owned(), ShaderType::Fragment);

    for shader in shader_list {
        match shader.shader_type {
            ShaderType::Vertex => vert_shader = shader,
            ShaderType::Fragment => frag_shader = shader,
            _ => {}
        }
    }
    let vert_spirv = vert_shader.compile();
    let frag_spirv = frag_shader.compile();

    (vert_spirv, frag_spirv)
}

struct PipelineState<B: Backend> {
    pipeline: Option<B::GraphicsPipeline>,
    pipeline_layout: Option<B::PipelineLayout>,
    device: std::rc::Rc<RefCell<renderer::DeviceState<B>>>,
}

//TODO: call this function using ptr, shaders etc...
unsafe fn reset_pipeline<B: Backend>(
    vert_spirv: Vec<u8>,
    frag_spirv: Vec<u8>,
    device_ptr: std::rc::Rc<RefCell<renderer::DeviceState<B>>>,
    render_pass: &B::RenderPass,
) -> PipelineState<B> {
    //take a ptr!
    let device = &device_ptr.borrow().device;

    let pipeline_layout = device
        .create_pipeline_layout(&[], &[])
        .expect("Coult not create pipeline layout");

    let vertex_shader_module = device
        .create_shader_module(&vert_spirv)
        .expect("Could not create vertex shader module");

    let fragment_shader_module = device
        .create_shader_module(&frag_spirv)
        .expect("Could not create fragment shader module");

    // A pipeline object encodes almost all the state you need in order to draw
    // geometry on screen. For now that's really only which shaders to use, what
    // kind of blending to do, and what kind of primitives to draw.
    let vs_entry = EntryPoint::<B> {
        entry: "main",
        module: &vertex_shader_module,
        specialization: Default::default(),
    };

    let fs_entry = EntryPoint::<B> {
        entry: "main",
        module: &fragment_shader_module,
        specialization: Default::default(),
    };

    let shader_entries = GraphicsShaderSet {
        vertex: vs_entry,
        hull: None,
        domain: None,
        geometry: None,
        fragment: Some(fs_entry),
    };

    let subpass = Subpass {
        index: 0,
        main_pass: render_pass,
    };

    let mut pipeline_desc = GraphicsPipelineDesc::new(
        shader_entries,
        Primitive::TriangleList,
        Rasterizer::FILL,
        &pipeline_layout,
        subpass,
    );

    pipeline_desc
        .blender
        .targets
        .push(ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA));

    let pipeline = device
        .create_graphics_pipeline(&pipeline_desc, None)
        .expect("Could not create graphics pipeline.");

    PipelineState {
        pipeline: Some(pipeline),
        pipeline_layout: Some(pipeline_layout),
        device: Rc::clone(&device_ptr),
    }
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
    let (vert_spirv, frag_spirv) = run_shader_code(&path);

    let child = thread::spawn(move || {
        watch(path, tx.clone());
    });

    let mut window_state = renderer::WindowState::new((640, 480), "TestWindow".to_owned());

    #[cfg(not(feature = "gl"))]
    let (_window, _instance, mut adapters, mut surface) = {
        let window = window_state
            .wb
            .take()
            .unwrap()
            .build(&window_state.events_loop)
            .unwrap();
        let instance = backend::Instance::create("Particle instance", 1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();
        (window, instance, adapters, surface)
    };

    #[cfg(feature = "gl")]
    let (mut adapters, mut surface) = {
        let window = {
            let builder = backend::config_context(
                backend::glutin::ContextBuilder::new(),
                ColorFormat::SELF,
                None,
            )
            .with_vsync(true);
            backend::glutin::GlWindow::new(
                window_state.wb.take().unwrap(),
                builder,
                &window_state.events_loop,
            )
            .unwrap()
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
    let device_ptr = Rc::new(RefCell::new(renderer::DeviceState::new(adapter, &surface)));

    //TODO: Move all of these:
    //Currently this is non-working as it pushes us to
    //borrow mutable references in many places.
    //Consider moving all this code to
    //a separate state-implementation
    let device = &device_state.device;
    let physical_device = &device_state.physical_device;

    let mut command_pool = unsafe {
        device.create_command_pool_typed(&device_state.queues, CommandPoolCreateFlags::empty())
    }
    .expect("Can't create command pool");

    let (caps, formats, _, _) = surface.compatibility(physical_device);

    let surface_color_format = {
        // We must pick a color format from the list of supported formats. If there
        // is no list, we default to Rgba8Srgb.
        match formats {
            Some(choices) => choices
                .into_iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .unwrap(),
            None => Format::Rgba8Srgb,
        }
    };

    // A render pass defines which attachments (images) are to be used for what
    // purposes. Right now, we only have a color attachment for the final output,
    // but eventually we might have depth/stencil attachments, or even other color
    // attachments for other purposes.
    let render_pass = {
        let color_attachment = Attachment {
            format: Some(surface_color_format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined..Layout::Present,
        };

        // A render pass could have multiple subpasses - but we're using one for now.
        let subpass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        // This expresses the dependencies between subpasses. Again, we only have
        // one subpass for now. Future tutorials may go into more detail.
        let dependency = SubpassDependency {
            passes: SubpassRef::External..SubpassRef::Pass(0),
            stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: Access::empty()
                ..(Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE),
        };

        unsafe { device.create_render_pass(&[color_attachment], &[subpass], &[dependency]) }
            .expect("Could not create render pass")
    };

    //uniforms and push constants go here:
    let pipeline_layout = unsafe { device.create_pipeline_layout(&[], &[]) }
        .expect("Coult not create pipeline layout");
    let vertex_shader_module = unsafe { device.create_shader_module(&vert_spirv) }
        .expect("Could not create vertex shader module");

    let fragment_shader_module = unsafe { device.create_shader_module(&frag_spirv) }
        .expect("Could not create fragment shader module");

    // A pipeline object encodes almost all the state you need in order to draw
    // geometry on screen. For now that's really only which shaders to use, what
    // kind of blending to do, and what kind of primitives to draw.
    let pipeline = {
        let vs_entry = EntryPoint::<backend::Backend> {
            entry: "main",
            module: &vertex_shader_module,
            specialization: Default::default(),
        };

        let fs_entry = EntryPoint::<backend::Backend> {
            entry: "main",
            module: &fragment_shader_module,
            specialization: Default::default(),
        };

        let shader_entries = GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };

        let subpass = Subpass {
            index: 0,
            main_pass: &render_pass,
        };

        let mut pipeline_desc = GraphicsPipelineDesc::new(
            shader_entries,
            Primitive::TriangleList,
            Rasterizer::FILL,
            &pipeline_layout,
            subpass,
        );

        pipeline_desc
            .blender
            .targets
            .push(ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA));

        unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }
            .expect("Could not create graphics pipeline")
    };

    let swap_config =
        SwapchainConfig::from_caps(&caps, surface_color_format, caps.current_extent.unwrap());

    let extent = swap_config.extent.to_extent();

    let (mut swapchain, backbuffer) =
        unsafe { device.create_swapchain(&mut surface, swap_config, None) }
            .expect("Could not create swapchain");

    let (frame_views, framebuffers) = match backbuffer {
        Backbuffer::Images(images) => {
            let color_range = SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            };

            let image_views = images
                .iter()
                .map(|image| {
                    unsafe {
                        device.create_image_view(
                            image,
                            ViewKind::D2,
                            surface_color_format,
                            Swizzle::NO,
                            color_range.clone(),
                        )
                    }
                    .expect("Could not create image view")
                })
                .collect::<Vec<_>>();

            let fbos = image_views
                .iter()
                .map(|image_view| {
                    unsafe { device.create_framebuffer(&render_pass, vec![image_view], extent) }
                        .expect("Could not create framebuffer")
                })
                .collect();

            (image_views, fbos)
        }

        // This arm of the branch is currently only used by the OpenGL backend,
        // which supplies an opaque framebuffer for you instead of giving you control
        // over individual images.
        Backbuffer::Framebuffer(fbo) => (vec![], vec![fbo]),
    };

    // The frame semaphore is used to allow us to wait for an image to be ready
    // before attempting to draw on it,
    //
    // The frame fence is used to to allow us to wait until our draw commands have
    // finished before attempting to display the image.
    let mut frame_semaphore = device.create_semaphore().unwrap();
    let mut frame_fence = device.create_fence(false).expect("Can't create fence"); // TODO: remove
    let present_semaphore = device.create_semaphore().unwrap();
    let mut recreate_swapchain = false;
    let mut quitting = false;
    while quitting == false {
        // match rx.try_recv() {
        //     Ok(event) => match event {
        //         notify::DebouncedEvent::Write(path) => {
        //             let some_shader = renderer::Shader::read(path);
        //             match some_shader.shadertype {
        //                 renderer::ShaderType::Fragment => frag_shader = some_shader.code.clone(),
        //                 renderer::ShaderType::Vertex => vert_shader = some_shader.code.clone(),
        //             }
        //             program =
        //                 glium::Program::from_source(&display, &vert_shader, &frag_shader, None)
        //                     .unwrap();
        //         }
        //         notify::DebouncedEvent::Remove(path) => println!("Removed path: {:?}", path),
        //         notify::DebouncedEvent::Create(path) => println!("Created to path: {:?}", path),
        //         _ => (),
        //     },
        //     _ => (),
        // }

        // If the window is closed, or Escape is pressed, quit
        window_state.poll_events(|event| {
            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => quitting = true,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => quitting = true,
                    _ => {}
                }
            }
        });
        // Start rendering
        if quitting == true {
            println!("Should quit!");
        }

        let frame: gfx_hal::SwapImageIndex = unsafe {
            device.reset_fence(&frame_fence).unwrap();
            command_pool.reset();
            match swapchain.acquire_image(!0, FrameSync::Semaphore(&mut frame_semaphore)) {
                Ok(i) => i,
                Err(_) => {
                    recreate_swapchain = true;
                    continue;
                }
            }
        };

        let mut cmd_buffer = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
        unsafe {
            cmd_buffer.begin();

            // Define a rectangle on screen to draw into.
            // In this case, the whole screen.
            let viewport = Viewport {
                rect: Rect {
                    x: 0,
                    y: 0,
                    w: extent.width as i16,
                    h: extent.height as i16,
                },
                depth: 0.0..1.0,
            };

            cmd_buffer.set_viewports(0, &[viewport.clone()]);
            cmd_buffer.set_scissors(0, &[viewport.rect]);

            // Choose a pipeline to use.
            cmd_buffer.bind_graphics_pipeline(&pipeline);

            {
                // Clear the screen and begin the render pass.
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &render_pass,
                    &framebuffers[frame as usize],
                    viewport.rect,
                    &[ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))],
                );

                //Shader has 3 vertices, indexlist is 0..1
                encoder.draw(0..3, 0..1);
            }

            // Finish building the command buffer - it's now ready to send to the GPU.
            cmd_buffer.finish();
            let submission = Submission {
                command_buffers: Some(&cmd_buffer),
                wait_semaphores: Some((&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: &[],
            };

            device_state.queues.queues[0].submit(submission, Some(&mut frame_fence));

            // TODO: replace with semaphore
            device.wait_for_fence(&frame_fence, !0).unwrap();
            command_pool.free(Some(cmd_buffer));

            // present frame
            if let Err(_) =
                swapchain.present_nosemaphores(&mut device_state.queues.queues[0], frame)
            {
                recreate_swapchain = true;
            }
        }
    }

    device.wait_idle().unwrap();
    unsafe {
        device.destroy_command_pool(command_pool.into_raw());

        device.destroy_fence(frame_fence);
        device.destroy_semaphore(frame_semaphore);
        device.destroy_render_pass(render_pass);

        device.destroy_graphics_pipeline(pipeline);
        device.destroy_pipeline_layout(pipeline_layout);
        for framebuffer in framebuffers {
            device.destroy_framebuffer(framebuffer);
        }

        device.destroy_swapchain(swapchain);
    }
    println!("Down here!");

    //How do we join our threads?
    //Just don't mind?
    // child.join().unwrap();
}
