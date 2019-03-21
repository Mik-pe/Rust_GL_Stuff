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
    adapter::PhysicalDevice,
    buffer,
    command::{ClearColor, ClearValue},
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{SubresourceRange, ViewKind},
    memory,
    pass::Subpass,
    pool::CommandPoolCreateFlags,
    pso::{
        AttributeDesc, BlendState, ColorBlendDesc, ColorMask, Element, EntryPoint,
        GraphicsPipelineDesc, GraphicsShaderSet, PipelineStage, Rasterizer, Rect, Viewport,
    },
    queue::Submission,
    window::Extent2D,
    Backbuffer, Backend, Device, FrameSync, Primitive, Surface, Swapchain, SwapchainConfig,
};
use glsl_to_spirv::ShaderType;
use winit::{Event, KeyboardInput, VirtualKeyCode, WindowEvent};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::cell::RefCell;
use std::env;
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

#[allow(dead_code)]
struct PipelineState<B: Backend> {
    pipeline: Option<B::GraphicsPipeline>,
    pipeline_layout: Option<B::PipelineLayout>,
    device: std::rc::Rc<RefCell<renderer::DeviceState<B>>>,
}

//TODO: move this func?
unsafe fn reset_pipeline<B: Backend>(
    vert_spirv: Vec<u8>,
    frag_spirv: Vec<u8>,
    pipeline_layout: &B::PipelineLayout,
    device: &B::Device,
    render_pass: &B::RenderPass,
) -> B::GraphicsPipeline {
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
    //TODO: Fix pipeline vertex attribs

    pipeline_desc
        .vertex_buffers
        .push(gfx_hal::pso::VertexBufferDesc {
            binding: 0,
            stride: std::mem::size_of::<math::Vec3>() as u32,
            rate: 0,
        });
    pipeline_desc.attributes.push(AttributeDesc {
        location: 0,
        binding: 0,
        element: Element {
            format: gfx_hal::format::Format::Rgb32Float,
            offset: 0,
        },
    });
    device
        .create_graphics_pipeline(&pipeline_desc, None)
        .expect("Could not create graphics pipeline.")
}

const WINDOW_DIMENSIONS: Extent2D = Extent2D {
    width: 640,
    height: 480,
};

fn main() {
    //TODO MATH STUFF:
    let mut my_emitter = particles::Emitter::new(20);
    let some_vec = math::Vec3::new(0.0, 1.0, 2.0);
    let some_other_vec = math::Vec3::new(0.0, 1.0, 2.0);
    let some_third_vec = some_vec.add(&some_other_vec);
    let some_vec4 = math::Vec4::from_xyz(0.0, 1.0, 5.0);
    let some_mat4 = math::Mat4::from_translation([10.0, 5.0, 5.0]);
    let some_other_mat4 = math::Mat4::from_translation([10.0, 0.0, 5.0]);
    let third_mat4 = some_mat4.mul(&some_other_mat4);
    let _multiplied_vec = math::mat4_mul_vec3(&third_mat4, &some_third_vec);
    let _multiplied_vec4 = math::mat4_mul_vec4(&third_mat4, &some_vec4);
    //TODO MATH STUFF

    let (tx, rx) = channel();

    let mut current_path = env::current_exe().unwrap();
    current_path.pop();
    let path = current_path.join("../../assets/shaders");
    let (vert_spirv, frag_spirv) = run_shader_code(&path);

    let _child = thread::spawn(move || {
        watch(path, tx.clone());
    });

    let mut window_state = renderer::WindowState::new(
        (
            WINDOW_DIMENSIONS.width as i32,
            WINDOW_DIMENSIONS.height as i32,
        ),
        "Playground_Window".to_owned(),
    );
    let (mut backend_state, _instance, mut adapter_state) =
        renderer::states::create_backend(&mut window_state);

    let mut device_state = renderer::DeviceState::new(
        adapter_state.adapter.take().unwrap(),
        &backend_state.surface,
    );

    let device = &device_state.device;
    let physical_device = &device_state.physical_device;
    let memory_types = physical_device.memory_properties().memory_types;

    let mut command_pool = unsafe {
        device.create_command_pool_typed(&device_state.queues, CommandPoolCreateFlags::empty())
    }
    .expect("Can't create command pool");

    let (caps, formats, _, _) = backend_state.surface.compatibility(physical_device);

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
    let fullscreen_pass =
        renderer::create_fullscreen_pass::<backend::Backend>(surface_color_format, &device);

    //uniforms and push constants go here:
    let pipeline_layout = unsafe { device.create_pipeline_layout(&[], &[]) }
        .expect("Coult not create pipeline layout");
    // A pipeline object encodes almost all the state you need in order to draw
    // geometry on screen. For now that's really only which shaders to use, what
    // kind of blending to do, and what kind of primitives to draw.
    let mut pipeline = unsafe {
        reset_pipeline::<backend::Backend>(
            vert_spirv,
            frag_spirv,
            &pipeline_layout,
            &device,
            &fullscreen_pass,
        )
    };

    let swap_config =
        SwapchainConfig::from_caps(&caps, surface_color_format, caps.current_extent.unwrap());

    let extent = swap_config.extent.to_extent();

    let (mut swapchain, backbuffer) =
        unsafe { device.create_swapchain(&mut backend_state.surface, swap_config, None) }
            .expect("Could not create swapchain");

    let (_frame_views, framebuffers) = match backbuffer {
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
                    unsafe { device.create_framebuffer(&fullscreen_pass, vec![image_view], extent) }
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

    let triangle: [math::Vec3; 3] = [
        math::Vec3::new(0.0, 0.1, 0.0),
        math::Vec3::new(0.1, 0.0, 0.0),
        math::Vec3::new(-0.1, 0.0, 0.0),
    ];
    let mut positions = vec![];
    let mut some_triangles = vec![];
    for particle in &my_emitter.particle_list {
        positions.push(particle.position.clone());
    }
    for pos in &positions {
        some_triangles.push(pos.add(&triangle[0]));
        some_triangles.push(pos.add(&triangle[1]));
        some_triangles.push(pos.add(&triangle[2]));
    }

    let buffer_stride = std::mem::size_of::<math::Vec3>() as u64;
    let buffer_len = some_triangles.len() as u64 * buffer_stride;
    let mut vertex_buffer =
        unsafe { device.create_buffer(buffer_len, buffer::Usage::VERTEX) }.unwrap();

    let buffer_req = unsafe { device.get_buffer_requirements(&vertex_buffer) };

    let upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            // type_mask is a bit field where each bit represents a memory type. If the bit is set
            // to 1 it means we can use that type for our buffer. So this code finds the first
            // memory type that has a `1` (or, is allowed), and is visible to the CPU.
            buffer_req.type_mask & (1 << id) != 0
                && mem_type
                    .properties
                    .contains(gfx_hal::memory::Properties::CPU_VISIBLE)
        })
        .unwrap()
        .into();

    let buffer_memory = unsafe { device.allocate_memory(upload_type, buffer_req.size) }.unwrap();

    unsafe { device.bind_buffer_memory(&buffer_memory, 0, &mut vertex_buffer) }.unwrap();

    // The frame semaphore is used to allow us to wait for an image to be ready
    // before attempting to draw on it,
    //
    // The frame fence is used to to allow us to wait until our draw commands have
    // finished before attempting to display the image.
    let mut frame_semaphore = device.create_semaphore().unwrap();
    let mut recreate_swapchain = false;
    let mut frame_fence = device.create_fence(false).expect("Can't create fence");
    let mut quitting = false;

    let mut last_frame_time = SystemTime::now();

    while quitting == false {
        let this_frame_time = SystemTime::now();
        let delta_time = last_frame_time.elapsed().unwrap() - this_frame_time.elapsed().unwrap();
        last_frame_time = this_frame_time;
        my_emitter.tick(((delta_time.as_micros() as f64) / 1_000_000.0) as f32);
        positions.clear();
        some_triangles.clear();
        for particle in &my_emitter.particle_list {
            positions.push(particle.position.clone());
        }
        for pos in &positions {
            some_triangles.push(pos.add(&triangle[0]));
            some_triangles.push(pos.add(&triangle[1]));
            some_triangles.push(pos.add(&triangle[2]));
        }
        dbg!(&positions[0]);
        // TODO: check transitions: read/write mapping and vertex buffer read
        unsafe {
            let mut vertices = device
                .acquire_mapping_writer::<math::Vec3>(&buffer_memory, 0..buffer_req.size)
                .unwrap();
            vertices[0..some_triangles.len()].copy_from_slice(&some_triangles);
            device.release_mapping_writer(vertices).unwrap();
        }

        match rx.try_recv() {
            Ok(event) => match event {
                notify::DebouncedEvent::Write(_watched_path) => {
                    let mut current_path = env::current_exe().unwrap();
                    current_path.pop();
                    let path = current_path.join("../../assets/shaders");
                    let (vert_spirv, frag_spirv) = run_shader_code(&path);
                    pipeline = unsafe {
                        reset_pipeline::<backend::Backend>(
                            vert_spirv,
                            frag_spirv,
                            &pipeline_layout,
                            &device,
                            &fullscreen_pass,
                        )
                    };
                }
                notify::DebouncedEvent::Remove(path) => println!("Removed path: {:?}", path),
                notify::DebouncedEvent::Create(path) => println!("Created to path: {:?}", path),
                _ => (),
            },
            _ => (),
        }

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
                    WindowEvent::Resized(logical_size) => {
                        dbg!(logical_size);
                    }
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
            cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));
            {
                // Clear the screen and begin the render pass.
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &fullscreen_pass,
                    &framebuffers[frame as usize],
                    viewport.rect,
                    &[ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))],
                );

                //Shader has 3 vertices, indexlist is 0..1
                encoder.draw(0..some_triangles.len() as u32, 0..1);
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
        device.destroy_render_pass(fullscreen_pass);

        device.destroy_graphics_pipeline(pipeline);
        device.destroy_pipeline_layout(pipeline_layout);
        for framebuffer in framebuffers {
            device.destroy_framebuffer(framebuffer);
        }

        device.destroy_swapchain(swapchain);
    }
    //How do we join our threads?
    //Just don't mind as the OS handles this! ...for now and forever, this is a hobby project, what do you expect?
    // child.join().unwrap();
}
