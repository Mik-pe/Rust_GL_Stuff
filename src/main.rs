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
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{SubresourceRange, ViewKind},
    pass::Subpass,
    pool::CommandPoolCreateFlags,
    pso,
    pso::{
        AttributeDesc, BlendState, ColorBlendDesc, ColorMask, Element, EntryPoint,
        GraphicsPipelineDesc, GraphicsShaderSet, PipelineStage, Rasterizer, Rect, ShaderStageFlags,
        Viewport,
    },
    queue::Submission,
    window::Extent2D,
    Backend, DescriptorPool, Primitive, Surface, Swapchain, SwapchainConfig,
};
use glsl_to_spirv::ShaderType;
use winit::{Event, KeyboardInput, VirtualKeyCode, WindowEvent};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::cell::RefCell;
use std::env;
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, SystemTime};

const COLOR_RANGE: SubresourceRange = SubresourceRange {
    aspects: Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

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
            rate: gfx_hal::pso::VertexInputRate::Vertex,
        });
    pipeline_desc.attributes.push(AttributeDesc {
        location: 0,
        binding: 0,
        element: Element {
            format: gfx_hal::format::Format::Rgb32Sfloat,
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
    let mut my_emitter = particles::Emitter::new(4000);
    let some_vec = math::Vec3::new(0.0, 1.0, 2.0);
    let some_other_vec = math::Vec3::new(0.0, 1.0, 2.0);
    let some_third_vec = some_vec.add(&some_other_vec);
    let some_vec4 = math::Vec4::from_xyz(0.0, 1.0, 5.0);
    let some_mat4 = math::Mat4::from_translation([10.0, 5.0, 5.0]);
    let some_other_mat4 = math::Mat4::from_translation([10.0, 0.0, 5.0]);
    let third_mat4 = some_mat4.mul(&some_other_mat4);
    let _multiplied_vec = math::mat4_mul_vec3(&third_mat4, &some_third_vec);
    let _multiplied_vec4 = math::mat4_mul_vec4(&third_mat4, &some_vec4);
    let mut _view_matrix = math::Mat4::new();
    let top = WINDOW_DIMENSIONS.height as f32 / 2.0;
    let right = WINDOW_DIMENSIONS.width as f32 / 2.0;

    let mut proj_matrix = math::Mat4::create_ortho(-top, top, -right, right, 0.1, 1000.0);
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

    let device: &backend::Device = &device_state.device;
    let physical_device = &device_state.physical_device;
    let memory_types = physical_device.memory_properties().memory_types;

    let command_pool = unsafe {
        device.create_command_pool_typed(&device_state.queues, CommandPoolCreateFlags::empty())
    }
    .expect("Can't create command pool");

    // Setup renderpass and pipeline
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &[pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
                immutable_samplers: false,
            }],
            &[],
        )
    }
    .expect("Can't create descriptor set layout");

    // Descriptors
    let mut desc_pool = unsafe {
        device.create_descriptor_pool(
            1, // sets
            &[pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
            }],
            pso::DescriptorPoolCreateFlags::empty(),
        )
    }
    .expect("Can't create descriptor pool");

    let desc_set = unsafe { desc_pool.allocate_set(&set_layout) }.unwrap();

    let _uniform_buffer = unsafe {
        renderer::BufferState::<backend::Backend>::new(
            &device,
            &proj_matrix.0,
            gfx_hal::buffer::Usage::UNIFORM,
            &adapter_state.memory_types,
        )
    };

    let queue_group = &mut device_state.queues;
    let (caps, formats, _) = backend_state.surface.compatibility(physical_device);

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

    let (frame_images, framebuffers) = {
        let pairs = backbuffer
            .into_iter()
            .map(|image| unsafe {
                let rtv = device
                    .create_image_view(
                        &image,
                        ViewKind::D2,
                        surface_color_format,
                        Swizzle::NO,
                        COLOR_RANGE.clone(),
                    )
                    .unwrap();
                (image, rtv)
            })
            .collect::<Vec<_>>();
        let fbos = pairs
            .iter()
            .map(|&(_, ref rtv)| unsafe {
                device
                    .create_framebuffer(&fullscreen_pass, Some(rtv), extent)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        (pairs, fbos)
    };
    let frames_in_flight = 3;

    // Number of image acquisition semaphores is based on the number of swapchain images, not frames in flight,
    // plus one extra which we can guarantee is unused at any given time by swapping it out with the ones
    // in the rest of the queue.
    let mut image_acquire_semaphores = Vec::with_capacity(frame_images.len());
    let mut free_acquire_semaphore = device
        .create_semaphore()
        .expect("Could not create semaphore");

    // The number of the rest of the resources is based on the frames in flight.
    let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
    let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
    // Note: We don't really need a different command pool per frame in such a simple demo like this,
    let mut cmd_pools = Vec::with_capacity(frames_in_flight);
    let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

    cmd_pools.push(command_pool);
    for _ in 1..frames_in_flight {
        unsafe {
            cmd_pools.push(
                device
                    .create_command_pool_typed(
                        &queue_group,
                        gfx_hal::pool::CommandPoolCreateFlags::empty(),
                    )
                    .expect("Can't create command pool"),
            );
        }
    }

    for _ in 0..frame_images.len() {
        image_acquire_semaphores.push(
            device
                .create_semaphore()
                .expect("Could not create semaphore"),
        );
    }

    for i in 0..frames_in_flight {
        submission_complete_semaphores.push(
            device
                .create_semaphore()
                .expect("Could not create semaphore"),
        );
        submission_complete_fences.push(
            device
                .create_fence(true)
                .expect("Could not create semaphore"),
        );
        cmd_buffers.push(cmd_pools[i].acquire_command_buffer::<gfx_hal::command::MultiShot>());
    }
    let scale = 5.0;
    let triangle: [math::Vec3; 3] = [
        math::Vec3::new(0.0, scale * 1.0, 0.0),
        math::Vec3::new(scale * 0.5, 0.0, 0.0),
        math::Vec3::new(scale * -0.5, 0.0, 0.0),
    ];
    let mut pos_rots = vec![];
    let mut some_triangles = vec![];
    for particle in &my_emitter.particle_list {
        pos_rots.push((particle.position.clone(), particle.rotation.clone()));
    }
    for (pos, _rot) in &pos_rots {
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

    let mut _recreate_swapchain = false;
    let mut quitting = false;

    let mut last_frame_time = SystemTime::now();
    let mut frame: u64 = 0;

    let viewport = Viewport {
        rect: Rect {
            x: 0,
            y: 0,
            w: extent.width as i16,
            h: extent.height as i16,
        },
        depth: 0.0..1.0,
    };

    while quitting == false {
        let this_frame_time = SystemTime::now();
        let delta_time = last_frame_time.elapsed().unwrap() - this_frame_time.elapsed().unwrap();
        last_frame_time = this_frame_time;
        my_emitter.tick(((delta_time.as_micros() as f64) / 1_000_000.0) as f32);
        pos_rots.clear();
        some_triangles.clear();
        for particle in &my_emitter.particle_list {
            pos_rots.push((particle.position.clone(), particle.rotation.clone()));
        }
        for (pos, rot) in &pos_rots {
            let rotmat = math::Mat4::from_rotaxis(rot, [0.0, 0.0, -1.0]);
            let mut local_triangle = [
                triangle[0].clone(),
                triangle[1].clone(),
                triangle[2].clone(),
            ];
            for vert in &mut local_triangle {
                *vert = math::mat4_mul_vec3(&rotmat, &vert);
            }
            some_triangles.push(math::mat4_mul_vec3(
                &proj_matrix,
                &pos.add(&local_triangle[0]),
            ));
            some_triangles.push(math::mat4_mul_vec3(
                &proj_matrix,
                &pos.add(&local_triangle[1]),
            ));
            some_triangles.push(math::mat4_mul_vec3(
                &proj_matrix,
                &pos.add(&local_triangle[2]),
            ));
        }
        // TODO: check transitions: read/write mapping and vertex buffer read
        unsafe {
            let mut vertices = device
                .acquire_mapping_writer::<math::Vec3>(&buffer_memory, 0..buffer_req.size)
                .unwrap();
            vertices[0..some_triangles.len()].copy_from_slice(&some_triangles);
            device.release_mapping_writer(vertices).unwrap();
        }

        //TODO: Move this:
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
                        let top = logical_size.height as f32 / 2.0;
                        let right = logical_size.width as f32 / 2.0;
                        proj_matrix =
                            math::Mat4::create_ortho(-top, top, -right, right, 0.1, 1000.0);
                    }
                    _ => {}
                }
            }
        });
        // Start rendering
        if quitting == true {
            println!("Should quit!");
        }

        let swap_image = unsafe {
            match swapchain.acquire_image(!0, Some(&free_acquire_semaphore), None) {
                Ok((i, _)) => i as usize,
                Err(_) => {
                    _recreate_swapchain = true;
                    continue;
                }
            }
        };
        // Swap the acquire semaphore with the one previously associated with the image we are acquiring
        core::mem::swap(
            &mut free_acquire_semaphore,
            &mut image_acquire_semaphores[swap_image],
        );

        let frame_idx = frame as usize % frames_in_flight;

        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
            device
                .wait_for_fence(&submission_complete_fences[frame_idx], !0)
                .expect("Failed to wait for fence");
            device
                .reset_fence(&submission_complete_fences[frame_idx])
                .expect("Failed to reset fence");
            cmd_pools[frame_idx].reset();
        }
        // Rendering
        let cmd_buffer = &mut cmd_buffers[frame_idx];
        unsafe {
            cmd_buffer.begin(false);

            cmd_buffer.set_viewports(0, &[viewport.clone()]);
            cmd_buffer.set_scissors(0, &[viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&pipeline);
            cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));
            //TODO: Fix descriptor sets:
            //Handle normalized viewspace coordinates better!
            cmd_buffer.bind_graphics_descriptor_sets(&pipeline_layout, 0, Some(&desc_set), &[]);

            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &fullscreen_pass,
                    &framebuffers[swap_image],
                    viewport.rect,
                    &[gfx_hal::command::ClearValue::Color(
                        gfx_hal::command::ClearColor::Float([0.0, 0.0, 0.0, 1.0]),
                    )],
                );
                encoder.draw(0..some_triangles.len() as u32, 0..1);
            }

            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: Some(&*cmd_buffer),
                wait_semaphores: Some((
                    &image_acquire_semaphores[swap_image],
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                )),
                signal_semaphores: Some(&submission_complete_semaphores[frame_idx]),
            };
            queue_group.queues[0].submit(submission, Some(&submission_complete_fences[frame_idx]));

            // present frame
            if let Err(_) = swapchain.present(
                &mut queue_group.queues[0],
                swap_image as gfx_hal::SwapImageIndex,
                Some(&submission_complete_semaphores[frame_idx]),
            ) {
                _recreate_swapchain = true;
            }
        }
        frame += 1;
    }

    device.wait_idle().unwrap();
    unsafe {
        for pool in cmd_pools {
            device.destroy_command_pool(pool.into_raw());
        }

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
