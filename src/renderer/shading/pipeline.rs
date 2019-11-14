use crate::math::Vec3;

use gfx_hal::{
    device::Device,
    pass::Subpass,
    pso::{
        AttributeDesc, BlendState, ColorBlendDesc, ColorMask, Element, EntryPoint,
        GraphicsPipelineDesc, GraphicsShaderSet, Primitive, Rasterizer,
    },
    Backend,
};
//TODO: Make a state-holder class, which later
//also has this function:
pub unsafe fn reset_pipeline<B: Backend>(
    vert_spirv: Vec<u32>,
    frag_spirv: Vec<u32>,
    pipeline_layout: &B::PipelineLayout,
    device: &B::Device,
    render_pass: &B::RenderPass,
) -> B::GraphicsPipeline {
    let vertex_shader_module = device
        .create_shader_module(vert_spirv.as_slice())
        .expect("Could not create vertex shader module");

    let fragment_shader_module = device
        .create_shader_module(frag_spirv.as_slice())
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

    pipeline_desc.blender.targets.push(ColorBlendDesc {
        mask: ColorMask::ALL,
        blend: Some(BlendState::ALPHA),
    });
    //TODO: Fix pipeline vertex attribs

    pipeline_desc
        .vertex_buffers
        .push(gfx_hal::pso::VertexBufferDesc {
            binding: 0,
            stride: std::mem::size_of::<Vec3>() as u32,
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
