#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(not(any(
  feature = "vulkan",
  feature = "dx12",
  feature = "metal",
  feature = "gl"
)))]
use gfx_backend_empty as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

use gfx_hal::{
  format::Format,
  image::{Access, Layout, SubresourceRange, ViewKind},
  pass::{
    Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
    SubpassDesc, SubpassRef,
  },
  pso::{
    BlendState, ColorBlendDesc, ColorMask, EntryPoint, GraphicsPipelineDesc, GraphicsShaderSet,
    PipelineStage, Rasterizer, Rect, Viewport,
  },
  Adapter, Backend, Device, Limits, MemoryType, PhysicalDevice,
};

pub fn create_fullscreen_pass<B: Backend>(
  surface_color_format: Format,
  device: &B::Device,
) -> B::RenderPass {
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
      accesses: Access::empty()..(Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE),
    };

    unsafe { device.create_render_pass(&[color_attachment], &[subpass], &[dependency]) }
      .expect("Could not create render pass")
  };
  render_pass
}
