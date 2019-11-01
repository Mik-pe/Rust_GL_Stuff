use gfx_hal::adapter::{Adapter, PhysicalDevice};
use gfx_hal::queue::{family::QueueFamily, QueueGroup};
use gfx_hal::window::Surface;
use gfx_hal::Backend;

pub struct DeviceState<B: Backend> {
    pub device: B::Device,
    pub physical_device: B::PhysicalDevice,
    pub queues: QueueGroup<B>,
}

impl<B: Backend> DeviceState<B> {
    pub fn new(adapter: Adapter<B>, surface: &B::Surface) -> Self {
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .unwrap();

        let mut gpu = unsafe {
            adapter
                .physical_device
                .open(&[(family, &[1.0])], gfx_hal::Features::empty())
                .unwrap()
        };

        DeviceState {
            device: gpu.device,
            queues: gpu.queue_groups.pop().unwrap(),
            physical_device: adapter.physical_device,
        }
    }
}
