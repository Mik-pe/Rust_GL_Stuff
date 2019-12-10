use gfx_hal::adapter::MemoryType;
use gfx_hal::device::Device;
use gfx_hal::{buffer, memory, Backend};
use std::mem::size_of;
use std::ptr;

#[allow(dead_code)]
pub struct BufferState<B: Backend> {
    memory: Option<B::Memory>,
    buffer: Option<B::Buffer>,
    size: u64,
}

impl<B: Backend> BufferState<B> {
    #[allow(dead_code)]
    pub fn get_buffer(&self) -> &B::Buffer {
        self.buffer.as_ref().unwrap()
    }

    //TODO: implement data as generic Vec<T>?
    pub unsafe fn new<T>(
        device: &B::Device,
        data_source: &[T],
        data_len: usize,
        usage: buffer::Usage,
        memory_types: &[MemoryType],
    ) -> Self
    where
        T: Copy,
    {
        let memory: B::Memory;
        let mut buffer: B::Buffer;
        let size: u64;

        let stride = size_of::<T>() as u64;
        let upload_size = data_len as u64 * stride;
        {
            buffer = device.create_buffer(upload_size, usage).unwrap();
            let mem_req = device.get_buffer_requirements(&buffer);

            // A note about performance: Using CPU_VISIBLE memory is convenient because it can be
            // directly memory mapped and easily updated by the CPU, but it is very slow and so should
            // only be used for small pieces of data that need to be updated very frequently. For something like
            // a vertex buffer that may be much larger and should not change frequently, you should instead
            // use a DEVICE_LOCAL buffer that gets filled by copying data from a CPU_VISIBLE staging buffer.
            let upload_type = memory_types
                .iter()
                .enumerate()
                .position(|(id, mem_type)| {
                    mem_req.type_mask & (1 << id) != 0
                        && mem_type
                            .properties
                            .contains(memory::Properties::CPU_VISIBLE)
                })
                .unwrap()
                .into();

            memory = device.allocate_memory(upload_type, mem_req.size).unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            size = mem_req.size;

            // TODO: check transitions: read/write mapping and vertex buffer read
            {
                let data_mapping = device.map_memory(&memory, 0..size).unwrap();
                ptr::copy_nonoverlapping(
                    data_source.as_ptr() as *const u8,
                    data_mapping.offset(0 as isize),
                    data_len,
                );
                device.unmap_memory(&memory);
            }
        }

        BufferState {
            memory: Some(memory),
            buffer: Some(buffer),
            size,
        }
    }

    #[allow(dead_code)]
    pub fn update_data<T>(
        &mut self,
        offset: u64,
        data_source: *const u8,
        data_len: usize,
        device: &B::Device,
    ) where
        T: Copy,
    {
        let stride = size_of::<T>() as u64;
        let upload_size = data_len as u64 * stride;

        assert!(offset + upload_size <= self.size);

        unsafe {
            let data_mapping = device
                .map_memory(&self.memory.as_ref().unwrap(), offset..self.size)
                .unwrap();
            ptr::copy_nonoverlapping(data_source, data_mapping.offset(0 as isize), data_len);
            device.unmap_memory(&self.memory.as_ref().unwrap());
        }
    }
}
