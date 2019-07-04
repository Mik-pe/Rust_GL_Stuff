use super::adapter::AdapterState;
use super::window::WindowState;

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
    Backend, Instance,
};

pub struct BackendState<B: Backend> {
    pub surface: B::Surface,
    #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
    #[allow(dead_code)]
    window: winit::Window,
}

#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
pub fn create_backend(
    window_state: &mut WindowState,
) -> (
    BackendState<back::Backend>,
    back::Instance,
    AdapterState<back::Backend>,
) {
    let window = window_state
        .wb
        .take()
        .unwrap()
        .build(&window_state.events_loop)
        .unwrap();
    let instance = back::Instance::create("Graphics instance", 1);
    let surface = instance.create_surface(&window);
    let mut adapters = instance.enumerate_adapters();
    (
        BackendState { surface, window },
        instance,
        AdapterState::new(&mut adapters),
    )
}

#[cfg(feature = "gl")]
pub fn create_backend(
    window_state: &mut WindowState,
) -> (BackendState<back::Backend>, (), AdapterState<back::Backend>) {
    let window = {
        let builder =
            back::config_context(back::glutin::ContextBuilder::new(), ColorFormat::SELF, None)
                .with_vsync(true);
        back::glutin::WindowedContext::new_windowed(
            window_state.wb.take().unwrap(),
            builder,
            &window_state.events_loop,
        )
        .unwrap()
    };

    let surface = back::Surface::from_window(window);
    let mut adapters = surface.enumerate_adapters();
    (
        BackendState { surface },
        (),
        AdapterState::new(&mut adapters),
    )
}
