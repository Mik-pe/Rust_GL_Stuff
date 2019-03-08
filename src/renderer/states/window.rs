use winit::{Event, EventsLoop, WindowBuilder};

pub struct WindowState {
    pub events_loop: winit::EventsLoop,
    pub wb: Option<winit::WindowBuilder>,
}

impl WindowState {
    pub fn new(wh: (i32, i32), title: String) -> Self {
        // Create a window with winit.
        let events_loop = EventsLoop::new();

        let wb = WindowBuilder::new()
            .with_dimensions(winit::dpi::LogicalSize::new(wh.0 as _, wh.1 as _))
            .with_title(title);

        WindowState {
            events_loop,
            wb: Some(wb),
        }
    }

    pub fn poll_events<F>(&mut self, callback: F)
    where
        F: FnMut(Event),
    {
        self.events_loop.poll_events(callback);
    }
}
