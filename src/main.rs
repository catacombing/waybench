use std::ffi::CString;
use std::mem;
use std::num::NonZeroU32;
use std::ops::{Div, Mul};
use std::time::{Duration, Instant};

use argh::FromArgs;
use glutin::config::{Api, Config, ConfigTemplateBuilder};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext,
    PossiblyCurrentGlContext, Version,
};
use glutin::display::{Display, DisplayApiPreference, GetGlDisplay, GlDisplay};
use glutin::prelude::GlSurface;
use glutin::surface::{Surface, SurfaceAttributesBuilder, WindowSurface};
use rand::Rng;
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState, Region};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::reexports::client::globals::{self, GlobalList};
use smithay_client_toolkit::reexports::client::protocol::wl_output::{Transform, WlOutput};
use smithay_client_toolkit::reexports::client::protocol::wl_subsurface::WlSubsurface;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface;
use smithay_client_toolkit::reexports::client::{Connection, Proxy, QueueHandle};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::shell::xdg::window::{
    Window as XdgWindow, WindowConfigure, WindowDecorations, WindowHandler,
};
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::subcompositor::SubcompositorState;
use smithay_client_toolkit::{
    delegate_compositor, delegate_output, delegate_registry, delegate_subcompositor,
    delegate_xdg_shell, delegate_xdg_window, registry_handlers,
};

#[derive(FromArgs)]
/// Wayland benchmark suite.
struct Cli {
    /// number of warmup frames [Default: 10]
    #[argh(option, default = "10")]
    warmup: usize,
    /// maximum fps target (should match output refresh rate) [Default: 60]
    #[argh(option, default = "60")]
    max_fps: usize,
    /// number of frames measured per benchmark run [Default: 128]
    #[argh(option, default = "128")]
    samples: usize,
    /// get framerate for a fixed number of surfaces
    #[argh(option)]
    surfaces: Option<usize>,
}

mod gl {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}

fn main() {
    // Get CLI arguments.
    let cli: Cli = argh::from_env();

    // Setup Wayland connection.
    let connection = Connection::connect_to_env().unwrap();
    let (globals, mut queue) = globals::registry_queue_init(&connection).unwrap();

    // Setup state.
    let mut state = State::new(&connection, &globals, queue.handle(), cli);

    // Start main event loop.
    while !state.terminated {
        queue.blocking_dispatch(&mut state).unwrap();
    }
}

/// Wayland client state.
struct State {
    protocol_states: ProtocolStates,
    queue: QueueHandle<Self>,
    terminated: bool,
    window: Window,

    target_frame_nanos: u128,
    min_frame_nanos: u128,
    frame_count: usize,
    last_frame: Instant,
    frames: Vec<u128>,
    bisect: Bisect,

    cli: Cli,
}

impl State {
    fn new(
        connection: &Connection,
        globals: &GlobalList,
        queue: QueueHandle<Self>,
        cli: Cli,
    ) -> Self {
        // Setup globals.
        let protocol_states = ProtocolStates::new(globals, &queue);

        // Create bisection tracker.
        let bisect = Bisect::new(2);

        // Create window.
        let window = Window::new(connection, &protocol_states, &queue);

        // Calculate target render times.
        let target_nanos = Duration::from_secs(1).as_nanos() / cli.max_fps as u128;

        // Allocate buffer for render times.
        let frames = vec![0; cli.samples];

        Self {
            protocol_states,
            frames,
            window,
            bisect,
            queue,
            cli,
            target_frame_nanos: target_nanos,
            min_frame_nanos: target_nanos,
            last_frame: Instant::now(),
            frame_count: Default::default(),
            terminated: Default::default(),
        }
    }

    fn resize(&mut self, size: Size) {
        self.window.size = size;

        self.update_surfaces();

        // Update opaque region.
        let logical_size = size / self.window.factor as f64;
        if let Ok(region) = Region::new(&self.protocol_states.compositor) {
            region.add(0, 0, logical_size.width, logical_size.height);
            self.window.window.wl_surface().set_opaque_region(Some(region.wl_region()));
        }

        // Resize OpenGL viewport.
        unsafe { gl::Viewport(0, 0, size.width, size.height) };

        self.request_frame();
    }

    /// Recreate all surfaces and subsurfaces at the correct size.
    fn update_surfaces(&mut self) {
        // Update or create the primary surface.
        let size = self.window.size;
        match &self.window.surface {
            None => {
                let wl_surface = self.window.window.wl_surface().clone();
                self.window.surface = Some(self.create_egl_surface(
                    wl_surface,
                    size.width as u32,
                    size.height as u32,
                    true,
                ));
            },
            Some(surface) => {
                surface.resize(
                    &self.window.context,
                    NonZeroU32::new(size.width as u32).unwrap(),
                    NonZeroU32::new(size.height as u32).unwrap(),
                );
            },
        }

        // Get desired number of (sub) surfaces.
        let num_surfaces = match self.cli.surfaces {
            Some(num_surfaces) => num_surfaces,
            None => self.bisect.current(),
        };

        // Re-create all subsurfaces.
        self.window.subsurfaces.clear();
        let logical_size = size / self.window.factor as f64;
        for i in 1..num_surfaces {
            let parent = self.window.window.wl_surface().clone();
            let subsurface = Subsurface::new(self, parent, logical_size, i);
            self.window.subsurfaces.push(subsurface);
        }
    }

    /// Create an EGL surface.
    fn create_egl_surface(
        &self,
        surface: WlSurface,
        width: u32,
        height: u32,
        primary: bool,
    ) -> Surface<WindowSurface> {
        // Setup surface attributes.
        let mut wayland_window_handle = WaylandWindowHandle::empty();
        wayland_window_handle.surface = surface.id().as_ptr().cast();
        let raw_window_handle = RawWindowHandle::Wayland(wayland_window_handle);
        let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle,
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        );

        // Create window surface.
        let display = self.window.context.display();
        let surface = unsafe {
            display.create_window_surface(&self.window.config, &surface_attributes).unwrap()
        };

        // Draw the surface's color.
        self.window.context.make_current(&surface).unwrap();
        let mut rng = rand::thread_rng();
        let (r, g, b, a) = if primary {
            (0., 0., 0., 1.)
        } else {
            (rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), 0.5)
        };
        unsafe {
            gl::ClearColor(r, g, b, a);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::Flush();
        }
        surface.swap_buffers(&self.window.context).unwrap();

        surface
    }

    /// Calculate the mean framerate of the sample window.
    fn mean_framerate(&self) -> Option<u128> {
        (self.frame_count >= self.frames.len() + self.cli.warmup).then(|| {
            let sum: u128 = self.frames.iter().sum();
            let mean = sum / self.frames.len() as u128;
            mean
        })
    }

    /// Send a new frame request.
    fn request_frame(&mut self) {
        if self.window.frame_pending {
            return;
        }
        self.window.frame_pending = true;

        let surface = self.window.window.wl_surface();
        surface.frame(&self.queue, surface.clone());
        surface.commit();
    }

    /// Calculate the relative variance of the sample window.
    fn relative_variance(&self) -> f64 {
        // Calculate mean of all samples.
        let sum: u128 = self.frames.iter().sum();
        let mean = sum / self.frames.len() as u128;

        // Calculate variance.
        let variance_sum = self.frames.iter().map(|value| (value - mean).pow(2)).sum::<u128>();
        let variance = variance_sum as f64 / self.frames.len() as f64;

        // Normalize std dev to disconnect it from the framerate.
        let relative_variance = variance.sqrt() / mean as f64;

        relative_variance
    }
}

struct Window {
    window: XdgWindow,

    context: PossiblyCurrentContext,
    config: Config,

    surface: Option<Surface<WindowSurface>>,
    subsurfaces: Vec<Subsurface>,

    factor: i32,
    size: Size,

    frame_pending: bool,
}

impl Window {
    fn new(
        connection: &Connection,
        protocol_states: &ProtocolStates,
        queue: &QueueHandle<State>,
    ) -> Self {
        // Initialize EGL context.

        let mut wayland_display = WaylandDisplayHandle::empty();
        wayland_display.display = connection.backend().display_ptr().cast();
        let raw_display = RawDisplayHandle::Wayland(wayland_display);
        let display = unsafe {
            Display::new(raw_display, DisplayApiPreference::Egl)
                .expect("Unable to create EGL display")
        };

        let config_template = ConfigTemplateBuilder::new().with_api(Api::GLES2).build();
        let config = unsafe {
            display
                .find_configs(config_template)
                .ok()
                .and_then(|mut configs| configs.next())
                .expect("No suitable configuration found")
        };

        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(Some(Version::new(2, 0))))
            .build(None);
        let context = unsafe {
            display
                .create_context(&config, &context_attributes)
                .expect("Failed to create EGL context")
        };

        // Load the OpenGL symbols.
        gl::load_with(|symbol| {
            let symbol = CString::new(symbol).unwrap();
            display.get_proc_address(symbol.as_c_str()).cast()
        });

        // Create Wayland window.

        let surface = protocol_states.compositor.create_surface(queue);

        let context = context.treat_as_possibly_current();

        let decorations = WindowDecorations::RequestServer;
        let window = protocol_states.xdg_shell.create_window(surface, decorations, queue);
        window.set_title("Waybench");
        window.set_app_id("Waybench");
        window.set_fullscreen(None);
        window.commit();

        // Default to reasonable initial size.
        let size = Size { width: 640, height: 480 };

        Self {
            context,
            config,
            window,
            size,
            factor: 1,
            frame_pending: Default::default(),
            subsurfaces: Default::default(),
            surface: Default::default(),
        }
    }
}

/// Wayland subsurface.
struct Subsurface {
    wl_subsurface: WlSubsurface,
    wl_surface: WlSurface,
    size: Size,
    x: i32,
    y: i32,
}

impl Subsurface {
    fn new(state: &State, parent: WlSurface, window_size: Size, index: usize) -> Self {
        // Create new Wayland subsurface.
        let (wl_subsurface, wl_surface) =
            state.protocol_states.subcompositor.create_subsurface(parent, &state.queue);

        // Create EGL surface.
        const MIN_SIZE: f64 = 0.05;
        const MAX_GROWTH: f64 = 0.2;
        const STEPS: usize = 64;
        let step = (index % STEPS) as f64 / STEPS as f64;
        let width = (window_size.width as f64 * (MIN_SIZE + MAX_GROWTH * step)) as u32;
        let height = (window_size.height as f64 * (MIN_SIZE + MAX_GROWTH * step)) as u32;
        state.create_egl_surface(wl_surface.clone(), width, height, false);

        // Set initial subsurface position.
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0..window_size.width - width as i32);
        let y = rng.gen_range(0..window_size.height - height as i32);
        wl_subsurface.set_position(x, y);

        let size = Size { width: width as i32, height: height as i32 };
        Self { wl_subsurface, wl_surface, size, x, y }
    }

    /// Move this surface to an new random location.
    fn update_position(&mut self, window_size: Size) {
        let mut rng = rand::thread_rng();
        self.x = rng.gen_range(0..window_size.width - self.size.width);
        self.y = rng.gen_range(0..window_size.height - self.size.height);
        self.wl_subsurface.set_position(self.x, self.y);
    }
}

impl Drop for Subsurface {
    fn drop(&mut self) {
        self.wl_subsurface.destroy();
        self.wl_surface.destroy();
    }
}

/// Surface bisection state.
struct Bisect {
    /// Lower bisection bound, inclusive.
    lower: usize,
    /// Upper bisection bound, exclusive.
    upper: Option<usize>,
    /// Current surface count.
    current: usize,
}

impl Bisect {
    /// Start a new surface count bisection.
    fn new(lower_bound: usize) -> Self {
        Self { lower: lower_bound, current: lower_bound, upper: None }
    }

    /// Mark current surface count as failed.
    ///
    /// This function updates the bisection state with the assumption that the
    /// surface count acquired using [`Self::current`] has rendering times
    /// *above* the desired threshold.
    ///
    /// Returns the highest successful surface count if bisection is done,
    /// otherwise returns [`None`].
    fn bad(&mut self) -> Option<usize> {
        self.upper = Some(self.current);

        if self.lower + 1 >= self.current {
            Some(self.lower)
        } else {
            self.current = (self.lower + self.current) / 2;
            None
        }
    }

    /// Mark current surface count as passed.
    ///
    /// This function updates the bisection state with the assumption that the
    /// surface count acquired using [`Self::current`] has rendering times
    /// *below* the desired threshold.
    ///
    /// Returns the highest successful surface count if bisection is done,
    /// otherwise returns [`None`].
    fn good(&mut self) -> Option<usize> {
        self.lower = self.current;

        match self.upper {
            Some(upper) if self.lower + 1 >= upper => Some(self.lower),
            Some(upper) => {
                self.current = (self.current + upper) / 2;
                None
            },
            None => {
                self.current *= 2;
                None
            },
        }
    }

    fn current(&self) -> usize {
        self.current
    }
}

impl CompositorHandler for State {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlSurface,
        factor: i32,
    ) {
        if self.window.factor == factor {
            return;
        }

        let factor_change = factor as f64 / self.window.factor as f64;
        self.window.factor = factor;

        self.resize(self.window.size * factor_change);
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlSurface,
        _: Transform,
    ) {
    }

    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlSurface, _: u32) {
        self.window.frame_pending = false;

        // TODO: Cleanup/Move?
        if let Some(mean_framerate) = self.mean_framerate() {
            if self.cli.surfaces.is_some() && !self.terminated {
                let fps = Duration::from_secs(1).as_nanos() as f64 / mean_framerate as f64;
                println!("MEAN FRAMERATE: {:.3} ms", mean_framerate as f64 / 1_000_000 as f64);
                println!("MEAN FRAMERATE: {:.3} FPS", fps);
                println!("FRAMES: {:?}", self.frames);
                self.terminated = true;
                return;
            }

            // Allow up to 3% framerate fluctuation.
            let desired_framerate = self.target_frame_nanos + self.target_frame_nanos / 33;
            let result = if mean_framerate <= desired_framerate {
                self.bisect.good()
            } else {
                self.bisect.bad()
            };

            // Handle individual framerate benchmark completions.
            if let Some(max_surfaces) = result {
                let fps = Duration::from_secs(1).as_nanos() / self.target_frame_nanos;
                println!("COMPLETED {fps} FPS: {max_surfaces}");

                self.target_frame_nanos += self.min_frame_nanos;

                self.bisect = Bisect::new(self.bisect.current());
            }

            // Update surface count.
            self.update_surfaces();

            // TODO: Remove?
            println!("SWITCHED TO {} SURFACES", self.bisect.current());

            // Reset framerate tracking.
            self.last_frame = Instant::now();
            self.frame_count = 0;
        } else {
            // Calculate time since last frame.
            self.frames.rotate_right(1);
            let prev_frame = mem::replace(&mut self.last_frame, Instant::now());
            self.frames[0] = (self.last_frame - prev_frame).as_nanos();
            self.frame_count += 1;
        }

        // Move all subsurfaces, to force recomposition.
        let logical_size = self.window.size / self.window.factor as f64;
        for surface in &mut self.window.subsurfaces {
            surface.update_position(logical_size);
        }

        self.request_frame();
    }
}
delegate_compositor!(State);

delegate_subcompositor!(State);

impl OutputHandler for State {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.protocol_states.output
    }

    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlOutput) {}

    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlOutput) {}

    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlOutput) {}
}
delegate_output!(State);

impl WindowHandler for State {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &XdgWindow) {
        self.terminated = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &XdgWindow,
        configure: WindowConfigure,
        _: u32,
    ) {
        // Use current size to trigger initial draw if no dimensions were provided.
        let size = configure.new_size.0.zip(configure.new_size.1);
        let size = size
            .map(|size| Size::from((size.0.get(), size.1.get())) * self.window.factor as f64)
            .unwrap_or(self.window.size);
        self.resize(size);
    }
}
delegate_xdg_shell!(State);

delegate_xdg_window!(State);

impl ProvidesRegistryState for State {
    registry_handlers![OutputState];

    fn registry(&mut self) -> &mut RegistryState {
        &mut self.protocol_states.registry
    }
}
delegate_registry!(State);

#[derive(Debug)]
struct ProtocolStates {
    subcompositor: SubcompositorState,
    compositor: CompositorState,
    registry: RegistryState,
    xdg_shell: XdgShell,
    output: OutputState,
}

impl ProtocolStates {
    fn new(globals: &GlobalList, queue: &QueueHandle<State>) -> Self {
        let registry = RegistryState::new(globals);
        let compositor = CompositorState::bind(globals, queue).unwrap();
        let wl_compositor = compositor.wl_compositor().clone();
        let subcompositor = SubcompositorState::bind(wl_compositor, globals, queue).unwrap();
        let xdg_shell = XdgShell::bind(globals, queue).unwrap();
        let output = OutputState::new(globals, queue);

        Self { registry, compositor, subcompositor, xdg_shell, output }
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Size<T = i32> {
    pub width: T,
    pub height: T,
}

impl From<(u32, u32)> for Size {
    fn from(tuple: (u32, u32)) -> Self {
        Self { width: tuple.0 as i32, height: tuple.1 as i32 }
    }
}

impl From<Size> for Size<f32> {
    fn from(from: Size) -> Self {
        Self { width: from.width as f32, height: from.height as f32 }
    }
}

impl Mul<f64> for Size {
    type Output = Self;

    fn mul(mut self, factor: f64) -> Self {
        self.width = (self.width as f64 * factor) as i32;
        self.height = (self.height as f64 * factor) as i32;
        self
    }
}

impl Div<f64> for Size {
    type Output = Self;

    fn div(mut self, factor: f64) -> Self {
        self.width = (self.width as f64 / factor).round() as i32;
        self.height = (self.height as f64 / factor).round() as i32;
        self
    }
}
