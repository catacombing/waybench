use std::ffi::CString;
use std::num::NonZeroU32;
use std::ops::{Div, Mul};
use std::time::{Duration, Instant};
use std::{cmp, fs, mem};

use argh::FromArgs;
use glutin::config::{Api, Config, ConfigTemplateBuilder};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext,
    PossiblyCurrentGlContext, Version,
};
use glutin::display::{Display, DisplayApiPreference, GetGlDisplay, GlDisplay};
use glutin::prelude::GlSurface;
use glutin::surface::{Surface, SurfaceAttributesBuilder, WindowSurface};
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

/// Number of samples checked to determine benchmark stability.
const STABLE_SAMPLES: usize = 128;

#[derive(FromArgs)]
/// Wayland benchmark suite.
struct Cli {
    /// use SHM instead of DMA buffers
    #[argh(switch)]
    shm: bool,
    /// number of warmup frames
    #[argh(option, short = 'w')]
    warmup_frames: Option<u8>,
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

    // TODO
    println!("{:?}", state.results);
    let mut csv_lines = vec![String::new()];
    for (surface_count, _) in &state.results {
        csv_lines[0] += &format!("{surface_count},");
    }
    for i in 0.. {
        let mut done = true;
        let mut line = String::new();
        for (_, results) in &state.results {
            match results.get(i) {
                Some(result) => {
                    let result_ms = *result as f64 / 1_000_000.;
                    line += &format!("{result_ms:.3},");
                    done = false;
                },
                None => line.push_str("-,"),
            }
        }
        if done {
            break;
        } else {
            csv_lines.push(line);
        }
    }
    for line in &mut csv_lines {
        line.truncate(line.len() - 1);
    }
    let csv = csv_lines.join("\n");
    fs::write("results.csv", csv).unwrap();
}

/// Wayland client state.
struct State {
    protocol_states: ProtocolStates,
    queue: QueueHandle<Self>,
    terminated: bool,
    window: Window,

    frames: [u128; STABLE_SAMPLES],
    bisect_bounds: Range<usize>,
    target_frame_nanos: u128,
    frame_count: usize,
    last_frame: Instant,
    warmup_frames: u8,
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

        // Create window.
        let window = Window::new(connection, &protocol_states, &queue);

        Self {
            protocol_states,
            window,
            queue,
            warmup_frames: cli.warmup_frames.unwrap_or(10),
            frames: [0; STABLE_SAMPLES],
            last_frame: Instant::now(),
            target_frame_nanos: 0,
            bisect_bounds: 1..,
            terminated: false,
            frame_count: 0,
        }
    }

    // TODO: This benchmark is kinda flawed because only a very limited number of
    // surfaces is tested, but since we're relying on buffer swaps the "step
    // size" is ~16ms so huge margin of error.
    //
    fn draw(&mut self) {
        // Terminate once frame or time limit was reached.
        const TEN_SECONDS: Duration = Duration::from_secs(10);
        let nanos_elapsed: u128 = self.frames[..self.frame_count].iter().sum();
        if self.frame_count == self.frames.len() || nanos_elapsed > TEN_SECONDS.as_nanos() {
            self.terminated = true;
            return;
        }

        if self.warmups_done < self.warmup_frames {
            // Ignore warmup frames.
            self.last_frame = Instant::now();
            self.warmups_done += 1;
        } else if self.is_stable() {
            // TODO: Means seem a bit high, are we measuring incorrectly?
            //
            // Collect results.
            let num_surfaces = self.num_surfaces();
            let results = self.frames[..self.frame_count].iter().copied().collect();
            self.results.push((num_surfaces, results));

            let mean: u128 =
                self.frames[..self.frame_count].iter().sum::<u128>() / self.frame_count as u128;
            println!(
                "COMPLETED {} WITH MEAN OF {} ({} FPS)",
                num_surfaces,
                mean as f64 / 1_000_000.0,
                1_000_000_000.0 / mean as f64,
            );

            // Increase surface count.
            self.bench_index += 1;
            self.update_surfaces();

            self.last_frame = Instant::now();
            self.warmups_done = 0;
            self.frame_count = 0;
        } else {
            // Calculate time since last frame.
            let last_frame = mem::replace(&mut self.last_frame, Instant::now());
            self.frames[self.frame_count] = (self.last_frame - last_frame).as_nanos();
            self.frame_count += 1;
        }

        // Draw the window.
        let num_surfaces = self.num_surfaces();
        let rows = f64::sqrt(num_surfaces as f64).floor();
        let columns = (num_surfaces as f64 / rows).ceil() as usize;
        for (i, surface) in self.window.surfaces.iter().enumerate() {
            self.window.context.make_current(surface).unwrap();

            // Calculate colors for checker-pattern.
            let row = i / columns;
            let column = i % columns;
            let color_index = (column % 2) ^ (row % 2);
            let (r, g, b) = if color_index == self.frame_count % 2 {
                (0., 0., 0.)
            } else if color_index == 0 {
                (0., 1., 0.)
            } else {
                (1., 0., 1.)
            };

            // TODO: Main surface is 1 frame ahead?
            //
            let (r, g, b) = if self.frame_count % 2 == 0 { (1., 0., 1.) } else { (0., 0., 0.) };

            // Clear entire buffer with new color.
            unsafe {
                gl::ClearColor(r, g, b, 1.);
                gl::Clear(gl::COLOR_BUFFER_BIT);
                gl::Flush(); // TODO: Do we need to flush?
            }

            surface.swap_buffers(&self.window.context).unwrap();
        }

        // Request redraw.
        let surface = self.window.window.wl_surface();
        surface.frame(&self.queue, surface.clone());
        surface.commit();
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

        self.draw();
    }

    /// Recreate all surfaces and subsurfaces at the correct size.
    fn update_surfaces(&mut self) {
        // Destroy all existing subsurfaces.
        for (subsurface, surface) in self.window.subsurfaces.drain(..) {
            subsurface.destroy();
            surface.destroy();
        }
        self.window.surfaces.clear();

        // Get width/height for each subsurface.
        let num_surfaces = self.num_surfaces();
        let size = self.window.size;
        let logical_size = size / self.window.factor as f64;
        let rows = f64::sqrt(num_surfaces as f64).floor();
        let columns = (num_surfaces as f64 / rows).ceil();
        let surface_width = (logical_size.width as f64 / columns).ceil() as u32;
        let surface_height = (logical_size.height as f64 / rows).ceil() as u32;

        // TODO: Resize old primary surface or destroy it?
        //
        // Create primary surface stretching behind the entire window.
        let wl_surface = self.window.window.wl_surface().clone();
        let surface = self.create_surface(wl_surface, size.width as u32, size.height as u32);
        self.window.surfaces.push(surface);

        // Re-create all subsurfaces.
        for i in 1..(num_surfaces) {
            // Create new Wayland subsurface.
            let parent = self.window.window.wl_surface().clone();
            let (subsurface, wl_surface) =
                self.protocol_states.subcompositor.create_subsurface(parent, &self.queue);

            // Create EGL surface.
            let surface = self.create_surface(wl_surface.clone(), surface_width, surface_height);
            self.window.surfaces.push(surface);

            // Update subsurface position.
            let x = surface_width as i32 * (i as i32 % columns as i32);
            let x = cmp::min(x, logical_size.width - surface_width as i32);
            let y = surface_height as i32 * (i as i32 / columns as i32);
            let y = cmp::min(y, logical_size.height - surface_height as i32);
            subsurface.set_position(x, y);

            // Store all subsurfaces for destruction on resize.
            self.window.subsurfaces.push((subsurface, wl_surface));
        }
    }

    /// Create an EGL surface.
    fn create_surface(
        &self,
        surface: WlSurface,
        width: u32,
        height: u32,
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
        unsafe { display.create_window_surface(&self.window.config, &surface_attributes).unwrap() }
    }

    /// Check if current benchmark is stable.
    fn is_stable(&self) -> bool {
        const MAX_VARIANCE: f64 = 0.2;

        if self.frame_count < STABLE_SAMPLES {
            return false;
        }

        // Calculate mean of all samples.
        let sample_window = &self.frames[self.frame_count - STABLE_SAMPLES..self.frame_count];
        let sum: u128 = sample_window.iter().sum();
        let mean = sum as f64 / STABLE_SAMPLES as f64;

        // Calculate variance.
        let variance_sum =
            sample_window.iter().map(|value| (*value as f64 - mean).powi(2)).sum::<f64>();
        let variance = variance_sum / STABLE_SAMPLES as f64;

        // Normalize std dev to disconnect it from the framerate.
        let relative_variance = variance.sqrt() / mean;

        relative_variance < MAX_VARIANCE
    }

    /// Get surface count for current benchmark index.
    fn num_surfaces(&self) -> usize {
        if self.bench_index < 10 {
            // To ensure good direct-scanout plane coverage, all counts from 1 to
            // 9 are tested.
            self.bench_index
        } else {
            // Afer 9, we only test square surface counts for and even grid
            // pattern.
            (3 + self.bench_index - 9).pow(2)
        }
    }
}

struct Window {
    context: PossiblyCurrentContext,
    config: Config,
    window: XdgWindow,
    surfaces: Vec<Surface<WindowSurface>>,
    subsurfaces: Vec<(WlSubsurface, WlSurface)>,
    factor: i32,
    size: Size,
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
            surfaces: Vec::new(),
            subsurfaces: Vec::new(),
        }
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
        self.draw();
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
