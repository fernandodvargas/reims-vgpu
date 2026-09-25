//! The display thread: take the connector, then present published frames on it.
//!
//! The same loop as the window's `App::draw`, without winit: the device's
//! publish wakes it through the [`WindowWaker`]'s channel, and
//! [`WINDOW_REDRAW_BACKSTOP`] bounds a wake that does not land. Taking the
//! screen happens before [`spawn`] returns, so a refusal reaches the caller
//! typed and in time to stop the boot; nothing falls back to a window.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread::JoinHandle;

use super::capture;
use super::cursor;
use super::drm::{self, Card};
use super::select::{self, SelectError};
use crate::backend::window::{SurfaceSource, WindowPresentOutcome, WindowSurface};
use crate::backend::Backend as _;
use crate::host_window::present::{
    needs_present, window_cpu_frame, FrameSlot, StopFlag, WindowWakeHandle, WINDOW_REDRAW_BACKSTOP,
};

/// Why the display could not take the screen.
#[derive(Debug)]
pub enum DisplayError {
    /// No connector or no mode (`no_connector`, `no_mode`).
    Select(SelectError),
    /// Another process is the card's DRM master (`drm_busy`). `holders` are the
    /// pids this namespace can see with the card open, for the log.
    Busy { holders: Vec<u32> },
    /// The card node did not open or answer (`drm_open`).
    Open(io::Error),
    /// The rail refused the display surface. `acquire_refused` when Vulkan
    /// would not hand over the connector, `display_attach` otherwise; `detail`
    /// is the rail's own decline line.
    Attach { acquire: bool, detail: String },
}

impl DisplayError {
    pub fn slug(&self) -> &'static str {
        match self {
            Self::Select(error) => error.slug(),
            Self::Busy { .. } => "drm_busy",
            Self::Open(_) => "drm_open",
            Self::Attach { acquire: true, .. } => "acquire_refused",
            Self::Attach { acquire: false, .. } => "display_attach",
        }
    }

    /// The process exit code for this refusal: the KMS output never falls back
    /// to QEMU's own display, so the boot ends, and the code names why.
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Open(_) => 70,
            Self::Busy { .. } => 71,
            Self::Select(SelectError::NoConnector { .. }) => 72,
            Self::Select(SelectError::NoMode { .. }) => 73,
            Self::Attach { acquire: true, .. } => 74,
            Self::Attach { acquire: false, .. } => 75,
        }
    }
}

impl std::fmt::Display for DisplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.slug())?;
        match self {
            Self::Select(error) => write!(f, " ({error})"),
            Self::Busy { holders } => write!(f, " holders={holders:?}"),
            Self::Open(error) => write!(f, " error={error}"),
            Self::Attach { detail, .. } => write!(f, " vk={detail}"),
        }
    }
}

impl crate::observe::Decline for DisplayError {
    fn slug(&self) -> &'static str {
        DisplayError::slug(self)
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let detail = match self {
            Self::Select(error) => error.to_string(),
            Self::Busy { holders } => format!("{holders:?}"),
            Self::Open(error) => error.to_string(),
            Self::Attach { detail, .. } => detail.clone(),
        };
        vec![("detail", crate::host_window::present::detail_field(&detail))]
    }
}

/// Type an attach refusal by the rail's slug: the two Vulkan calls that take
/// the connector are the acquire; anything after them is the attach.
fn attach_refusal(slug: &str, detail: String) -> DisplayError {
    DisplayError::Attach {
        acquire: matches!(slug, "vk_display_get" | "vk_display_acquire"),
        detail,
    }
}

/// Everything the display loop holds while it presents.
struct Taken {
    /// Kept open for the presenter's life: the acquired display is bound to it.
    card: Card,
    connector: String,
    connector_id: u32,
    width: u32,
    height: u32,
    refresh_mhz: u32,
}

/// Take the connector (`REIMS_VGPU_DRM_CARD`, `REIMS_VGPU_CONNECTOR`) and start
/// the thread `reims-vgpu-display` presenting from `frames` until `stop`. A
/// refusal is returned here, after the thread has ended, and also logged.
pub fn spawn(
    frames: FrameSlot,
    stop: StopFlag,
    wake: WindowWakeHandle,
) -> Result<JoinHandle<()>, DisplayError> {
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), DisplayError>>(1);
    let handle = std::thread::Builder::new()
        .name("reims-vgpu-display".into())
        .spawn(move || {
            let taken = match take_screen() {
                Ok(taken) => {
                    let _ = ready_tx.send(Ok(()));
                    taken
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            present_loop(taken, frames, stop, wake);
        })
        .map_err(DisplayError::Open)?;
    let ready = ready_rx.recv().unwrap_or_else(|_| {
        Err(DisplayError::Open(io::Error::other(
            "display thread ended before taking the screen",
        )))
    });
    if let Err(error) = ready {
        let _ = handle.join();
        crate::observe::Emit::decline("host_display", &error).fail();
        eprintln!(
            "reims-vgpu-display: refused reason={} {error}",
            error.slug()
        );
        return Err(error);
    }
    Ok(handle)
}

fn take_screen() -> Result<Taken, DisplayError> {
    let card_path = PathBuf::from(
        std::env::var("REIMS_VGPU_DRM_CARD").unwrap_or_else(|_| "/dev/dri/card1".into()),
    );
    let wanted = std::env::var("REIMS_VGPU_CONNECTOR")
        .ok()
        .filter(|s| !s.is_empty());
    let card = Card::open(&card_path).map_err(DisplayError::Open)?;
    // Diagnostic switch for gate D6: skip the master check so the refusal that
    // fires is Vulkan's own acquire. Loud, because it is never a product state.
    if std::env::var("REIMS_VGPU_DRM_HOLDERS").as_deref() == Ok("ignore") {
        crate::observe::off("host_display_master_check status=ignored");
        eprintln!("reims-vgpu-display: WARN REIMS_VGPU_DRM_HOLDERS=ignore, master check skipped");
    } else if !card.is_master() {
        return Err(DisplayError::Busy {
            holders: drm::holders(&card_path),
        });
    }
    let connectors = card.connectors().map_err(DisplayError::Open)?;
    let choice = select::choose(&connectors, wanted.as_deref()).map_err(DisplayError::Select)?;
    let (width, height) = (u32::from(choice.mode.width), u32::from(choice.mode.height));
    let refresh_mhz = choice.mode.refresh_mhz;
    crate::backend::selected()
        .window_attach(&WindowSurface {
            source: SurfaceSource::Display {
                drm_fd: card.fd(),
                connector_id: choice.connector_id,
                width,
                height,
                refresh_mhz,
            },
            width,
            height,
        })
        .map_err(|decline| {
            attach_refusal(crate::observe::Decline::slug(&decline), decline.to_string())
        })?;
    eprintln!(
        "reims-vgpu-display: took {} {width}x{height}@{refresh_mhz}mHz on {}",
        choice.connector,
        card_path.display()
    );
    crate::observe::off(format!(
        "host_display_taken connector={} mode={width}x{height} refresh_mhz={refresh_mhz}",
        choice.connector
    ));
    Ok(Taken {
        card,
        connector: choice.connector,
        connector_id: choice.connector_id,
        width,
        height,
        refresh_mhz,
    })
}

/// The guest cursor as last published, and the plane showing it.
#[derive(Default)]
struct CursorView {
    plane: Option<super::drm::CursorPlane>,
    glyph: Option<std::sync::Arc<cursor::Glyph>>,
    new_glyph: bool,
    pos: (u16, u16),
    show: bool,
    dirty: bool,
}

impl CursorView {
    fn update(&mut self, change: cursor::CursorChange) {
        self.pos = (change.x, change.y);
        self.show = change.show;
        if let Some(glyph) = change.glyph {
            self.glyph = Some(glyph);
            self.new_glyph = true;
        }
        self.dirty = true;
    }

    /// Put the cursor on the plane. The plane is made lazily: the connector
    /// has no CRTC until the first present's modeset, and until then the
    /// change stays pending.
    fn apply(&mut self, taken: &Taken, frame: (u32, u32)) {
        if !self.dirty {
            return;
        }
        if self.plane.is_none() {
            match taken.card.crtc_of(taken.connector_id) {
                Ok(Some(crtc)) => match taken.card.cursor_plane(crtc) {
                    Ok(plane) => self.plane = Some(plane),
                    Err(error) => {
                        cursor::note_error(format!("cursor_plane: {error}"));
                        self.dirty = false;
                        return;
                    }
                },
                Ok(None) => return,
                Err(error) => {
                    cursor::note_error(format!("crtc_of: {error}"));
                    return;
                }
            }
        }
        let Some(plane) = self.plane.as_mut() else {
            return;
        };
        if self.new_glyph {
            if let Some(glyph) = &self.glyph {
                plane.set_image(&cursor::plane_image(glyph, plane.size()));
            }
            self.new_glyph = false;
        }
        let result = match (&self.glyph, self.show) {
            (Some(glyph), true) => {
                let (x, y) = cursor::screen_position(
                    self.pos,
                    (glyph.hot_x, glyph.hot_y),
                    frame,
                    (taken.width, taken.height),
                );
                plane.show_at(x, y)
            }
            _ => plane.hide(),
        };
        match result {
            Ok(()) => cursor::note_applied(),
            Err(error) => cursor::note_error(format!("cursor2: {error}")),
        }
        self.dirty = false;
    }
}

fn present_loop(taken: Taken, frames: FrameSlot, stop: StopFlag, wake: WindowWakeHandle) {
    let (tx, rx) = mpsc::sync_channel(1);
    let cursor_shared = cursor::shared();
    cursor_shared.set_wake(tx.clone());
    cursor_shared.set_active(true);
    let mut cursor_view = CursorView::default();
    wake.arm_channel(tx);
    let backend = crate::backend::selected();
    let mut last_presented: Option<u64> = None;
    let mut redraw = true;
    let mut error_logged = false;
    let mut first_logged = false;
    let mut rebuilds = 0u32;
    while !stop.load(Ordering::Acquire) {
        let _ = rx.recv_timeout(WINDOW_REDRAW_BACKSTOP);
        if stop.load(Ordering::Acquire) {
            break;
        }
        if serve_capture_request(Path::new(capture::REQUEST)) {
            redraw = true;
        }
        let frame = frames.lock().ok().and_then(|slot| slot.clone());
        let incoming = frame.as_ref().map(|f| f.seq);
        if needs_present(last_presented, redraw, incoming) {
            match backend.window_present(
                frame.as_ref().and_then(|f| f.resident.as_ref()),
                frame.as_deref().map(window_cpu_frame),
            ) {
                Ok(WindowPresentOutcome::Busy) => {}
                Ok(WindowPresentOutcome::Presented {
                    width,
                    height,
                    suboptimal,
                    ..
                }) => {
                    error_logged = false;
                    rebuilds = 0;
                    last_presented = incoming;
                    redraw = suboptimal;
                    if !first_logged {
                        eprintln!(
                            "reims-vgpu-display: first frame presented on {} ({width}x{height})",
                            taken.connector
                        );
                        first_logged = true;
                    }
                }
                Err(error) => {
                    if !error_logged {
                        crate::observe::Emit::decline("host_display_present", &error).fail();
                        eprintln!("reims-vgpu-display: present failed: {error}");
                        error_logged = true;
                    }
                    if error.presenter_lost() && rebuilds < backend.window_reattach_budget() {
                        rebuilds += 1;
                        let _ = backend.window_attach(&WindowSurface {
                            source: SurfaceSource::Display {
                                drm_fd: taken.card.fd(),
                                connector_id: taken.connector_id,
                                width: taken.width,
                                height: taken.height,
                                refresh_mhz: taken.refresh_mhz,
                            },
                            width: taken.width,
                            height: taken.height,
                        });
                        redraw = true;
                    }
                }
            }
        }
        if let Some(shot) = crate::backend::vulkan::engine::window_capture_take() {
            write_capture(shot);
        }
        if let Some(change) = cursor_shared.take() {
            cursor_view.update(change);
        }
        let frame_size = frame.as_ref().map_or((taken.width, taken.height), |f| {
            (f.width.max(1), f.height.max(1))
        });
        cursor_view.apply(&taken, frame_size);
    }
    cursor_shared.set_active(false);
    // The plane hides and frees its buffer while the card is still open.
    drop(cursor_view);
    backend.window_detach();
    drop(taken);
}

/// Read and remove the lab's capture request; queue the capture when the name
/// is acceptable. Returns whether a capture was queued (the next present must
/// happen even without a new frame).
fn serve_capture_request(request: &Path) -> bool {
    let Ok(contents) = std::fs::read_to_string(request) else {
        return false;
    };
    let _ = std::fs::remove_file(request);
    match capture::parse_request(&contents) {
        Ok(name) => {
            crate::backend::vulkan::engine::window_capture_next(capture::output_path(&name));
            true
        }
        Err(refusal) => {
            crate::observe::fail(format!("host_display_capture reason={refusal}"));
            eprintln!("reims-vgpu-display: capture refused: {refusal}");
            false
        }
    }
}

fn write_capture(shot: crate::backend::vulkan::engine::CapturedFrame) {
    let name = shot
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    match capture::write_ppm(&shot.path, shot.width, shot.height, &shot.bgra) {
        Ok(()) => crate::observe::off(format!(
            "host_display_capture name={name} w={} h={}",
            shot.width, shot.height
        )),
        Err(error) => crate::observe::fail(format!(
            "host_display_capture_write name={name} error={error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_display::select::SelectError;

    #[test]
    fn every_refusal_has_the_slug_the_spec_names() {
        let no_connector = DisplayError::Select(SelectError::NoConnector { wanted: None });
        let no_mode = DisplayError::Select(SelectError::NoMode {
            connector: "HDMI-A-1".into(),
        });
        let busy = DisplayError::Busy { holders: vec![42] };
        let open = DisplayError::Open(std::io::Error::from_raw_os_error(libc::ENOENT));
        assert_eq!(no_connector.slug(), "no_connector");
        assert_eq!(no_mode.slug(), "no_mode");
        assert_eq!(busy.slug(), "drm_busy");
        assert_eq!(open.slug(), "drm_open");
    }

    /// The Ryzen's TV: needs `/dev/dri` and the Radeon ICD in the container.
    /// A 64x64 CPU frame of one colour is published; the display takes the
    /// connector, letterboxes it into the mode, and a capture request returns
    /// that colour in the centre and slate at the corner.
    #[test]
    #[ignore]
    fn the_display_presents_and_captures_a_published_frame() {
        use crate::host_window::present::{Frame, WindowWaker};
        use std::sync::atomic::AtomicBool;
        use std::sync::{Arc, Mutex};

        let (w, h) = (64u32, 64u32);
        let bgra: Vec<u8> = [0x10, 0x80, 0xF0, 0xFF].repeat((w * h) as usize);
        let frames: FrameSlot = Arc::new(Mutex::new(Some(Arc::new(Frame {
            seq: 1,
            width: w,
            height: h,
            bgra,
            resident: None,
        }))));
        let stop: StopFlag = Arc::new(AtomicBool::new(false));
        let handle = spawn(frames, stop.clone(), WindowWaker::new()).expect("take the screen");
        let out = capture::output_path("m3-run-test");
        let _ = std::fs::remove_file(&out);
        std::fs::write(capture::REQUEST, "m3-run-test\n").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !out.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        stop.store(true, Ordering::Release);
        handle.join().unwrap();
        let ppm = std::fs::read(&out).expect("capture written");
        let header = b"P6\n1920 1080\n255\n";
        assert!(ppm.starts_with(header), "{:?}", &ppm[..20.min(ppm.len())]);
        let px = |x: usize, y: usize| {
            let at = header.len() + (y * 1920 + x) * 3;
            [ppm[at], ppm[at + 1], ppm[at + 2]]
        };
        assert_eq!(
            px(960, 540),
            [0xF0, 0x80, 0x10],
            "centre is the frame, as RGB"
        );
        assert_ne!(px(0, 0), [0xF0, 0x80, 0x10], "the corner is the letterbox");
    }

    #[test]
    fn every_refusal_exits_with_its_own_code() {
        let errors = [
            DisplayError::Open(std::io::Error::from_raw_os_error(libc::ENOENT)),
            DisplayError::Busy { holders: vec![] },
            DisplayError::Select(SelectError::NoConnector { wanted: None }),
            DisplayError::Select(SelectError::NoMode {
                connector: "HDMI-A-1".into(),
            }),
            attach_refusal("vk_display_acquire", "x".into()),
            attach_refusal("display_plane_missing", "x".into()),
        ];
        let mut codes: Vec<i32> = errors.iter().map(DisplayError::exit_code).collect();
        assert!(codes.iter().all(|&c| c > 2 && c < 126), "{codes:?}");
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), errors.len());
    }

    /// A present with nothing to show (no resident, no complete CPU frame) must
    /// not paint slate over what the connector already scans out: the display
    /// keeps its last image, and a capture then returns that image.
    #[test]
    #[ignore]
    fn a_present_without_a_source_keeps_the_last_scanout() {
        use crate::host_window::present::{Frame, WindowWaker};
        use std::sync::atomic::AtomicBool;
        use std::sync::{Arc, Mutex};

        let (w, h) = (64u32, 64u32);
        let frame = |seq, bgra| {
            Some(Arc::new(Frame {
                seq,
                width: w,
                height: h,
                bgra,
                resident: None,
            }))
        };
        let frames: FrameSlot = Arc::new(Mutex::new(frame(
            1,
            [0x10, 0x80, 0xF0, 0xFF].repeat((w * h) as usize),
        )));
        let stop: StopFlag = Arc::new(AtomicBool::new(false));
        let handle =
            spawn(frames.clone(), stop.clone(), WindowWaker::new()).expect("take the screen");
        let shoot = |name: &str| {
            let out = capture::output_path(name);
            let _ = std::fs::remove_file(&out);
            std::fs::write(capture::REQUEST, format!("{name}\n")).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !out.exists() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            std::fs::read(&out).expect("capture written")
        };
        let centre = |ppm: &[u8]| {
            let at = b"P6\n1920 1080\n255\n".len() + (540 * 1920 + 960) * 3;
            [ppm[at], ppm[at + 1], ppm[at + 2]]
        };
        assert_eq!(centre(&shoot("m3-keep-1")), [0xF0, 0x80, 0x10]);
        // A newer frame with no usable source: an empty CPU buffer, no resident.
        *frames.lock().unwrap() = frame(2, Vec::new());
        std::thread::sleep(std::time::Duration::from_millis(300));
        let kept = shoot("m3-keep-2");
        stop.store(true, Ordering::Release);
        handle.join().unwrap();
        assert_eq!(
            centre(&kept),
            [0xF0, 0x80, 0x10],
            "slate painted over the scanout"
        );
    }

    /// The guest's cursor reaches the CRTC's cursor plane: a published glyph
    /// and position are applied with the DRM cursor ioctls, without error.
    #[test]
    #[ignore]
    fn a_published_cursor_lands_on_the_cursor_plane() {
        use crate::host_display::cursor::{self, Glyph};
        use crate::host_window::present::{Frame, WindowWaker};
        use std::sync::atomic::AtomicBool;
        use std::sync::{Arc, Mutex};

        let frames: FrameSlot = Arc::new(Mutex::new(Some(Arc::new(Frame {
            seq: 1,
            width: 1920,
            height: 1080,
            bgra: [0x40, 0x40, 0x40, 0xFF].repeat(1920 * 1080),
            resident: None,
        }))));
        let stop: StopFlag = Arc::new(AtomicBool::new(false));
        let handle = spawn(frames, stop.clone(), WindowWaker::new()).expect("take the screen");
        std::thread::sleep(std::time::Duration::from_millis(300));
        let before = cursor::applied_count();
        cursor::publish_glyph(Glyph {
            width: 16,
            height: 16,
            hot_x: 0,
            hot_y: 0,
            pixels: vec![0xFFFF_0000; 256],
        });
        for step in 0..20u16 {
            cursor::publish_position(800 + step * 10, 500, true);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let applied = cursor::applied_count() - before;
        let error = cursor::last_error();
        stop.store(true, Ordering::Release);
        handle.join().unwrap();
        assert!(error.is_none(), "{error:?}");
        assert!(applied >= 10, "applied {applied} cursor changes");
    }

    #[test]
    fn only_the_vulkan_acquire_is_an_acquire_refusal() {
        for slug in ["vk_display_get", "vk_display_acquire"] {
            assert_eq!(attach_refusal(slug, "x".into()).slug(), "acquire_refused");
        }
        assert_eq!(
            attach_refusal("display_plane_missing", "x".into()).slug(),
            "display_attach"
        );
    }
}
