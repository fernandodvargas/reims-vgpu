//! The guest's hardware cursor on the connector's cursor plane.
//!
//! macOS draws its pointer as a hardware cursor: the glyph and the position
//! travel apart from the frame, and the device hands both to QEMU's console
//! (`HostAction::cursor*`). In a window the host's own pointer stood in for it;
//! on a bare connector nothing draws it unless this does. The drain publishes
//! each change here ([`publish_position`], [`publish_glyph`]), the display loop
//! takes the latest ([`take`]) and puts it on the CRTC's cursor plane with the
//! legacy DRM cursor ioctls, beside the Vulkan scanout — the way a real Mac's
//! cursor is a plane of its own.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex};

/// One cursor image, as the device holds it: `0xAARRGGBB` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glyph {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub pixels: Vec<u32>,
}

/// What changed since the display loop last looked. `glyph` is `Some` only
/// when a new image arrived.
#[derive(Clone, Debug)]
pub struct CursorChange {
    pub x: u16,
    pub y: u16,
    pub show: bool,
    pub glyph: Option<Arc<Glyph>>,
}

#[derive(Default)]
struct State {
    x: u16,
    y: u16,
    show: bool,
    glyph: Option<Arc<Glyph>>,
    moved: bool,
    new_glyph: bool,
}

/// The publication point between the drain and the display loop.
#[derive(Default)]
pub struct Shared {
    /// A display loop is running; publications before that are dropped.
    active: AtomicBool,
    state: Mutex<State>,
    wake: Mutex<Option<SyncSender<()>>>,
}

impl Shared {
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Release);
        if !active {
            *self.lock() = State::default();
            *self.wake.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    pub fn set_wake(&self, tx: SyncSender<()>) {
        *self.wake.lock().unwrap_or_else(|e| e.into_inner()) = Some(tx);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn woken(&self) {
        if let Some(tx) = self.wake.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let _ = tx.try_send(());
        }
    }

    pub fn position(&self, x: u16, y: u16, show: bool) {
        if !self.active.load(Ordering::Acquire) {
            return;
        }
        {
            let mut state = self.lock();
            if (state.x, state.y, state.show) == (x, y, show) && !state.moved {
                return;
            }
            (state.x, state.y, state.show) = (x, y, show);
            state.moved = true;
        }
        self.woken();
    }

    pub fn glyph(&self, glyph: Glyph) {
        if !self.active.load(Ordering::Acquire) {
            return;
        }
        {
            let mut state = self.lock();
            state.glyph = Some(Arc::new(glyph));
            state.new_glyph = true;
        }
        self.woken();
    }

    /// The latest cursor, once per change.
    pub fn take(&self) -> Option<CursorChange> {
        let mut state = self.lock();
        if !state.moved && !state.new_glyph {
            return None;
        }
        let change = CursorChange {
            x: state.x,
            y: state.y,
            show: state.show,
            glyph: if state.new_glyph {
                state.glyph.clone()
            } else {
                None
            },
        };
        state.moved = false;
        state.new_glyph = false;
        Some(change)
    }
}

/// The process's one publication point.
pub fn shared() -> &'static Shared {
    static SHARED: std::sync::OnceLock<Shared> = std::sync::OnceLock::new();
    SHARED.get_or_init(Shared::default)
}

static APPLIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Cursor changes the display loop has put on the plane.
pub fn applied_count() -> u64 {
    APPLIED.load(Ordering::Relaxed)
}

/// The last cursor-plane failure, if any.
pub fn last_error() -> Option<String> {
    LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub(crate) fn note_applied() {
    APPLIED.fetch_add(1, Ordering::Relaxed);
}

/// Logged once per distinct message: a plane that refuses keeps refusing.
pub(crate) fn note_error(message: String) {
    let mut last = LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner());
    if last.as_deref() != Some(message.as_str()) {
        crate::observe::fail(format!(
            "host_display_cursor reason=plane_refused detail={}",
            crate::host_window::present::detail_field(&message)
        ));
        eprintln!("reims-vgpu-display: cursor plane: {message}");
        *last = Some(message);
    }
}

/// The drain's cursor position (`HostAction::cursor`), for the display.
pub fn publish_position(x: u16, y: u16, show: bool) {
    shared().position(x, y, show);
}

/// The drain's new cursor image (`HostAction::cursor_glyph`), for the display.
pub fn publish_glyph(glyph: Glyph) {
    shared().glyph(glyph);
}

/// `glyph` in the top-left corner of a `size`x`size` cursor-plane image,
/// transparent elsewhere; a glyph larger than the plane is cropped.
pub fn plane_image(glyph: &Glyph, size: u32) -> Vec<u32> {
    let size = size as usize;
    let mut image = vec![0u32; size * size];
    let (w, h) = (usize::from(glyph.width), usize::from(glyph.height));
    if glyph.pixels.len() < w * h {
        return image; // a malformed glyph shows nothing rather than a torn one
    }
    for y in 0..h.min(size) {
        let row = &glyph.pixels[y * w..y * w + w];
        image[y * size..y * size + w.min(size)].copy_from_slice(&row[..w.min(size)]);
    }
    image
}

/// Where the plane's top-left goes on the mode so the glyph's hotspot sits on
/// the guest position `pos`: through the same aspect-fit the frame's blit
/// uses (`backend::window::viewport`), then back by the hotspot. The plane is
/// not scaled, so on a scaled frame the glyph keeps its guest size.
pub fn screen_position(
    pos: (u16, u16),
    hot: (u16, u16),
    frame: (u32, u32),
    mode: (u32, u32),
) -> (i32, i32) {
    let vp = crate::backend::window::viewport::aspect_fit(frame, mode);
    let scale = |p: u16, offset: u32, span: u32, full: u32| {
        i64::from(offset) + i64::from(p) * i64::from(span) / i64::from(full.max(1))
    };
    let x = scale(pos.0, vp.x, vp.width, frame.0) - i64::from(hot.0);
    let y = scale(pos.1, vp.y, vp.height, frame.1) - i64::from(hot.1);
    (x as i32, y as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glyph(w: u16, h: u16, hot: (u16, u16), px: u32) -> Glyph {
        Glyph {
            width: w,
            height: h,
            hot_x: hot.0,
            hot_y: hot.1,
            pixels: vec![px; usize::from(w) * usize::from(h)],
        }
    }

    #[test]
    fn a_small_glyph_sits_top_left_in_the_plane_image() {
        let image = plane_image(&glyph(2, 2, (0, 0), 0xFF11_2233), 64);
        assert_eq!(image.len(), 64 * 64);
        assert_eq!(&image[0..3], &[0xFF11_2233, 0xFF11_2233, 0]);
        assert_eq!(&image[64..66], &[0xFF11_2233, 0xFF11_2233]);
        assert_eq!(image[2 * 64], 0);
    }

    #[test]
    fn a_glyph_wider_than_the_plane_is_cropped_not_wrapped() {
        let image = plane_image(&glyph(70, 1, (0, 0), 0xFFFF_FFFF), 64);
        assert!(image[..64].iter().all(|&p| p == 0xFFFF_FFFF));
        assert!(image[64..].iter().all(|&p| p == 0));
    }

    #[test]
    fn the_hotspot_lands_on_the_guest_position_on_an_unscaled_frame() {
        // Guest frame = mode: no letterbox, no scale.
        assert_eq!(
            screen_position((500, 300), (4, 2), (1920, 1080), (1920, 1080)),
            (496, 298)
        );
    }

    #[test]
    fn a_letterboxed_frame_offsets_and_scales_the_position() {
        // 1440x900 (16:10) fits 1920x1080 as 1728x1080 at x=96.
        assert_eq!(
            screen_position((720, 450), (4, 4), (1440, 900), (1920, 1080)),
            (956, 536)
        );
    }

    #[test]
    fn a_hotspot_past_the_left_edge_goes_negative() {
        assert_eq!(
            screen_position((0, 0), (5, 3), (1920, 1080), (1920, 1080)),
            (-5, -3)
        );
    }

    #[test]
    fn a_publication_is_taken_once_per_change() {
        let shared = Shared::default();
        shared.set_active(true);
        assert!(shared.take().is_none());
        shared.position(10, 20, true);
        let seen = shared.take().expect("a new position");
        assert_eq!((seen.x, seen.y, seen.show), (10, 20, true));
        assert!(seen.glyph.is_none());
        assert!(shared.take().is_none());
        shared.glyph(glyph(1, 1, (0, 0), 1));
        assert!(shared.take().expect("a new glyph").glyph.is_some());
    }

    #[test]
    fn nothing_is_kept_while_no_display_runs() {
        let shared = Shared::default();
        shared.position(10, 20, true);
        shared.set_active(true);
        assert!(shared.take().is_none());
    }
}
