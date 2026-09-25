//! The few DRM ioctls the display needs: resources, connectors and their modes.
//!
//! None of these need DRM master: they only read. Structs are the `#[repr(C)]`
//! layouts of `drm_mode.h`, and each list ioctl is called twice, the first time
//! to count and the second to fill.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use super::select::{Connector, Mode};

const DRM_IOCTL_MODE_GETRESOURCES: libc::c_ulong = 0xC040_64A0;
const DRM_IOCTL_MODE_GETCONNECTOR: libc::c_ulong = 0xC050_64A7;
/// `DRM_IOW(0x11, struct drm_auth)`.
const DRM_IOCTL_AUTH_MAGIC: libc::c_ulong = 0x4004_6411;
const DRM_IOCTL_GET_CAP: libc::c_ulong = 0xC010_640C;
const DRM_IOCTL_MODE_GETENCODER: libc::c_ulong = 0xC014_64A6;
const DRM_IOCTL_MODE_CREATE_DUMB: libc::c_ulong = 0xC020_64B2;
const DRM_IOCTL_MODE_MAP_DUMB: libc::c_ulong = 0xC010_64B3;
const DRM_IOCTL_MODE_DESTROY_DUMB: libc::c_ulong = 0xC004_64B4;
const DRM_IOCTL_MODE_CURSOR2: libc::c_ulong = 0xC024_64BB;
const DRM_CAP_CURSOR_WIDTH: u64 = 0x8;
const DRM_MODE_CURSOR_BO: u32 = 0x01;
const DRM_MODE_CURSOR_MOVE: u32 = 0x02;
const DRM_MODE_TYPE_PREFERRED: u32 = 1 << 3;
const DRM_MODE_CONNECTED: u32 = 1;
const MODEINFO_SIZE: usize = 68;

#[repr(C)]
#[derive(Default)]
struct CardRes {
    fb_id_ptr: u64,
    crtc_id_ptr: u64,
    connector_id_ptr: u64,
    encoder_id_ptr: u64,
    count_fbs: u32,
    count_crtcs: u32,
    count_connectors: u32,
    count_encoders: u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
}

#[repr(C)]
#[derive(Default)]
struct GetConnector {
    encoders_ptr: u64,
    modes_ptr: u64,
    props_ptr: u64,
    prop_values_ptr: u64,
    count_modes: u32,
    count_props: u32,
    count_encoders: u32,
    encoder_id: u32,
    connector_id: u32,
    connector_type: u32,
    connector_type_id: u32,
    connection: u32,
    mm_width: u32,
    mm_height: u32,
    subpixel: u32,
    pad: u32,
}

/// An open DRM card node.
pub struct Card {
    file: File,
}

impl Card {
    /// Open the card read-write, close-on-exec.
    pub fn open(path: &Path) -> io::Result<Card> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(path)?;
        Ok(Card { file })
    }

    pub fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    /// Whether this open file is the card's DRM master, the way libdrm's
    /// `drmIsMaster` asks: `DRM_IOCTL_AUTH_MAGIC` is master-only, so it fails
    /// with `EACCES` for anyone else, and magic 0 is never valid, so a master
    /// gets a harmless `EINVAL`. The first opener of a card with no master
    /// becomes master; a card another process already drives stays theirs.
    pub fn is_master(&self) -> bool {
        let mut magic: u32 = 0;
        !matches!(
            ioctl(self.fd(), DRM_IOCTL_AUTH_MAGIC, &mut magic),
            Err(error) if error.raw_os_error() == Some(libc::EACCES)
        )
    }

    /// Every connector of the card, with its modes in kernel order.
    pub fn connectors(&self) -> io::Result<Vec<Connector>> {
        let ids = self.connector_ids()?;
        ids.into_iter().map(|id| self.connector(id)).collect()
    }

    fn connector_ids(&self) -> io::Result<Vec<u32>> {
        loop {
            let mut res = CardRes::default();
            ioctl(self.fd(), DRM_IOCTL_MODE_GETRESOURCES, &mut res)?;
            let count = res.count_connectors;
            let mut ids = vec![0u32; count as usize];
            let mut fill = CardRes {
                connector_id_ptr: ids.as_mut_ptr() as u64,
                count_connectors: count,
                ..CardRes::default()
            };
            ioctl(self.fd(), DRM_IOCTL_MODE_GETRESOURCES, &mut fill)?;
            // A connector hot-added between the two calls: count again.
            if fill.count_connectors <= count {
                ids.truncate(fill.count_connectors as usize);
                return Ok(ids);
            }
        }
    }

    fn connector(&self, id: u32) -> io::Result<Connector> {
        loop {
            // count_modes == 0 makes the kernel probe the connector first.
            let mut probe = GetConnector {
                connector_id: id,
                ..GetConnector::default()
            };
            ioctl(self.fd(), DRM_IOCTL_MODE_GETCONNECTOR, &mut probe)?;
            let count = probe.count_modes;
            let mut raw = vec![0u8; count as usize * MODEINFO_SIZE];
            let mut fill = GetConnector {
                connector_id: id,
                modes_ptr: raw.as_mut_ptr() as u64,
                count_modes: count,
                ..GetConnector::default()
            };
            ioctl(self.fd(), DRM_IOCTL_MODE_GETCONNECTOR, &mut fill)?;
            if fill.count_modes > count {
                continue;
            }
            let modes = raw
                .chunks_exact(MODEINFO_SIZE)
                .take(fill.count_modes as usize)
                .enumerate()
                .map(|(index, chunk)| decode_mode(chunk.try_into().expect("68-byte chunk"), index))
                .collect();
            return Ok(Connector {
                id,
                name: connector_name(fill.connector_type, fill.connector_type_id),
                connected: fill.connection == DRM_MODE_CONNECTED,
                modes,
            });
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct GetEncoder {
    encoder_id: u32,
    encoder_type: u32,
    crtc_id: u32,
    possible_crtcs: u32,
    possible_clones: u32,
}

#[repr(C)]
#[derive(Default)]
struct CreateDumb {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32,
    handle: u32,
    pitch: u32,
    size: u64,
}

#[repr(C)]
#[derive(Default)]
struct MapDumb {
    handle: u32,
    pad: u32,
    offset: u64,
}

#[repr(C)]
#[derive(Default)]
struct Cursor2 {
    flags: u32,
    crtc_id: u32,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    handle: u32,
    hot_x: i32,
    hot_y: i32,
}

#[repr(C)]
#[derive(Default)]
struct GetCap {
    capability: u64,
    value: u64,
}

impl Card {
    /// The CRTC driving `connector_id` right now; `None` before the first
    /// modeset has bound one.
    pub fn crtc_of(&self, connector_id: u32) -> io::Result<Option<u32>> {
        let mut conn = GetConnector {
            connector_id,
            ..GetConnector::default()
        };
        // count_modes stays 1 so the kernel does not re-probe the connector.
        let mut one_mode = [0u8; MODEINFO_SIZE];
        conn.modes_ptr = one_mode.as_mut_ptr() as u64;
        conn.count_modes = 1;
        ioctl(self.fd(), DRM_IOCTL_MODE_GETCONNECTOR, &mut conn)?;
        if conn.encoder_id == 0 {
            return Ok(None);
        }
        let mut enc = GetEncoder {
            encoder_id: conn.encoder_id,
            ..GetEncoder::default()
        };
        ioctl(self.fd(), DRM_IOCTL_MODE_GETENCODER, &mut enc)?;
        Ok((enc.crtc_id != 0).then_some(enc.crtc_id))
    }

    /// A cursor plane on `crtc_id`: a dumb ARGB8888 buffer of the size the
    /// driver's cursor takes (`DRM_CAP_CURSOR_WIDTH`, 64 when unsaid), mapped.
    pub fn cursor_plane(&self, crtc_id: u32) -> io::Result<CursorPlane> {
        let mut cap = GetCap {
            capability: DRM_CAP_CURSOR_WIDTH,
            value: 0,
        };
        let size = match ioctl(self.fd(), DRM_IOCTL_GET_CAP, &mut cap) {
            Ok(()) if (1..=512).contains(&cap.value) => cap.value as u32,
            _ => 64,
        };
        let mut dumb = CreateDumb {
            height: size,
            width: size,
            bpp: 32,
            ..CreateDumb::default()
        };
        ioctl(self.fd(), DRM_IOCTL_MODE_CREATE_DUMB, &mut dumb)?;
        let mut map = MapDumb {
            handle: dumb.handle,
            ..MapDumb::default()
        };
        let destroy = |fd, handle| {
            let mut d = handle;
            let _ = ioctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &mut d);
        };
        if let Err(error) = ioctl(self.fd(), DRM_IOCTL_MODE_MAP_DUMB, &mut map) {
            destroy(self.fd(), dumb.handle);
            return Err(error);
        }
        // SAFETY: mapping the dumb buffer the kernel just made, at the offset it
        // named, for the size it reported.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                dumb.size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.fd(),
                map.offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            let error = io::Error::last_os_error();
            destroy(self.fd(), dumb.handle);
            return Err(error);
        }
        Ok(CursorPlane {
            fd: self.fd(),
            crtc_id,
            size,
            handle: dumb.handle,
            pitch: dumb.pitch,
            map: ptr.cast(),
            map_len: dumb.size as usize,
            shown: false,
        })
    }
}

/// The CRTC's cursor plane, fed from a mapped dumb buffer. Hidden and freed on
/// drop. Borrows the card's fd: the [`Card`] must outlive it.
pub struct CursorPlane {
    fd: RawFd,
    crtc_id: u32,
    size: u32,
    handle: u32,
    pitch: u32,
    map: *mut u8,
    map_len: usize,
    shown: bool,
}

// SAFETY: the mapping is owned by this value alone and only written through
// `&mut self`.
unsafe impl Send for CursorPlane {}

impl CursorPlane {
    /// Side of the square image the plane takes.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Write a `size`x`size` ARGB8888 image into the buffer.
    pub fn set_image(&mut self, argb: &[u32]) {
        let size = self.size as usize;
        for (y, row) in argb.chunks_exact(size).take(size).enumerate() {
            let at = y * self.pitch as usize;
            if at + size * 4 > self.map_len {
                break;
            }
            // SAFETY: `at + size*4` is inside the mapping, checked above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr().cast::<u8>(),
                    self.map.add(at),
                    size * 4,
                );
            }
        }
        // The next show must hand the buffer over again for the new image.
        self.shown = false;
    }

    /// Show the image with its top-left at `(x, y)` on the CRTC.
    pub fn show_at(&mut self, x: i32, y: i32) -> io::Result<()> {
        let mut cursor = Cursor2 {
            flags: if self.shown {
                DRM_MODE_CURSOR_MOVE
            } else {
                DRM_MODE_CURSOR_BO | DRM_MODE_CURSOR_MOVE
            },
            crtc_id: self.crtc_id,
            x,
            y,
            width: self.size,
            height: self.size,
            handle: self.handle,
            ..Cursor2::default()
        };
        ioctl(self.fd, DRM_IOCTL_MODE_CURSOR2, &mut cursor)?;
        self.shown = true;
        Ok(())
    }

    pub fn hide(&mut self) -> io::Result<()> {
        let mut cursor = Cursor2 {
            flags: DRM_MODE_CURSOR_BO,
            crtc_id: self.crtc_id,
            ..Cursor2::default()
        };
        ioctl(self.fd, DRM_IOCTL_MODE_CURSOR2, &mut cursor)?;
        self.shown = false;
        Ok(())
    }
}

impl Drop for CursorPlane {
    fn drop(&mut self) {
        let _ = self.hide();
        // SAFETY: unmapping the mapping this value made.
        unsafe { libc::munmap(self.map.cast(), self.map_len) };
        let mut handle = self.handle;
        let _ = ioctl(self.fd, DRM_IOCTL_MODE_DESTROY_DUMB, &mut handle);
    }
}

fn ioctl<T>(fd: RawFd, request: libc::c_ulong, arg: &mut T) -> io::Result<()> {
    loop {
        // SAFETY: `arg` is a live, exclusively borrowed `#[repr(C)]` struct of the
        // size encoded in `request`, and any pointers inside it address buffers
        // the caller keeps alive for the call.
        let rc = unsafe { libc::ioctl(fd, request as _, arg as *mut T) };
        if rc == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EINTR) | Some(libc::EAGAIN) => continue,
            _ => return Err(err),
        }
    }
}

/// The kernel's name for a connector: `drm_connector_enum_list` plus the
/// per-type index.
pub fn connector_name(kind: u32, kind_id: u32) -> String {
    const NAMES: [&str; 21] = [
        "Unknown",
        "VGA",
        "DVI-I",
        "DVI-D",
        "DVI-A",
        "Composite",
        "SVIDEO",
        "LVDS",
        "Component",
        "DIN",
        "DP",
        "HDMI-A",
        "HDMI-B",
        "TV",
        "eDP",
        "Virtual",
        "DSI",
        "DPI",
        "Writeback",
        "SPI",
        "USB",
    ];
    let name = NAMES.get(kind as usize).copied().unwrap_or("Unknown");
    format!("{name}-{kind_id}")
}

/// Decode one `struct drm_mode_modeinfo`.
pub fn decode_mode(raw: &[u8; MODEINFO_SIZE], index: usize) -> Mode {
    let u16_at = |o: usize| u16::from_ne_bytes([raw[o], raw[o + 1]]);
    let u32_at = |o: usize| u32::from_ne_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
    Mode {
        width: u16_at(4),
        height: u16_at(14),
        refresh_mhz: u32_at(24).saturating_mul(1000),
        preferred: u32_at(32) & DRM_MODE_TYPE_PREFERRED != 0,
        index,
    }
}

/// Pids, other than this process, that hold `path` open, found through
/// `/proc/*/fd`. Only the processes this pid namespace can see.
pub fn holders(path: &Path) -> Vec<u32> {
    let Ok(card) = path.canonicalize() else {
        return Vec::new();
    };
    let me = std::process::id();
    let Ok(procs) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut pids: Vec<u32> = procs
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .filter(|&pid| pid != me)
        .filter(|pid| {
            std::fs::read_dir(format!("/proc/{pid}/fd")).is_ok_and(|fds| {
                fds.flatten()
                    .any(|fd| std::fs::read_link(fd.path()).is_ok_and(|target| target == card))
            })
        })
        .collect();
    pids.sort_unstable();
    pids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_connectors_like_the_kernel() {
        assert_eq!(connector_name(11, 1), "HDMI-A-1");
        assert_eq!(connector_name(10, 1), "DP-1");
        assert_eq!(connector_name(14, 2), "eDP-2");
        assert_eq!(connector_name(99, 3), "Unknown-3");
    }

    #[test]
    fn a_mode_info_decodes_size_refresh_and_preference() {
        // struct drm_mode_modeinfo, 68 bytes: clock, hdisplay.., vrefresh, flags, type, name[32]
        let mut raw = [0u8; 68];
        raw[0..4].copy_from_slice(&148_500u32.to_ne_bytes());
        raw[4..6].copy_from_slice(&1920u16.to_ne_bytes());
        raw[14..16].copy_from_slice(&1080u16.to_ne_bytes());
        raw[24..28].copy_from_slice(&60u32.to_ne_bytes());
        raw[32..36].copy_from_slice(&(1u32 << 3).to_ne_bytes()); // DRM_MODE_TYPE_PREFERRED
        let mode = decode_mode(&raw, 4);
        assert_eq!(
            (
                mode.width,
                mode.height,
                mode.refresh_mhz,
                mode.preferred,
                mode.index
            ),
            (1920, 1080, 60_000, true, 4)
        );
    }

    /// The Ryzen's TV: needs `/dev/dri/card1` in the container.
    #[test]
    #[ignore]
    fn the_ryzen_card_shows_the_tv_on_hdmi() {
        let card = Card::open(Path::new("/dev/dri/card1")).unwrap();
        let connectors = card.connectors().unwrap();
        let hdmi = connectors.iter().find(|c| c.name == "HDMI-A-1").unwrap();
        assert!(hdmi.connected, "{connectors:?}");
        assert!(hdmi
            .modes
            .iter()
            .any(|m| (m.width, m.height) == (1920, 1080)));
    }
}
