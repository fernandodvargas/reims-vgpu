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
