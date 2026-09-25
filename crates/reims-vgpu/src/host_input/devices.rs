//! Opening, grabbing and following evdev devices.
//!
//! One thread, `reims-vgpu-input`, scans `/dev/input/event*` and then follows
//! `/dev/input` with inotify, without udev: a new node is tried on `IN_CREATE`
//! and again on `IN_ATTRIB`, which is when udev has set its permissions. Each
//! keyboard or mouse gets `EVIOCGRAB` and a reader thread of its own; a device
//! that disappears has every key and button it held released in the guest.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use super::evdev::{self, Kind, SharedPointer, Translator, EVENT_SIZE};
use crate::host_window::pointer::Pointer;
use crate::host_window::present::{InputSink, StopFlag};

const INPUT_DIR: &str = "/dev/input";
/// How often a blocked reader looks at the stop flag.
const POLL_MS: i32 = 200;
/// `_IOW('E', 0x90, int)`.
pub(crate) const EVIOCGRAB: libc::c_ulong = 0x4004_4590;
/// Bytes of the `EV_KEY` bitmap (`KEY_MAX` 0x2ff) and the `EV_REL` one
/// (`REL_MAX` 0x0f).
const KEY_BITS: usize = 96;
const REL_BITS: usize = 2;

/// `EVIOCGBIT(ev, len)`: `_IOC(_IOC_READ, 'E', 0x20 + ev, len)`.
fn eviocgbit(ev: u32, len: usize) -> libc::c_ulong {
    0x8000_4520 | ((len as libc::c_ulong) << 16) | libc::c_ulong::from(ev)
}

/// `EVIOCGNAME(len)`: `_IOC(_IOC_READ, 'E', 0x06, len)`.
fn eviocgname(len: usize) -> libc::c_ulong {
    0x8000_4506 | ((len as libc::c_ulong) << 16)
}

/// Grab every keyboard and mouse, now and as they appear, and feed their
/// events to `on_input` until `stop`. `width`x`height` is the guest frame the
/// pointer is held inside.
pub fn spawn(on_input: InputSink, width: u32, height: u32, stop: StopFlag) -> JoinHandle<()> {
    spawn_filtered(on_input, width, height, stop, None)
}

/// [`spawn`], taking only devices whose name is `only` (the tests' own uinput
/// devices, so a test never grabs the machine's keyboard).
pub(crate) fn spawn_filtered(
    on_input: InputSink,
    width: u32,
    height: u32,
    stop: StopFlag,
    only: Option<String>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("reims-vgpu-input".into())
        .spawn(move || {
            let watcher = Watcher {
                on_input,
                stop,
                only,
                attached: Arc::new(Mutex::new(HashSet::new())),
                pointer: Arc::new(Mutex::new(Pointer::new(width, height))),
                workers: Vec::new(),
            };
            watcher.run();
        })
        .expect("spawn reims-vgpu-input")
}

struct Watcher {
    on_input: InputSink,
    stop: StopFlag,
    only: Option<String>,
    attached: Arc<Mutex<HashSet<PathBuf>>>,
    /// The guest's one cursor, moved by every mouse.
    pointer: SharedPointer,
    workers: Vec<JoinHandle<()>>,
}

impl Watcher {
    fn run(mut self) {
        // Watch before the scan, so a device added in between is not missed.
        let inotify = match inotify_on(INPUT_DIR) {
            Ok(fd) => Some(fd),
            Err(error) => {
                eprintln!("reims-vgpu-input: WARN no hotplug, inotify on {INPUT_DIR}: {error}");
                None
            }
        };
        let mut nodes: Vec<PathBuf> = std::fs::read_dir(INPUT_DIR)
            .map(|dir| {
                dir.flatten()
                    .map(|e| e.path())
                    .filter(|p| is_event_node(p))
                    .collect()
            })
            .unwrap_or_default();
        nodes.sort();
        for node in nodes {
            self.try_attach(&node);
        }
        if self.only.is_none() && self.attached.lock().map_or(true, |a| a.is_empty()) {
            eprintln!("reims-vgpu-input: WARN no keyboard or mouse");
            crate::observe::off("host_input_devices count=0");
        }
        while !self.stop.load(Ordering::Acquire) {
            let Some(fd) = inotify.as_ref() else {
                std::thread::sleep(Duration::from_millis(POLL_MS as u64));
                continue;
            };
            if !readable(fd.as_raw_fd()) {
                continue;
            }
            for (name, created) in read_inotify(fd.as_raw_fd()) {
                let path = Path::new(INPUT_DIR).join(&name);
                if !is_event_node(&path) {
                    continue;
                }
                if created {
                    // Give udev a moment to set the node's permissions.
                    std::thread::sleep(Duration::from_millis(100));
                }
                self.try_attach(&path);
            }
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }

    fn try_attach(&mut self, path: &Path) {
        if self.attached.lock().map_or(true, |a| a.contains(path)) {
            return;
        }
        let Ok(file) = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)
        else {
            return; // not ours yet; IN_ATTRIB brings it back
        };
        let fd = file.as_raw_fd();
        let name = device_name(fd);
        if self.only.as_ref().is_some_and(|only| *only != name) {
            return;
        }
        let mut keys = [0u8; KEY_BITS];
        let mut rels = [0u8; REL_BITS];
        let _ = ioctl_buf(fd, eviocgbit(1, KEY_BITS), &mut keys);
        let _ = ioctl_buf(fd, eviocgbit(2, REL_BITS), &mut rels);
        let kind = evdev::classify(&keys, &rels);
        if kind == Kind::Other {
            return;
        }
        // SAFETY: EVIOCGRAB takes an int by value on a live evdev fd.
        if unsafe { libc::ioctl(fd, EVIOCGRAB as _, 1 as libc::c_int) } != 0 {
            let error = io::Error::last_os_error();
            eprintln!(
                "reims-vgpu-input: WARN cannot grab {name:?} {}: {error}",
                path.display()
            );
            crate::observe::fail(format!(
                "host_input_grab reason=grab_refused path={} errno={}",
                path.display(),
                error.raw_os_error().unwrap_or(0)
            ));
            return;
        }
        if let Ok(mut attached) = self.attached.lock() {
            attached.insert(path.to_owned());
        }
        eprintln!(
            "reims-vgpu-input: grabbed {name:?} ({kind:?}) {}",
            path.display()
        );
        crate::observe::off(format!(
            "host_input_grab kind={kind:?} path={}",
            path.display()
        ));
        let device = Device {
            file,
            path: path.to_owned(),
            name,
            translator: Translator::with_pointer(self.pointer.clone()),
            on_input: self.on_input.clone(),
            stop: self.stop.clone(),
            attached: self.attached.clone(),
        };
        let thread_name = format!(
            "reims-vgpu-input-{}",
            path.file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
        );
        match std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || device.run())
        {
            Ok(handle) => self.workers.push(handle),
            Err(error) => eprintln!("reims-vgpu-input: WARN reader thread: {error}"),
        }
    }
}

struct Device {
    file: File,
    path: PathBuf,
    name: String,
    translator: Translator,
    on_input: InputSink,
    stop: StopFlag,
    attached: Arc<Mutex<HashSet<PathBuf>>>,
}

impl Device {
    fn run(mut self) {
        let mut buf = [0u8; EVENT_SIZE * 64];
        let lost = loop {
            if self.stop.load(Ordering::Acquire) {
                break false;
            }
            if !readable(self.file.as_raw_fd()) {
                continue;
            }
            match self.file.read(&mut buf) {
                Ok(0) => break true,
                Ok(n) => {
                    for chunk in buf[..n].chunks_exact(EVENT_SIZE) {
                        let event = evdev::decode(chunk.try_into().expect("24-byte chunk"));
                        for action in self.translator.feed(event) {
                            (self.on_input)(action);
                        }
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                // ENODEV: unplugged; anything else is as good as unplugged.
                Err(_) => break true,
            }
        };
        if lost {
            for action in self.translator.lost() {
                (self.on_input)(action);
            }
            eprintln!(
                "reims-vgpu-input: lost {:?} {}",
                self.name,
                self.path.display()
            );
            crate::observe::off(format!("host_input_lost path={}", self.path.display()));
        } else {
            // SAFETY: releasing our own grab on our own live fd.
            unsafe { libc::ioctl(self.file.as_raw_fd(), EVIOCGRAB as _, 0 as libc::c_int) };
        }
        if let Ok(mut attached) = self.attached.lock() {
            attached.remove(&self.path);
        }
    }
}

fn is_event_node(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("event"))
}

fn device_name(fd: RawFd) -> String {
    let mut raw = [0u8; 256];
    match ioctl_buf(fd, eviocgname(raw.len()), &mut raw) {
        Ok(()) => {
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            String::from_utf8_lossy(&raw[..end]).into_owned()
        }
        Err(_) => String::new(),
    }
}

fn ioctl_buf(fd: RawFd, request: libc::c_ulong, buf: &mut [u8]) -> io::Result<()> {
    // SAFETY: `request` encodes `buf.len()` as its size, and `buf` is live and
    // exclusively borrowed for the call.
    if unsafe { libc::ioctl(fd, request as _, buf.as_mut_ptr()) } < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Wait up to [`POLL_MS`] for `fd` to have something to read (or an error).
fn readable(fd: RawFd) -> bool {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live pollfd.
    let rc = unsafe { libc::poll(&mut pfd, 1, POLL_MS) };
    rc > 0 && pfd.revents != 0
}

fn inotify_on(dir: &str) -> io::Result<OwnedFd> {
    // SAFETY: plain syscalls; the fd is owned right away.
    let raw = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let path = std::ffi::CString::new(dir).expect("no NUL in dir");
    let wd =
        unsafe { libc::inotify_add_watch(raw, path.as_ptr(), libc::IN_CREATE | libc::IN_ATTRIB) };
    if wd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

/// Drain pending inotify events: `(name, created)` per event.
fn read_inotify(fd: RawFd) -> Vec<(String, bool)> {
    let mut buf = [0u8; 4096];
    let mut out = Vec::new();
    loop {
        // SAFETY: reading into a live buffer of the given length.
        let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
        if n <= 0 {
            return out;
        }
        let mut at = 0usize;
        let n = n as usize;
        // struct inotify_event: wd i32, mask u32, cookie u32, len u32, name[len].
        while at + 16 <= n {
            let mask = u32::from_ne_bytes(buf[at + 4..at + 8].try_into().unwrap());
            let len = u32::from_ne_bytes(buf[at + 12..at + 16].try_into().unwrap()) as usize;
            let name_bytes = &buf[at + 16..(at + 16 + len).min(n)];
            let end = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            out.push((
                String::from_utf8_lossy(&name_bytes[..end]).into_owned(),
                mask & libc::IN_CREATE != 0,
            ));
            at += 16 + len;
        }
    }
}

#[cfg(test)]
mod tests {
    //! On the Ryzen, as root, with `/dev/uinput` and the host's `/dev/input`:
    //! each test makes its own uinput keyboard and only grabs that one.
    use super::*;
    use crate::runtime::host::HostAction;
    use std::os::fd::AsRawFd;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const UI_SET_EVBIT: libc::c_ulong = 0x4004_5564;
    const UI_SET_KEYBIT: libc::c_ulong = 0x4004_5565;
    const UI_DEV_CREATE: libc::c_ulong = 0x5501;
    const UI_DEV_DESTROY: libc::c_ulong = 0x5502;

    struct Uinput(std::fs::File);

    impl Uinput {
        fn keyboard(name: &str) -> Self {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open("/dev/uinput")
                .expect("/dev/uinput");
            let fd = file.as_raw_fd();
            unsafe {
                assert_eq!(libc::ioctl(fd, UI_SET_EVBIT as _, 1), 0); // EV_KEY
                for key in [28, 29, 30] {
                    assert_eq!(libc::ioctl(fd, UI_SET_KEYBIT as _, key), 0);
                }
            }
            let mut dev = vec![0u8; 1116];
            dev[..name.len()].copy_from_slice(name.as_bytes());
            dev[80..82].copy_from_slice(&3u16.to_ne_bytes()); // BUS_USB
            use std::io::Write;
            (&file).write_all(&dev).unwrap();
            assert_eq!(unsafe { libc::ioctl(fd, UI_DEV_CREATE as _) }, 0);
            Self(file)
        }

        fn emit(&self, kind: u16, code: u16, value: i32) {
            let mut raw = [0u8; 24];
            raw[16..18].copy_from_slice(&kind.to_ne_bytes());
            raw[18..20].copy_from_slice(&code.to_ne_bytes());
            raw[20..24].copy_from_slice(&value.to_ne_bytes());
            use std::io::Write;
            (&self.0).write_all(&raw).unwrap();
        }

        fn key(&self, code: u16, down: bool) {
            self.emit(1, code, i32::from(down));
            self.emit(0, 0, 0);
        }

        fn destroy(self) {
            unsafe { libc::ioctl(self.0.as_raw_fd(), UI_DEV_DESTROY as _) };
        }
    }

    /// The event node the kernel made for the device called `name`.
    fn node(name: &str) -> std::path::PathBuf {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            for entry in std::fs::read_dir("/sys/class/input").unwrap().flatten() {
                let file = entry.file_name().to_string_lossy().into_owned();
                if !file.starts_with("event") {
                    continue;
                }
                let dev_name =
                    std::fs::read_to_string(entry.path().join("device/name")).unwrap_or_default();
                if dev_name.trim() == name {
                    return std::path::Path::new("/dev/input").join(file);
                }
            }
            assert!(Instant::now() < deadline, "no event node for {name}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Whether another `EVIOCGRAB` on `path` is refused with `EBUSY`.
    fn grabbed_by_someone(path: &std::path::Path) -> bool {
        let file = std::fs::File::open(path).unwrap();
        let rc = unsafe { libc::ioctl(file.as_raw_fd(), EVIOCGRAB as _, 1 as libc::c_int) };
        if rc == 0 {
            unsafe { libc::ioctl(file.as_raw_fd(), EVIOCGRAB as _, 0 as libc::c_int) };
            return false;
        }
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EBUSY)
    }

    fn wait_for(what: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !what() {
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        true
    }

    fn collector() -> (
        crate::host_window::present::InputSink,
        Arc<Mutex<Vec<HostAction>>>,
    ) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink_seen = seen.clone();
        let sink: crate::host_window::present::InputSink =
            Arc::new(move |action| sink_seen.lock().unwrap().push(action));
        (sink, seen)
    }

    #[test]
    #[ignore]
    fn a_hotplugged_keyboard_is_grabbed_translated_and_released_on_stop() {
        let name = "lumiere-test-kbd-hotplug";
        let (sink, seen) = collector();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = spawn_filtered(sink, 1920, 1080, stop.clone(), Some(name.into()));
        let kbd = Uinput::keyboard(name);
        let path = node(name);
        assert!(
            wait_for(|| grabbed_by_someone(&path)),
            "grab never happened"
        );
        kbd.key(30, true);
        kbd.key(30, false);
        let expected = vec![
            HostAction::input_key(30, true),
            HostAction::input_key(30, false),
        ];
        assert!(
            wait_for(|| *seen.lock().unwrap() == expected),
            "{:?}",
            seen.lock().unwrap()
        );
        stop.store(true, Ordering::Release);
        handle.join().unwrap();
        assert!(!grabbed_by_someone(&path), "grab kept after stop");
        kbd.destroy();
    }

    /// Gate D2's second half: the lab's own injector (`scripts/uinput_inject.py
    /// type a`, run beside this test) reaches `host_input`.
    #[test]
    #[ignore]
    fn d2_the_lab_injector_reaches_host_input() {
        let (sink, seen) = collector();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = spawn_filtered(
            sink,
            1920,
            1080,
            stop.clone(),
            Some("lumiere-uinput-kbd".into()),
        );
        let expected = [
            HostAction::input_key(30, true),
            HostAction::input_key(30, false),
        ];
        let deadline = Instant::now() + Duration::from_secs(30);
        while !seen.lock().unwrap().ends_with(&expected) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        stop.store(true, Ordering::Release);
        handle.join().unwrap();
        assert!(
            seen.lock().unwrap().ends_with(&expected),
            "{:?}",
            seen.lock().unwrap()
        );
    }

    #[test]
    #[ignore]
    fn a_keyboard_that_disappears_releases_its_held_keys() {
        let name = "lumiere-test-kbd-lost";
        let (sink, seen) = collector();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = spawn_filtered(sink, 1920, 1080, stop.clone(), Some(name.into()));
        let kbd = Uinput::keyboard(name);
        let path = node(name);
        assert!(
            wait_for(|| grabbed_by_someone(&path)),
            "grab never happened"
        );
        kbd.key(29, true); // LEFTCTRL, never released by the device
        assert!(wait_for(|| seen.lock().unwrap().len() == 1));
        kbd.destroy();
        assert!(
            wait_for(|| seen
                .lock()
                .unwrap()
                .contains(&HostAction::input_key(29, false))),
            "{:?}",
            seen.lock().unwrap()
        );
        stop.store(true, Ordering::Release);
        handle.join().unwrap();
    }
}
