//! Host keyboard and mouse straight from evdev (M3 "Headless Display").
//!
//! With no window system there is no window to receive input, so the device
//! reads `/dev/input/event*` itself and takes each keyboard and mouse with
//! `EVIOCGRAB`: while the grab holds, neither the Linux console nor any other
//! process sees those events. Events become the same [`HostAction`]s the
//! `winit` window produces and go to the same queue.
//!
//! Keys pass through [`crate::host_window::keyboard::Keyboard`], so every key
//! pressed is eventually released, including when its device disappears. The
//! window's Ctrl+Alt+Esc ungrab chord is consumed the same way; here it only
//! releases the held keys, because there is no window grab to give up — the
//! real escape (switching back to the Linux console) is the supervisor's, in
//! M4.
//!
//! [`HostAction`]: crate::runtime::host::HostAction

pub mod devices;
pub mod evdev;
