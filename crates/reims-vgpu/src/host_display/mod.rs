//! Direct-to-display presentation (M3 "Headless Display").
//!
//! The guest's frames go straight to a DRM connector, with this process as the
//! DRM master: no Wayland, no X, no compositor. The card is `/dev/dri/card1`
//! unless `REIMS_VGPU_DRM_CARD` names another. The connector is the first one
//! whose state is `connected`, or the one `REIMS_VGPU_CONNECTOR` names (for
//! example `HDMI-A-1`). The mode is 1920x1080 when the connector offers it,
//! otherwise the preferred mode, otherwise the first one listed.
//!
//! `vkAcquireDrmDisplayEXT` turns the connector into a `VkDisplayKHR`; the plane
//! and mode come from `VK_KHR_display`, and the display plane surface goes to
//! the same `Backend::window_attach` the `winit` window uses. The publish →
//! fit → present loop and the letterbox do not change: only the origin of the
//! surface does.
//!
//! Every failure to take the screen is a typed refusal with its own slug, never
//! a silent fall back to the window or to QEMU's console.

pub mod capture;
pub mod cursor;
pub mod drm;
pub mod run;
pub mod select;
