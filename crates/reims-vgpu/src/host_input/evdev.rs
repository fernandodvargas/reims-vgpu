//! evdev events → guest input. Pure: bytes and bitmaps in, `HostAction`s out.

use crate::host_window::keyboard::Keyboard;
use crate::host_window::pointer::Pointer;
use crate::runtime::host::HostAction;
use crate::runtime::input::ReimsVgpuButton;
use std::sync::{Arc, Mutex};

/// The one guest cursor every mouse moves.
pub type SharedPointer = Arc<Mutex<Pointer>>;

const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_REL: u16 = 0x02;
const SYN_REPORT: u16 = 0;
const REL_X: u16 = 0x00;
const REL_Y: u16 = 0x01;
const REL_HWHEEL: u16 = 0x06;
const REL_WHEEL: u16 = 0x08;
const KEY_ENTER: u16 = 28;
const KEY_A: u16 = 30;
const BTN_MISC: u16 = 0x100;
const BTN_LEFT: u16 = 0x110;
const BTN_RIGHT: u16 = 0x111;
const BTN_MIDDLE: u16 = 0x112;
const BTN_SIDE: u16 = 0x113;
const BTN_EXTRA: u16 = 0x114;
/// Past the button block (`BTN_MISC` .. `BTN_DIGI` and the rest up to
/// `KEY_OK`), codes are keys again.
const KEY_OK: u16 = 0x160;

/// One `struct input_event`, without its timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputEvent {
    pub kind: u16,
    pub code: u16,
    pub value: i32,
}

/// Size of `struct input_event` on a 64-bit kernel: `timeval` (16) + type,
/// code (2 + 2) + value (4).
pub const EVENT_SIZE: usize = 24;

/// Decode one 64-bit `struct input_event`.
pub fn decode(raw: &[u8; EVENT_SIZE]) -> InputEvent {
    InputEvent {
        kind: u16::from_ne_bytes([raw[16], raw[17]]),
        code: u16::from_ne_bytes([raw[18], raw[19]]),
        value: i32::from_ne_bytes([raw[20], raw[21], raw[22], raw[23]]),
    }
}

/// What a device is, by what it can send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Keyboard,
    Mouse,
    Other,
}

fn bit(bitmap: &[u8], code: u16) -> bool {
    bitmap
        .get(usize::from(code / 8))
        .is_some_and(|byte| byte & (1 << (code % 8)) != 0)
}

/// Classify by the `EVIOCGBIT` bitmaps of `EV_KEY` and `EV_REL`: a keyboard has
/// letter and Enter keys; a mouse has a left button and relative X and Y.
pub fn classify(keys: &[u8], rels: &[u8]) -> Kind {
    if bit(keys, KEY_A) && bit(keys, KEY_ENTER) {
        Kind::Keyboard
    } else if bit(keys, BTN_LEFT) && bit(rels, REL_X) && bit(rels, REL_Y) {
        Kind::Mouse
    } else {
        Kind::Other
    }
}

fn button(code: u16) -> Option<ReimsVgpuButton> {
    Some(match code {
        BTN_LEFT => ReimsVgpuButton::Left,
        BTN_RIGHT => ReimsVgpuButton::Right,
        BTN_MIDDLE => ReimsVgpuButton::Middle,
        BTN_SIDE => ReimsVgpuButton::Side,
        BTN_EXTRA => ReimsVgpuButton::Extra,
        _ => return None,
    })
}

/// One device's events → guest input: keys through [`Keyboard`], relative
/// motion through [`Pointer`] one move per `SYN_REPORT`, buttons and wheel
/// notches as neutral buttons.
pub struct Translator {
    keyboard: Keyboard,
    /// Shared by every device: the guest has one cursor, and a mouse moved
    /// after another carries on from where that one left it.
    pointer: SharedPointer,
    pending: (i32, i32),
    moved: bool,
    buttons_held: Vec<ReimsVgpuButton>,
}

impl Translator {
    pub fn new(width: u32, height: u32) -> Self {
        Self::with_pointer(Arc::new(Mutex::new(Pointer::new(width, height))))
    }

    /// A translator moving the given shared pointer.
    pub fn with_pointer(pointer: SharedPointer) -> Self {
        Self {
            keyboard: Keyboard::new(),
            pointer,
            pending: (0, 0),
            moved: false,
            buttons_held: Vec::new(),
        }
    }

    /// Feed one event; returns the guest input it completes, in order.
    pub fn feed(&mut self, ev: InputEvent) -> Vec<HostAction> {
        match (ev.kind, ev.code) {
            (EV_KEY, code) => self.key(code, ev.value),
            (EV_REL, REL_X) => self.motion(ev.value, 0),
            (EV_REL, REL_Y) => self.motion(0, ev.value),
            (EV_REL, REL_WHEEL) => notches(
                ev.value,
                ReimsVgpuButton::WheelUp,
                ReimsVgpuButton::WheelDown,
            ),
            (EV_REL, REL_HWHEEL) => notches(
                ev.value,
                ReimsVgpuButton::WheelRight,
                ReimsVgpuButton::WheelLeft,
            ),
            (EV_SYN, SYN_REPORT) if self.moved => {
                self.moved = false;
                let (dx, dy) = std::mem::take(&mut self.pending);
                let mut pointer = self.pointer.lock().unwrap_or_else(|e| e.into_inner());
                let (x, y) = pointer.moved(dx, dy);
                let (w, h) = pointer.size();
                vec![HostAction::input_pointer_move(x, y, w, h)]
            }
            _ => Vec::new(),
        }
    }

    /// The device is gone: release every key and button it still holds.
    pub fn lost(&mut self) -> Vec<HostAction> {
        let mut out = self.keyboard.shutdown().actions;
        out.extend(
            self.buttons_held
                .drain(..)
                .map(|b| HostAction::input_pointer_button(b, false)),
        );
        self.pending = (0, 0);
        self.moved = false;
        out
    }

    fn motion(&mut self, dx: i32, dy: i32) -> Vec<HostAction> {
        self.pending.0 = self.pending.0.saturating_add(dx);
        self.pending.1 = self.pending.1.saturating_add(dy);
        self.moved = true;
        Vec::new()
    }

    fn key(&mut self, code: u16, value: i32) -> Vec<HostAction> {
        // 2 is autorepeat: the guest keeps its own repeat, as with the window.
        let down = match value {
            0 => false,
            1 => true,
            _ => return Vec::new(),
        };
        if let Some(b) = button(code) {
            let held = self.buttons_held.contains(&b);
            if down == held {
                return Vec::new();
            }
            if down {
                self.buttons_held.push(b);
            } else {
                self.buttons_held.retain(|&h| h != b);
            }
            return vec![HostAction::input_pointer_button(b, down)];
        }
        if (BTN_MISC..KEY_OK).contains(&code) {
            // Joystick, tablet and touch buttons: not a keyboard's, not a mouse's.
            return Vec::new();
        }
        self.keyboard.key(u32::from(code), down).actions
    }
}

/// A down+up pair per notch, `pos` for positive values.
fn notches(value: i32, pos: ReimsVgpuButton, neg: ReimsVgpuButton) -> Vec<HostAction> {
    let b = if value > 0 { pos } else { neg };
    (0..value.unsigned_abs())
        .flat_map(|_| {
            [
                HostAction::input_pointer_button(b, true),
                HostAction::input_pointer_button(b, false),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::HostAction;
    use crate::runtime::input::ReimsVgpuButton;

    const EV_SYN: u16 = 0;
    const EV_KEY: u16 = 1;
    const EV_REL: u16 = 2;
    const KEY_A: u16 = 30;
    const BTN_LEFT: u16 = 0x110;
    const REL_X: u16 = 0;
    const REL_Y: u16 = 1;
    const REL_WHEEL: u16 = 8;

    fn ev(kind: u16, code: u16, value: i32) -> InputEvent {
        InputEvent { kind, code, value }
    }

    #[test]
    fn decodes_a_64_bit_input_event() {
        let mut raw = [0u8; 24];
        raw[16..18].copy_from_slice(&EV_KEY.to_ne_bytes());
        raw[18..20].copy_from_slice(&KEY_A.to_ne_bytes());
        raw[20..24].copy_from_slice(&1i32.to_ne_bytes());
        assert_eq!(decode(&raw), ev(EV_KEY, KEY_A, 1));
    }

    #[test]
    fn a_key_press_and_release_become_guest_keys() {
        let mut t = Translator::new(1920, 1080);
        assert_eq!(
            t.feed(ev(EV_KEY, KEY_A, 1)),
            vec![HostAction::input_key(30, true)]
        );
        assert_eq!(t.feed(ev(EV_KEY, KEY_A, 2)), vec![]); // autorepeat: the guest repeats on its own
        assert_eq!(
            t.feed(ev(EV_KEY, KEY_A, 0)),
            vec![HostAction::input_key(30, false)]
        );
    }

    #[test]
    fn relative_motion_is_one_absolute_move_per_report() {
        let mut t = Translator::new(1920, 1080);
        assert_eq!(t.feed(ev(EV_REL, REL_X, 40)), vec![]);
        assert_eq!(t.feed(ev(EV_REL, REL_Y, -40)), vec![]);
        assert_eq!(
            t.feed(ev(EV_SYN, 0, 0)),
            vec![HostAction::input_pointer_move(1000, 500, 1920, 1080)]
        );
    }

    #[test]
    fn buttons_and_wheel_map_to_neutral_buttons() {
        let mut t = Translator::new(1920, 1080);
        assert_eq!(
            t.feed(ev(EV_KEY, BTN_LEFT, 1)),
            vec![HostAction::input_pointer_button(
                ReimsVgpuButton::Left,
                true
            )]
        );
        assert_eq!(
            t.feed(ev(EV_REL, REL_WHEEL, -1)),
            vec![
                HostAction::input_pointer_button(ReimsVgpuButton::WheelDown, true),
                HostAction::input_pointer_button(ReimsVgpuButton::WheelDown, false)
            ]
        );
    }

    #[test]
    fn a_lost_device_releases_every_held_key() {
        let mut t = Translator::new(1920, 1080);
        t.feed(ev(EV_KEY, 29, 1)); // LEFTCTRL
        t.feed(ev(EV_KEY, KEY_A, 1));
        let released = t.lost();
        assert_eq!(released.len(), 2);
        assert!(released.contains(&HostAction::input_key(29, false)));
        assert!(released.contains(&HostAction::input_key(30, false)));
    }

    #[test]
    fn a_lost_mouse_releases_its_held_buttons() {
        let mut t = Translator::new(1920, 1080);
        t.feed(ev(EV_KEY, BTN_LEFT, 1));
        assert_eq!(
            t.lost(),
            vec![HostAction::input_pointer_button(
                ReimsVgpuButton::Left,
                false
            )]
        );
    }

    /// The guest has one cursor: a second mouse carries on from where the
    /// first one left it, instead of jumping to a position of its own.
    #[test]
    fn two_mice_share_one_pointer() {
        let shared = std::sync::Arc::new(std::sync::Mutex::new(
            crate::host_window::pointer::Pointer::new(1920, 1080),
        ));
        let mut script = Translator::with_pointer(shared.clone());
        let mut dell = Translator::with_pointer(shared);
        script.feed(ev(EV_REL, REL_X, -10_000));
        script.feed(ev(EV_REL, REL_Y, -10_000));
        script.feed(ev(EV_SYN, 0, 0));
        script.feed(ev(EV_REL, REL_X, 1500));
        script.feed(ev(EV_REL, REL_Y, 700));
        assert_eq!(
            script.feed(ev(EV_SYN, 0, 0)),
            vec![HostAction::input_pointer_move(1500, 700, 1920, 1080)]
        );
        dell.feed(ev(EV_REL, REL_X, 1));
        assert_eq!(
            dell.feed(ev(EV_SYN, 0, 0)),
            vec![HostAction::input_pointer_move(1501, 700, 1920, 1080)]
        );
    }

    #[test]
    fn keyboards_and_mice_are_told_apart_by_capabilities() {
        const KEY_ENTER: u16 = 28;
        let mut keys = vec![0u8; 96];
        keys[(KEY_A / 8) as usize] |= 1 << (KEY_A % 8);
        keys[(KEY_ENTER / 8) as usize] |= 1 << (KEY_ENTER % 8);
        assert_eq!(classify(&keys, &[0u8; 2]), Kind::Keyboard);
        let mut mouse_keys = vec![0u8; 96];
        mouse_keys[(BTN_LEFT / 8) as usize] |= 1 << (BTN_LEFT % 8);
        assert_eq!(classify(&mouse_keys, &[0b11, 0]), Kind::Mouse);
        assert_eq!(classify(&[0u8; 96], &[0u8; 2]), Kind::Other);
    }
}
