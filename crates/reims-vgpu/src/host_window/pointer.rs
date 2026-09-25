//! A pointer driven by relative motion (an evdev mouse), held inside the
//! guest frame.

/// Absolute pointer position from relative deltas, starting at the centre and
/// clamped to `[0, width-1] x [0, height-1]`. Sums in `i64`, so no delta a
/// device can send overflows it.
#[derive(Clone, Copy, Debug)]
pub struct Pointer {
    x: i64,
    y: i64,
    width: u32,
    height: u32,
}

impl Pointer {
    pub fn new(width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        Self {
            x: i64::from(width / 2),
            y: i64::from(height / 2),
            width,
            height,
        }
    }

    /// Move by `(dx, dy)` and return the new position.
    pub fn moved(&mut self, dx: i32, dy: i32) -> (u32, u32) {
        let clamp = |v: i64, dim: u32| v.clamp(0, i64::from(dim) - 1);
        self.x = clamp(self.x.saturating_add(i64::from(dx)), self.width);
        self.y = clamp(self.y.saturating_add(i64::from(dy)), self.height);
        (self.x as u32, self.y as u32)
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_motion_starts_at_the_centre_and_is_clamped() {
        let mut p = Pointer::new(1920, 1080);
        assert_eq!(p.moved(0, 0), (960, 540));
        assert_eq!(p.moved(10_000, 10_000), (1919, 1079));
        assert_eq!(p.moved(-100_000, -100_000), (0, 0));
        assert_eq!(p.moved(i32::MAX, i32::MIN), (1919, 0));
    }
}
