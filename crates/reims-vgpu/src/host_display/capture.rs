//! Capture on request: what the display sent to scanout, as a PPM.
//!
//! The lab has no compositor to screenshot, so the capture comes from here.
//! Writing a bare name to [`REQUEST`] asks for the next presented image; the
//! display loop hands the path to the presenter, which copies the swapchain
//! image it is about to present and writes `/tmp/<name>.ppm`.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Where the lab writes the name of the capture it wants.
pub const REQUEST: &str = "/tmp/reims-capture.req";

/// Longest name a request may carry.
const MAX_NAME: usize = 128;

/// Why a capture request was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureRefusal {
    Empty,
    /// `.`, `..`, a `/`, or a byte outside `[A-Za-z0-9._-]`.
    BadName(String),
    TooLong(usize),
}

impl std::fmt::Display for CaptureRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("capture_request_empty"),
            Self::BadName(name) => write!(f, "capture_request_bad_name name={name:?}"),
            Self::TooLong(len) => write!(f, "capture_request_too_long len={len}"),
        }
    }
}

/// The capture name in a request: trimmed, one path component that stays in
/// `/tmp`, at most 128 bytes of `[A-Za-z0-9._-]`.
pub fn parse_request(contents: &str) -> Result<String, CaptureRefusal> {
    let name = contents.trim();
    if name.is_empty() {
        return Err(CaptureRefusal::Empty);
    }
    if name.len() > MAX_NAME {
        return Err(CaptureRefusal::TooLong(name.len()));
    }
    let allowed = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-');
    if name == "." || name == ".." || !name.bytes().all(allowed) {
        return Err(CaptureRefusal::BadName(name.to_owned()));
    }
    Ok(name.to_owned())
}

/// `/tmp/<name>.ppm` for an accepted name.
pub fn output_path(name: &str) -> PathBuf {
    Path::new("/tmp").join(format!("{name}.ppm"))
}

/// Write `bgra` (tightly packed, `width * height * 4` bytes) as a binary PPM.
/// Written to `<path>.part` and renamed, so a reader never sees half a file.
pub fn write_ppm(path: &Path, width: u32, height: u32, bgra: &[u8]) -> io::Result<()> {
    let pixels = width as usize * height as usize;
    if bgra.len() < pixels * 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} bytes for {width}x{height}", bgra.len()),
        ));
    }
    let mut out = Vec::with_capacity(pixels * 3 + 32);
    write!(out, "P6\n{width} {height}\n255\n")?;
    for px in bgra[..pixels * 4].chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0]]);
    }
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    std::fs::write(&part, &out)?;
    std::fs::rename(&part, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_name_is_accepted_and_trimmed() {
        assert_eq!(parse_request("03-wallpaper\n").unwrap(), "03-wallpaper");
    }

    #[test]
    fn names_that_leave_tmp_or_are_empty_are_refused() {
        let long = "x".repeat(129);
        for bad in [
            "",
            "   ",
            "../x",
            "a/b",
            "/etc/passwd",
            ".",
            "..",
            long.as_str(),
        ] {
            assert!(parse_request(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_ppm_is_rgb_from_bgra() {
        let dir = std::env::temp_dir().join(format!("reims-ppm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("one.ppm");
        write_ppm(&path, 2, 1, &[1, 2, 3, 255, 4, 5, 6, 255]).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"P6\n2 1\n255\n\x03\x02\x01\x06\x05\x04".to_vec()
        );
    }
}
