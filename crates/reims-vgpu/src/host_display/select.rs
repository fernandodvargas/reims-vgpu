//! Which connector and which mode the display takes. Pure: the lists come from
//! the DRM ioctls (`super::drm`) or from a test.

/// One display mode of a connector, in the order the kernel listed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    pub width: u16,
    pub height: u16,
    /// Vertical refresh in millihertz.
    pub refresh_mhz: u32,
    /// `DRM_MODE_TYPE_PREFERRED` was set.
    pub preferred: bool,
    /// Position in the connector's mode list.
    pub index: usize,
}

/// One DRM connector, named the way the kernel names it (`HDMI-A-1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connector {
    pub id: u32,
    pub name: String,
    pub connected: bool,
    pub modes: Vec<Mode>,
}

/// The connector and mode the display will take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub connector_id: u32,
    pub connector: String,
    pub mode: Mode,
}

/// Why no connector or mode could be chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError {
    /// No connected connector, or the one named is absent or has no screen.
    NoConnector { wanted: Option<String> },
    /// The connector is connected but lists no mode.
    NoMode { connector: String },
}

/// Take the named connector (or the first connected one) and its 1920x1080
/// mode, else its preferred mode, else its first mode.
pub fn choose(connectors: &[Connector], wanted: Option<&str>) -> Result<Choice, SelectError> {
    let connector = connectors
        .iter()
        .filter(|c| c.connected)
        .find(|c| wanted.map_or(true, |name| c.name == name))
        .ok_or_else(|| SelectError::NoConnector {
            wanted: wanted.map(str::to_owned),
        })?;
    let mode = connector
        .modes
        .iter()
        .find(|m| (m.width, m.height) == (1920, 1080))
        .or_else(|| connector.modes.iter().find(|m| m.preferred))
        .or_else(|| connector.modes.first())
        .copied()
        .ok_or_else(|| SelectError::NoMode {
            connector: connector.name.clone(),
        })?;
    Ok(Choice {
        connector_id: connector.id,
        connector: connector.name.clone(),
        mode,
    })
}

impl SelectError {
    pub fn slug(&self) -> &'static str {
        match self {
            SelectError::NoConnector { .. } => "no_connector",
            SelectError::NoMode { .. } => "no_mode",
        }
    }
}

impl std::fmt::Display for SelectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectError::NoConnector { wanted: Some(name) } => {
                write!(f, "no_connector wanted={name}")
            }
            SelectError::NoConnector { wanted: None } => f.write_str("no_connector"),
            SelectError::NoMode { connector } => write!(f, "no_mode connector={connector}"),
        }
    }
}

impl crate::observe::Decline for SelectError {
    fn slug(&self) -> &'static str {
        SelectError::slug(self)
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            SelectError::NoConnector { wanted: Some(name) } => vec![("wanted", name.clone())],
            SelectError::NoConnector { wanted: None } => Vec::new(),
            SelectError::NoMode { connector } => vec![("connector", connector.clone())],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(w: u16, h: u16, preferred: bool, index: usize) -> Mode {
        Mode {
            width: w,
            height: h,
            refresh_mhz: 60_000,
            preferred,
            index,
        }
    }

    fn hdmi(connected: bool, modes: Vec<Mode>) -> Connector {
        Connector {
            id: 90,
            name: "HDMI-A-1".into(),
            connected,
            modes,
        }
    }

    fn dp() -> Connector {
        Connector {
            id: 80,
            name: "DP-1".into(),
            connected: false,
            modes: vec![],
        }
    }

    #[test]
    fn takes_1080p_over_the_preferred_mode() {
        let tv = hdmi(
            true,
            vec![mode(1360, 768, true, 0), mode(1920, 1080, false, 1)],
        );
        let choice = choose(&[dp(), tv], None).unwrap();
        assert_eq!(choice.connector, "HDMI-A-1");
        assert_eq!(
            (choice.mode.width, choice.mode.height, choice.mode.index),
            (1920, 1080, 1)
        );
    }

    #[test]
    fn falls_back_to_the_preferred_mode_without_1080p() {
        let tv = hdmi(
            true,
            vec![mode(1280, 720, false, 0), mode(1360, 768, true, 1)],
        );
        assert_eq!(choose(&[tv], None).unwrap().mode.index, 1);
    }

    #[test]
    fn falls_back_to_the_first_mode_when_none_is_preferred() {
        let tv = hdmi(
            true,
            vec![mode(1280, 720, false, 0), mode(1024, 768, false, 1)],
        );
        assert_eq!(choose(&[tv], None).unwrap().mode.index, 0);
    }

    #[test]
    fn a_named_connector_without_a_screen_is_refused() {
        let tv = hdmi(true, vec![mode(1920, 1080, true, 0)]);
        let error = choose(&[dp(), tv], Some("DP-1")).unwrap_err();
        assert_eq!(error.slug(), "no_connector");
    }

    #[test]
    fn no_connected_connector_is_refused() {
        let error = choose(&[dp()], None).unwrap_err();
        assert_eq!(error, SelectError::NoConnector { wanted: None });
    }

    #[test]
    fn a_connected_connector_without_modes_is_refused() {
        let error = choose(&[hdmi(true, vec![])], None).unwrap_err();
        assert_eq!(error.slug(), "no_mode");
    }
}
