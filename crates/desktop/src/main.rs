//! # mzekezeke-desktop
//!
//! Desktop application for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This application provides a graphical user interface for interacting with the MDTEC system.

#![deny(missing_docs)]
#![deny(unsafe_code)]

use eframe::egui;
use tracing::info;

mod app;
mod ui;

pub use app::MzekezekeApp;

fn main() -> Result<(), eframe::Error> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    info!("Starting mzekezeke desktop application");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("mzekezeke - Environmental Cryptography"),
        ..Default::default()
    };

    eframe::run_native(
        "mzekezeke",
        options,
        Box::new(|cc| {
            // Configure fonts
            configure_fonts(&cc.egui_ctx);
            
            // Create the application
            Ok(Box::new(MzekezekeApp::new(cc)))
        }),
    )
}

fn configure_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    
    // Set default font sizes
    fonts.font_data.insert(
        "default".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/default.ttf"))
            .unwrap_or_else(|_| egui::FontData::default()),
    );
    
    ctx.set_fonts(fonts);
} 