//! # mzekezeke-server
//!
//! Server implementation for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This server coordinates environmental challenges between clients and manages
//! the distributed cryptographic protocol.

use clap::Parser;
use std::net::SocketAddr;
use tracing::{info, warn};

mod app;
mod config;
mod handlers;
mod middleware;
mod state;

pub use app::create_app;
pub use config::Config;
pub use state::AppState;

/// CLI arguments for the mzekezeke server
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Server configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Server bind address
    #[arg(short, long, default_value = "0.0.0.0:8080")]
    bind: SocketAddr,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Number of worker threads
    #[arg(short, long, default_value = "4")]
    workers: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(&args.log_level)
        .init();

    info!("Starting mzekezeke server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = Config::from_file(&args.config)?;
    info!("Loaded configuration from {}", args.config);

    // Create application state
    let state = AppState::new(config).await?;
    info!("Initialized application state");

    // Create the application
    let app = create_app(state);

    // Create the listener
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!("Server listening on {}", args.bind);

    // Start the server
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            warn!("Received Ctrl+C, shutting down...");
        }
        _ = terminate => {
            warn!("Received SIGTERM, shutting down...");
        }
    }
} 