//! # mzekezeke-web-api
//!
//! REST API server for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This server provides a REST API interface for the MDTEC system.

use clap::Parser;
use std::net::SocketAddr;
use tracing::{info, warn};

mod app;
mod handlers;
mod middleware;
mod state;

pub use app::create_app;
pub use state::AppState;

/// CLI arguments for the web API server
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Server bind address
    #[arg(short, long, default_value = "0.0.0.0:3000")]
    bind: SocketAddr,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Number of worker threads
    #[arg(short, long, default_value = "4")]
    workers: usize,

    /// Enable CORS
    #[arg(long)]
    cors: bool,

    /// Static file directory
    #[arg(long, default_value = "static")]
    static_dir: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(&args.log_level)
        .init();

    info!("Starting mzekezeke web API server v{}", env!("CARGO_PKG_VERSION"));

    // Create application state
    let state = AppState::new().await?;
    info!("Initialized application state");

    // Create the application
    let app = create_app(state, args.cors, &args.static_dir);

    // Create the listener
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!("Web API server listening on {}", args.bind);

    // Start the server
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Web API server shutdown complete");
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