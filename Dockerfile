# Multi-stage build for mzekezeke Environmental Cryptography
# Stage 1: Build environment
FROM rust:1.75-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    libwebkit2gtk-4.0-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libasound2-dev \
    libxcb1-dev \
    libxrandr-dev \
    libxss-dev \
    libglib2.0-dev \
    libgdk-pixbuf2.0-dev \
    libsoup2.4-dev \
    libjavascriptcoregtk-4.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install wasm-pack for WebAssembly builds
RUN cargo install wasm-pack

# Set working directory
WORKDIR /usr/src/app

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./
COPY clippy.toml ./

# Copy source code
COPY crates/ crates/

# Build the project
RUN cargo build --release --package mzekezeke-server
RUN cargo build --release --package mzekezeke-web-api

# Build WebAssembly bindings
WORKDIR /usr/src/app/crates/wasm
RUN wasm-pack build --target web --out-dir pkg --release

# Stage 2: Runtime environment
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    libgtk-3-0 \
    libwebkit2gtk-4.0-37 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false -m -d /app mzekezeke

# Set working directory
WORKDIR /app

# Copy built binaries from builder stage
COPY --from=builder /usr/src/app/target/release/mzekezeke-server /app/
COPY --from=builder /usr/src/app/target/release/mzekezeke-web-api /app/

# Copy WebAssembly package
COPY --from=builder /usr/src/app/crates/wasm/pkg /app/wasm/

# Copy assets if they exist
COPY --from=builder /usr/src/app/assets /app/assets/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/config

# Change ownership to app user
RUN chown -R mzekezeke:mzekezeke /app

# Switch to app user
USER mzekezeke

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./mzekezeke-server"]

# Labels for metadata
LABEL maintainer="Kundai Farai Sachikonye <kundai@fullscreen-triangle.com>"
LABEL description="mzekezeke Environmental Cryptography Server"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/fullscreen-triangle/mzekezeke"
LABEL org.opencontainers.image.description="Zero-cost multi-dimensional environmental cryptography framework"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0" 