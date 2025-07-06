#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    elif [[ "$OSTYPE" == "msys" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Install Rust
install_rust() {
    if command_exists rustc; then
        print_status "Rust is already installed ($(rustc --version))"
        return
    fi
    
    print_status "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    print_success "Rust installed successfully"
}

# Install wasm-pack
install_wasm_pack() {
    if command_exists wasm-pack; then
        print_status "wasm-pack is already installed ($(wasm-pack --version))"
        return
    fi
    
    print_status "Installing wasm-pack..."
    cargo install wasm-pack
    print_success "wasm-pack installed successfully"
}

# Install additional Rust components
install_rust_components() {
    print_status "Installing additional Rust components..."
    
    # Install targets for cross-compilation
    rustup target add wasm32-unknown-unknown
    rustup target add aarch64-apple-darwin
    rustup target add x86_64-pc-windows-gnu
    rustup target add aarch64-unknown-linux-gnu
    
    # Install useful tools
    cargo install cargo-watch
    cargo install cargo-edit
    cargo install cargo-audit
    cargo install cargo-deny
    cargo install cargo-expand
    cargo install cargo-tree
    cargo install cargo-outdated
    
    print_success "Additional Rust components installed"
}

# Install system dependencies for Linux
install_linux_deps() {
    print_status "Installing Linux system dependencies..."
    
    # Update package manager
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y \
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
            libjavascriptcoregtk-4.0-dev
    elif command_exists yum; then
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            openssl-devel \
            webkit2gtk4.0-devel \
            gtk3-devel \
            libappindicator-gtk3-devel \
            librsvg2-devel \
            alsa-lib-devel
    elif command_exists pacman; then
        sudo pacman -S --needed \
            base-devel \
            webkit2gtk \
            gtk3 \
            libappindicator-gtk3 \
            librsvg \
            alsa-lib
    fi
    
    print_success "Linux system dependencies installed"
}

# Install system dependencies for macOS
install_macos_deps() {
    print_status "Installing macOS system dependencies..."
    
    # Install Xcode command line tools
    if ! command_exists xcode-select; then
        print_status "Installing Xcode command line tools..."
        xcode-select --install
    fi
    
    # Install Homebrew if not present
    if ! command_exists brew; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install required packages
    brew install pkg-config openssl
    
    print_success "macOS system dependencies installed"
}

# Install Docker
install_docker() {
    if command_exists docker; then
        print_status "Docker is already installed ($(docker --version))"
        return
    fi
    
    print_status "Installing Docker..."
    OS=$(detect_os)
    
    case $OS in
        "linux")
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            ;;
        "macos")
            print_warning "Please install Docker Desktop for macOS from https://www.docker.com/products/docker-desktop"
            ;;
        *)
            print_warning "Please install Docker manually for your operating system"
            ;;
    esac
    
    print_success "Docker installation completed"
}

# Install Node.js (for web development)
install_nodejs() {
    if command_exists node; then
        print_status "Node.js is already installed ($(node --version))"
        return
    fi
    
    print_status "Installing Node.js..."
    
    # Install Node Version Manager (nvm)
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    # Install latest LTS Node.js
    nvm install --lts
    nvm use --lts
    
    print_success "Node.js installed successfully"
}

# Setup development environment
setup_dev_environment() {
    print_status "Setting up development environment..."
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p tmp
    mkdir -p config
    
    # Create environment files
    if [ ! -f ".env.example" ]; then
        cat > .env.example << EOF
# mzekezeke Environmental Cryptography Configuration
RUST_LOG=info
RUST_BACKTRACE=1

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8080
SERVER_WORKERS=4

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/mzekezeke

# Environmental Sensors Configuration
ENABLE_GPS_SENSOR=true
ENABLE_CELLULAR_SENSOR=true
ENABLE_WIFI_SENSOR=true
ENABLE_HARDWARE_SENSOR=true

# Security Configuration
ENCRYPTION_KEY_SIZE=256
REALITY_DIMENSIONS=12
CHALLENGE_TIMEOUT=30

# Performance Configuration
WORKER_THREADS=4
BUFFER_SIZE=8192
CACHE_SIZE=1024
EOF
    fi
    
    # Set up Git hooks
    mkdir -p .git/hooks
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for mzekezeke Environmental Cryptography

# Run clippy
echo "Running clippy..."
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
echo "Running tests..."
cargo test --all

# Format code
echo "Formatting code..."
cargo fmt --all -- --check

echo "Pre-commit checks passed!"
EOF
    chmod +x .git/hooks/pre-commit
    
    print_success "Development environment setup completed"
}

# Main setup function
main() {
    print_status "Setting up mzekezeke Environmental Cryptography development environment..."
    echo
    
    OS=$(detect_os)
    print_status "Detected OS: $OS"
    echo
    
    # Install Rust and related tools
    install_rust
    install_wasm_pack
    install_rust_components
    echo
    
    # Install system dependencies
    case $OS in
        "linux")
            install_linux_deps
            ;;
        "macos")
            install_macos_deps
            ;;
        *)
            print_warning "Please install system dependencies manually for your OS"
            ;;
    esac
    echo
    
    # Install additional tools
    install_docker
    install_nodejs
    echo
    
    # Setup development environment
    setup_dev_environment
    echo
    
    print_success "mzekezeke Environmental Cryptography development environment setup completed!"
    echo
    print_status "Next steps:"
    echo "1. Copy .env.example to .env and configure your settings"
    echo "2. Run './scripts/build.sh' to build the project"
    echo "3. Run 'cargo test' to run the test suite"
    echo "4. Start developing your Environmental Cryptography applications!"
    echo
    print_warning "Note: You may need to restart your terminal or run 'source ~/.cargo/env' to use Rust tools"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Sets up the development environment for mzekezeke Environmental Cryptography"
            echo ""
            echo "Options:"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Run main function
main 