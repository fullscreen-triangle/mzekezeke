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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists cargo; then
        print_error "Rust/Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    if ! command_exists wasm-pack; then
        print_warning "wasm-pack not found. Installing..."
        cargo install wasm-pack
    fi
    
    print_success "Prerequisites check completed"
}

# Build core library
build_core() {
    print_status "Building core cryptography library..."
    cargo build --package mzekezeke-core --release
    print_success "Core library built successfully"
}

# Build server
build_server() {
    print_status "Building environmental cryptography server..."
    cargo build --package mzekezeke-server --release
    print_success "Server built successfully"
}

# Build client library
build_client() {
    print_status "Building client library..."
    cargo build --package mzekezeke-client --release
    print_success "Client library built successfully"
}

# Build sensors library
build_sensors() {
    print_status "Building environmental sensors library..."
    cargo build --package mzekezeke-sensors --release
    print_success "Sensors library built successfully"
}

# Build protocols library
build_protocols() {
    print_status "Building network protocols library..."
    cargo build --package mzekezeke-protocols --release
    print_success "Protocols library built successfully"
}

# Build web API
build_web_api() {
    print_status "Building web API server..."
    cargo build --package mzekezeke-web-api --release
    print_success "Web API server built successfully"
}

# Build WebAssembly bindings
build_wasm() {
    print_status "Building WebAssembly bindings..."
    cd crates/wasm
    wasm-pack build --target web --out-dir pkg --release
    cd ../..
    print_success "WebAssembly bindings built successfully"
}

# Build mobile bindings
build_mobile() {
    print_status "Building mobile platform bindings..."
    cargo build --package mzekezeke-mobile --release
    print_success "Mobile bindings built successfully"
}

# Build desktop application
build_desktop() {
    print_status "Building desktop application..."
    cargo build --package mzekezeke-desktop --release
    print_success "Desktop application built successfully"
}

# Run tests
run_tests() {
    print_status "Running test suite..."
    cargo test --workspace --release
    print_success "All tests passed"
}

# Generate documentation
generate_docs() {
    print_status "Generating documentation..."
    cargo doc --workspace --no-deps --release
    print_success "Documentation generated successfully"
}

# Main build function
main() {
    print_status "Starting mzekezeke Environmental Cryptography build..."
    echo
    
    # Parse command line arguments
    BUILD_ALL=true
    RUN_TESTS=false
    GENERATE_DOCS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --core)
                BUILD_ALL=false
                build_core
                shift
                ;;
            --server)
                BUILD_ALL=false
                build_server
                shift
                ;;
            --client)
                BUILD_ALL=false
                build_client
                shift
                ;;
            --wasm)
                BUILD_ALL=false
                build_wasm
                shift
                ;;
            --desktop)
                BUILD_ALL=false
                build_desktop
                shift
                ;;
            --test)
                RUN_TESTS=true
                shift
                ;;
            --docs)
                GENERATE_DOCS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --core      Build only core library"
                echo "  --server    Build only server"
                echo "  --client    Build only client library"
                echo "  --wasm      Build only WebAssembly bindings"
                echo "  --desktop   Build only desktop application"
                echo "  --test      Run tests after building"
                echo "  --docs      Generate documentation"
                echo "  --help      Show this help message"
                echo ""
                echo "If no specific component is specified, all components are built."
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    echo
    
    if [ "$BUILD_ALL" = true ]; then
        build_core
        build_sensors
        build_protocols
        build_client
        build_server
        build_web_api
        build_wasm
        build_mobile
        build_desktop
    fi
    
    if [ "$RUN_TESTS" = true ]; then
        echo
        run_tests
    fi
    
    if [ "$GENERATE_DOCS" = true ]; then
        echo
        generate_docs
    fi
    
    echo
    print_success "mzekezeke Environmental Cryptography build completed successfully!"
    echo
    print_status "Build artifacts are available in:"
    echo "  - target/release/ (Native binaries)"
    echo "  - crates/wasm/pkg/ (WebAssembly package)"
    echo "  - target/doc/ (Documentation)"
}

# Run main function with all arguments
main "$@"
