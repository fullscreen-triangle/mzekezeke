# PowerShell build script for mzekezeke Environmental Cryptography
param(
    [switch]$Core,
    [switch]$Server,
    [switch]$Client,
    [switch]$Wasm,
    [switch]$Desktop,
    [switch]$Test,
    [switch]$Docs,
    [switch]$Help
)

# Colors for output
$Colors = @{
    Info = "Blue"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Info
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Success
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Warning
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Error
}

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Check prerequisites
function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    if (!(Test-Command "cargo")) {
        Write-Error "Rust/Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    }
    
    if (!(Test-Command "wasm-pack")) {
        Write-Warning "wasm-pack not found. Installing..."
        cargo install wasm-pack
    }
    
    Write-Success "Prerequisites check completed"
}

# Build core library
function Build-Core {
    Write-Status "Building core cryptography library..."
    cargo build --package mzekezeke-core --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Core library built successfully"
}

# Build server
function Build-Server {
    Write-Status "Building environmental cryptography server..."
    cargo build --package mzekezeke-server --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Server built successfully"
}

# Build client library
function Build-Client {
    Write-Status "Building client library..."
    cargo build --package mzekezeke-client --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Client library built successfully"
}

# Build sensors library
function Build-Sensors {
    Write-Status "Building environmental sensors library..."
    cargo build --package mzekezeke-sensors --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Sensors library built successfully"
}

# Build protocols library
function Build-Protocols {
    Write-Status "Building network protocols library..."
    cargo build --package mzekezeke-protocols --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Protocols library built successfully"
}

# Build web API
function Build-WebApi {
    Write-Status "Building web API server..."
    cargo build --package mzekezeke-web-api --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Web API server built successfully"
}

# Build WebAssembly bindings
function Build-Wasm {
    Write-Status "Building WebAssembly bindings..."
    Push-Location crates/wasm
    wasm-pack build --target web --out-dir pkg --release
    if ($LASTEXITCODE -ne 0) { 
        Pop-Location
        exit $LASTEXITCODE 
    }
    Pop-Location
    Write-Success "WebAssembly bindings built successfully"
}

# Build mobile bindings
function Build-Mobile {
    Write-Status "Building mobile platform bindings..."
    cargo build --package mzekezeke-mobile --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Mobile bindings built successfully"
}

# Build desktop application
function Build-Desktop {
    Write-Status "Building desktop application..."
    cargo build --package mzekezeke-desktop --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Desktop application built successfully"
}

# Run tests
function Test-Project {
    Write-Status "Running test suite..."
    cargo test --workspace --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "All tests passed"
}

# Generate documentation
function Build-Documentation {
    Write-Status "Generating documentation..."
    cargo doc --workspace --no-deps --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Success "Documentation generated successfully"
}

# Show help
function Show-Help {
    Write-Host "Usage: ./build.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Core      Build only core library"
    Write-Host "  -Server    Build only server"
    Write-Host "  -Client    Build only client library"
    Write-Host "  -Wasm      Build only WebAssembly bindings"
    Write-Host "  -Desktop   Build only desktop application"
    Write-Host "  -Test      Run tests after building"
    Write-Host "  -Docs      Generate documentation"
    Write-Host "  -Help      Show this help message"
    Write-Host ""
    Write-Host "If no specific component is specified, all components are built."
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

Write-Status "Starting mzekezeke Environmental Cryptography build..."
Write-Host ""

Test-Prerequisites
Write-Host ""

$BuildAll = $true

# Check if any specific component flags were provided
if ($Core -or $Server -or $Client -or $Wasm -or $Desktop) {
    $BuildAll = $false
}

# Build specific components or all components
if ($Core -or $BuildAll) { Build-Core }
if ($BuildAll) { Build-Sensors }
if ($BuildAll) { Build-Protocols }
if ($Client -or $BuildAll) { Build-Client }
if ($Server -or $BuildAll) { Build-Server }
if ($BuildAll) { Build-WebApi }
if ($Wasm -or $BuildAll) { Build-Wasm }
if ($BuildAll) { Build-Mobile }
if ($Desktop -or $BuildAll) { Build-Desktop }

if ($Test) {
    Write-Host ""
    Test-Project
}

if ($Docs) {
    Write-Host ""
    Build-Documentation
}

Write-Host ""
Write-Success "mzekezeke Environmental Cryptography build completed successfully!"
Write-Host ""
Write-Status "Build artifacts are available in:"
Write-Host "  - target/release/ (Native binaries)"
Write-Host "  - crates/wasm/pkg/ (WebAssembly package)"
Write-Host "  - target/doc/ (Documentation)" 