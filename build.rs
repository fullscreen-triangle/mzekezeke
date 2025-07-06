use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=proto/");
    
    // Set up protocol buffer compilation
    let proto_root = "proto";
    let proto_files = [
        "proto/mdtec.proto",
        "proto/sensors.proto",
        "proto/challenge.proto",
    ];
    
    // Create proto directory if it doesn't exist
    let proto_dir = PathBuf::from(&proto_root);
    if !proto_dir.exists() {
        std::fs::create_dir_all(&proto_dir).unwrap();
    }
    
    // Generate protocol buffer code
    if proto_files.iter().all(|f| PathBuf::from(f).exists()) {
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .compile(&proto_files, &[proto_root])
            .unwrap_or_else(|e| panic!("Failed to compile protos: {}", e));
    }
    
    // Platform-specific configurations
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=ssl");
            println!("cargo:rustc-link-lib=dylib=crypto");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=framework=Security");
            println!("cargo:rustc-link-lib=framework=SystemConfiguration");
        }
        "windows" => {
            println!("cargo:rustc-link-lib=dylib=ws2_32");
            println!("cargo:rustc-link-lib=dylib=userenv");
        }
        _ => {}
    }
    
    // WebAssembly specific configurations
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "wasm32" {
        println!("cargo:rustc-cfg=web_sys_unstable_apis");
    }
    
    // Development vs Release configurations
    if env::var("PROFILE").unwrap() == "debug" {
        println!("cargo:rustc-cfg=debug_assertions");
    }
} 