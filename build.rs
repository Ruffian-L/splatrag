use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/dream_kernels.cu");
    println!("cargo:rerun-if-changed=cuda/dream_kernels.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(cuda_dream)");

    if env::var("CARGO_FEATURE_NO_CUDA").is_ok() {
        return;
    }

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    if Command::new(&nvcc).arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found; dream stays CPU-only");
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out_dir.join("dream_kernels.o");
    let lib = out_dir.join("libdream_cuda.a");
    let cccl = PathBuf::from("/usr/local/cuda/include/cccl");

    let mut cmd = Command::new(&nvcc);
    cmd.args([
        "-O3",
        "--std=c++17",
        "-arch=sm_121",
        "-Xcompiler",
        "-fPIC",
        "-I",
        manifest.join("cuda").to_str().unwrap(),
        "-c",
        manifest.join("cuda/dream_kernels.cu").to_str().unwrap(),
        "-o",
        obj.to_str().unwrap(),
    ]);
    if cccl.is_dir() {
        cmd.arg("-I").arg(&cccl);
    }

    let status = cmd.status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=nvcc sm_121 compile failed ({s}); dream stays CPU-only");
            return;
        }
        Err(err) => {
            println!("cargo:warning=nvcc launch failed ({err}); dream stays CPU-only");
            return;
        }
    }

    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let ar_status = Command::new(&ar)
        .args(["crus", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status();
    match ar_status {
        Ok(s) if s.success() => {}
        other => {
            println!("cargo:warning=ar libdream_cuda.a failed ({other:?}); dream stays CPU-only");
            return;
        }
    }

    let cuda_lib = cuda_lib_dir();
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=dream_cuda");
    if let Some(dir) = cuda_lib {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=cuda_dream");
}

fn cuda_lib_dir() -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &[
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/sbsa-linux/lib",
        "/usr/local/cuda/lib",
    ];
    CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|p| p.join("libcudart.so").exists() || p.join("libcudart_static.a").exists())
        .or_else(|| {
            env::var("CUDA_HOME").ok().and_then(|home| {
                let p = Path::new(&home).join("lib64");
                p.exists().then_some(p)
            })
        })
}
