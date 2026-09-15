fn main() {
    println!("cargo:rerun-if-changed=third_party/soxr");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").ok().as_deref() == Some("msvc");
    // The `cmake` crate replaces `CMAKE_C_FLAGS_RELEASE` instead of extending it,
    // which drops CMake's own `/O2`: the resampler was built with optimisation
    // off (~1.4 s for a 3-minute 44.1 kHz clip).  Put it back — `CMAKE_C_FLAGS`
    // is a separate variable, so this cannot be clobbered the same way.
    let opt = if msvc { "/O2" } else { "-O2" };
    let dst = cmake::Config::new("third_party/soxr")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("WITH_OPENMP", "OFF")
        .define("WITH_LSR_BINDINGS", "OFF")
        .define("WITH_DEV_TRACE", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .cflag(opt)
        .profile("Release")
        .build();
    let lib = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=soxr");
    // soxr's CMake links libm privately; rustc does not inherit that.
    if !msvc {
        println!("cargo:rustc-link-lib=m");
    }
}
