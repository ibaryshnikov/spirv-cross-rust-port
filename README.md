# spirv-cross-rust-port

It's a port of [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)
to Rust

## Motivation

Provide a library which can be used from Rust in wasm.
Currently it's quite tricky to link C/C++ and Rust for `wasm32-unknown-unknown` target.


## Status

Work is highly in progress. Any help is welcome. At the moment the SPIRV parser is mostly finished,
and the focus is switched to make a minimal GLSL compiler, which will
transform intermediate representation to GLSL shader.

## Contributing

To build use `cargo build`. Before you submit a PR, make sure to run
`cargo clippy` and `cargo fmt`

## License

Both Apache License and MIT license
