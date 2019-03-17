use crate::spirv_cross::Compiler;
use crate::spirv_cross_parsed_ir::ParsedIR;

enum PlsFormat {
    None = 0,

    R11FG11FB10F,
    R32F,
    RG16F,
    RGB10A2,
    RGBA8,
    RG16,

    RGBA8I,
    RG16I,

    RGB10A2UI,
    RGBA8UI,
    RG16UI,
    R32UI,
}

struct PlsRemap {
    id: u32,
    format: PlsFormat,
}

enum AccessChainFlagBits {
    INDEX_IS_LITERAL_BIT = 1,                   // 1
    CHAIN_ONLY_BIT = 1 << 1,                    // 2
    PTR_CHAIN_BIT = 1 << 2,                     // 4
    SKIP_REGISTER_EXPRESSION_READ_BIT = 1 << 3, // 8
}

type AccessChainFlags = u32;

enum Precision {
    DontCare,
    Lowp,
    Mediump,
    Highp,
}

struct OptionsVertex {
    // GLSL: In vertex shaders, rewrite [0, w] depth (Vulkan/D3D style) to [-w, w] depth (GL style).
    // MSL: In vertex shaders, rewrite [-w, w] depth (GL style) to [0, w] depth.
    // HLSL: In vertex shaders, rewrite [-w, w] depth (GL style) to [0, w] depth.
    fixup_clipspace: bool,

    // Inverts gl_Position.y or equivalent.
    flip_vert_y: bool,

    // GLSL only, for HLSL version of this option, see CompilerHLSL.
    // If true, the backend will assume that InstanceIndex will need to apply
    // a base instance offset. Set to false if you know you will never use base instance
    // functionality as it might remove some internal uniforms.
    support_nonzero_base_instance: bool,
}

impl Default for OptionsVertex {
    fn default() -> Self {
        Self {
            fixup_clipspace: false,
            flip_vert_y: false,
            support_nonzero_base_instance: true,
        }
    }
}

struct OptionsFragment {
    // Add precision mediump float in ES targets when emitting GLES source.
    // Add precision highp int in ES targets when emitting GLES source.
    default_float_precision: Precision,
    default_int_precision: Precision,
}

impl Default for OptionsFragment {
    fn default() -> Self {
        Self {
            default_float_precision: Precision::Mediump,
            default_int_precision: Precision::Highp,
        }
    }
}

struct Options {
    // The shading language version. Corresponds to #version $VALUE.
    version: u32,

    // Emit the OpenGL ES shading language instead of desktop OpenGL.
    es: bool,

    // Debug option to always emit temporary variables for all expressions.
    force_temporary: bool,

    // If true, Vulkan GLSL features are used instead of GL-compatible features.
    // Mostly useful for debugging SPIR-V files.
    vulkan_semantics: bool,

    // If true, gl_PerVertex is explicitly redeclared in vertex, geometry and tessellation shaders.
    // The members of gl_PerVertex is determined by which built-ins are declared by the shader.
    // This option is ignored in ES versions, as redeclaration in ES is not required, and it depends on a different extension
    // (EXT_shader_io_blocks) which makes things a bit more fuzzy.
    separate_shader_objects: bool,

    // Flattens multidimensional arrays, e.g. float foo[a][b][c] into single-dimensional arrays,
    // e.g. float foo[a * b * c].
    // This function does not change the actual SPIRType of any object.
    // Only the generated code, including declarations of interface variables are changed to be single array dimension.
    flatten_multidimensional_arrays: bool,

    // For older desktop GLSL targets than version 420, the
    // GL_ARB_shading_language_420pack extensions is used to be able to support
    // layout(binding) on UBOs and samplers.
    // If disabled on older targets, binding decorations will be stripped.
    enable_420pack_extension: bool,

    vertex: OptionsVertex,
    fragment: OptionsFragment,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            version: 450,
            es: false,
            force_temporary: false,
            vulkan_semantics: false,
            separate_shader_objects: false,
            flatten_multidimensional_arrays: false,
            enable_420pack_extension: true,
            vertex: OptionsVertex::default(),
            fragment: OptionsFragment::default(),
        }
    }
}

struct CompilerGLSL {
    ir: ParsedIR,
}

impl Compiler for CompilerGLSL {
    fn get_ir(&self) -> &ParsedIR {
        &self.ir
    }
    fn get_ir_mut(&mut self) -> &mut ParsedIR {
        &mut self.ir
    }
}
