use std::collections::HashMap;

use crate::spirv_common::{
    Meta,
    SPIREntryPoint,
    Types,
    Variant,
};
use crate::spirv as spv;

enum BlockMetaFlagBits {
    BLOCK_META_LOOP_HEADER_BIT = 1 << 0,
    BLOCK_META_CONTINUE_BIT = 1 << 1,
    BLOCK_META_LOOP_MERGE_BIT = 1 << 2,
    BLOCK_META_SELECTION_MERGE_BIT = 1 << 3,
    BLOCK_META_MULTISELECT_MERGE_BIT = 1 << 4,
}
type BlockMetaFlags = u8;

struct Source {
    version: u32,
    es: bool,
    known: bool,
    hlsl: bool,
}

impl Default for Source {
    fn default() -> Self {
        Source {
            version: 0,
            es: false,
            known: false,
            hlsl: false,
        }
    }
}

pub struct ParsedIR {
    pub spirv: Vec<u32>,
    pub ids: Vec<Variant>,
    pub meta: HashMap<u32, Meta>,
    pub ids_for_type: [u32; Types::TypeCount as usize],
    pub ids_for_constant_or_type: Vec<u32>,
    pub ids_for_constant_or_variable: Vec<u32>,
    pub declared_capabilities: Vec<spv::Capability>,
    pub declared_extensions: Vec<String>,
    pub block_meta: BlockMetaFlags,
    pub continue_block_to_loop_header: HashMap<u32, u32>,
    pub entry_points: HashMap<u32, SPIREntryPoint>,
    pub default_entry_point: u32,
    pub source: Source,

}

impl ParsedIR {
}
