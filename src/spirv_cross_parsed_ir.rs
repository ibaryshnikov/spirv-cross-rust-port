use std::collections::HashMap;

use crate::spirv_common::{
    Meta,
    SPIREntryPoint,
    SPIRVariable,
    Types,
    Variant,
};
use crate::spirv as spv;

enum BlockMetaFlagBits {
    BLOCK_META_LOOP_HEADER_BIT = 1 << 0,       // 1
    BLOCK_META_CONTINUE_BIT = 1 << 1,          // 2
    BLOCK_META_LOOP_MERGE_BIT = 1 << 2,        // 4
    BLOCK_META_SELECTION_MERGE_BIT = 1 << 3,   // 8
    BLOCK_META_MULTISELECT_MERGE_BIT = 1 << 4, // 16
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
    pub block_meta: Vec<BlockMetaFlags>,
    pub continue_block_to_loop_header: HashMap<u32, u32>,
    pub entry_points: HashMap<u32, SPIREntryPoint>,
    pub default_entry_point: u32,
    pub source: Source,
    loop_iteration_depth: u32,
    empty_string: String,
    cleared_bitset: Bitset,
}

impl ParsedIR {
    fn set_id_bounds(&mut self, bounds: u32) {
        self.ids.resize(
            bounds as usize,
            Variant::default(),
        );
        self.block_meta.resize(
            bounds as usize,
            BlockMetaFlags::default(),
        );
    }
    fn set_name(&mut self, id: u32, name: String) {
        let meta = self.meta
            .get_mut(&id)
            .unwrap();
        let mut _str = &meta
            .decoration
            .alias;
        _str.clear();

        if name.is_empty() {
            return;
        }

        if name[0] == '_' && name.len() >= 2 && isdigit(name[1]) {
            return;
        }

        meta.decoration
            .alias = ensure_valid_identifier(name, false);
    }
    fn get_name(&self, id: u32) -> String {
        if let Some(m) = self.find_meta(id) {
            m.decoration.alias
        } else {
            self.empty_string.to_owned()
        }
    }
    fn set_decoration(
        id: u32,
        decoration: spv::Decoration,
        argument: impl Into<Option<u32>>,
    ) {
        let argument = argument.unwrap_or(0);
    }
    fn set_decoration_string(
        id: u32,
        decoration: spv::Decoration,
        argument: String,
    ) {}
    fn has_decoration(
        id: u32,
        decoration: spv::Decoration,
    ) -> bool {}
    fn get_decoration(
        id: u32,
        decoration: spv::Decoration,
    ) -> u32 {}
    fn get_decoration_string(
        id: u32,
        decoration: spv::Decoration,
    ) -> String {}
    fn get_decoration_bitset(
        id: u32,
    ) -> Bitset {}
    fn unset_decoration(
        id: u32,
        decoration: spv::Decoration,
    ) {}

    fn set_member_name(
        &self,
        id: u32,
        index: u32,
        name: String,
    ) {}
    fn get_member_name(
        &self,
        id: u32,
        index: u32,
    ) -> String {
        if let Some(m) = self.find_meta(id) {
            if index >= m.members.len() {
                self.empty_string().to_owned()
            } else {
                m.members[i].alias
            }
        } else {
            self.empty_string.to_owned()
        }
    }
    fn set_member_decoration(
        id: u32,
        index: u32,
        decoration: spv::Decoration,
        argument: impl Into<Option<u32>>,
    ) {
        let argument = argument.unwrap_or(0);
    }
    fn set_member_decoration_string(
        id: u32,
        index: u32,
        decoration: spv::Decoration,
        argument: String,
    ) {

    }
    fn get_member_decoration(
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) -> u32 {

    }
    fn get_member_decoration_string(
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) -> String {}
    fn has_member_decoration(
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) -> bool {}
    fn get_member_decoration_bitset(
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) -> Bitset {}
    fn unset_member_decoration(
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) {}

    fn mark_used_as_array_length(id: u32) {}
    fn increase_bound_by(count: u32) -> u32 {}
    fn get_buffer_block_flags(var: SPIRVariable) -> Bitset {}

    fn add_typed_id(_type: Types, id: u32) {}
    fn remove_typed_id(_type: Types, id: u32) {}

    fn for_each_typed_id<T>(&mut self, op: T) {
        self.loop_iteration_depth += 1;

        for id in self.ids_for_type[T::_type] {
            if self.ids[id].get_type() == T::_type {
                op(id, self.get<T>(id));
            }
        }

        loop_iteration_depth -= 1;
    }

    fn reset_all_of_type(&mut self, _type: Types) {}
    fn find_meta(id: u32) -> Meta {}

    fn get_empty_string(&self) -> String {
        self.empty_string.to_owned()
    }

}

fn ensure_valid_identifier(
    name: String,
    member: bool,
) -> String {
    // Functions in glslangValidator are mangled with name(<mangled> stuff.
    // Normally, we would never see '(' in any legal identifiers, so just strip them out.
    let mut _str = name[0..name.find('(').unwrap_or(name.len())]
        .to_owned();
    for i in 0.._str.len() {
        let mut c = _str[i];
        if member {
            // _m<num> variables are reserved by the internal implementation,
            // otherwise, make sure the name is a valid identifier.
            if i == 0 {
                c = if isalpha(c) { c } else { '_' };
            } else if i == 2 && _str[0] == '_' && _str[1] == 'm' {
                c = if isalpha(c) { c } else { '_' };
            } else {
                c = if isalnum(c) { c } else { '_' };
            }
        } else {
            // _<num> variables are reserved by the internal implementation,
            // otherwise, make sure the name is a valid identifier.
            if i == 0 || (_str[0] == '_' && i == 1) {
                c = if isalpha(c) { c } else { '_' };
            } else {
                c = if isalnum(c) { c } else { '_' };
            }
        }
        _str[i] = c;
    }
    return _str;
}