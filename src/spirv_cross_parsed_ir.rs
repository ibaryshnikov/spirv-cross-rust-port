use std::cmp::max;
use std::collections::HashMap;

use num::FromPrimitive;

use crate::spirv_common::{
    BaseType,
    Bitset,
    Decoration,
    HasType,
    Meta,
    SPIRConstant,
    SPIRConstantOp,
    SPIREntryPoint,
    SPIRType,
    SPIRVariable,
    Types,
    IVariant,
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

pub struct Source {
    pub version: u32,
    pub es: bool,
    pub known: bool,
    pub hlsl: bool,
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
    pub ids_for_type: Vec<Vec<u32>>,
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

fn make_ids_of_size(len: usize) -> Vec<Vec<u32>> {
    let mut result = vec![];
    for i in 0..len {
        result.push(vec![]);
    }
    result
}

impl Default for ParsedIR {
    fn default() -> Self {
        ParsedIR {
            spirv: vec![],
            ids: vec![],
            meta: HashMap::new(),
            ids_for_type: make_ids_of_size(Types::TypeCount as usize),
            ids_for_constant_or_type: vec![],
            ids_for_constant_or_variable: vec![],
            declared_capabilities: vec![],
            declared_extensions: vec![],
            block_meta: vec![],
            continue_block_to_loop_header: HashMap::new(),
            entry_points: HashMap::new(),
            default_entry_point: 0,
            source: Source::default(),
            loop_iteration_depth: 0,
            empty_string: String::new(),
            cleared_bitset: Bitset::default(),
        }
    }
}

impl ParsedIR {
    pub fn set_id_bounds(&mut self, bounds: u32) {
        self.ids.resize(
            bounds as usize,
            Variant::default(),
        );
        self.block_meta.resize(
            bounds as usize,
            BlockMetaFlags::default(),
        );
    }

    pub fn set_name(&mut self, id: u32, name: String) {
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

        if name.as_bytes()[0] as char == '_' && name.len() >= 2 && (name.as_bytes()[1] as char).is_digit(10) {
            return;
        }

        meta.decoration
            .alias = ensure_valid_identifier(name, false);
    }

    fn get_name(&self, id: u32) -> String {
        if let Some(m) = self.find_meta(id) {
            m.decoration.alias.to_owned()
        } else {
            self.empty_string.to_owned()
        }
    }

    pub fn set_decoration(
        &mut self,
        id: u32,
        decoration: spv::Decoration,
        argument: impl Into<Option<u32>>,
    ) {
        let argument = argument.into().unwrap_or(0);
        let meta = self.meta
            .get_mut(&id)
            .unwrap();

        let dec = &mut meta.decoration;

        dec
            .decoration_flags
            .set(decoration as u32);

        use spv::Decoration::*;

        match decoration {
            DecorationBuiltIn => {
                dec.builtin = true;
                dec.builtin_type = FromPrimitive::from_u32(argument).unwrap();
            }
            DecorationLocation => {
                dec.location = argument;
            }
            DecorationComponent => {
                dec.component = argument;
            }
            DecorationOffset => {
                dec.offset = argument;
            }
            DecorationArrayStride => {
                dec.array_stride = argument;
            }
            DecorationMatrixStride => {
                dec.matrix_stride = argument;
            }
            DecorationBinding => {
                dec.binding = argument;
            }
            DecorationDescriptorSet => {
                dec.set = argument;
            }
            DecorationInputAttachmentIndex => {
                dec.input_attachment = argument;
            }
            DecorationSpecId => {
                dec.spec_id = argument;
            }
            DecorationIndex => {
                dec.index = argument;
            }
            DecorationHlslCounterBufferGOOGLE => {
                meta.hlsl_magic_counter_buffer = argument;
                meta.hlsl_is_magic_counter_buffer = true;
            }
            DecorationFPRoundingMode => {
                dec.fp_rounding_mode = FromPrimitive::from_u32(argument).unwrap();;
            }
            _ => ()
        }
    }

    pub fn set_decoration_string(
        &mut self,
        id: u32,
        decoration: spv::Decoration,
        argument: String,
    ) {
        let meta = self.meta
            .get_mut(&id)
            .unwrap();
        meta
            .decoration
            .decoration_flags.set(decoration as u32);

        match decoration {
            spv::Decoration::DecorationHlslSemanticGOOGLE => {
                meta
                    .decoration
                    .hlsl_semantic = argument;
            }
            _ => ()
        }
    }

    fn has_decoration(
        &self,
        id: u32,
        decoration: spv::Decoration,
    ) -> bool {
        self.get_decoration_bitset(id)
            .get(decoration as u32)
    }

    pub fn get_decoration(
        &self,
        id: u32,
        decoration: spv::Decoration,
    ) -> u32 {
        let meta = match self.find_meta(id) {
            None => return 0,
            Some(m) => m,
        };

        let dec = &meta.decoration;
        if !dec.decoration_flags
            .get(decoration as u32) {
            return 0;
        }

        use spv::Decoration::*;

        match decoration {
            DecorationBuiltIn=> dec.builtin_type as u32,
            DecorationLocation=> dec.location,
            DecorationComponent=> dec.component,
            DecorationOffset=> dec.offset,
            DecorationBinding=> dec.binding,
            DecorationDescriptorSet=> dec.set,
            DecorationInputAttachmentIndex=> dec.input_attachment,
            DecorationSpecId=> dec.spec_id,
            DecorationArrayStride=> dec.array_stride,
            DecorationMatrixStride=> dec.matrix_stride,
            DecorationIndex=> dec.index,
            DecorationFPRoundingMode=> dec.fp_rounding_mode as u32,
            _ => 1,
        }
    }

    pub fn get_decoration_string(
        &self,
        id: u32,
        decoration: spv::Decoration,
    ) -> String {
        let meta = match self.find_meta(id) {
            None => return self.empty_string.to_owned(),
            Some(m) => m,
        };

        let dec = &meta.decoration;
        if !dec.decoration_flags.get(decoration as u32) {
            return self.empty_string.to_owned();
        }

        use spv::Decoration::*;

        match decoration {
            DecorationHlslSemanticGOOGLE => dec.hlsl_semantic.to_owned(),
            _ => self.empty_string.to_owned(),
        }
    }

    fn get_decoration_bitset(
        &self,
        id: u32,
    ) -> &Bitset {
        if let Some(meta) = self.find_meta(id) {
            &meta.decoration.decoration_flags
        } else {
            &self.cleared_bitset
        }
    }

    fn unset_decoration(
        &mut self,
        id: u32,
        decoration: spv::Decoration,
    ) {
        let meta = self
            .meta
            .get_mut(&id)
            .unwrap();
        let dec = &mut meta.decoration;
        dec.decoration_flags.clear(decoration as u32);

        use spv::Decoration::*;

        match decoration {
            DecorationBuiltIn=> {
                dec.builtin = false;
            }
            DecorationLocation=> {
                dec.location = 0;
            }
            DecorationComponent=> {
                dec.component = 0;
            }
            DecorationOffset=> {
                dec.offset = 0;
            }
            DecorationBinding=> {
                dec.binding = 0;
            }
            DecorationDescriptorSet=> {
                dec.set = 0;
            }
            DecorationInputAttachmentIndex=> {
                dec.input_attachment = 0;
            }
            DecorationSpecId=> {
                dec.spec_id = 0;
            }
            DecorationHlslSemanticGOOGLE=> {
                dec.hlsl_semantic.clear();
            }
            DecorationFPRoundingMode=> {
                dec.fp_rounding_mode = spv::FPRoundingMode::FPRoundingModeMax;
            }
            DecorationHlslCounterBufferGOOGLE=> {
                let counter = meta.hlsl_magic_counter_buffer;
                if counter != 0 {
                    self
                        .meta
                        .get_mut(&counter)
                        .unwrap()
                        .hlsl_is_magic_counter_buffer = false;
                    meta.hlsl_magic_counter_buffer = 0;
                }
            }
            _ => (),
        }
    }

    pub fn set_member_name(
        &mut self,
        id: u32,
        index: u32,
        name: String,
    ) {
        let meta = self.meta
            .get_mut(&id)
            .unwrap();
        meta.members.resize(
            max(
                meta.members.len(),
                index as usize + 1,
            ),
            Decoration::default(),
        );
        let mut _str = &meta.members[index as usize].alias;
        _str.clear();
        if name.is_empty() {
            return;
        }

        // Reserved for unnamed members.
        if name.as_bytes()[0] as char == '_' && name.len() >= 3
            && name.as_bytes()[1] as char == 'm'
            && (name.as_bytes()[2] as char).is_digit(10) {
            return;
        }

        meta.members[index as usize].alias
            = ensure_valid_identifier(name, true);
    }

    fn get_member_name(
        &self,
        id: u32,
        index: usize,
    ) -> String {
        if let Some(m) = self.find_meta(id) {
            if index >= m.members.len() {
                self.empty_string.to_owned()
            } else {
                m.members[index].alias
            }
        } else {
            self.empty_string.to_owned()
        }
    }

    fn set_member_decoration(
        &mut self,
        id: u32,
        index: u32,
        decoration: spv::Decoration,
        argument: impl Into<Option<u32>>,
    ) {
        let argument = argument.into().unwrap_or(0);
        let meta = self.meta
            .get_mut(&id)
            .unwrap();
        meta.members.resize(
            max(
                meta.members.len(),
                index as usize,
            ),
            Decoration::default(),
        );
        let dec = &mut meta.members[index as usize];
        dec.decoration_flags.set(decoration as u32);

        match decoration {
            DecorationBuiltIn=> {
                dec.builtin = true;
                dec.builtin_type = FromPrimitive::from_u32(argument).unwrap();
            }
            DecorationLocation=> {
                dec.location = argument;
            }
            DecorationComponent=> {
                dec.component = argument;
            }
            DecorationBinding=> {
                dec.binding = argument;
            }
            DecorationOffset=> {
                dec.offset = argument;
            }
            DecorationSpecId=> {
                dec.spec_id = argument;
            }
            DecorationMatrixStride=> {
                dec.matrix_stride = argument;
            }
            DecorationIndex=> {
                dec.index = argument;
            }
            _ => ()
        }
    }

    fn set_member_decoration_string(
        &mut self,
        id: u32,
        index: u32,
        decoration: spv::Decoration,
        argument: String,
    ) {
        let meta = self.meta
            .get_mut(&id)
            .unwrap();

        meta.members.resize(
            max(
                meta.members.len(),
                index as usize,
            ),
            Decoration::default(),
        );

        let dec = &mut meta.members[index as usize];
        dec.decoration_flags.set(decoration as u32);

        use spv::Decoration::*;

        match decoration {
            DecorationHlslSemanticGOOGLE=> {
                dec.hlsl_semantic = argument;
            }
            _ => (),
        }
    }

    fn get_member_decoration(
        &self,
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) -> u32 {
        let meta = match self.find_meta(id) {
            None => return 0,
            Some(m) => m,
        };

        if index >= meta.members.len() as u32 {
            return 0;
        }

        let dec = &meta.members[index as usize];
        if !dec.decoration_flags.get(decoration as u32) {
            return 0;
        }

        match decoration {
            DecorationBuiltIn=> dec.builtin_type as u32,
            DecorationLocation=> dec.location,
            DecorationComponent=> dec.component,
            DecorationBinding=> dec.binding,
            DecorationOffset=> dec.offset,
            DecorationSpecId=> dec.spec_id,
            DecorationIndex=> dec.index,
            _ => 1,
        }
    }

    fn get_member_decoration_string(
        &self,
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) -> String {
        let meta = match self.find_meta(id) {
            None => return self.empty_string.to_owned(),
            Some(m) => m,
        };

        if !self.has_member_decoration(id, index, decoration) {
            return self.empty_string.to_owned();
        }

        let dec = &meta.members[index as usize];

        use spv::Decoration::*;

        match decoration {
            DecorationHlslSemanticGOOGLE => dec.hlsl_semantic.to_owned(),
            _ => self.empty_string.to_owned(),
        }
    }

    fn has_member_decoration(
        &self,
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) -> bool {
        self.get_member_decoration_bitset(id, index)
            .get(decoration as u32)
    }

    fn get_member_decoration_bitset(
        &self,
        id: u32,
        index: u32,
    ) -> &Bitset {
        if let Some(meta) = self.find_meta(id) {
            if index >= meta.members.len() as u32 {
                return &self.cleared_bitset;
            }
            return &meta.members[index as usize]
                .decoration_flags;
        }
        return &self.cleared_bitset;
    }

    fn unset_member_decoration(
        &mut self,
        id: u32,
        index: u32,
        decoration: spv::Decoration,
    ) {
        let meta = self.meta
            .get_mut(&id)
            .unwrap();

        if index >= meta.members.len() as u32 {
            return;
        }

        let dec = &mut meta.members[index as usize];
        dec.decoration_flags.clear(decoration as u32);

        use spv::Decoration::*;

        match decoration {
            DecorationBuiltIn=> {
                dec.builtin = false;
            }
            DecorationLocation=> {
                dec.location = 0;
            }
            DecorationComponent=> {
                dec.component = 0;
            }
            DecorationOffset=> {
                dec.offset = 0;
            }
            DecorationSpecId=> {
                dec.spec_id = 0;
            }
            DecorationHlslSemanticGOOGLE=> {
                dec.hlsl_semantic.clear();
            }
            _ => (),
        }
    }

    // Recursively marks any constants referenced by the specified constant instruction as being used
    // as an array length. The id must be a constant instruction (SPIRConstant or SPIRConstantOp).
    fn mark_used_as_array_length(&mut self, id: u32) {
        let variant = &mut self.ids[id as usize];
        match variant.get_type() {
            Types::TypeConstant => {
                (variant
                    .holder
                    .unwrap() as SPIRConstant)
                    .is_used_as_array_length = true;
            }
            Types::TypeConstantOp => {
                let cop = (
                    variant
                        .holder
                        .unwrap() as SPIRConstantOp
                    );
                for arg_id in cop.arguments {
                    self.mark_used_as_array_length(arg_id);
                }
            }
            Types::TypeUndef => (),
            _ => panic!(),
        }
    }
    fn increase_bound_by(
        &mut self,
        incr_amount: u32,
    ) -> u32 {
        let curr_bound = self.ids.len() as u32;
        let new_bound = curr_bound + incr_amount;
        self.ids
            .resize(new_bound, Variant::default());
        self.block_meta
            .resize(new_bound as usize, BlockMetaFlags::default());
        curr_bound as u32
    }
    fn get_buffer_block_flags(&self, var: SPIRVariable) -> Bitset {
        let _type = self.ids[var.basetype as usize]
            .holder
            .unwrap() as SPIRType;
        assert!(_type.basetype as u32 == BaseType::Struct as u32);

        // Some flags like non-writable, non-readable are actually found
        // as member decorations. If all members have a decoration set, propagate
        // the decoration up as a regular variable decoration.
        let mut base_flags = Bitset::default();
        if let Some(meta) = self.find_meta(var.get_self()) {
            base_flags = meta.decoration.decoration_flags;
        }

        if _type.member_types.is_empty() {
            return base_flags;
        }

        let mut all_member_flags = self.get_member_decoration_bitset(
            _type.get_self(),
            0,
        );
        for i in 1.._type.member_types.len() as u32 {
            all_member_flags.merge_and(
                self.get_member_decoration_bitset(
                    _type.get_self(),
                    i,
                )
            )
        }

        base_flags.merge_or(all_member_flags);
        base_flags
    }

    pub fn add_typed_id<T: HasType>(
        &mut self,
        id: u32,
    ) {
        if self.loop_iteration_depth != 0 {
            panic!("Cannot add typed ID while looping over it.");
        }

        match T::get_type() {
            Types::TypeConstant => {
                self.ids_for_constant_or_variable.push(id);
                self.ids_for_constant_or_type.push(id);
            }
            Types::TypeVariable => {
                self.ids_for_constant_or_variable.push(id);
            }
            Types::TypeType => {
                self.ids_for_constant_or_type.push(id);
            }
            Types::TypeConstantOp => {
                self.ids_for_constant_or_type.push(id);
            }
            _ => (),
        }

        if self.ids[id as usize].empty() {
            self.ids_for_type[T::get_type() as usize]
                .push(id);
        } else if self.ids[id as usize].get_type() as u32 != T::get_type() as u32 {
            self.remove_typed_id(
                self.ids[id as usize].get_type(),
                id,
            );
            self.ids_for_type[T::get_type() as usize].push(id);
        }
    }
    fn remove_typed_id(
        &mut self,
        _type: Types,
        id: u32,
    ) {
        let type_ids = &mut self.ids_for_type[_type as usize];
        type_ids.retain(|x| *x != id);
    }

    fn for_each_typed_id<T: HasType>(&mut self, op: impl Fn(u32, T)) {
        self.loop_iteration_depth += 1;

        // todo: do something with it
        for id in self.ids_for_type[T::get_type() as usize] {
            if self.ids[id as usize].get_type() as u32 == T::get_type() as u32 {
                op(id, self.get::<T>(id));
            }
        }

        self.loop_iteration_depth -= 1;
    }

    fn reset_all_of_type(&mut self, _type: Types) {
        for id in self.ids_for_type[_type as usize] {
            if self.ids[id as usize].get_type() as u32 == _type as u32 {
                self.ids[id as usize].reset();
            }
        }
        self.ids_for_type[_type as usize].clear();
    }
    fn find_meta(&self, id: u32) -> Option<&Meta> {
        self.meta.get(&id)
    }

    fn get_empty_string(&self) -> String {
        self.empty_string.to_owned()
    }

    fn get<T: HasType>(&self, id: u32) -> T {
        self.ids[id as usize].get::<T>()
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
        let mut c = _str.as_bytes()[i] as char;
        if member {
            // _m<num> variables are reserved by the internal implementation,
            // otherwise, make sure the name is a valid identifier.
            if i == 0 {
                c = if c.is_alphabetic() { c } else { '_' };
            } else if i == 2 && _str.as_bytes()[0] as char == '_' && _str.as_bytes()[1]  as char == 'm' {
                c = if c.is_alphabetic() { c } else { '_' };
            } else {
                c = if c.is_alphanumeric() { c } else { '_' };
            }
        } else {
            // _<num> variables are reserved by the internal implementation,
            // otherwise, make sure the name is a valid identifier.
            if i == 0 || (_str.as_bytes()[0] as char == '_' && i == 1) {
                c = if c.is_alphabetic() { c } else { '_' };
            } else {
                c = if c.is_alphanumeric() { c } else { '_' };
            }
        }
        _str.as_bytes()[i] = c as u8;
    }
    return _str;
}
