use num::FromPrimitive;

use crate::spirv_common::{
    BaseType,
    IVariant,
    Instruction,
    SPIRBlock,
    SPIREntryPoint,
    SPIRExtension,
    SPIRType,
    Extension,
    SPIRFunction,
    SPIRUndef,
    Types,
    to_signed_basetype,
    to_unsigned_basetype,
};
use crate::spirv_cross_parsed_ir::{
    ParsedIR,
};
use crate::spirv::{
    self as spv,
    AccessQualifier,
    ExecutionMode::*,
    Op::*,
    SourceLanguage::*,
    StorageClass,
};
use crate::spirv::Capability::CapabilityKernel;
use crate::spirv_common::VariantHolder;

struct Parser {
    ir: ParsedIR,
    current_function: Option<SPIRFunction>,
    current_block: Option<SPIRBlock>,
    global_struct_cache: Vec<u32>,
}

impl Default for Parser {
    fn default() -> Self {
        Parser {
            ir: ParsedIR::default(),
            current_function: None,
            current_block: None,
            global_struct_cache: vec![],
        }
    }
}

impl Parser {
    fn new_with_spirv(spirv: Vec<u32>) -> Self {
        let mut parser = Parser::default();
        parser.ir.spirv = spirv;
        parser
    }
    fn decoration_is_string(decoration: spv::Decoration) -> bool {
        match decoration {
            spv::Decoration::DecorationHlslSemanticGOOGLE => true,
            _ => false,
        }
    }
    fn is_valid_spirv_version(version: u32) -> bool {
        match version {
            // Allow v99 since it tends to just work.
            99 => true,
            0x10000 => true, // SPIR-V 1.0
            0x10100 => true, // SPIR-V 1.1
            0x10200 => true, // SPIR-V 1.2
            0x10300 => true, // SPIR-V 1.3
            _ => false,
        }
    }
    fn swap_endian(v: u32) -> u32 {
        v.swap_bytes()
    }
    #[allow(clippy::cyclomatic_complexity)]
    fn parse_instruction(&mut self, instruction: &Instruction) {
        let ops = self.stream(instruction);
        let op = FromPrimitive::from_u16(instruction.op)
            .unwrap();
        let length = instruction.length;

        // TODO: choose the behavior for None
        let ops = ops.unwrap();

        match op {
            OpMemoryModel => (),
            OpSourceContinued => (),
            OpSourceExtension => (),
            OpNop => (),
            OpLine => (),
            OpNoLine => (),
            OpString => (),
            OpModuleProcessed => (),

            OpSource => {
                let lang = FromPrimitive::from_u32(ops[0])
                    .unwrap();

                match lang {
                    SourceLanguageESSL => {
                        self.ir.source.es = true;
                        self.ir.source.version = ops[1];
                        self.ir.source.known = true;
                        self.ir.source.hlsl = false;
                    }
                    SourceLanguageGLSL => {
                        self.ir.source.es = false;
                        self.ir.source.version = ops[1];
                        self.ir.source.known = true;
                        self.ir.source.hlsl = false;
                    }
                    SourceLanguageHLSL => {
                        // For purposes of cross-compiling, this is GLSL 450.
                        self.ir.source.es = false;
                        self.ir.source.version = 450;
                        self.ir.source.known = true;
                        self.ir.source.hlsl = true;
                    }
                    _ => {
                        self.ir.source.known = false;
                    }
                }

            }

            OpUndef => {
                let result_type = ops[0];
                let id = ops[1];
                let mut value = Box::new(SPIRUndef::new(result_type));
                value.set_self(id);

                self.set(id, VariantHolder::SPIRUndef(value));
            }

            OpCapability => {
                let cap = FromPrimitive::from_u32(ops[0])
                    .unwrap();
                if let CapabilityKernel = cap {
                    panic!("Kernel capability not supported.");
                }
                self.ir.declared_capabilities.push(cap);
            }

            OpExtension => {
                let ext = Self::extract_string(
                    &self.ir.spirv,
                    instruction.offset,
                );
                self.ir.declared_extensions.push(ext);
            }

            OpExtInstImport => {
                let id = ops[0];
                let ext = Self::extract_string(
                    &self.ir.spirv,
                    instruction.offset + 1,
                );
                let kind = Extension::from_str(&ext);
                let mut value = Box::new(SPIRExtension::new(kind));
                value.set_self(id);
                self.set(id, VariantHolder::SPIRExtension(value));

                // Other SPIR-V extensions which have ExtInstrs are currently not supported.
            }

            OpEntryPoint => {
                let model = FromPrimitive::from_u32(ops[0])
                    .unwrap();

                let mut entry_point = SPIREntryPoint::new(
                    ops[1],
                    model,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 2),
                );

                // Strings need nul-terminator and consume the whole word.
                let strlen_words = (entry_point.name.len() + 1 + 3) >> 2;
                for _i in 0..ops[strlen_words + 2] {
                    entry_point.interface_variables.push(ops[instruction.length as usize]);
                }


                // Set the name of the entry point in case OpName is not provided later.
                self.ir.set_name(ops[1], entry_point.name.clone());

                // If we don't have an entry, make the first one our "default".
                if self.ir.default_entry_point == 0 {
                    self.ir.default_entry_point = ops[1];
                }

                self.ir.entry_points.insert(ops[1], entry_point);
            }

            OpExecutionMode => {
                let entry_point = self.ir.entry_points
                    .get_mut(&ops[0])
                    .unwrap();
                let mode: spv::ExecutionMode = FromPrimitive::from_u32(ops[1])
                    .unwrap();
                entry_point.flags.set(mode.clone() as u32);

                match mode {
                    ExecutionModeInvocations => {
                        entry_point.invocations = ops[2];
                    }

                    ExecutionModeLocalSize => {
                        entry_point.workgroup_size.x = ops[2];
                        entry_point.workgroup_size.y = ops[3];
                        entry_point.workgroup_size.z = ops[4];
                    }

                    ExecutionModeOutputVertices => {
                        entry_point.output_vertices = ops[2];
                    }

                    _ => (),
                }
            }

            OpName => {
                let id = ops[0];
                self.ir.set_name(
                    id,
                    Self::extract_string(
                        &self.ir.spirv,
                        instruction.offset + 1,
                    ),
                );
            }

            OpMemberName => {
                let id = ops[0];
                let member = ops[1];
                self.ir.set_member_name(
                    id,
                    member,
                    Self::extract_string(
                        &self.ir.spirv,
                        instruction.offset + 2,
                    ),
                )
            }

            OpDecorationGroup => {
                // Noop, this simply means an ID should be a collector of decorations.
                // The meta array is already a flat array of decorations which will contain the relevant decorations.
            }

            OpGroupDecorate => {
                let group_id = ops[0];
                let decorations = &self.ir.meta
                    .get_mut(&group_id)
                    .unwrap()
                    .decoration;
                let flags = &decorations.decoration_flags.clone();

                // Copies decorations from one ID to another. Only copy decorations which are set in the group,
                // i.e., we cannot just copy the meta structure directly.
                for i in 1..length {
                    let target = ops[i as usize];
                    flags.for_each_bit(|bit: u32| {
                    let decoration = FromPrimitive::from_u32(bit)
                        .unwrap();

                    if Self::decoration_is_string(decoration) {
                        self.ir.set_decoration_string(
                            target,
                            decoration,
                            self.ir.get_decoration_string(group_id, decoration),
                        );
                    } else {
                        let offset = self.ir.meta[&group_id]
                            .decoration_word_offset[&(decoration as u32)];
                        self.ir.meta.get_mut(&target)
                            .unwrap()
                            .decoration_word_offset
                            .insert(
                                decoration as u32,
                                offset,
                            );
                        self.ir.set_decoration(
                            target,
                            decoration,
                            self.ir.get_decoration(group_id, decoration),
                        );
                    }
                });
                }
            }

            OpGroupMemberDecorate => {
                let group_id = ops[0];
                let flags = &self.ir.meta[&group_id]
                    .decoration.decoration_flags
                    .clone();

                // Copies decorations from one ID to another. Only copy decorations which are set in the group,
                // i.e., we cannot just copy the meta structure directly.
                let mut i = 1;
                let limit = length as usize - 1;
                while i < limit {
                    let target = ops[i];
                    let index = ops[i + 1];
                    flags.for_each_bit(|bit: u32| {
                        let decoration = FromPrimitive::from_u32(bit)
                            .unwrap();
                        if Self::decoration_is_string(decoration) {
                            self.ir.set_member_decoration_string(
                                target,
                                index,
                                decoration,
                                self.ir.get_decoration_string(group_id, decoration),
                            );
                        } else {
                            self.ir.set_member_decoration(
                                target,
                                index,
                                decoration,
                                self.ir.get_decoration(group_id, decoration));
                        }
                    });
                    i += 2;
                }
            }

            OpDecorate => {
                // OpDecorateId technically supports an array of arguments, but our only supported decorations are single uint,
                // so merge decorate and decorate-id here.
                let id = ops[0];

                let decoration: spv::Decoration = FromPrimitive::from_u32(ops[1])
                    .unwrap();
                if length >= 3 {
//                    self.ir.meta
//                        .get_mut(&id)
//                        .unwrap()
//                        .decoration_word_offset
//                        // TODO: fix data() thing
//                        .insert(decoration as u32, (&ops[2] - self.ir.spirv.data()) as u32);
                    self.ir.set_decoration(id, decoration, ops[2]);
                } else {
                    self.ir.set_decoration(id, decoration, None);
                }
            }

            OpDecorateId => {
                // OpDecorateId technically supports an array of arguments, but our only supported decorations are single uint,
                // so merge decorate and decorate-id here.
                let id = ops[0];

                let decoration: spv::Decoration = FromPrimitive::from_u32(ops[1])
                    .unwrap();
                if length >= 3 {
//                    self.ir.meta
//                        .get_mut(&id)
//                        .unwrap()
//                        .decoration_word_offset
//                        // TODO: fix data thing
//                        .insert(decoration as u32, (&ops[2] - self.ir.spirv.data()) as u32);
//                    self.ir.set_decoration(id, decoration, ops[2]);
                } else {
                    self.ir.set_decoration(id, decoration, None);
                }
            }

            OpDecorateStringGOOGLE => {
                let id = ops[0];
                let decoration = FromPrimitive::from_u32(ops[1])
                    .unwrap();
                self.ir.set_decoration_string(
                    id,
                    decoration,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 2),
                );
            }

            OpMemberDecorate => {
                let id = ops[0];
                let member = ops[1];
                let decoration = FromPrimitive::from_u32(ops[2])
                    .unwrap();
                if length >= 4 {
                    self.ir.set_member_decoration(
                        id,
                        member,
                        decoration,
                        ops[3],
                    );
                } else {
                    self.ir.set_member_decoration(
                        id,
                        member,
                        decoration,
                        None,
                    );
                }
            }

            OpMemberDecorateStringGOOGLE => {
                let id = ops[0];
                let member = ops[1];
                let decoration = FromPrimitive::from_u32(ops[2])
                    .unwrap();
                self.ir.set_member_decoration_string(
                    id,
                    member,
                    decoration,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 3),
                );
            }

            OpTypeVoid => {
                let id = ops[0];
                let mut value = Box::new(SPIRType::default());
                value.basetype = BaseType::Void;
                self.set(id, VariantHolder::SPIRType(value));
            }

            OpTypeBool => {
                let id = ops[0];
                let mut value = Box::new(SPIRType::default());
                value.basetype = BaseType::Boolean;
                value.width = 1;
                self.set(id, VariantHolder::SPIRType(value));
            }

            OpTypeFloat => {
                let id = ops[0];
		        let width = ops[1];

                let mut value = Box::new(SPIRType::default());
                if width == 64 {
                    value.basetype = BaseType::Double;
                } else if width == 32 {
                    value.basetype = BaseType::Float;
                } else if width == 16 {
                    value.basetype = BaseType::Half;
                } else {
                    panic!("Unrecognized bit-width of floating point type.");
                }
                value.width = width;
		        self.set(id, VariantHolder::SPIRType(value));
            }

            OpTypeInt => {
                let id = ops[0];
                let width = ops[1];
                let signedness = ops[2] != 0;
                let mut value = Box::new(SPIRType::default());
                value.basetype = if signedness {
                    to_signed_basetype(width)
                 } else {
                     to_unsigned_basetype(width)
                 };
                value.width = width;
                self.set(id, VariantHolder::SPIRType(value));
            }

            // Build composite types by "inheriting".
	        // NOTE: The self member is also copied! For pointers and array modifiers this is a good thing
	        // since we can refer to decorations on pointee classes which is needed for UBO/SSBO, I/O blocks in geometry/tess etc.
            OpTypeVector => {
                let id = ops[0];
                let vecsize = ops[2];

                let base = match self.get(ops[1] as usize) {
                    VariantHolder::SPIRType(value) => value,
                    _ => panic!("Bad cast"),
                };

                let mut vecbase = base.clone();
                vecbase.vecsize = vecsize;
                vecbase.set_self(id);
                vecbase.parent_type = ops[1];

                self.set(id, VariantHolder::SPIRType(vecbase));
            }

            OpTypeMatrix => {
                let id = ops[0];
                let colcount = ops[2];

                let base = match self.get(ops[1] as usize) {
                    VariantHolder::SPIRType(value) => value,
                    _ => panic!("Bad case"),
                };
                let mut matrixbase = base.clone();

                matrixbase.columns = colcount;
                matrixbase.set_self(id);
                matrixbase.parent_type = ops[1];

                self.set(id, VariantHolder::SPIRType(matrixbase));
            }

            OpTypeArray => {
                let id = ops[0];

                let tid = ops[1];
                let base = match self.get(tid as usize) {
                    VariantHolder::SPIRType(value) => value,
                    _ => panic!("Bad cast"),
                };

                let mut arraybase = base.clone();
                arraybase.parent_type = tid;

                let cid = ops[2];
                self.ir.mark_used_as_array_length(cid);
                let c = match self.get(cid as usize) {
                    VariantHolder::SPIRConstant(value) => Some(value),
                    _ => None,
                };
                let literal = c.is_some() && !c.unwrap().specialization;

                arraybase.array_size_literal.push(literal);
                arraybase.array.push(if literal { c.unwrap().scalar(None, None) } else { cid });
                // Do NOT set arraybase.self!

                self.set(id, VariantHolder::SPIRType(arraybase));
            }

            OpTypeRuntimeArray => {
                let id = ops[0];

                let base = match self.get(ops[1] as usize) {
                    VariantHolder::SPIRType(value) => value,
                    _ => panic!("Bad type"),
                };

                let mut arraybase = base.clone();
                arraybase.array.push(0);
                arraybase.array_size_literal.push(true);
                arraybase.parent_type = ops[1];
                // Do NOT set arraybase.self!

                self.set(id, VariantHolder::SPIRType(arraybase));
            }

            OpTypeImage => {
                let id = ops[0];

                let mut _type = Box::new(SPIRType::default());
                _type.basetype = BaseType::Image;
                _type.image._type = ops[1];
                _type.image.dim = FromPrimitive::from_u32(ops[2]).unwrap();
                _type.image.depth = ops[3] == 1;
                _type.image.arrayed = ops[4] != 0;
                _type.image.ms = ops[5] != 0;
                _type.image.sampled = ops[6];
                _type.image.format = FromPrimitive::from_u32(ops[7]).unwrap();
                _type.image.access = if length >= 9 {
                    FromPrimitive::from_u32(ops[8]).unwrap()
                } else {
                    AccessQualifier::AccessQualifierMax
                };

                if _type.image.sampled == 0 {
                    panic!("OpTypeImage Sampled parameter must not be zero.");
                }

                self.set(id, VariantHolder::SPIRType(_type));
            }

            OpTypeSampledImage => {
                let id = ops[0];
                let imagetype = ops[1];
                let base = match self.get(imagetype as usize) {
                    VariantHolder::SPIRType(value) => value,
                    _ => panic!("Bad cast"),
                };
                let mut _type = base.clone();
                _type.basetype = BaseType::SampledImage;
                _type.set_self(id);
                self.set(id, VariantHolder::SPIRType(_type));
            }

            OpTypeSampler => {
                let id = ops[0];
                let mut _type = Box::new(SPIRType::default());
                _type.basetype = BaseType::Sampler;
                self.set(id, VariantHolder::SPIRType(_type));
            }

            OpTypePointer => {
                let id = ops[0];

                let base = match self.get(ops[2] as usize) {
                    VariantHolder::SPIRType(value) => value,
                    _ => panic!("Bad cast"),
                };

                let mut ptrbase = base.clone();
                ptrbase.pointer = true;
                ptrbase.pointer_depth += 1;
                ptrbase.storage = FromPrimitive::from_u32(ops[1]).unwrap();

                if let StorageClass::StorageClassAtomicCounter = ptrbase.storage {
                    ptrbase.basetype = BaseType::AtomicCounter;
                };

                ptrbase.parent_type = ops[2];

                // Do NOT set ptrbase.self!
                self.set(id, VariantHolder::SPIRType(ptrbase));
            }

            OpTypeStruct => {
                // ...
            }

            _ => {

            }

        }
    }
    fn parse(&mut self) {
        let (bound, len) = {
            let spirv = &mut self.ir.spirv;

            let len = spirv.len();
            if len < 5 {
                panic!("SPIRV file too small.");
            }

            // Endian-swap if we need to.
            if spirv[0] == spv::MAGIC_NUMBER.swap_bytes() {
                #[allow(clippy::needless_range_loop)]
                for i in 0..len {
                    spirv[i] = spirv[i].swap_bytes();
                }
            }

            if spirv[0] != spv::MAGIC_NUMBER
                || !Parser::is_valid_spirv_version(spirv[1]) {
                panic!("Invalid SPIRV format.");
            }

            let bound = spirv[3];

            (bound, len)
        };
        self.ir.set_id_bounds(bound);

        let spirv = &self.ir.spirv;
        let mut offset = 5;

        let mut instructions = vec![];

        while offset < len {
            let mut instr = Instruction::default();
            instr.op = spirv[offset] as u16;
            instr.count = (spirv[offset] >> 16) as u16;

            if instr.count == 0 {
                panic!("SPIR-V instructions cannot consume 0 words. Invalid SPIR-V file.");
            }

            instr.offset = offset as u32 + 1;
            instr.length = u32::from(instr.count) - 1;

            offset += instr.count as usize;

            if offset > len {
                panic!("SPIR-V instruction goes out of bounds.");
            }

            instructions.push(instr);
        }

        for instr in instructions {
            self.parse_instruction(&instr);
        }

        if self.current_function.is_some() {
            panic!("Function was not terminated.");
        }
        if self.current_block.is_some() {
            panic!("Block was not terminated.");
        }
    }

    fn stream(&self, instr: &Instruction) -> Option<Vec<u32>> {
        // If we're not going to use any arguments, just return nullptr.
        // We want to avoid case where we return an out of range pointer
        // that trips debug assertions on some platforms.
        if instr.length == 0 {
            return None;
        }

        if (instr.offset + instr.length) as usize > self.ir.spirv.len() {
            panic!("Compiler::stream() out of range.");
        }
        Some(self.ir.spirv[instr.offset as usize..].to_vec())
    }

    fn extract_string(spirv: &[u32], offset: u32) -> String {
        let mut ret = String::new();

        for item in spirv.iter().skip(offset as usize) {
            let mut w = *item;

            for _j in 0..4 {
                w >>= 8;

                let c: char = (w & 0xff) as u8 as char;
                if c == '\0' {
                    return ret;
                }
                ret.push(c);
            }
        }

        panic!("String was not terminated before EOF");
    }

    fn get_parsed_ir(&self) -> &ParsedIR {
        &self.ir
    }

    fn set(&mut self, id: u32, mut holder: VariantHolder) {
        self.ir.add_typed_id(id, holder.get_type());
        holder.set_self(id);
        self
            .ir
            .ids[id as usize]
            .set(holder);
    }

    fn get(&self, id: usize) -> &VariantHolder {
        self.ir.ids[id].get()
    }

    fn maybe_get(&self, id: usize, _type: Types) -> Option<&VariantHolder> {
        if self.ir.ids[id].get_type() as u32 == _type as u32 {
            Some(self.ir.ids[id].get())
        } else {
            None
        }
    }
}
