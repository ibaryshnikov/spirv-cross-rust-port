use num::FromPrimitive;

use crate::spirv_common::{
    HasType,
    Instruction,
    SPIRBlock,
    SPIREntryPoint,
    SPIRExtension,
    Extension,
    SPIRFunction,
    SPIRUndef,
};
use crate::spirv_cross_parsed_ir::{
    ParsedIR,
};
use crate::spirv::{
    self as spv,
    MAGIC_NUMBER,
    ExecutionMode::{self, *},
    ExecutionModel::{self, *},
    Capability::{self, *},
    Op::{self, *},
    SourceLanguage::{self, *},
};
use crate::spirv::Capability::CapabilityKernel;
use crate::spirv::Capability;

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
                self.set::<SPIRUndef>(id, result_type);
            }

            OpCapability => {
                let cap = FromPrimitive::from_u32(ops[0])
                    .unwrap();
                if cap == CapabilityKernel {
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
                self.set::<SPIRExtension>(id, kind as u32);

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
                for i in 0..ops[strlen_words + 2] {
                    entry_point.interface_variables.push(ops[instruction.length]);
                }


                // Set the name of the entry point in case OpName is not provided later.
                self.ir.set_name(ops[1], entry_point.name.clone());

                // If we don't have an entry, make the first one our "default".
                if self.ir.default_entry_point == 0 {
                    self.ir.default_entry_point = ops[1];
                }

                self.ir.entry_points.insert(ops[1], entry_point)
            }

            OpExecutionMode => {
                let mut entry_point = self.ir.entry_points[ops[0]];
                let mode = FromPrimitive::from_u32(ops[1])
                    .unwrap();
                entry_point.flags.set(mode);

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
                        execution.output_vertices = ops[2];
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
                        &self.ir.spirve,
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
                let flags = &decorations.decoration_flags;

                // Copies decorations from one ID to another. Only copy decorations which are set in the group,
                // i.e., we cannot just copy the meta structure directly.
                for i in 1..length {
                    let target = ops[i];
                    flags.for_each_bit(|bit: u32| {
                    let decoration = FromPrimitive::from_u32(bit)
                        .unwrap();

                    if Self::decoration_is_string(decoration) {
                        self.ir.set_decoration_string(target, decoration, ir.get_decoration_string(group_id, decoration));
                    } else {
                        self.ir.meta.get_mut(target)
                            .unwrap()
                            .decoration_word_offset
                            .insert(
                                decoration as u32,
                                self.ir.meta.get(&group_id)
                                    .unwrap()
                                    .decoration_word_offset
                                    .get(decoration as &u32)
                                    .unwrap()
                                    .clone()
                            );
                        self.ir.set_decoration(target, decoration, ir.get_decoration(group_id, decoration));
                    }
                });
                }
            }

        }
    }
    fn parse(&mut self) {
        let spirv = &self.ir.spirv;

        let len = spirv.len();
        if len < 5 {
            panic!("SPIRV file too small.");
        }

        let mut s = spirv;

        // Endian-swap if we need to.
        if s[0] == spv::MAGIC_NUMBER.swap_bytes() {
            s = s
                .iter()
                .map(|x| x.swap_bytes())
                .collect();
        }

        if s[0] != spv::MAGIC_NUMBER
            || !Parser::is_valid_spirv_version(s[1]) {
            panic!("Invalid SPIRV format.");
        }

        let bound = s[3];
        self.ir.set_id_bounds(bound);

        let mut offset = 5;

        let mut instructions = vec![];

        while offset < len {
            let mut instr = Instruction::default();
            instr.op = spirv[offset] as u16 & 0xffff;
            instr.count = (spirv[offset] >> 16) as u16 & 0xffff;

            if instr.count == 0 {
                panic!("SPIR-V instructions cannot consume 0 words. Invalid SPIR-V file.");
            }

            instr.offset = offset as u32 + 1;
            instr.length = instr.count as u32 - 1;

            offset += instr.count;

            if offset > s.len() {
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

    fn stream(&self, instr: &Instruction) -> Option<&[u32]> {
        // If we're not going to use any arguments, just return nullptr.
        // We want to avoid case where we return an out of range pointer
        // that trips debug assertions on some platforms.
        if instr.len() == 0 {
            return None;
        }

        if (instr.offset + instr.length) as usize > self.ir.spirv.len() {
            panic!("Compiler::stream() out of range.");
        }
        Some(&self.ir.spirv[instr.offset as usize..])
    }

    fn extract_string(spirv: &Vec<u32>, offset: u32) -> String {
        let mut ret = String::new();

        for i in offset..spirv.len() {
            let mut w = spirv[i];

            for j in 0..4 {
                w >>= 8;

                let c: char = w & 0xff;
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

    fn set<T: HasType>(&mut self, id: u32, _type: u32) -> &T {
        self.ir.add_typed_id::<T>(id);
        let mut var = &self.ir.ids[id as usize];
        // TODO: fix this
        //  var.set::<T>(id, )
    }
}
