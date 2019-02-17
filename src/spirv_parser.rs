use crate::spirv_common::{
    Instruction,
    SPIRBlock,
    SPIRFunction,
};
use crate::spirv_cross_parsed_ir::{
    ParsedIR,
};
use crate::spirv as spv;
use crate::spirv::MAGIC_NUMBER;

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
    fn parse_instruction(&self, instr: &Instruction) {

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

    fn stream(&self, instr: &Instruction) -> Option<u32> {
        // If we're not going to use any arguments, just return nullptr.
        // We want to avoid case where we return an out of range pointer
        // that trips debug assertions on some platforms.
        if instr.len() == 0 {
            return None;
        }
    }

    fn get_parsed_ir(&self) -> &ParsedIR {
        &self.ir
    }
}