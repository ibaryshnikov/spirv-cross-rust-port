use crate::spirv_common::{
    VariantHolder,
};

use crate::spirv_cross_parsed_ir::{
    ParsedIR,
};

pub struct Compiler {
    ir: ParsedIR,
}

impl Compiler {
    pub fn get(&self, id: usize) -> &VariantHolder {
        self.ir.ids[id].get()
    }
}
