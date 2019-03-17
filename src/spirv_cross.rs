use crate::spirv_common::VariantHolder;

use crate::spirv_cross_parsed_ir::ParsedIR;

pub trait Compiler {
    fn get_ir(&self) -> &ParsedIR;
    fn get_ir_mut(&mut self) -> &mut ParsedIR;
    fn get(&self, id: usize) -> &VariantHolder {
        self.get_ir().ids[id].get()
    }
}
