use num::FromPrimitive;

use crate::spirv::{
    self as spv, AccessQualifier, Capability, ExecutionMode, Op, SourceLanguage, StorageClass,
};
use crate::spirv_common::{
    to_signed_basetype, to_unsigned_basetype, BaseType, Extension, IVariant, Instruction, Phi,
    SPIRBlock, SPIRConstant, SPIREntryPoint, SPIRExtension, SPIRFunction, SPIRFunctionPrototype,
    SPIRType, SPIRUndef, SPIRVariable, Types, VariantHolder,
};
use crate::spirv_cross_parsed_ir::ParsedIR;

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
            spv::Decoration::HlslSemantic => true,
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
        let op = FromPrimitive::from_u16(instruction.op).unwrap();
        let length = instruction.length;

        // TODO: choose the behavior for None
        let ops = ops.unwrap();

        match op {
            Op::MemoryModel
            | Op::SourceContinued
            | Op::SourceExtension
            | Op::Nop
            | Op::Line
            | Op::NoLine
            | Op::String
            | Op::ModuleProcessed => {}

            Op::Source => {
                let lang = FromPrimitive::from_u32(ops[0]).unwrap();

                match lang {
                    SourceLanguage::ESSL => {
                        self.ir.source.es = true;
                        self.ir.source.version = ops[1];
                        self.ir.source.known = true;
                        self.ir.source.hlsl = false;
                    }
                    SourceLanguage::GLSL => {
                        self.ir.source.es = false;
                        self.ir.source.version = ops[1];
                        self.ir.source.known = true;
                        self.ir.source.hlsl = false;
                    }
                    SourceLanguage::HLSL => {
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

            Op::Undef => {
                let result_type = ops[0];
                let id = ops[1];
                let mut value = Box::new(SPIRUndef::new(result_type));
                value.set_self(id);

                self.set(id, VariantHolder::Undef(value));
            }

            Op::Capability => {
                let cap = FromPrimitive::from_u32(ops[0]).unwrap();
                if let Capability::Kernel = cap {
                    panic!("Kernel capability not supported.");
                }
                self.ir.declared_capabilities.push(cap);
            }

            Op::Extension => {
                let ext = Self::extract_string(&self.ir.spirv, instruction.offset);
                self.ir.declared_extensions.push(ext);
            }

            Op::ExtInstImport => {
                let id = ops[0];
                let ext = Self::extract_string(&self.ir.spirv, instruction.offset + 1);
                let kind = Extension::from_str(&ext);
                let mut value = Box::new(SPIRExtension::new(kind));
                value.set_self(id);
                self.set(id, VariantHolder::Extension(value));

                // Other SPIR-V extensions which have ExtInstrs are currently not supported.
            }

            Op::EntryPoint => {
                let model = FromPrimitive::from_u32(ops[0]).unwrap();

                let mut entry_point = SPIREntryPoint::new(
                    ops[1],
                    model,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 2),
                );

                // Strings need nul-terminator and consume the whole word.
                let strlen_words = (entry_point.name.len() + 1 + 3) >> 2;
                for _i in 0..ops[strlen_words + 2] {
                    entry_point
                        .interface_variables
                        .push(ops[instruction.length as usize]);
                }

                // Set the name of the entry point in case OpName is not provided later.
                self.ir.set_name(ops[1], entry_point.name.clone());

                // If we don't have an entry, make the first one our "default".
                if self.ir.default_entry_point == 0 {
                    self.ir.default_entry_point = ops[1];
                }

                self.ir.entry_points.insert(ops[1], entry_point);
            }

            Op::ExecutionMode => {
                let entry_point = self.ir.entry_points.get_mut(&ops[0]).unwrap();
                let mode: ExecutionMode = FromPrimitive::from_u32(ops[1]).unwrap();
                entry_point.flags.set(mode.clone() as u32);

                match mode {
                    ExecutionMode::Invocations => {
                        entry_point.invocations = ops[2];
                    }

                    ExecutionMode::LocalSize => {
                        entry_point.workgroup_size.x = ops[2];
                        entry_point.workgroup_size.y = ops[3];
                        entry_point.workgroup_size.z = ops[4];
                    }

                    ExecutionMode::OutputVertices => {
                        entry_point.output_vertices = ops[2];
                    }

                    _ => (),
                }
            }

            Op::Name => {
                let id = ops[0];
                self.ir.set_name(
                    id,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 1),
                );
            }

            Op::MemberName => {
                let id = ops[0];
                let member = ops[1];
                self.ir.set_member_name(
                    id,
                    member,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 2),
                )
            }

            Op::DecorationGroup => {
                // Noop, this simply means an ID should be a collector of decorations.
                // The meta array is already a flat array of decorations which will contain the relevant decorations.
            }

            Op::GroupDecorate => {
                let group_id = ops[0];
                let decorations = &self.ir.meta.get_mut(&group_id).unwrap().decoration;
                let flags = &decorations.decoration_flags.clone();

                // Copies decorations from one ID to another. Only copy decorations which are set in the group,
                // i.e., we cannot just copy the meta structure directly.
                for i in 1..length {
                    let target = ops[i as usize];
                    flags.for_each_bit(|bit: u32| {
                        let decoration = FromPrimitive::from_u32(bit).unwrap();

                        if Self::decoration_is_string(decoration) {
                            self.ir.set_decoration_string(
                                target,
                                decoration,
                                self.ir.get_decoration_string(group_id, decoration),
                            );
                        } else {
                            let offset = self.ir.meta[&group_id].decoration_word_offset
                                [&(decoration as u32)];
                            self.ir
                                .meta
                                .get_mut(&target)
                                .unwrap()
                                .decoration_word_offset
                                .insert(decoration as u32, offset);
                            self.ir.set_decoration(
                                target,
                                decoration,
                                self.ir.get_decoration(group_id, decoration),
                            );
                        }
                    });
                }
            }

            Op::GroupMemberDecorate => {
                let group_id = ops[0];
                let flags = &self.ir.meta[&group_id].decoration.decoration_flags.clone();

                // Copies decorations from one ID to another. Only copy decorations which are set in the group,
                // i.e., we cannot just copy the meta structure directly.
                let mut i = 1;
                let limit = length as usize - 1;
                while i < limit {
                    let target = ops[i];
                    let index = ops[i + 1];
                    flags.for_each_bit(|bit: u32| {
                        let decoration = FromPrimitive::from_u32(bit).unwrap();
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
                                self.ir.get_decoration(group_id, decoration),
                            );
                        }
                    });
                    i += 2;
                }
            }

            Op::Decorate | Op::DecorateId => {
                // Op::DecorateId technically supports an array of arguments, but our only supported decorations are single uint,
                // so merge decorate and decorate-id here.
                let id = ops[0];

                let decoration: spv::Decoration = FromPrimitive::from_u32(ops[1]).unwrap();
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

            Op::DecorateString => {
                let id = ops[0];
                let decoration = FromPrimitive::from_u32(ops[1]).unwrap();
                self.ir.set_decoration_string(
                    id,
                    decoration,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 2),
                );
            }

            Op::MemberDecorate => {
                let id = ops[0];
                let member = ops[1];
                let decoration = FromPrimitive::from_u32(ops[2]).unwrap();
                if length >= 4 {
                    self.ir
                        .set_member_decoration(id, member, decoration, ops[3]);
                } else {
                    self.ir.set_member_decoration(id, member, decoration, None);
                }
            }

            Op::MemberDecorateString => {
                let id = ops[0];
                let member = ops[1];
                let decoration = FromPrimitive::from_u32(ops[2]).unwrap();
                self.ir.set_member_decoration_string(
                    id,
                    member,
                    decoration,
                    Self::extract_string(&self.ir.spirv, instruction.offset + 3),
                );
            }

            Op::TypeVoid => {
                let id = ops[0];
                let mut value = Box::new(SPIRType::default());
                value.basetype = BaseType::Void;
                self.set(id, VariantHolder::Type(value));
            }

            Op::TypeBool => {
                let id = ops[0];
                let mut value = Box::new(SPIRType::default());
                value.basetype = BaseType::Boolean;
                value.width = 1;
                self.set(id, VariantHolder::Type(value));
            }

            Op::TypeFloat => {
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
                self.set(id, VariantHolder::Type(value));
            }

            Op::TypeInt => {
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
                self.set(id, VariantHolder::Type(value));
            }

            // Build composite types by "inheriting".
            // NOTE: The self member is also copied! For pointers and array modifiers this is a good thing
            // since we can refer to decorations on pointee classes which is needed for UBO/SSBO, I/O blocks in geometry/tess etc.
            Op::TypeVector => {
                let id = ops[0];
                let vecsize = ops[2];

                let base = self.get::<SPIRType>(ops[1] as usize);

                let mut vecbase = base.clone();
                vecbase.vecsize = vecsize;
                vecbase.set_self(id);
                vecbase.parent_type = ops[1];

                self.set(id, VariantHolder::Type(vecbase));
            }

            Op::TypeMatrix => {
                let id = ops[0];
                let colcount = ops[2];

                let base = self.get::<SPIRType>(ops[1] as usize);
                let mut matrixbase = base.clone();

                matrixbase.columns = colcount;
                matrixbase.set_self(id);
                matrixbase.parent_type = ops[1];

                self.set(id, VariantHolder::Type(matrixbase));
            }

            Op::TypeArray => {
                let id = ops[0];

                let tid = ops[1];
                let base = self.get::<SPIRType>(tid as usize);

                let mut arraybase = base.clone();
                arraybase.parent_type = tid;

                let cid = ops[2];
                self.ir.mark_used_as_array_length(cid);
                let c = match self.ir.ids[cid as usize].get() {
                    VariantHolder::Constant(value) => Some(value),
                    _ => None,
                };
                let literal = c.is_some() && !c.unwrap().specialization;

                arraybase.array_size_literal.push(literal);
                arraybase.array.push(if literal {
                    c.unwrap().scalar(None, None)
                } else {
                    cid
                });
                // Do NOT set arraybase.self!

                self.set(id, VariantHolder::Type(arraybase));
            }

            Op::TypeRuntimeArray => {
                let id = ops[0];

                let base = self.get::<SPIRType>(ops[1] as usize);

                let mut arraybase = base.clone();
                arraybase.array.push(0);
                arraybase.array_size_literal.push(true);
                arraybase.parent_type = ops[1];
                // Do NOT set arraybase.self!

                self.set(id, VariantHolder::Type(arraybase));
            }

            Op::TypeImage => {
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
                    AccessQualifier::Max
                };

                if _type.image.sampled == 0 {
                    panic!("OpTypeImage Sampled parameter must not be zero.");
                }

                self.set(id, VariantHolder::Type(_type));
            }

            Op::TypeSampledImage => {
                let id = ops[0];
                let imagetype = ops[1];
                let base = self.get::<SPIRType>(imagetype as usize);
                let mut _type = base.clone();
                _type.basetype = BaseType::SampledImage;
                _type.set_self(id);
                self.set(id, VariantHolder::Type(_type));
            }

            Op::TypeSampler => {
                let id = ops[0];
                let mut _type = Box::new(SPIRType::default());
                _type.basetype = BaseType::Sampler;
                self.set(id, VariantHolder::Type(_type));
            }

            Op::TypePointer => {
                let id = ops[0];

                let base = self.get::<SPIRType>(ops[2] as usize);

                let mut ptrbase = base.clone();
                ptrbase.pointer = true;
                ptrbase.pointer_depth += 1;
                ptrbase.storage = FromPrimitive::from_u32(ops[1]).unwrap();

                if let StorageClass::AtomicCounter = ptrbase.storage {
                    ptrbase.basetype = BaseType::AtomicCounter;
                };

                ptrbase.parent_type = ops[2];

                // Do NOT set ptrbase.self!
                self.set(id, VariantHolder::Type(ptrbase));
            }

            Op::TypeStruct => {
                let id = ops[0];

                let mut _type = Box::new(SPIRType::default());
                _type.set_self(id);
                _type.basetype = BaseType::Struct;
                for i in 1..length {
                    _type.member_types.push(ops[i as usize]);
                }

                // Check if we have seen this struct type before, with just different
                // decorations.
                //
                // Add workaround for issue #17 as well by looking at OpName for the struct
                // types, which we shouldn't normally do.
                // We should not normally have to consider type aliases like this to begin with
                // however ... glslang issues #304, #307 cover this.

                // For stripped names, never consider struct type aliasing.
                // We risk declaring the same struct multiple times, but type-punning is not allowed
                // so this is safe.
                let consider_aliasing = !self.ir.get_name(_type.get_self()).is_empty();
                if consider_aliasing {
                    for other in &self.global_struct_cache {
                        if self.ir.get_name(_type.get_self()) == self.ir.get_name(*other)
                            && self.types_are_logically_equivalent(
                                &_type,
                                self.get::<SPIRType>(*other as usize),
                            )
                        {
                            _type.type_alias = *other;
                            break;
                        }
                    }

                    if _type.type_alias == 0 {
                        self.global_struct_cache.push(id);
                    }
                }
            }

            Op::TypeFunction => {
                let id = ops[0];
                let ret = ops[1];

                let mut func = Box::new(SPIRFunctionPrototype::new(ret));

                for i in 2..length {
                    func.parameter_types.push(ops[i as usize]);
                }

                self.set(id, VariantHolder::FunctionPrototype(func));
            }

            // Variable declaration
            // All variables are essentially pointers with a storage qualifier.
            Op::Variable => {
                let _type = ops[0];
                let id = ops[1];
                let storage: StorageClass = FromPrimitive::from_u32(ops[2]).unwrap();
                let initializer = if length == 4 { ops[3] } else { 0 };

                if let StorageClass::Function = storage {
                    if let Some(value) = &mut self.current_function {
                        value.add_local_variable(id);
                    } else {
                        panic!("No function currently in scope");
                    }
                }

                let variable = Box::new(SPIRVariable::new(_type, storage, initializer, None));

                // hlsl based shaders don't have those decorations. force them and then reset when reading/writing images
                let ttype = self.get::<SPIRType>(_type as usize);
                if let BaseType::Image = ttype.basetype {
                    self.ir
                        .set_decoration(id, spv::Decoration::NonWritable, None);
                    self.ir
                        .set_decoration(id, spv::Decoration::NonReadable, None);
                }

                self.set(id, VariantHolder::Variable(variable));
            }

            Op::Phi => {
                let current_function = if let Some(value) = &mut self.current_function {
                    value
                } else {
                    panic!("No function currently in scope");
                };
                let current_block = if let Some(value) = &mut self.current_block {
                    value
                } else {
                    panic!("No block currently in scope");
                };

                let result_type = ops[0];
                let id = ops[1];

                // Instead of a temporary, create a new function-wide temporary with this ID instead.
                let mut var = Box::new(SPIRVariable::new(
                    result_type,
                    StorageClass::Function,
                    None,
                    None,
                ));

                var.phi_variable = true;

                current_function.add_local_variable(id);

                let mut i = 2usize;
                while i + 2 <= length as usize {
                    current_block
                        .phi_variables
                        .push(Phi::new(ops[i], ops[i + 1], id));
                    i += 2;
                }

                self.set(id, VariantHolder::Variable(var));
            }

            Op::SpecConstant | Op::Constant => {
                let id = ops[1];
                let _type = self.get::<SPIRType>(ops[0] as usize);

                let constant = if _type.width > 32 {
                    SPIRConstant::new_with_scalar_u64_and_specialization(
                        ops[0],
                        u64::from(ops[2]) | (u64::from(ops[3]) << 32),
                        op == Op::SpecConstant,
                    )
                } else {
                    SPIRConstant::new_with_scalar_u32_and_specialization(
                        ops[0],
                        ops[2],
                        op == Op::SpecConstant,
                    )
                };
                self.set(id, VariantHolder::Constant(Box::new(constant)));
            }

            Op::SpecConstantFalse | Op::ConstantFalse => {
                let id = ops[1];
                let constant = SPIRConstant::new_with_scalar_u32_and_specialization(
                    ops[0],
                    0u32,
                    op == Op::SpecConstantFalse,
                );
                self.set(id, VariantHolder::Constant(Box::new(constant)));
            }

            Op::SpecConstantTrue | Op::ConstantTrue => {
                let id = ops[1];
                let constant = SPIRConstant::new_with_scalar_u32_and_specialization(
                    ops[0],
                    1u32,
                    op == Op::SpecConstantFalse,
                );
                self.set(id, VariantHolder::Constant(Box::new(constant)));
            }

            Op::ConstantNull => {
                let id = ops[1];
                let _type = ops[0];
                self.make_constant_null(id, _type);
            }

            Op::SpecConstantComposite | Op::ConstantComposite => {
                let id = ops[1];
                let _type = ops[0];

                let ctype = self.get::<SPIRType>(_type as usize);

                // We can have constants which are structs and arrays.
                // In this case, our SPIRConstant will be a list of other SPIRConstant ids which we
                // can refer to.
                let constant = if ctype.basetype == BaseType::Struct || !ctype.array.is_empty() {
                    let subconstants = ops[2..length as usize].to_vec();
                    SPIRConstant::new_with_subconstants_and_specialization(
                        _type,
                        subconstants,
                        op == Op::SpecConstantComposite,
                    )
                } else {
                    let elements = length - 2;
                    if elements > 4 {
                        panic!("Op::ConstantComposite only supports 1, 2, 3 and 4 elements.");
                    }

                    let mut remapped_constant_ops: Vec<SPIRConstant> = vec![];
                    for _i in 0..elements as usize {
                        // constant type here is a placeholder, it will be overwritten later
                        remapped_constant_ops.push(SPIRConstant::new(0));
                    }
                    for i in 0..elements as usize {
                        // Specialization constants operations can also be part of this.
                        // We do not know their value, so any attempt to query SPIRConstant later
                        // will fail. We can only propagate the ID of the expression and use to_expression on it.
                        if let VariantHolder::ConstantOp(constant_op) =
                            self.ir.ids[ops[2 + i] as usize].get()
                        {
                            if op == Op::ConstantComposite {
                                panic!("Specialization constant operation used in Op::ConstantComposite.");
                            }
                            remapped_constant_ops[i]
                                .make_null(self.get::<SPIRType>(constant_op.basetype as usize));
                            remapped_constant_ops[i].set_self(constant_op.get_self());
                            remapped_constant_ops[i].constant_type = constant_op.basetype;
                            remapped_constant_ops[i].specialization = true;
                        }
                    }

                    let mut c: Vec<&SPIRConstant> = vec![];
                    // Run loop second time to avoid borrowing as mutable and as immutable simultaneously
                    for i in 0..elements as usize {
                        if let VariantHolder::ConstantOp(_) = self.ir.ids[ops[2 + i] as usize].get()
                        {
                            c.push(&remapped_constant_ops[i]);
                        } else {
                            c.push(self.get::<SPIRConstant>(ops[2 + i] as usize));
                        }
                    }
                    SPIRConstant::new_with_vector_elements_and_specialization(
                        _type,
                        &c,
                        op == Op::SpecConstantComposite,
                    )
                };
                self.set(id, VariantHolder::Constant(Box::new(constant)));
            }

            Op::Function => {
                let res = ops[0];
                let id = ops[1];
                // Control
                let _type = ops[3];

                if self.current_function.is_some() {
                    panic!("Must end a function before starting a new one!");
                }

                let current_function = SPIRFunction::new(res, _type);
                self.set(
                    id,
                    VariantHolder::Function(Box::new(current_function.clone())),
                );

                self.current_function = Some(current_function);
            }

            Op::FunctionParameter => {
                // ...
            }

            _ => {}
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

            if spirv[0] != spv::MAGIC_NUMBER || !Parser::is_valid_spirv_version(spirv[1]) {
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
        self.ir.ids[id as usize].set(holder);
    }

    #[allow(clippy::borrowed_box)]
    fn get<T: IVariant>(&self, id: usize) -> &Box<T> {
        T::cast(self.ir.ids[id].get())
    }

    fn maybe_get(&self, id: usize, _type: Types) -> Option<&VariantHolder> {
        if self.ir.ids[id].get_type() as u32 == _type as u32 {
            Some(self.ir.ids[id].get())
        } else {
            None
        }
    }

    fn types_are_logically_equivalent(&self, a: &SPIRType, b: &SPIRType) -> bool {
        if a.basetype != b.basetype {
            return false;
        }
        if a.width != b.width {
            return false;
        }
        if a.vecsize != b.vecsize {
            return false;
        }
        if a.columns != b.columns {
            return false;
        }
        if a.array.as_slice() != b.array.as_slice() {
            return false;
        }

        match a.basetype {
            BaseType::Image => {
                if a.image != b.image {
                    return false;
                }
            }
            BaseType::SampledImage => {
                if a.image != b.image {
                    return false;
                }
            }
            _ => (),
        }

        if a.member_types.len() != b.member_types.len() {
            return false;
        }

        for i in 0..a.member_types.len() {
            let type_a = self.get::<SPIRType>(a.member_types[i] as usize);
            let type_b = self.get::<SPIRType>(b.member_types[i] as usize);
            if !self.types_are_logically_equivalent(type_a, type_b) {
                return false;
            }
        }

        true
    }

    fn make_constant_null(&mut self, id: u32, _type: u32) {
        let constant_type = *self.get::<SPIRType>(_type as usize).clone();

        let constant = if constant_type.pointer {
            let mut constant = SPIRConstant::new(_type);
            constant.make_null(&constant_type);
            constant
        } else if !constant_type.array.is_empty() {
            assert_ne!(constant_type.parent_type, 0);
            let parent_id = self.ir.increase_bound_by(1);
            self.make_constant_null(parent_id, constant_type.parent_type);

            if !constant_type.array_size_literal.last().unwrap() {
                panic!("Array size of Op::ConstantNull must be a literal.");
            }

            let array_size = *constant_type.array.last().unwrap() as usize;
            let subconstants = vec![parent_id; array_size];

            SPIRConstant::new_with_subconstants_and_specialization(_type, subconstants, false)
        } else if !constant_type.member_types.is_empty() {
            let members_len = constant_type.member_types.len();
            let member_ids = self.ir.increase_bound_by(members_len as u32);

            let mut subconstants = vec![0; members_len];
            for (i, item) in subconstants.iter_mut().enumerate() {
                self.make_constant_null(member_ids + i as u32, constant_type.member_types[i]);
                *item = member_ids + i as u32;
            }

            SPIRConstant::new_with_subconstants_and_specialization(_type, subconstants, false)
        } else {
            let mut constant = SPIRConstant::new(_type);
            constant.make_null(&constant_type);
            constant
        };

        self.set(id, VariantHolder::Constant(Box::new(constant)));
    }
}
