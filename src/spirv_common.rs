use std::collections::{HashMap, HashSet};
use std::cmp::PartialEq;

use crate::spirv as spv;

struct CompilerError(String);

struct Bitset {
    // The most common bits to set are all lower than 64,
    // so optimize for this case. Bits spilling outside 64 go into a slower data structure.
	// In almost all cases, higher data structure will not be used.
    lower: u64,
    higher: HashSet<u32>,
}

impl Default for Bitset {
    fn default() -> Self {
        Bitset { lower: 0, higher: HashSet::new() }
    }
}

impl Bitset {
    fn new(lower: u64) -> Self {
        Bitset { lower, higher: HashSet::new() }
    }

    fn get(&self, bit: u32) -> bool {
        if bit < 64 {
            // TODO: ull is what the type? u64?
            (self.lower & (1u64 << bit)) != 0
        } else {
            self.higher.contains(&bit)
        }
    }

    fn set(&mut self, bit: u32) {
        if bit < 64 {
            // TODO: ull is what type?
            self.lower |= 1u64 << bit;
        } else {
            self.higher.insert(bit);
        }
    }

    fn clear(&mut self) {
        if bit < 64 {
            // todo: ull what type?
            // how to do bitwise invert, like "~" operator
            self.lower &= (~(1u64 << bit));
        } else {
            self.higher.remove(bit);
        }
    }

    fn get_lower(&self) -> u64 {
        self.lower
    }

    fn reset(&mut self) {
        self.lower = 0;
        self.higher.clear();
    }

    fn merge_and(&mut self, other: &Bitset) {
        self.lower &= other.lower;
        let mut tmp_set = HashSet::new();
        for v in self.higher {
            if other.higher.contains(&v) {
                tmp_set.insert(v);
            }
        }
        self.higher = tmp_set;
    }

    fn merge_or(&mut self, other: &Bitset) {
        self.lower |= other.lower;
        for v in other.higher {
            self.higher.insert(v);
        }
    }

    fn for_each_bit<T>(&self, op: T) {
        // TODO: Add ctz-based iteration.
        for i in 0..64 {
            // TODO: i hope 1ull is really 1u64
            if lower & (1u64 << i) {
                op(i);
            }
        }

        if higher.empty() {
            return;
        }

        // Need to enforce an order here for reproducible results,
        // but hitting this path should happen extremely rarely, so having this slow path is fine.
        let mut bits: Vec<u32> = vec![];
        bits.reserve(self.higher.len());
        for v in self.higher {
            bits.push(v);
        }
        bits.sort();

        for v in bits {
            op(v);
        }
    }

    fn empty(&self) -> bool {
        self.value == 0 && self.higher.len() == 0
    }

}

impl PartialEq for Bitset {
    fn eq(&self, other: &Bitset) -> bool {
        if self.lower != other.lower {
            return false;
        }

        if self.higher.len() != other.higher.len() {
            return false;
        }

        for v in self.higher {
            if !other.higher.contains(&v) {
                return false;
            }
        }

        true
    }

    fn ne(&self, other: &Bitset) -> bool {
        !(self == other)
    }
}

fn merge(list: &Vec<String>) -> String {
    list.join(", ")
}


struct Instruction {
    op: u16,
    count: u16,
    offset: u32,
    length: u32,
}

impl Default for Instruction {
    fn default() -> Self {
        Instruction { op: 0, count: 0, offset: 0, length: 0 }
    }
}

struct IVariant {
    _self: u32,
}

impl Default for IVariant {
    fn default() -> Self {
        IVariant { _self: 0 }
    }
}



#[derive(Clone, Copy)]
pub enum Types {
    TypeNone,
    TypeType,
    TypeVariable,
    TypeConstant,
    TypeFunction,
    TypeFunctionPrototype,
    TypeBlock,
    TypeExtension,
    TypeExpression,
    TypeConstantOp,
    TypeCombinedImageSampler,
    TypeAccessChain,
    TypeUndef,
    TypeCount,
}

use Types::TypeUndef;

#[derive(Clone)]
struct SPIRUndef {
    _type: Types,
    basetype: u32,
}

#[derive(Clone)]
impl SPIRUndef {
    fn new(basetype: u32) -> Self {
        SPIRUndef { _type: Types::TypeUndef, basetype }
    }
}

struct SPIRCombinedImageSampler {
    _type: Types,
    combined_type: u32,
    image: u32,
    sampler: u32,

}

impl SPIRCombinedImageSampler {
    fn new(type_: u32, image: u32, sampler: u32) -> Self {
        SPIRCombinedImageSampler {
            _type: Types::TypeCombinedImageSampler,
            combined_type: type_,
            image,
            sampler,
        }
    }
}

#[derive(Clone)]
struct SPIRConstantOp {
    _type: Types,
    opcode: spv::Op,
    arguments: Vec<u32>,
    basetype: u32,
}

impl SPIRConstantOp {
    fn new(result_type: u32, op: spv::Op, args: Vec<u32>) -> Self {
        SPIRConstantOp {
            _type: Types::TypeConstantOp,
            opcode: op,
            arguments: args,
            basetype: result_type,
        }
    }
}

enum BaseType {
    Unknown,
    Void,
    Boolean,
    Char,
    SByte,
    UByte,
    Short,
    UShort,
    Int,
    UInt,
    Int64,
    UInt64,
    AtomicCounter,
    Half,
    Float,
    Double,
    Struct,
    Image,
    SampledImage,
    Sampler,
}

#[derive(Default)]
struct ImageType {
    _type: u32,
    dim: spv::Dim,
    depth: bool,
    arrayed: bool,
    ms: bool,
    sampled: u32,
    format: spv::ImageFormat,
    access: spv::AccessQualifier,
}

struct SpirType {
    _type: Types,

    // Scalar/vector/matrix support.
    basetype: BaseType,
    width: u32,
    vecsize: u32,
    columns: u32,

    // Arrays, support array of arrays by having a vector of array sizes.
    array: Vec<u32>,

    // Array elements can be either specialization constants or specialization ops.
    // This array determines how to interpret the array size.
    // If an element is true, the element is a literal,
    // otherwise, it's an expression, which must be resolved on demand.
    // The actual size is not really known until runtime.
    array_size_literal: Vec<bool>,

    // Pointers
    // Keep track of how many pointer layers we have.
    pointer_depth: u32,
    pointer: bool,

    storage: spv::StorageClass,

    member_types: Vec<u32>,

    image: ImageType,

    // Structs can be declared multiple times if they are used as part of interface blocks.
    // We want to detect this so that we only emit the struct definition once.
    // Since we cannot rely on OpName to be equal, we need to figure out aliases.
    type_alias: u32,

    // Denotes the type which this type is based on.
    // Allows the backend to traverse how a complex type is built up during access chains.
    parent_type: u32,

    // Used in backends to avoid emitting members with conflicting names.
    member_name_cache: HashSet<String>,
}

impl Default for SpirType {
    fn default() -> Self {
        SpirType {
            _type: Types::TypeType,
            basetype: BaseType::Unknown,
            width: 0,
            vecsize: 1,
            columns: 1,
            array: vec![],
            array_size_literal: vec![],
            pointer_depth: 0,
            pointer: false,
            storage: spv::StorageClassGeneric,
            member_types: vec![],
            image: Default::default(),
            type_alias: 0,
            parent_type: 0,
            member_name_cache: HashSet::new(),
        }
    }
}

enum Extension {
    Unsupported,
    GLSL,
    SPV_AMD_shader_ballot,
    SPV_AMD_shader_explicit_vertex_parameter,
    SPV_AMD_shader_trinary_minmax,
    SPV_AMD_gcn_shader,
}

#[derive(Clone)]
struct SPIRExtension {
    _type: Types,
    ext: Extension,
}

impl SPIRExtension {
    fn new(ext: Extension) -> Self {
        SPIRExtension {
            _type: Types:: TypeExtension,
            ext,
        }
    }
}


struct WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
    constant: u32,
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        WorkgroupSize {
            x: 0,
            y: 0,
            z: 0,
            constant: 0,
        }
    }
}

pub struct SPIREntryPoint {
    _self: u32,
    name: String,
    orig_name: String,
    interface_variables: Vec<u32>,
    flags: Bitset,
    workgroup_size: WorkgroupSize,
    invocations: u32,
    output_vertices: u32,
    model: spv::ExecutionModel,
}



impl SPIREntryPoint {
    pub fn new(
        _self: u32,
        execution_model: spv:: ExecutionModel,
        entry_name: String,
    ) -> Self {
        SPIREntryPoint {
            _self,
            name: entry_name.clone(),
            orig_name: entry_name,
            interface_variables: vec![],
            flags: Default::default(),
            workgroup_size: Default::default(),
            invocations: 0,
            output_vertices: 0,
            model: execution_model,
        }
    }
}

struct SPIRExpression {
    _type: Types,
    // If non-zero, prepend expression with to_expression(base_expression).
    // Used in amortizing multiple calls to to_expression()
    // where in certain cases that would quickly force a temporary when not needed.
    base_expression: u32,

    expression: String,
    expression_type: u32,

    // If this expression is a forwarded load,
    // allow us to reference the original variable.
    loaded_from: u32,

    // If this expression will never change, we can avoid lots of temporaries
    // in high level source.
    // An expression being immutable can be speculative,
    // it is assumed that this is true almost always.
    immutable: bool,

    // Before use, this expression must be transposed.
    // This is needed for targets which don't support row_major layouts.
    need_transpose: bool,

    // Whether or not this is an access chain expression.
    access_chain: bool,

    // A list of expressions which this expression depends on.
    expression_dependencies: Vec<u32>,

    // By reading this expression, we implicitly read these expressions as well.
    // Used by access chain Store and Load since we read multiple expressions in this case.
    implied_read_expressions: Vec<u32>,
}

impl SPIRExpression {
    fn new(expression: String, expression_type: u32, immutable: bool) -> Self {
        SPIRExpression {
            _type: Types::TypeExpression,
            expression,
            expression_type,
            immutable,
            base_expression: 0,
            loaded_from: 0,
            need_transpose: false,
            access_chain: false,
            expression_dependencies: vec![],
            implied_read_expressions: vec![],
        }
    }
}

struct SPIRFunctionPrototype {
    _type: Types,
    return_type: u32,
    parameter_types: Vec<u32>,
}

impl SPIRFunctionPrototype {
    fn new(return_type: u32) -> Self {
        SPIRFunctionPrototype {
            _type: Types::TypeFunctionPrototype,
            return_type,
            parameter_types: vec![],
        }
    }
}

enum Terminator {
    Unknown,
    Direct, // Emit next block directly without a particular condition.

    Select, // Block ends with an if/else block.
    MultiSelect, // Block ends with switch statement.

    Return, // Block ends with return.
    Unreachable, // Noop
    Kill, // Discard
}

enum Merge {
    MergeNone,
    MergeLoop,
    MergeSelection,
}

enum Hints {
    HintNone,
    HintUnroll,
    HintDontUnroll,
    HintFlatten,
    HintDontFlatten,
}

enum Method {
    MergeToSelectForLoop,
    MergeToDirectForLoop,
    MergeToSelectContinueForLoop,
}

enum ContinueBlockType {
    ContinueNone,

    // Continue block is branchless and has at least one instruction.
    ForLoop,

    // Noop continue block.
    WhileLoop,

    // Continue block is conditional.
    DoWhileLoop,

    // Highly unlikely that anything will use this,
    // since it is really awkward/impossible to express in GLSL.
    ComplexLoop,
}

#[derive(Default)]
struct Phi {
    local_variable: u32, // flush local variable ...
    parent: u32, // If we're in from_block and want to branch into this block ...
    function_variable: u32, // to this function-global "phi" variable first.
}

#[derive(Default)]
struct Case {
    value: u32,
    block: u32,
}



struct SPIRBlock {
    _type: Types,
    terminator: Terminator,
    merge: Merge,
    hint: Hints,
    next_block: u32,
    merge_block: u32,
    continue_block: u32,

    return_value: u32, // If 0, return nothing (void).
    condition: u32,
    true_block: u32,
    false_block: u32,
    default_block: u32,

    ops: Vec<Instruction>,

    // Before entering this block flush out local variables to magical "phi" variables.
    phi_variables: Vec<Phi>,

    // Declare these temporaries before beginning the block.
    // Used for handling complex continue blocks which have side effects.
    declare_temporary: Vec<(u32, u32)>,

    // Declare these temporaries, but only conditionally if this block turns out to be
    // a complex loop header.
    potential_declare_temporary: Vec<(u32, u32)>,

    cases: Vec<Case>,

    // If we have tried to optimize code for this block but failed,
    // keep track of this.
    disable_block_optimization: bool,

    // If the continue block is complex, fallback to "dumb" for loops.
    complex_continue: bool,

    // Do we need a ladder variable to defer breaking out of a loop construct after a switch block?
    need_ladder_break: bool,

    // The dominating block which this block might be within.
    // Used in continue; blocks to determine if we really need to write continue.
    loop_dominator: u32,

    // All access to these variables are dominated by this block,
    // so before branching anywhere we need to make sure that we declare these variables.
    dominated_variables: Vec<u32>,

    // These are variables which should be declared in a for loop header, if we
    // fail to use a classic for-loop,
    // we remove these variables, and fall back to regular variables outside the loop.
    loop_variables: Vec<u32>,

    // Some expressions are control-flow dependent, i.e. any instruction which relies on derivatives or
    // sub-group-like operations.
    // Make sure that we only use these expressions in the original block.
    invalidate_expressions: Vec<u32>,
}

impl Default for SPIRBlock {
    fn default() -> Self {
        SPIRBlock {
            _type: Types::TypeBlock,
            terminator: Terminator::Unknown,
            merge: Merge::MergeNone,
            hint: Hints::HintNone,
            next_block: 0,
            merge_block: 0,
            continue_block: 0,
            return_value: 0, // If 0, return nothing (void).
            condition: 0,
            true_block: 0,
            false_block: 0,
            default_block: 0,
            ops: vec![],
            phi_variables: vec![],
            declare_temporary: vec![],
            potential_declare_temporary: vec![],
            cases: vec![],
            disable_block_optimization: false,
            complex_continue: false,
            need_ladder_break: false,
            loop_dominator: 0,
            dominated_variables: vec![],
            loop_variables: vec![],
            invalidate_expressions: vec![],
        }
    }
}

struct Parameter {
    _type: u32,
    id: u32,
    read_count: u32,
    write_count: u32,

    // Set to true if this parameter aliases a global variable,
    // used mostly in Metal where global variables
    // have to be passed down to functions as regular arguments.
    // However, for this kind of variable, we should not care about
    // read and write counts as access to the function arguments
    // is not local to the function in question.
    alias_global_variable: bool,
}

// When calling a function, and we're remapping separate image samplers,
// resolve these arguments into combined image samplers and pass them
// as additional arguments in this order.
// It gets more complicated as functions can pull in their own globals
// and combine them with parameters,
// so we need to distinguish if something is local parameter index
// or a global ID.
struct CombinedImageSamplerParameter {
    id: u32,
    image_id: u32,
    sampler_id: u32,
    global_image: bool,
    global_sampler: bool,
    depth: bool,
};

struct SPIRFunction {
    _type: Types,
    return_type: u32,
    function_type: u32,
    arguments: Vec<Parameter>,

    // Can be used by backends to add magic arguments.
    // Currently used by combined image/sampler implementation.

    shadow_arguments: Vec<Parameter>,
    local_variables: Vec<u32>,
    entry_block: u32,
    blocks: Vec<u32>,
    combined_parameters: Vec<CombinedImageSamplerParameter>,

    // Hooks to be run when the function returns.
    // Mostly used for lowering internal data structures onto flattened structures.
    // Need to defer this, because they might rely on things which change during compilation.
    fixup_hooks_out: Vec<impl Fn() -> ()>,

    // Hooks to be run when the function begins.
    // Mostly used for populating internal data structures from flattened structures.
    // Need to defer this, because they might rely on things which change during compilation.
    fixup_hooks_in: Vec<impl Fn() -> ()>,

    // On function entry, make sure to copy a constant array into thread addr space to work around
    // the case where we are passing a constant array by value to a function on backends which do not
    // consider arrays value types.
    constant_arrays_needed_on_stack: Vec<u32>,

    active: bool,
    flush_undeclare: bool,
    do_combined_parameter: bool,
}

impl SPIRFunction {
    fn new(return_type: u32, function_type: u32) -> Self {
        SPIRFunction {
            _type: Types::TypeFunction,
            return_type,
            function_type,
            arguments: vec![],
            shadow_arguments: vec![],
            local_variables: vec![],
            entry_block: 0,
            blocks: vec![],
            combined_parameters: vec![],
            fixup_hooks_out: vec![],
            fixup_hooks_in: vec![],
            constant_arrays_needed_on_stack: vec![],
            active: false,
            flush_undeclare: true,
            do_combined_parameter: true,
        }
    }

    fn add_local_variable(&mut self, id: u32) {
        self.local_variables.push(id);
    }

    fn add_parameter(
        &mut self,
        parameter_type: u32,
        id: u32,
        alias_global_variable: impl Into<Option<bool>>,
    ) {
        let alias_global_variable = (alias_global_variable as Option<bool>).unwrap_or(false);
        // Arguments are read-only until proven otherwise.
        self.arguments.push(Parameter {
            _type: parameter_type,
            id,
            read_count: 0u32,
            write_count: 0u32,
            alias_global_variable,
        });
    }
}

struct SPIRAccessChain {
    _type: Types,
    basetype: u32,
    storage: spv::StorageClass,
    base: String,
    dynamic_index: String,
    static_index: i32,

    loaded_from: u32,
    matrix_stride: u32,
    row_major_matrix: bool,
    immutable: bool,

    // By reading this expression, we implicitly read these expressions as well.
    // Used by access chain Store and Load since we read multiple expressions in this case.
    implied_read_expressions: Vec<u32>,
}

impl SPIRAccessChain {
    fn new(
        basetype: u32,
        storage: spv::StorageClass,
        base: String,
        dynamic_index: String,
        static_index: i32,
    ) -> Self {
        SPIRAccessChain {
            _type: Types::TypeAccessChain,
            basetype,
            storage,
            base,
            dynamic_index,
            static_index,
            loaded_from: 0,
            matrix_stride: 0,
            row_major_matrix: false,
            immutable: false,
            implied_read_expressions: vec![],
        }
    }
}

pub struct Variant {
    holder: IVariant,
    _type: Types,
    allow_type_rewrite: bool,
}

impl Variant {
    pub fn set(&mut self, new_type: Types) {
        if !self.allow_type_rewrite && self._type != Types::TypeNone && self._type != new_type {
            panic!("Overwriting a variant with new type.");
        }
        self._type = new_type;
        self.allow_type_rewrite = false;
    }

    pub fn get_type(&self) -> Types {
        self._type
    }

    pub fn reset(&mut self) {
        self._type = Types::TypeNone;
    }

    pub fn set_allow_type_rewrite(&mut self) {
        self.allow_type_rewrite = true;
    }
}

struct DecorationExtended {
    packed: bool,
    packed_type: u32,
}

pub struct Decoration {
    alias: String,
    qualified_alias: String,
    hlsl_semantic: String,
    decoration_flags: Bitset,
    builtin_type: spv::BuiltIn,
    location: u32,
    component: u32,
    set: u32,
    binding: u32,
    offset: u32,
    array_stride: u32,
    matrix_stride: u32,
    input_attachment: u32,
    spec_id: u32,
    idex: u32,
    fp_rounding_mode: spv::FPRoundingMode,
    builtin: bool,
    extended: DecorationExtended,
}

pub struct Meta {
    decoration: Decoration,
    members: Vec<Decoration>,
    decoration_word_offset: HashMap<u32, u32>,
    hlsl_is_magic_counter_buffer: bool,
    hlsl_magic_counter_buffer: u32,
}

