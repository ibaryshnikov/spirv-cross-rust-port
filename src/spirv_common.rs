use std::collections::{HashMap, HashSet};
use std::cmp::PartialEq;

use crate::spirv as spv;

struct CompilerError(String);

#[derive(Clone)]
pub struct Bitset {
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

    pub fn get(&self, bit: u32) -> bool {
        if bit < 64 {
            (self.lower & (1u64 << bit)) != 0
        } else {
            self.higher.contains(&bit)
        }
    }

    pub fn set(&mut self, bit: u32) {
        if bit < 64 {
            self.lower |= 1u64 << bit;
        } else {
            self.higher.insert(bit);
        }
    }

    pub fn clear(&mut self, bit: u32) {
        if bit < 64 {
            // how to do bitwise invert, like "~" operator
            self.lower &= !(1u64 << bit);
        } else {
            self.higher.remove(&bit);
        }
    }

    fn get_lower(&self) -> u64 {
        self.lower
    }

    fn reset(&mut self) {
        self.lower = 0;
        self.higher.clear();
    }

    pub fn merge_and(&mut self, other: &Bitset) {
        self.lower &= other.lower;
        let mut tmp_set = HashSet::new();
        for v in &self.higher {
            if other.higher.contains(v) {
                tmp_set.insert(*v);
            }
        }
        self.higher = tmp_set;
    }

    pub fn merge_or(&mut self, other: &Bitset) {
        self.lower |= other.lower;
        for v in &other.higher {
            self.higher.insert(*v);
        }
    }

    pub fn for_each_bit(&self, mut op: impl FnMut(u32) -> ()) {
        // TODO: Add ctz-based iteration.
        for i in 0..64 {
            if (self.lower & (1u64 << i)) != 0 {
                op(i as u32);
            }
        }

        if self.higher.is_empty() {
            return;
        }

        // Need to enforce an order here for reproducible results,
        // but hitting this path should happen extremely rarely, so having this slow path is fine.
        let mut bits: Vec<u32> = vec![];
        bits.reserve(self.higher.len());
        for v in &self.higher {
            bits.push(*v);
        }
        bits.sort();

        for v in bits {
            op(v);
        }
    }

    fn empty(&self) -> bool {
        self.lower == 0 && self.higher.is_empty()
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

        for v in &self.higher {
            if !other.higher.contains(v) {
                return false;
            }
        }

        true
    }
}

fn merge(list: &[String]) -> String {
    list.join(", ")
}

#[derive(Clone)]
pub struct Instruction {
    pub op: u16,
    pub count: u16,
    pub offset: u32,
    pub length: u32,
}

impl Default for Instruction {
    fn default() -> Self {
        Instruction { op: 0, count: 0, offset: 0, length: 0 }
    }
}

pub trait IVariant {
    fn get_self(&self) -> u32;
    fn set_self(&mut self, _self: u32);
}

pub trait HasType {
    fn get_type() -> Types;
}

#[derive(Clone, Copy)]
pub enum Types {
    None,
    Type,
    Variable,
    Constant,
    Function,
    FunctionPrototype,
    Block,
    Extension,
    Expression,
    ConstantOp,
    CombinedImageSampler,
    AccessChain,
    Undef,
    TypeCount, // required to init some arrays
}

#[derive(Clone)]
pub enum VariantHolder {
    Undef(Box<SPIRUndef>),
    CombinedImageSampler(Box<SPIRCombinedImageSampler>),
    ConstantOp(Box<SPIRConstantOp>),
    Type(Box<SPIRType>),
    Extension(Box<SPIRExtension>),
    Expression(Box<SPIRExpression>),
    FunctionPrototype(Box<SPIRFunctionPrototype>),
    Block(Box<SPIRBlock>),
    Function(Box<SPIRFunction>),
    AccessChain(Box<SPIRAccessChain>),
    Variable(Box<SPIRVariable>),
    Constant(Box<SPIRConstant>),
}

impl VariantHolder {
    pub fn get_self(&self) -> u32 {
        match self {
            VariantHolder::Undef(variant) => variant.get_self(),
            VariantHolder::CombinedImageSampler(variant) => variant.get_self(),
            VariantHolder::ConstantOp(variant) => variant.get_self(),
            VariantHolder::Type(variant) => variant.get_self(),
            VariantHolder::Extension(variant) => variant.get_self(),
            VariantHolder::Expression(variant) => variant.get_self(),
            VariantHolder::FunctionPrototype(variant) => variant.get_self(),
            VariantHolder::Block(variant) => variant.get_self(),
            VariantHolder::Function(variant) => variant.get_self(),
            VariantHolder::AccessChain(variant) => variant.get_self(),
            VariantHolder::Variable(variant) => variant.get_self(),
            VariantHolder::Constant(variant) => variant.get_self(),
        }
    }
    pub fn set_self(&mut self, value: u32) {
        match self {
            VariantHolder::Undef(variant) => variant.set_self(value),
            VariantHolder::CombinedImageSampler(variant) => variant.set_self(value),
            VariantHolder::ConstantOp(variant) => variant.set_self(value),
            VariantHolder::Type(variant) => variant.set_self(value),
            VariantHolder::Extension(variant) => variant.set_self(value),
            VariantHolder::Expression(variant) => variant.set_self(value),
            VariantHolder::FunctionPrototype(variant) => variant.set_self(value),
            VariantHolder::Block(variant) => variant.set_self(value),
            VariantHolder::Function(variant) => variant.set_self(value),
            VariantHolder::AccessChain(variant) => variant.set_self(value),
            VariantHolder::Variable(variant) => variant.set_self(value),
            VariantHolder::Constant(variant) => variant.set_self(value),
        }
    }
    pub fn get_type(&self) -> Types {
        match self {
            VariantHolder::Undef(_) => Types::Undef,
            VariantHolder::CombinedImageSampler(_) => Types::CombinedImageSampler,
            VariantHolder::ConstantOp(_) => Types::ConstantOp,
            VariantHolder::Type(_) => Types::Type,
            VariantHolder::Extension(_) => Types::Extension,
            VariantHolder::Expression(_) => Types::Expression,
            VariantHolder::FunctionPrototype(_) => Types::FunctionPrototype,
            VariantHolder::Block(_) => Types::Block,
            VariantHolder::Function(_) => Types::Function,
            VariantHolder::AccessChain(_) => Types::AccessChain,
            VariantHolder::Variable(_) => Types::Variable,
            VariantHolder::Constant(_) => Types::Constant,
        }
    }
}

#[derive(Clone, Default)]
pub struct SPIRUndef {
    basetype: u32,
    _self: u32,
}

impl IVariant for SPIRUndef {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRUndef {
    fn get_type() -> Types {
        Types::Undef
    }
}

impl SPIRUndef {
    pub fn new(basetype: u32) -> Self {
        SPIRUndef { basetype, _self: 0 }
    }
}

#[derive(Clone)]
pub struct SPIRCombinedImageSampler {
    combined_type: u32,
    image: u32,
    sampler: u32,
    _self: u32,
}

impl IVariant for SPIRCombinedImageSampler {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRCombinedImageSampler {
    fn get_type() -> Types {
        Types::CombinedImageSampler
    }
}

impl SPIRCombinedImageSampler {
    fn new(type_: u32, image: u32, sampler: u32) -> Self {
        SPIRCombinedImageSampler {
            combined_type: type_,
            image,
            sampler,
            _self: 0,
        }
    }
}

#[derive(Clone)]
pub struct SPIRConstantOp {
    opcode: spv::Op,
    pub arguments: Vec<u32>,
    basetype: u32,
    _self: u32,
}

impl IVariant for SPIRConstantOp {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRConstantOp {
    fn get_type() -> Types {
        Types::ConstantOp
    }
}

impl SPIRConstantOp {
    fn new(result_type: u32, op: spv::Op, args: Vec<u32>) -> Self {
        SPIRConstantOp {
            opcode: op,
            arguments: args,
            basetype: result_type,
            _self: 0,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum BaseType {
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

#[derive(Clone, Default, PartialEq, Eq)]
pub struct ImageType {
    pub _type: u32,
    pub dim: spv::Dim,
    pub depth: bool,
    pub arrayed: bool,
    pub ms: bool,
    pub sampled: u32,
    pub format: spv::ImageFormat,
    pub access: spv::AccessQualifier,
}

pub struct SPIRType {
    // Scalar/vector/matrix support.
    pub basetype: BaseType,
    pub width: u32,
    pub vecsize: u32,
    pub columns: u32,

    // Arrays, support array of arrays by having a vector of array sizes.
    pub array: Vec<u32>,

    // Array elements can be either specialization constants or specialization ops.
    // This array determines how to interpret the array size.
    // If an element is true, the element is a literal,
    // otherwise, it's an expression, which must be resolved on demand.
    // The actual size is not really known until runtime.
    pub array_size_literal: Vec<bool>,

    // Pointers
    // Keep track of how many pointer layers we have.
    pub pointer_depth: u32,
    pub pointer: bool,

    pub storage: spv::StorageClass,

    pub member_types: Vec<u32>,

    pub image: ImageType,

    // Structs can be declared multiple times if they are used as part of interface blocks.
    // We want to detect this so that we only emit the struct definition once.
    // Since we cannot rely on OpName to be equal, we need to figure out aliases.
    pub type_alias: u32,

    // Denotes the type which this type is based on.
    // Allows the backend to traverse how a complex type is built up during access chains.
    pub parent_type: u32,

    // Used in backends to avoid emitting members with conflicting names.
    member_name_cache: HashSet<String>,

    _self: u32,
}

impl Clone for SPIRType {
    fn clone(&self) -> Self {
        Self {
            basetype: self.basetype.clone(),
            array: self.array.to_vec(),
            array_size_literal: self.array_size_literal.to_vec(),
            storage: self.storage.clone(),
            member_types: self.member_types.clone(),
            image: self.image.clone(),
            member_name_cache: self.member_name_cache.clone(),
            ..*self
        }
    }
}

impl IVariant for SPIRType {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRType {
    fn get_type() -> Types {
        Types::Type
    }
}

impl Default for SPIRType {
    fn default() -> Self {
        SPIRType {
            basetype: BaseType::Unknown,
            width: 0,
            vecsize: 1,
            columns: 1,
            array: vec![],
            array_size_literal: vec![],
            pointer_depth: 0,
            pointer: false,
            storage: spv::StorageClass::Generic,
            member_types: vec![],
            image: Default::default(),
            type_alias: 0,
            parent_type: 0,
            member_name_cache: HashSet::new(),
            _self: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub enum Extension {
    Unsupported,
    GLSL,
    SPV_AMD_shader_ballot,
    SPV_AMD_shader_explicit_vertex_parameter,
    SPV_AMD_shader_trinary_minmax,
    SPV_AMD_gcn_shader,
}

impl Extension {
    pub fn from_str(value: &str) -> Extension {
        match value {
            "GLSL.std.450" => Extension::GLSL,
            "SPV_AMD_shader_ballot" => Extension::SPV_AMD_shader_ballot,
            "SPV_AMD_shader_explicit_vertex_parameter" => Extension::SPV_AMD_shader_explicit_vertex_parameter,
            "SPV_AMD_shader_trinary_minmax" => Extension::SPV_AMD_shader_trinary_minmax,
            "SPV_AMD_gcn_shader" => Extension::SPV_AMD_gcn_shader,
            _ => Extension::Unsupported,
        }
    }
}

#[derive(Clone)]
pub struct SPIRExtension {
    ext: Extension,
    _self: u32,
}

impl IVariant for SPIRExtension {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRExtension {
    fn get_type() -> Types {
        Types::Extension
    }
}

impl SPIRExtension {
    pub fn new(ext: Extension) -> Self {
        SPIRExtension {
            ext,
            _self: 0,
        }
    }
}


pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
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
    pub name: String,
    orig_name: String,
    pub interface_variables: Vec<u32>,
    pub flags: Bitset,
    pub workgroup_size: WorkgroupSize,
    pub invocations: u32,
    pub output_vertices: u32,
    model: spv::ExecutionModel,
}

// SPIREntryPoint is not a variant since its IDs are used to decorate OpFunction,
// so in order to avoid conflicts, we can't stick them in the ids array.
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

#[derive(Clone)]
pub struct SPIRExpression {
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

    _self: u32,
}

impl IVariant for SPIRExpression {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRExpression {
    fn get_type() -> Types {
        Types::Expression
    }
}

impl SPIRExpression {
    fn new(expression: String, expression_type: u32, immutable: bool) -> Self {
        SPIRExpression {
            expression,
            expression_type,
            immutable,
            base_expression: 0,
            loaded_from: 0,
            need_transpose: false,
            access_chain: false,
            expression_dependencies: vec![],
            implied_read_expressions: vec![],
            _self: 0,
        }
    }
}

#[derive(Clone)]
pub struct SPIRFunctionPrototype {
    pub return_type: u32,
    pub parameter_types: Vec<u32>,
    _self: u32,
}

impl IVariant for SPIRFunctionPrototype {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRFunctionPrototype {
    fn get_type() -> Types {
        Types::FunctionPrototype
    }
}

impl SPIRFunctionPrototype {
    pub fn new(return_type: u32) -> Self {
        SPIRFunctionPrototype {
            return_type,
            parameter_types: vec![],
            _self: 0,
        }
    }
}

#[derive(Clone)]
pub enum Terminator {
    Unknown,
    Direct, // Emit next block directly without a particular condition.

    Select, // Block ends with an if/else block.
    MultiSelect, // Block ends with switch statement.

    Return, // Block ends with return.
    Unreachable, // Noop
    Kill, // Discard
}

#[derive(Clone)]
pub enum Merge {
    None,
    Loop,
    Selection,
}

#[derive(Clone)]
enum Hints {
    None,
    Unroll,
    DontUnroll,
    Flatten,
    DontFlatten,
}

// TODO: remove MergeTo prefix
#[allow(clippy::enum_variant_names)]
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

#[derive(Clone, Default)]
struct Phi {
    local_variable: u32, // flush local variable ...
    parent: u32, // If we're in from_block and want to branch into this block ...
    function_variable: u32, // to this function-global "phi" variable first.
}

#[derive(Clone, Default)]
pub struct Case {
    value: u32,
    pub block: u32,
}

#[derive(Clone)]
pub struct SPIRBlock {
    pub terminator: Terminator,
    pub merge: Merge,
    hint: Hints,
    pub next_block: u32,
    pub merge_block: u32,
    continue_block: u32,

    return_value: u32, // If 0, return nothing (void).
    condition: u32,
    pub true_block: u32,
    pub false_block: u32,
    pub default_block: u32,

    ops: Vec<Instruction>,

    // Before entering this block flush out local variables to magical "phi" variables.
    phi_variables: Vec<Phi>,

    // Declare these temporaries before beginning the block.
    // Used for handling complex continue blocks which have side effects.
    declare_temporary: Vec<(u32, u32)>,

    // Declare these temporaries, but only conditionally if this block turns out to be
    // a complex loop header.
    potential_declare_temporary: Vec<(u32, u32)>,

    pub cases: Vec<Case>,

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

    _self: u32,
}

impl IVariant for SPIRBlock {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}

impl HasType for SPIRBlock {
    fn get_type() -> Types {
        Types::Block
    }
}

impl Default for SPIRBlock {
    fn default() -> Self {
        SPIRBlock {
            terminator: Terminator::Unknown,
            merge: Merge::None,
            hint: Hints::None,
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
            _self: 0,
        }
    }
}

#[derive(Clone)]
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
#[derive(Clone)]
struct CombinedImageSamplerParameter {
    id: u32,
    image_id: u32,
    sampler_id: u32,
    global_image: bool,
    global_sampler: bool,
    depth: bool,
}

#[derive(Clone)]
pub struct SPIRFunction {
    return_type: u32,
    function_type: u32,
    arguments: Vec<Parameter>,

    // Can be used by backends to add magic arguments.
    // Currently used by combined image/sampler implementation.

    shadow_arguments: Vec<Parameter>,
    local_variables: Vec<u32>,
    pub entry_block: u32,
    blocks: Vec<u32>,
    combined_parameters: Vec<CombinedImageSamplerParameter>,

    // Hooks to be run when the function returns.
    // Mostly used for lowering internal data structures onto flattened structures.
    // Need to defer this, because they might rely on things which change during compilation.
    fixup_hooks_out: Vec<Box<fn() -> ()>>,

    // Hooks to be run when the function begins.
    // Mostly used for populating internal data structures from flattened structures.
    // Need to defer this, because they might rely on things which change during compilation.
    fixup_hooks_in: Vec<Box<fn() -> ()>>,

    // On function entry, make sure to copy a constant array into thread addr space to work around
    // the case where we are passing a constant array by value to a function on backends which do not
    // consider arrays value types.
    constant_arrays_needed_on_stack: Vec<u32>,

    active: bool,
    flush_undeclare: bool,
    do_combined_parameter: bool,

    _self: u32,
}

impl IVariant for SPIRFunction {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRFunction {
    fn get_type() -> Types {
        Types::Function
    }
}

impl SPIRFunction {
    fn new(return_type: u32, function_type: u32) -> Self {
        SPIRFunction {
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
            _self: 0,
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
        let alias_global_variable = alias_global_variable.into().unwrap_or(false);
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

#[derive(Clone)]
pub struct SPIRAccessChain {
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

    _self: u32,
}

impl IVariant for SPIRAccessChain {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRAccessChain {
    fn get_type() -> Types {
        Types::AccessChain
    }
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
            _self: 0,
        }
    }
}

#[derive(Clone)]
pub struct SPIRVariable {
    pub basetype: u32,
    storage: spv::StorageClass,
    decoration: u32,
    initializer: u32,
    basevariable: u32,

    dereference_chain: Vec<u32>,
    compat_builtin: bool,

    // If a variable is shadowed, we only statically assign to it
    // and never actually emit a statement for it.
    // When we read the variable as an expression, just forward
    // shadowed_id as the expression.
    statically_assigned: bool,
    static_expression: u32,

    // Temporaries which can remain forwarded as long as this variable is not modified.
    dependees: Vec<u32>,
    forwardable: bool,

    deferred_declaration: bool,
    phi_variable: bool,

    // Used to deal with Phi variable flushes. See flush_phi().
    allocate_temporary_copy: bool,

    remapped_variable: bool,
    remapped_components: u32,

    // The block which dominates all access to this variable.
    dominator: u32,
    // If true, this variable is a loop variable, when accessing the variable
    // outside a loop,
    // we should statically forward it.
    loop_variable: bool,
    // Set to true while we're inside the for loop.
    loop_variable_enable: bool,

    parameter: Option<Parameter>,

    _self: u32,
}

impl IVariant for SPIRVariable {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRVariable {
    fn get_type() -> Types {
        Types::Variable
    }
}

impl SPIRVariable {
    fn new(
        basetype: u32,
        storage: spv::StorageClass,
        initializer: impl Into<Option<u32>>, // 0
        basevariable: impl Into<Option<u32>>, // 0
    ) -> Self {
        let initializer = initializer
            .into()
            .unwrap_or(0);
        let basevariable = basevariable
            .into()
            .unwrap_or(0);
        SPIRVariable {
            basetype,
            storage,
            initializer,
            basevariable,
            decoration: 0,
            dereference_chain: vec![],
            compat_builtin: false,
            statically_assigned: false,
            static_expression: 0,
            dependees: vec![],
            forwardable: true,
            deferred_declaration: false,
            phi_variable: false,
            allocate_temporary_copy: false,
            remapped_variable: false,
            remapped_components: 0,
            dominator: 0,
            loop_variable: false,
            loop_variable_enable: false,
            parameter: None,
            _self: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct Constant(u64);

impl Constant {
    fn from_i32(value: i32) -> Self {
        Self(u64::from(u32::from_ne_bytes(value.to_ne_bytes())))
    }
    fn from_u32(value: u32) -> Self {
        Self(u64::from(value))
    }
    fn from_i64(value: i64) -> Self {
        Self(u64::from_ne_bytes(value.to_ne_bytes()))
    }
    fn from_u64(value: u64) -> Self {
        Self(value)
    }
    fn from_f32(value: f32) -> Self {
        Self(u64::from(value.to_bits()))
    }
    fn from_f64(value: f64) -> Self {
        Self(value.to_bits())
    }
    fn to_u32(self) -> u32 {
        self.0 as u32
    }
    fn to_i32(self) -> i32 {
        i32::from_ne_bytes((self.0 as u32).to_ne_bytes())
    }
    fn to_i64(self) -> i64 {
        i64::from_ne_bytes(self.0.to_ne_bytes())
    }
    fn to_u64(self) -> u64 {
        self.0
    }
    fn to_f32(self) -> f32 {
        f32::from_bits(self.0 as u32)
    }
    fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

#[derive(Clone, Copy)]
struct ConstantVector {
    r: [Constant; 4],
    id: [u32; 4],
    vecsize: u32,
}

impl Default for ConstantVector {
    fn default() -> Self {
        ConstantVector {
            r: [Constant(0); 4],
            id: [0; 4],
            vecsize: 1,
        }
    }
}

#[derive(Clone)]
struct ConstantMatrix {
    c: [ConstantVector; 4],
    id: [u32; 4],
    columns: u32,
}

impl Default for ConstantMatrix {
    fn default() -> Self {
        ConstantMatrix {
            c: [ConstantVector::default(); 4],
            id: [0; 4],
            columns: 1,
        }
    }
}

#[derive(Clone)]
pub struct SPIRConstant {
    constant_type: u32,
    m: ConstantMatrix,

    // If this constant is a specialization constant (i.e. created with OpSpecConstant*).
    pub specialization: bool,
    // If this constant is used as an array length which creates specialization restrictions on some backends.
    pub is_used_as_array_length: bool,

    // If true, this is a LUT, and should always be declared in the outer scope.
    is_used_as_lut: bool,

    // For composites which are constant arrays, etc.
    subconstants: Vec<u32>,

    // Non-Vulkan GLSL, HLSL and sometimes MSL emits defines for each specialization constant,
    // and uses them to initialize the constant. This allows the user
    // to still be able to specialize the value by supplying corresponding
    // preprocessor directives before compiling the shader.
    specialization_constant_macro_name: String,

    _self: u32,
}

impl IVariant for SPIRConstant {
    fn get_self(&self) -> u32 {
        self._self
    }
    fn set_self(&mut self, value: u32) {
        self._self = value;
    }
}
impl HasType for SPIRConstant {
    fn get_type() -> Types {
        Types::Constant
    }
}

impl SPIRConstant {
    fn new(constant_type: u32) -> Self {
        SPIRConstant {
            constant_type,
            m: ConstantMatrix::default(),
            specialization: false,
            is_used_as_array_length: false,
            is_used_as_lut: false,
            subconstants: vec![],
            specialization_constant_macro_name: String::new(),
            _self: 0,
        }
    }

    fn new_with_subconstants_and_specialization(
        constant_type: u32,
        elements: Vec<u32>,
        specialized: bool,
    ) -> Self {
        SPIRConstant {
            constant_type,
            m: ConstantMatrix::default(),
            specialization: specialized,
            is_used_as_array_length: false,
            is_used_as_lut: false,
            subconstants: elements,
            specialization_constant_macro_name: String::new(),
            _self: 0,
        }
    }

    // Construct scalar (32-bit).
    fn new_with_scalar_u32_and_specialization(
        constant_type: u32,
        v0: u32,
        specialized: bool,
    ) -> Self {
        let mut constant = SPIRConstant {
            constant_type,
            m: ConstantMatrix::default(),
            specialization: specialized,
            is_used_as_array_length: false,
            is_used_as_lut: false,
            subconstants: vec![],
            specialization_constant_macro_name: String::new(),
            _self: 0,
        };
        constant.m.c[0].r[0] = Constant::from_u32(v0);
        constant.m.c[0].vecsize = 1;
        constant.m.columns = 1;
        constant
    }

    // Construct scalar (64-bit).
    fn new_with_scalar_u64_and_specialization(
        constant_type: u32,
        v0: u64,
        specialized: bool,
    ) -> Self {
        let mut constant = SPIRConstant {
            constant_type,
            m: ConstantMatrix::default(),
            specialization: specialized,
            is_used_as_array_length: false,
            is_used_as_lut: false,
            subconstants: vec![],
            specialization_constant_macro_name: String::new(),
            _self: 0,
        };
        constant.m.c[0].r[0] = Constant::from_u64(v0);
        constant.m.c[0].vecsize = 1;
        constant.m.columns = 1;
        constant
    }

    fn new_with_vector_elements_and_specialization(
        constant_type: u32,
        vector_elements: &[SPIRConstant],
        specialized: bool,
    ) -> Self {
        let mut constant = SPIRConstant {
            constant_type,
            m: ConstantMatrix::default(),
            specialization: specialized,
            is_used_as_array_length: false,
            is_used_as_lut: false,
            subconstants: vec![],
            specialization_constant_macro_name: String::new(),
            _self: 0,
        };

        let matrix = vector_elements[0].m.c[0].vecsize > 1;
        if matrix {
            constant.m.columns = vector_elements.len() as u32;
            for (i, element) in vector_elements.iter().enumerate() {
                constant.m.c[i] = element.m.c[0];
                if element.specialization {
                    constant.m.id[i] = element.get_self();
                    constant.m.id[i] = 0;
                }
            }
        } else {
            constant.m.c[0].vecsize = vector_elements.len() as u32;
            constant.m.columns = 1;
            for (i, element) in vector_elements.iter().enumerate() {
                constant.m.c[0].r[i] = element.m.c[0].r[0];
                if element.specialization {
                    constant.m.c[0].id[i] = element.get_self();
                    constant.m.c[0].id[i] = 0;
                }
            }
        }

        constant
    }

    fn f16_to_f32(u16_value: u16) -> f32 {
        // Based on the GLM implementation.
        let s = i32::from((u16_value >> 15) & 0x1);
        let mut e = i32::from((u16_value >> 10) & 0x1f);
        let mut m = i32::from(u16_value & 0x3ff);

        if e == 0 {
            if m == 0 {
                let u: u32 = (s as u32) << 31;
                return f32::from_bits(u);
            } else {
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }

                e += 1;
                m &= !0x400;
            }
        }
        else if e == 31 {
            if m == 0 {
                let u: u32 = ((s as u32) << 31) | 0x7f80_0000u32;
                return f32::from_bits(u);
            } else {
                let u: u32 = ((s as u32) << 31) | 0x7f80_0000u32 | ((m as u32) << 13);
                return f32::from_bits(u);
            }
        }

        e += 127 - 15;
        m <<= 13;
        let u = ((s as u32) << 31) | ((e as u32) << 23) | m as u32;
        f32::from_bits(u)
    }

    fn specialization_constant_id(&self, col: usize, row: usize) -> u32 {
        self.m.c[col].id[row]
    }

    fn specialization_constant_id_col(&self, col: usize) -> u32 {
        self.m.id[col]
    }

    fn get_constant(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> Constant {
        self.m
            .c[col.into().unwrap_or(0)]
            .r[row.into().unwrap_or(0)]
    }

    pub fn scalar(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> u32 {
        self.get_constant(col, row).to_u32()
    }

    fn scalar_i8(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> i8 {
        (self.get_constant(col, row).to_u32() & 0xffu32) as i8
    }

    fn scalar_u8(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> u8 {
        (self.get_constant(col, row).to_u32() & 0xffu32) as u8
    }

    fn scalar_i16(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> i16 {
        (self.get_constant(col, row).to_u32() & 0xffffu32) as i16
    }

    fn scalar_u16(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> u16 {
        (self.get_constant(col, row).to_u32() & 0xffffu32) as u16
    }

    fn scalar_f16(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> f32 {
        Self::f16_to_f32(self.scalar_u16(col, row))
    }

    fn scalar_i32(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> i32 {
        self.get_constant(col, row).to_i32()
    }

    fn scalar_f32(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> f32 {
        self.get_constant(col, row).to_f32()
    }

    fn scalar_i64(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> i64 {
        self.get_constant(col.into(), row.into()).to_i64()
    }

    fn scalar_u64(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> u64 {
        self.get_constant(col, row).to_u64()
    }

    fn scalar_f64(&self, col: impl Into<Option<usize>>, row: impl Into<Option<usize>>) -> f64 {
        self.get_constant(col, row).to_f64()
    }

    fn vector(&self) -> ConstantVector {
        self.m.c[0]
    }

    fn vector_size(&self) -> u32 {
        self.m.c[0].vecsize
    }

    fn columns(&self) -> u32 {
        self.m.columns
    }

    fn make_null(&mut self, constant_type: SPIRType) {
        self.m = ConstantMatrix::default();
        self.m.columns = constant_type.columns;
        for column in self.m.c.iter_mut() {
            column.vecsize = constant_type.vecsize
        }
    }

    fn constant_is_null(&self) -> bool {
        if self.specialization {
            return false;
        }
        if !self.subconstants.is_empty() {
            return false;
        }

        for col in 0..self.columns() as usize {
            for row in 0..self.vector_size() as usize {
                if self.scalar_u64(col, row) != 0 {
                    return false;
                }
            }
        }

        true
    }
}

#[derive(Clone)]
pub struct Variant {
    pub holder: Option<VariantHolder>,
    _type: Types,
    allow_type_rewrite: bool,
}

impl Variant {
    pub fn set(&mut self, val: VariantHolder) {
        if !self.allow_type_rewrite
            && self._type as u32 != Types::None as u32
            && self._type as u32 != val.get_type() as u32 {
            panic!("Overwriting a variant with new type.");
        }
        self._type = val.get_type();
        self.holder = Some(val);
        self.allow_type_rewrite = false;
    }
    pub fn get(&self) -> &VariantHolder {
        match &self.holder {
            Some(value) => value,
            None => panic!("Holder is None"),
        }
    }

    pub fn get_mut(&mut self) -> &mut VariantHolder {
        match &mut self.holder {
            Some(value) => value,
            None => panic!("Holder is None"),
        }
    }

    pub fn get_type(&self) -> Types {
        self._type
    }

    pub fn get_id(&self) -> u32 {
        if let Some(ref holder) = self.holder {
            holder.get_self()
        } else {
            0
        }
    }

    pub fn empty(&self) -> bool {
        self.holder.is_none()
    }

    pub fn reset(&mut self) {
        self.holder = None;
        self._type = Types::None;
    }

    pub fn set_allow_type_rewrite(&mut self) {
        self.allow_type_rewrite = true;
    }
}

impl Default for Variant {
    fn default() -> Self {
        Self {
            holder: None,
            _type: Types::None,
            allow_type_rewrite: false,
        }
    }
}

struct AccessChainMeta {
    storage_packed_type: u32,
    need_transpose: bool,
    storage_is_packed: bool,
    storage_is_invariant: bool,
}

impl Default for AccessChainMeta {
    fn default() -> Self {
        Self {
            storage_packed_type: 0,
            need_transpose: false,
            storage_is_packed: false,
            storage_is_invariant: false,
        }
    }
}

#[derive(Clone)]
struct DecorationExtended {
    packed: bool,
    packed_type: u32,
}

impl Default for DecorationExtended {
    fn default() -> Self {
        Self {
            packed_type: 0,
            packed: false,
        }
    }
}

#[derive(Clone)]
pub struct Decoration {
    pub alias: String,
    qualified_alias: String,
    pub hlsl_semantic: String,
    pub decoration_flags: Bitset,
    pub builtin_type: spv::BuiltIn,
    pub location: u32,
    pub component: u32,
    pub set: u32,
    pub binding: u32,
    pub offset: u32,
    pub array_stride: u32,
    pub matrix_stride: u32,
    pub input_attachment: u32,
    pub spec_id: u32,
    pub index: u32,
    pub fp_rounding_mode: spv::FPRoundingMode,
    pub builtin: bool,
    extended: DecorationExtended,
}

impl Default for Decoration {
    fn default() -> Self {
        Self {
            alias: String::new(),
            qualified_alias: String::new(),
            hlsl_semantic: String::new(),
            decoration_flags: Bitset::default(),
            builtin_type: spv::BuiltIn::Max,
            location: 0,
            component: 0,
            set: 0,
            binding: 0,
            offset: 0,
            array_stride: 0,
            matrix_stride: 0,
            input_attachment: 0,
            spec_id: 0,
            index: 0,
            fp_rounding_mode: spv::FPRoundingMode::Max,
            builtin: false,
            extended: DecorationExtended::default(),
        }
    }
}

pub struct Meta {
    pub decoration: Decoration,
    pub members: Vec<Decoration>,
    pub decoration_word_offset: HashMap<u32, u32>,
    pub hlsl_is_magic_counter_buffer: bool,
    pub hlsl_magic_counter_buffer: u32,
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            decoration: Decoration::default(),
            members: Vec::new(),
            decoration_word_offset: HashMap::new(),
            hlsl_is_magic_counter_buffer: false,
            hlsl_magic_counter_buffer: 0,
        }
    }
}

// A user callback that remaps the type of any variable.
// var_name is the declared name of the variable.
// name_of_type is the textual name of the type which will be used in the code unless written to by the callback.
type VariableTypeRemapCallback = Fn(&SPIRType, &String, String) -> bool;

struct Hasher {
    h: u64,
}

impl Hasher {
    fn u32(&mut self, value: u32) {
        self.h = (self.h * 0x0100_0000_01b3u64) ^ (u64::from(value));
    }
    fn get(&self) -> u64 {
        self.h
    }
}

fn type_is_floating_point(_type: &SPIRType) -> bool {
    match _type.basetype {
        BaseType::Half => true,
        BaseType::Float => true,
        BaseType::Double => true,
        _ => false,
    }
}

fn type_is_integral(_type: &SPIRType) -> bool {
    match _type.basetype {
        BaseType::SByte => true,
        BaseType::UByte => true,
        BaseType::Short => true,
        BaseType::UShort => true,
        BaseType::Int => true,
        BaseType::UInt => true,
        BaseType::Int64 => true,
        BaseType::UInt64 => true,
        _ => false,
    }
}

pub fn to_signed_basetype(width: u32) -> BaseType {
    match width {
        8 => BaseType::SByte,
        16 => BaseType::Short,
        32 => BaseType::Int,
        64 => BaseType::Int64,
        _ => panic!("Invalid bit width."),
    }
}

pub fn to_unsigned_basetype(width: u32) -> BaseType {
    match width {
        8 => BaseType::UByte,
        16 => BaseType::UShort,
        32 => BaseType::UInt,
        64 => BaseType::UInt64,
        _ => panic!("Invalid bit width."),
    }
}

// Returns true if an arithmetic operation does not change behavior depending on signedness.
fn opcode_is_sign_invariant(opcode: spv::Op) -> bool {
    match opcode {
        spv::Op::IEqual => true,
        spv::Op::INotEqual => true,
        spv::Op::ISub => true,
        spv::Op::IAdd => true,
        spv::Op::IMul => true,
        spv::Op::ShiftLeftLogical => true,
        spv::Op::BitwiseOr => true,
        spv::Op::BitwiseXor => true,
        spv::Op::BitwiseAnd => true,
        _ => false,
    }
}
