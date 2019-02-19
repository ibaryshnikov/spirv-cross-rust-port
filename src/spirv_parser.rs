use crate::spirv_common::{
    HasType,
    Instruction,
    SPIRBlock,
    SPIRFunction,
    SPIRUndef,
};
use crate::spirv_cross_parsed_ir::{
    ParsedIR,
};
use crate::spirv::{
    self as spv,
    MAGIC_NUMBER,
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
        let op = get_op_from_u32(instruction.op as u32)
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
                let lang = get_source_language_from_u32(ops[0])
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
                let cap = get_capability_from_u32(ops[0])
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

fn get_capability_from_u32(value: u32) -> Option<Capability> {
    match value {
        0 => CapabilityMatrix,
        1 => CapabilityShader,
        2 => CapabilityGeometry,
        3 => CapabilityTessellation,
        4 => CapabilityAddresses,
        5 => CapabilityLinkage,
        6 => CapabilityKernel,
        7 => CapabilityVector16,
        8 => CapabilityFloat16Buffer,
        9 => CapabilityFloat16,
        10 => CapabilityFloat64,
        11 => CapabilityInt64,
        12 => CapabilityInt64Atomics,
        13 => CapabilityImageBasic,
        14 => CapabilityImageReadWrite,
        15 => CapabilityImageMipmap,
        17 => CapabilityPipes,
        18 => CapabilityGroups,
        19 => CapabilityDeviceEnqueue,
        20 => CapabilityLiteralSampler,
        21 => CapabilityAtomicStorage,
        22 => CapabilityInt16,
        23 => CapabilityTessellationPointSize,
        24 => CapabilityGeometryPointSize,
        25 => CapabilityImageGatherExtended,
        27 => CapabilityStorageImageMultisample,
        28 => CapabilityUniformBufferArrayDynamicIndexing,
        29 => CapabilitySampledImageArrayDynamicIndexing,
        30 => CapabilityStorageBufferArrayDynamicIndexing,
        31 => CapabilityStorageImageArrayDynamicIndexing,
        32 => CapabilityClipDistance,
        33 => CapabilityCullDistance,
        34 => CapabilityImageCubeArray,
        35 => CapabilitySampleRateShading,
        36 => CapabilityImageRect,
        37 => CapabilitySampledRect,
        38 => CapabilityGenericPointer,
        39 => CapabilityInt8,
        40 => CapabilityInputAttachment,
        41 => CapabilitySparseResidency,
        42 => CapabilityMinLod,
        43 => CapabilitySampled1D,
        44 => CapabilityImage1D,
        45 => CapabilitySampledCubeArray,
        46 => CapabilitySampledBuffer,
        47 => CapabilityImageBuffer,
        48 => CapabilityImageMSArray,
        49 => CapabilityStorageImageExtendedFormats,
        50 => CapabilityImageQuery,
        51 => CapabilityDerivativeControl,
        52 => CapabilityInterpolationFunction,
        53 => CapabilityTransformFeedback,
        54 => CapabilityGeometryStreams,
        55 => CapabilityStorageImageReadWithoutFormat,
        56 => CapabilityStorageImageWriteWithoutFormat,
        57 => CapabilityMultiViewport,
        58 => CapabilitySubgroupDispatch,
        59 => CapabilityNamedBarrier,
        60 => CapabilityPipeStorage,
        61 => CapabilityGroupNonUniform,
        62 => CapabilityGroupNonUniformVote,
        63 => CapabilityGroupNonUniformArithmetic,
        64 => CapabilityGroupNonUniformBallot,
        65 => CapabilityGroupNonUniformShuffle,
        66 => CapabilityGroupNonUniformShuffleRelative,
        67 => CapabilityGroupNonUniformClustered,
        68 => CapabilityGroupNonUniformQuad,
        4423 => CapabilitySubgroupBallotKHR,
        4427 => CapabilityDrawParameters,
        4431 => CapabilitySubgroupVoteKHR,
        4433 => CapabilityStorageBuffer16BitAccess,
        4433 => CapabilityStorageUniformBufferBlock16,
        4434 => CapabilityStorageUniform16,
        4434 => CapabilityUniformAndStorageBuffer16BitAccess,
        4435 => CapabilityStoragePushConstant16,
        4436 => CapabilityStorageInputOutput16,
        4437 => CapabilityDeviceGroup,
        4439 => CapabilityMultiView,
        4441 => CapabilityVariablePointersStorageBuffer,
        4442 => CapabilityVariablePointers,
        4445 => CapabilityAtomicStorageOps,
        4447 => CapabilitySampleMaskPostDepthCoverage,
        5008 => CapabilityFloat16ImageAMD,
        5009 => CapabilityImageGatherBiasLodAMD,
        5010 => CapabilityFragmentMaskAMD,
        5013 => CapabilityStencilExportEXT,
        5015 => CapabilityImageReadWriteLodAMD,
        5249 => CapabilitySampleMaskOverrideCoverageNV,
        5251 => CapabilityGeometryShaderPassthroughNV,
        5254 => CapabilityShaderViewportIndexLayerEXT,
        5254 => CapabilityShaderViewportIndexLayerNV,
        5255 => CapabilityShaderViewportMaskNV,
        5259 => CapabilityShaderStereoViewNV,
        5260 => CapabilityPerViewAttributesNV,
        5265 => CapabilityFragmentFullyCoveredEXT,
        5568 => CapabilitySubgroupShuffleINTEL,
        5569 => CapabilitySubgroupBufferBlockIOINTEL,
        5570 => CapabilitySubgroupImageBlockIOINTEL,
        0x7fffffff => CapabilityMax,
        _ => None,
    }.into()
}

fn get_op_from_u32(value: u32) -> Option<Op> {
    match value {
        0 => OpNop,
        1 => OpUndef,
        2 => OpSourceContinued,
        3 => OpSource,
        4 => OpSourceExtension,
        5 => OpName,
        6 => OpMemberName,
        7 => OpString,
        8 => OpLine,
        10 => OpExtension,
        11 => OpExtInstImport,
        12 => OpExtInst,
        14 => OpMemoryModel,
        15 => OpEntryPoint,
        16 => OpExecutionMode,
        17 => OpCapability,
        19 => OpTypeVoid,
        20 => OpTypeBool,
        21 => OpTypeInt,
        22 => OpTypeFloat,
        23 => OpTypeVector,
        24 => OpTypeMatrix,
        25 => OpTypeImage,
        26 => OpTypeSampler,
        27 => OpTypeSampledImage,
        28 => OpTypeArray,
        29 => OpTypeRuntimeArray,
        30 => OpTypeStruct,
        31 => OpTypeOpaque,
        32 => OpTypePointer,
        33 => OpTypeFunction,
        34 => OpTypeEvent,
        35 => OpTypeDeviceEvent,
        36 => OpTypeReserveId,
        37 => OpTypeQueue,
        38 => OpTypePipe,
        39 => OpTypeForwardPointer,
        41 => OpConstantTrue,
        42 => OpConstantFalse,
        43 => OpConstant,
        44 => OpConstantComposite,
        45 => OpConstantSampler,
        46 => OpConstantNull,
        48 => OpSpecConstantTrue,
        49 => OpSpecConstantFalse,
        50 => OpSpecConstant,
        51 => OpSpecConstantComposite,
        52 => OpSpecConstantOp,
        54 => OpFunction,
        55 => OpFunctionParameter,
        56 => OpFunctionEnd,
        57 => OpFunctionCall,
        59 => OpVariable,
        60 => OpImageTexelPointer,
        61 => OpLoad,
        62 => OpStore,
        63 => OpCopyMemory,
        64 => OpCopyMemorySized,
        65 => OpAccessChain,
        66 => OpInBoundsAccessChain,
        67 => OpPtrAccessChain,
        68 => OpArrayLength,
        69 => OpGenericPtrMemSemantics,
        70 => OpInBoundsPtrAccessChain,
        71 => OpDecorate,
        72 => OpMemberDecorate,
        73 => OpDecorationGroup,
        74 => OpGroupDecorate,
        75 => OpGroupMemberDecorate,
        77 => OpVectorExtractDynamic,
        78 => OpVectorInsertDynamic,
        79 => OpVectorShuffle,
        80 => OpCompositeConstruct,
        81 => OpCompositeExtract,
        82 => OpCompositeInsert,
        83 => OpCopyObject,
        84 => OpTranspose,
        86 => OpSampledImage,
        87 => OpImageSampleImplicitLod,
        88 => OpImageSampleExplicitLod,
        89 => OpImageSampleDrefImplicitLod,
        90 => OpImageSampleDrefExplicitLod,
        91 => OpImageSampleProjImplicitLod,
        92 => OpImageSampleProjExplicitLod,
        93 => OpImageSampleProjDrefImplicitLod,
        94 => OpImageSampleProjDrefExplicitLod,
        95 => OpImageFetch,
        96 => OpImageGather,
        97 => OpImageDrefGather,
        98 => OpImageRead,
        99 => OpImageWrite,
        100 => OpImage,
        101 => OpImageQueryFormat,
        102 => OpImageQueryOrder,
        103 => OpImageQuerySizeLod,
        104 => OpImageQuerySize,
        105 => OpImageQueryLod,
        106 => OpImageQueryLevels,
        107 => OpImageQuerySamples,
        109 => OpConvertFToU,
        110 => OpConvertFToS,
        111 => OpConvertSToF,
        112 => OpConvertUToF,
        113 => OpUConvert,
        114 => OpSConvert,
        115 => OpFConvert,
        116 => OpQuantizeToF16,
        117 => OpConvertPtrToU,
        118 => OpSatConvertSToU,
        119 => OpSatConvertUToS,
        120 => OpConvertUToPtr,
        121 => OpPtrCastToGeneric,
        122 => OpGenericCastToPtr,
        123 => OpGenericCastToPtrExplicit,
        124 => OpBitcast,
        126 => OpSNegate,
        127 => OpFNegate,
        128 => OpIAdd,
        129 => OpFAdd,
        130 => OpISub,
        131 => OpFSub,
        132 => OpIMul,
        133 => OpFMul,
        134 => OpUDiv,
        135 => OpSDiv,
        136 => OpFDiv,
        137 => OpUMod,
        138 => OpSRem,
        139 => OpSMod,
        140 => OpFRem,
        141 => OpFMod,
        142 => OpVectorTimesScalar,
        143 => OpMatrixTimesScalar,
        144 => OpVectorTimesMatrix,
        145 => OpMatrixTimesVector,
        146 => OpMatrixTimesMatrix,
        147 => OpOuterProduct,
        148 => OpDot,
        149 => OpIAddCarry,
        150 => OpISubBorrow,
        151 => OpUMulExtended,
        152 => OpSMulExtended,
        154 => OpAny,
        155 => OpAll,
        156 => OpIsNan,
        157 => OpIsInf,
        158 => OpIsFinite,
        159 => OpIsNormal,
        160 => OpSignBitSet,
        161 => OpLessOrGreater,
        162 => OpOrdered,
        163 => OpUnordered,
        164 => OpLogicalEqual,
        165 => OpLogicalNotEqual,
        166 => OpLogicalOr,
        167 => OpLogicalAnd,
        168 => OpLogicalNot,
        169 => OpSelect,
        170 => OpIEqual,
        171 => OpINotEqual,
        172 => OpUGreaterThan,
        173 => OpSGreaterThan,
        174 => OpUGreaterThanEqual,
        175 => OpSGreaterThanEqual,
        176 => OpULessThan,
        177 => OpSLessThan,
        178 => OpULessThanEqual,
        179 => OpSLessThanEqual,
        180 => OpFOrdEqual,
        181 => OpFUnordEqual,
        182 => OpFOrdNotEqual,
        183 => OpFUnordNotEqual,
        184 => OpFOrdLessThan,
        185 => OpFUnordLessThan,
        186 => OpFOrdGreaterThan,
        187 => OpFUnordGreaterThan,
        188 => OpFOrdLessThanEqual,
        189 => OpFUnordLessThanEqual,
        190 => OpFOrdGreaterThanEqual,
        191 => OpFUnordGreaterThanEqual,
        194 => OpShiftRightLogical,
        195 => OpShiftRightArithmetic,
        196 => OpShiftLeftLogical,
        197 => OpBitwiseOr,
        198 => OpBitwiseXor,
        199 => OpBitwiseAnd,
        200 => OpNot,
        201 => OpBitFieldInsert,
        202 => OpBitFieldSExtract,
        203 => OpBitFieldUExtract,
        204 => OpBitReverse,
        205 => OpBitCount,
        207 => OpDPdx,
        208 => OpDPdy,
        209 => OpFwidth,
        210 => OpDPdxFine,
        211 => OpDPdyFine,
        212 => OpFwidthFine,
        213 => OpDPdxCoarse,
        214 => OpDPdyCoarse,
        215 => OpFwidthCoarse,
        218 => OpEmitVertex,
        219 => OpEndPrimitive,
        220 => OpEmitStreamVertex,
        221 => OpEndStreamPrimitive,
        224 => OpControlBarrier,
        225 => OpMemoryBarrier,
        227 => OpAtomicLoad,
        228 => OpAtomicStore,
        229 => OpAtomicExchange,
        230 => OpAtomicCompareExchange,
        231 => OpAtomicCompareExchangeWeak,
        232 => OpAtomicIIncrement,
        233 => OpAtomicIDecrement,
        234 => OpAtomicIAdd,
        235 => OpAtomicISub,
        236 => OpAtomicSMin,
        237 => OpAtomicUMin,
        238 => OpAtomicSMax,
        239 => OpAtomicUMax,
        240 => OpAtomicAnd,
        241 => OpAtomicOr,
        242 => OpAtomicXor,
        245 => OpPhi,
        246 => OpLoopMerge,
        247 => OpSelectionMerge,
        248 => OpLabel,
        249 => OpBranch,
        250 => OpBranchConditional,
        251 => OpSwitch,
        252 => OpKill,
        253 => OpReturn,
        254 => OpReturnValue,
        255 => OpUnreachable,
        256 => OpLifetimeStart,
        257 => OpLifetimeStop,
        259 => OpGroupAsyncCopy,
        260 => OpGroupWaitEvents,
        261 => OpGroupAll,
        262 => OpGroupAny,
        263 => OpGroupBroadcast,
        264 => OpGroupIAdd,
        265 => OpGroupFAdd,
        266 => OpGroupFMin,
        267 => OpGroupUMin,
        268 => OpGroupSMin,
        269 => OpGroupFMax,
        270 => OpGroupUMax,
        271 => OpGroupSMax,
        274 => OpReadPipe,
        275 => OpWritePipe,
        276 => OpReservedReadPipe,
        277 => OpReservedWritePipe,
        278 => OpReserveReadPipePackets,
        279 => OpReserveWritePipePackets,
        280 => OpCommitReadPipe,
        281 => OpCommitWritePipe,
        282 => OpIsValidReserveId,
        283 => OpGetNumPipePackets,
        284 => OpGetMaxPipePackets,
        285 => OpGroupReserveReadPipePackets,
        286 => OpGroupReserveWritePipePackets,
        287 => OpGroupCommitReadPipe,
        288 => OpGroupCommitWritePipe,
        291 => OpEnqueueMarker,
        292 => OpEnqueueKernel,
        293 => OpGetKernelNDrangeSubGroupCount,
        294 => OpGetKernelNDrangeMaxSubGroupSize,
        295 => OpGetKernelWorkGroupSize,
        296 => OpGetKernelPreferredWorkGroupSizeMultiple,
        297 => OpRetainEvent,
        298 => OpReleaseEvent,
        299 => OpCreateUserEvent,
        300 => OpIsValidEvent,
        301 => OpSetUserEventStatus,
        302 => OpCaptureEventProfilingInfo,
        303 => OpGetDefaultQueue,
        304 => OpBuildNDRange,
        305 => OpImageSparseSampleImplicitLod,
        306 => OpImageSparseSampleExplicitLod,
        307 => OpImageSparseSampleDrefImplicitLod,
        308 => OpImageSparseSampleDrefExplicitLod,
        309 => OpImageSparseSampleProjImplicitLod,
        310 => OpImageSparseSampleProjExplicitLod,
        311 => OpImageSparseSampleProjDrefImplicitLod,
        312 => OpImageSparseSampleProjDrefExplicitLod,
        313 => OpImageSparseFetch,
        314 => OpImageSparseGather,
        315 => OpImageSparseDrefGather,
        316 => OpImageSparseTexelsResident,
        317 => OpNoLine,
        318 => OpAtomicFlagTestAndSet,
        319 => OpAtomicFlagClear,
        320 => OpImageSparseRead,
        321 => OpSizeOf,
        322 => OpTypePipeStorage,
        323 => OpConstantPipeStorage,
        324 => OpCreatePipeFromPipeStorage,
        325 => OpGetKernelLocalSizeForSubgroupCount,
        326 => OpGetKernelMaxNumSubgroups,
        327 => OpTypeNamedBarrier,
        328 => OpNamedBarrierInitialize,
        329 => OpMemoryNamedBarrier,
        330 => OpModuleProcessed,
        331 => OpExecutionModeId,
        332 => OpDecorateId,
        333 => OpGroupNonUniformElect,
        334 => OpGroupNonUniformAll,
        335 => OpGroupNonUniformAny,
        336 => OpGroupNonUniformAllEqual,
        337 => OpGroupNonUniformBroadcast,
        338 => OpGroupNonUniformBroadcastFirst,
        339 => OpGroupNonUniformBallot,
        340 => OpGroupNonUniformInverseBallot,
        341 => OpGroupNonUniformBallotBitExtract,
        342 => OpGroupNonUniformBallotBitCount,
        343 => OpGroupNonUniformBallotFindLSB,
        344 => OpGroupNonUniformBallotFindMSB,
        345 => OpGroupNonUniformShuffle,
        346 => OpGroupNonUniformShuffleXor,
        347 => OpGroupNonUniformShuffleUp,
        348 => OpGroupNonUniformShuffleDown,
        349 => OpGroupNonUniformIAdd,
        350 => OpGroupNonUniformFAdd,
        351 => OpGroupNonUniformIMul,
        352 => OpGroupNonUniformFMul,
        353 => OpGroupNonUniformSMin,
        354 => OpGroupNonUniformUMin,
        355 => OpGroupNonUniformFMin,
        356 => OpGroupNonUniformSMax,
        357 => OpGroupNonUniformUMax,
        358 => OpGroupNonUniformFMax,
        359 => OpGroupNonUniformBitwiseAnd,
        360 => OpGroupNonUniformBitwiseOr,
        361 => OpGroupNonUniformBitwiseXor,
        362 => OpGroupNonUniformLogicalAnd,
        363 => OpGroupNonUniformLogicalOr,
        364 => OpGroupNonUniformLogicalXor,
        365 => OpGroupNonUniformQuadBroadcast,
        366 => OpGroupNonUniformQuadSwap,
        4421 => OpSubgroupBallotKHR,
        4422 => OpSubgroupFirstInvocationKHR,
        4428 => OpSubgroupAllKHR,
        4429 => OpSubgroupAnyKHR,
        4430 => OpSubgroupAllEqualKHR,
        4432 => OpSubgroupReadInvocationKHR,
        5000 => OpGroupIAddNonUniformAMD,
        5001 => OpGroupFAddNonUniformAMD,
        5002 => OpGroupFMinNonUniformAMD,
        5003 => OpGroupUMinNonUniformAMD,
        5004 => OpGroupSMinNonUniformAMD,
        5005 => OpGroupFMaxNonUniformAMD,
        5006 => OpGroupUMaxNonUniformAMD,
        5007 => OpGroupSMaxNonUniformAMD,
        5011 => OpFragmentMaskFetchAMD,
        5012 => OpFragmentFetchAMD,
        5571 => OpSubgroupShuffleINTEL,
        5572 => OpSubgroupShuffleDownINTEL,
        5573 => OpSubgroupShuffleUpINTEL,
        5574 => OpSubgroupShuffleXorINTEL,
        5575 => OpSubgroupBlockReadINTEL,
        5576 => OpSubgroupBlockWriteINTEL,
        5577 => OpSubgroupImageBlockReadINTEL,
        5578 => OpSubgroupImageBlockWriteINTEL,
        5632 => OpDecorateStringGOOGLE,
        5633 => OpMemberDecorateStringGOOGLE,
        0x7fffffff => OpMax,
        _ => None,
    }.into()
}

fn get_source_language_from_u32(value: u32) -> Option<SourceLanguage> {
    match value {
        0 => SourceLanguageUnknown,
        1 => SourceLanguageESSL,
        2 => SourceLanguageGLSL,
        3 => SourceLanguageOpenCL_C,
        4 => SourceLanguageOpenCL_CPP,
        5 => SourceLanguageHLSL,
        0x7fffffff => SourceLanguageMax,
        _ => None,
    }.into()
}
