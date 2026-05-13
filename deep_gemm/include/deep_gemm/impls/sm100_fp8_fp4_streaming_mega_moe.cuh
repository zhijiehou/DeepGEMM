#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/layout/streaming_mega_moe.cuh>
#include <deep_gemm/scheduler/streaming_mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumChunks,
    uint32_t kChunkTokensPerRank,
    uint32_t kNumSMs, uint32_t kNumRanks,
    uint32_t kActivationClampBits,
    bool kFastMath,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_streaming_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const uint32_t chunk_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(kNumMaxTokensPerRank == kNumChunks * kChunkTokensPerRank, "Invalid token count for streaming");
    DG_STATIC_ASSERT(kChunkTokensPerRank % BLOCK_M == 0, "Chunk tokens must be aligned to BLOCK_M");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
    }

    // Workspaces
    const auto streaming_workspace = layout::StreamingWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumChunks, kChunkTokensPerRank, kNumTopk);
    // Also create a regular workspace for grid_sync/nvlink_barrier compatibility
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    // Token and buffer layouts
    constexpr auto fp8_token_layout = layout::Data(kHidden);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Registered inputs
    constexpr uint32_t kTotalTokensPerRank = kNumChunks * kChunkTokensPerRank;
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kTotalTokensPerRank,
        streaming_workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kTotalTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kTotalTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kTotalTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // L1 inputs
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());

    // L2 inputs
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr()
    );
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr()
    );

    // Combine inputs
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kTotalTokensPerRank,
        l2_sf_buffer.get_end_ptr()
    );

    // Data types
    // NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t kSwizzleBMode = BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE = SMEM_CD_L1_SIZE / kNumTMAStoreStages;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemoryAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemoryAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemoryAlignment == 0,
                     "Shared memory of CD/A/B must be aligned to 1024 bytes");

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Assign shared memory for dispatch warps
    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    // GEMM shared memory: C/D, A, B
    // NOTES: GEMM shared memory starts after the dispatch region, aligned to 1024 bytes
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE
    );

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(smem_gemm_base, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_cd_l2 = smem_cd[0];
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SF shared memory: SFA and SFB per pipeline stage
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Epilogue amax reduction shared memory
    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumStages]);

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + STORE_BLOCK_M * kNumEpilogueWarps / 2);
    auto dispatch_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages + i); });
    auto combine_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + i); });
    auto tmem_ptr_in_smem       = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + kNumEpilogueWarps * 2);

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync())
            ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at all CTAs
                full_barriers[i]->init(2 * 2);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                tmem_full_barriers[i]->init(1);
                // Arrive only at the leader CTA
                tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    // NOTES: Using `.relaxed` is allowed here since `fence_barrier_init` is `.release.cluster`,
    // and `barrier.cluster.wait.aligned` is by default `.acquire`
    comm::cluster_sync_with_relaxed_arrive();

    // Task scheduler
    auto scheduler = sched::StreamingMegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumChunks,
        kNumSMs, kNumRanks>(streaming_workspace);

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;
    constexpr uint32_t kInterChunkBarrierTag = 4;

    // Adjust registers
    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;
    constexpr uint32_t kInterChunkGridSyncIndex = 2;
    constexpr uint32_t kInterChunkBarrierIdx = 5;

    // Different warp roles - restructured for streaming with chunk loop
    // Phase 1: Dispatch-send for ALL chunks, then single NVLink barrier
    // Phase 2: Chunk loop: dispatch-pull + GEMM + epilogue per chunk with inter-chunk sync
    // Phase 3: Workspace clean for all chunks + combine-reduce for total tokens

    // ================================================================
    // Phase 1: Dispatch-send for ALL chunks (dispatch warps only)
    // ================================================================
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;

        // Process each chunk's dispatch-send
        for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
            const uint32_t chunk_token_start = chunk_idx * kChunkTokensPerRank;
            const uint32_t chunk_token_end = chunk_token_start + chunk_tokens;

            const auto read_topk_idx_chunk = [&](const auto& process) {
                #pragma unroll
                for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < chunk_tokens;
                     i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    const uint32_t global_i = chunk_token_start + i;
                    // NOTES: each warp processes kNumTokensPerWarp consecutive tokens
                    // (lanes 0..K-1 cover token global_i, lanes K..2K-1 cover global_i+1, ...).
                    // The bound must stay within both the current chunk and num_tokens
                    // to avoid spilling into the next chunk's metadata.
                    const uint32_t per_lane_token = global_i + (lane_idx / kNumTopk);
                    if (per_lane_token < chunk_token_end and per_lane_token < num_tokens
                        and lane_idx < kNumActivateLanes) {
                        expert_idx = static_cast<int>(
                            __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + global_i * kNumTopk + lane_idx));
                        if (expert_idx >= 0)
                            process(global_i * kNumTopk + lane_idx, expert_idx);
                    }
                    __syncwarp();
                }
            };

            // Clean smem_expert_count for this chunk iteration.
            // NOTES: `st.bulk` is async; without a CTA-wide barrier the following
            // `atomicAdd_block` on other dispatch warps may read stale values
            // (e.g. previous chunk's per-expert offsets), corrupting this chunk's
            // dispatch counts. The post-bulk `sync_aligned` fences the bulk
            // store and synchronizes all dispatch warps before counting begins.
            if (cute::elect_one_sync())
                ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Count experts' tokens for this chunk
            read_topk_idx_chunk([&](const uint32_t& token_topk_idx, const int& expert_idx) {
               atomicAdd_block(smem_expert_count + expert_idx, 1);
            });
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Get SM offset for this chunk
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
                smem_expert_count[i] = static_cast<uint32_t>(
                    ptx::atomic_add(streaming_workspace.get_expert_send_count_ptr(chunk_idx, i), send_value));
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Write source indices for this chunk
            read_topk_idx_chunk([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
                const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
                const auto dst_ptr = streaming_workspace.get_src_token_topk_idx_ptr(
                    chunk_idx, expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            });
            // NOTES: sync ALL dispatch warps at the chunk boundary so that the
            // next chunk's `st_shared_bulk` doesn't wipe `smem_expert_count`
            // while in-flight `atomicAdd_block` from this chunk's second loop
            // (allocating dst_slot_idx) is still running on slower warps.
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        }

        // Grid sync (all chunks' dispatch-send complete)
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write per-chunk expert recv counts (SM 0 only)
        if (sm_idx == 0) {
            for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                    const auto dst_rank_idx = i / kNumExpertsPerRank;
                    const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                    const auto expert_status = *streaming_workspace.get_expert_send_count_ptr(chunk_idx, i);
                    *sym_buffer.map(
                        streaming_workspace.get_expert_recv_count_ptr(chunk_idx, sym_buffer.rank_idx, dst_local_expert_idx),
                        dst_rank_idx) = expert_status & 0xffffffff;
                    ptx::atomic_add_sys(
                        sym_buffer.map(streaming_workspace.get_expert_recv_count_sum_ptr(chunk_idx, dst_local_expert_idx), dst_rank_idx),
                        expert_status);
                }
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Single NVLink barrier before dispatch-pull (barrier compression)
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true
        );

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // ================================================================
        // Phase 2: Chunk loop - dispatch-pull per chunk
        // ================================================================
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
            scheduler.init_chunk(chunk_idx);

            constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
            const uint32_t chunk_token_start = chunk_idx * kChunkTokensPerRank;

            current_expert_idx = -1;
            expert_start_idx = 0;
            expert_end_idx = 0;
            expert_pool_block_offset = 0;

            for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }

                if (current_expert_idx >= kNumExpertsPerRank)
                    break;

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*streaming_workspace.get_expert_recv_count_ptr(chunk_idx, j, current_expert_idx)) : 0;
                    }
                }

                // Round-robin rank selection
                uint32_t current_rank_in_expert_idx;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t slot_idx = token_idx_in_expert;
                uint32_t token_idx_in_rank;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);
                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }

                const uint32_t src_token_topk_idx = *streaming_workspace.get_src_token_topk_idx_ptr(
                    chunk_idx, current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                constexpr uint32_t kNumSFUint32 = kHidden / 128;
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
                const auto sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M +
                    transform_sf_token_idx(token_idx_in_expert);
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFUint32)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
                }
                __syncwarp();

                const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                if (cute::elect_one_sync()) {
                    const auto weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *streaming_workspace.get_token_src_metadata_ptr(chunk_idx, pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        streaming_workspace.get_l1_arrival_count_ptr(chunk_idx, expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                }
                __syncwarp();
            }

            // Inter-chunk sync
            if (chunk_idx < kNumChunks - 1) {
                ptx::sync_unaligned(kNumThreads, kInterChunkBarrierIdx);
                comm::grid_sync<kNumSMs, kInterChunkGridSyncIndex>(
                    workspace, sm_idx, thread_idx,
                    [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
                );
                __syncwarp();
            }
        }

        // Phase 3: Workspace clean for all chunks
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                    *streaming_workspace.get_expert_send_count_ptr(chunk_idx, i) = 0;
            }
        } else {
            for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
                for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                    const auto num_recv_tokens = static_cast<uint32_t>(
                        *streaming_workspace.get_expert_recv_count_sum_ptr(chunk_idx, i));
                    const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                    // Manually fill stored_num_tokens_per_expert for pool offset calculation
                    // (Cannot use init_chunk here — it spins on recv count sums which are being cleared)
                    constexpr uint32_t num_experts_per_lane_clean = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
                    #pragma unroll
                    for (uint32_t j = 0; j < num_experts_per_lane_clean; ++ j) {
                        const auto expert_idx_j = j * 32 + ptx::get_lane_idx();
                        scheduler.stored_num_tokens_per_expert[j] = (expert_idx_j < kNumExpertsPerRank) ?
                            static_cast<uint32_t>(*streaming_workspace.get_expert_recv_count_sum_ptr(chunk_idx, expert_idx_j)) : 0;
                    }
                    __syncwarp();
                    expert_pool_block_offset = scheduler.get_pool_block_offset(i);
                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                    DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                    if (warp_idx == 0) {
                        *streaming_workspace.get_expert_recv_count_sum_ptr(chunk_idx, i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }

                    for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                        *streaming_workspace.get_expert_recv_count_ptr(chunk_idx, j, i) = 0;
                    __syncwarp();

                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                        *streaming_workspace.get_l1_arrival_count_ptr(chunk_idx, expert_pool_block_offset + j) = 0;
                        *streaming_workspace.get_l2_arrival_mask_ptr(chunk_idx, expert_pool_block_offset + j) = 0;
                    }
                    __syncwarp();
                }
            }
        }

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false
        );

    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
            scheduler.init_chunk(chunk_idx);
            scheduler.for_each_block_in_chunk([&](const sched::BlockPhase& block_phase,
                                                 const uint32_t& local_expert_idx,
                                                 const uint32_t& num_k_blocks,
                                                 const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_l2_acts : &tensor_map_l1_acts;
                const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

                const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
                const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

                if (block_phase == sched::BlockPhase::Linear1) {
                    const auto ptr = streaming_workspace.get_l1_arrival_count_ptr(chunk_idx, pool_block_idx);
                    const auto expected = scheduler.template get_valid_m<false>();
                    while (ptx::ld_acq(ptr) != expected);
                } else {
                    DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");
                    const auto ptr = streaming_workspace.get_l2_arrival_mask_ptr(chunk_idx, pool_block_idx);
                    const uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1;
                    while (ptx::ld_acq_gpu(ptr) != expected);
                }

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    uint32_t m_idx = pool_block_idx * BLOCK_M;
                    uint32_t k_idx = k_block_idx * BLOCK_K;
                    uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                    uint32_t sfa_k_idx = k_block_idx;
                    if (not is_leader_cta)
                        m_idx += scheduler.template get_valid_m<true>() / 2;
                    if (cute::elect_one_sync()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 2);
                        tma::copy<SF_BLOCK_M, 1, 0>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                        if (is_leader_cta) {
                            full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE * 2 + SF_BLOCK_M * sizeof(uint32_t) * 2);
                        } else {
                            full_barriers[stage_idx]->arrive(0u);
                        }
                    }
                    __syncwarp();
                }
            });
            if (chunk_idx < kNumChunks - 1) {
                ptx::sync_unaligned(kNumThreads, kInterChunkBarrierIdx);
                __syncwarp();
            }
        }

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
            scheduler.init_chunk(chunk_idx);
            scheduler.for_each_block_in_chunk([&](const sched::BlockPhase& block_phase,
                                                 const uint32_t& local_expert_idx,
                                                 const uint32_t& num_k_blocks,
                                                 const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                const auto tensor_map_b_ptr =
                    block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;
                const auto tensor_map_sfb_ptr =
                    block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;
                const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
                const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;
                const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    uint32_t k_idx = k_block_idx * BLOCK_K;
                    uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                    uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx;
                    if (cute::elect_one_sync()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t) * 2);
                        } else {
                            full_barriers[stage_idx]->arrive(0u);
                        }
                    }
                    __syncwarp();
                }
            });
            if (chunk_idx < kNumChunks - 1) {
                ptx::sync_unaligned(kNumThreads, kInterChunkBarrierIdx);
                __syncwarp();
            }
        }

    } else if (warp_idx == kNumDispatchWarps + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        if (is_leader_cta) {
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K>();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);
            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            uint32_t current_iter_idx = 0;
            for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
                scheduler.init_chunk(chunk_idx);
                scheduler.for_each_block_in_chunk([&](const sched::BlockPhase& block_phase,
                                                     const uint32_t& local_expert_idx,
                                                     const uint32_t& num_k_blocks,
                                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());
                    const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                    const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                    tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                    ptx::tcgen05_after_thread_sync();

                    auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                        auto umma_arrive = [](const uint64_t* barrier) {
                            constexpr uint16_t kCTAMask = (1 << 2) - 1;
                            cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                        };
                        umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                        if (do_tmem_full_arrive)
                            umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                        __syncwarp();
                    };

                    #pragma unroll 2
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);
                        ptx::tcgen05_after_thread_sync();
                        const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                        const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                        if (cute::elect_one_sync()) {
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                                b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                                ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                    b_desc, a_desc, accum_stage_idx * UMMA_N,
                                    k_block_idx > 0 or k > 0, runtime_instr_desc,
                                    kTmemStartColOfSFB, kTmemStartColOfSFA);
                            }
                        }
                        __syncwarp();
                        empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                    }
                });
                if (current_iter_idx > 0) {
                    const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                    tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);
                }
                if (chunk_idx < kNumChunks - 1) {
                    ptx::sync_unaligned(kNumThreads, kInterChunkBarrierIdx);
                    __syncwarp();
                }
            }
        } else {
            // Non-leader CTA: must participate in inter-chunk barriers but no MMA work
            for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks - 1; ++chunk_idx) {
                ptx::sync_unaligned(kNumThreads, kInterChunkBarrierIdx);
                __syncwarp();
            }
        }

    } else if (warp_idx == kNumDispatchWarps + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks - 1; ++chunk_idx) {
            ptx::sync_unaligned(kNumThreads, kInterChunkBarrierIdx);
            __syncwarp();
        }

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;

        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        uint32_t current_iter_idx = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++chunk_idx) {
            scheduler.init_chunk(chunk_idx);
            scheduler.for_each_block_in_chunk([&](const sched::BlockPhase& block_phase,
                                                 const uint32_t& local_expert_idx,
                                                 const uint32_t& num_k_blocks,
                                                 const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
                ptx::tcgen05_after_thread_sync();
                const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
                const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t n_idx = n_block_idx * BLOCK_N;

                if (block_phase == sched::BlockPhase::Linear1) {
                    // L1 SwiGLU epilogue (same as original, per-chunk workspace)
                    float stored_cached_weight = 0;
                    #pragma unroll
                    for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                        if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                            break;
                        }
                        float2 swiglu_values[kNumAtomsPerStore * 2];
                        float2 amax_values[kNumAtomsPerStore];
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                            const uint32_t j = s * kNumAtomsPerStore + i;
                            if ((j * ATOM_M) % 32 == 0 and (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                                stored_cached_weight = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                    .get_base_ptr<float>();
                            }
                            const float2 weights = {
                                ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                                ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                            };
                            uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                            uint32_t values[ATOM_M];
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, values[4], values[5], values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();
                            if (j == WG_BLOCK_M / ATOM_M - 1) {
                                ptx::tcgen05_before_thread_sync();
                                tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                            }
                            auto fp32_values = reinterpret_cast<float*>(values);
                            #pragma unroll
                            for (uint32_t k = 0; k < 2; ++ k) {
                                auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                                auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));
                                if constexpr (kActivationClampBits != 0x7F800000u) {
                                    const float kActivationClamp = __uint_as_float(kActivationClampBits);
                                    bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                    bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                    bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                                }
                                auto gate = __bfloat1622float2(bf16_gate);
                                auto neg_gate_exp = make_float2(
                                    kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                    kFastMath ? __expf(-gate.y) : expf(-gate.y));
                                const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                                if constexpr (kFastMath) {
                                    gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                                } else {
                                    gate = {gate.x / denom.x, gate.y / denom.y};
                                }
                                const auto up = __bfloat1622float2(bf16_up);
                                swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                            }
                            amax_values[i].x = math::warp_reduce<4, true>(
                                cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                                math::ReduceMax<float>());
                            amax_values[i].y = math::warp_reduce<4, true>(
                                cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                                math::ReduceMax<float>());
                            if (lane_idx < 4)
                                smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                            __syncwarp();
                        }
                        const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                        ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                            const float2 wp_amax =
                                smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                            amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                            amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);
                            float2 sf, sf_inv;
                            math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);
                            const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                            const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                            const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));
                            uint32_t row = lane_idx;
                            uint32_t col = warp_idx_in_wg;
                            const auto smem_ptr = smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                                                         + i * ATOM_M * L1_OUT_BLOCK_N
                                                                         + row * L1_OUT_BLOCK_N
                                                                         + (col ^ (row / 2)) * kNumBankGroupBytes;
                            ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);
                            if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                                const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                                const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                                const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                                const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                                const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                                __builtin_assume(token_base_idx < BLOCK_M);
                                const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                                    + m_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                                const auto sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                                sf_base_ptr[sf_addr] = (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                                sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] = (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                            }
                            __syncwarp();
                        }
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                                out_n_idx,
                                m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                    }
                    ptx::tma_store_wait<0>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                        ptx::red_or_rel_gpu(
                            streaming_workspace.get_l2_arrival_mask_ptr(chunk_idx, pool_block_idx),
                            1ull << n_block_idx);
                    }
                    __syncwarp();

                } else {
                    // L2 BF16 epilogue: write to remote combine buffer (per-chunk src metadata)
                    DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                    constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;
                    #pragma unroll
                    for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                        if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                            break;
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                            uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                            uint32_t values[ATOM_M];
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, values[4], values[5], values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();
                            if (i == 0 and s > 0)
                                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                            if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                                ptx::tcgen05_before_thread_sync();
                                tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                            }
                            uint32_t row = lane_idx % 8;
                            uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                            const auto smem_ptr = smem_cd_l2 +
                                epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                                i * ATOM_M * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) +
                                (col ^ row) * kNumBankGroupBytes;
                            ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                math::cast_into_bf16_and_pack(values[0], values[1]),
                                math::cast_into_bf16_and_pack(values[2], values[3]),
                                math::cast_into_bf16_and_pack(values[4], values[5]),
                                math::cast_into_bf16_and_pack(values[6], values[7]),
                                smem_ptr);
                        }
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                        const uint32_t bank_group_idx = lane_idx % 8;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                            const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;
                            if (m_idx_in_block >= valid_m)
                                break;
                            const auto src_metadata = *streaming_workspace.get_token_src_metadata_ptr(chunk_idx, m_idx + m_idx_in_block);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            const uint32_t dst_token_idx = src_metadata.token_idx;
                            const uint32_t dst_topk_idx = src_metadata.topk_idx;
                            const auto smem_ptr2 = smem_cd_l2 +
                                epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                                row_in_store * kSwizzleCDMode +
                                (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                            const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr2));
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            const auto dst_ptr = math::advance_ptr<float4>(
                                dst_token.get_base_ptr(),
                                n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            });
            if (chunk_idx < kNumChunks - 1) {
                ptx::sync_unaligned(kNumThreads, kInterChunkBarrierIdx);
                __syncwarp();
            }
        }

        // Phase 3: Combine-reduce for total tokens (epilogue warps)
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Combine: reduce top-k results for ALL tokens
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);
        constexpr uint32_t kNumCombineChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kNumCombineChunks =
            kNumCombineChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumCombineChunkBytes = kNumHiddenBytes / kNumCombineChunks;
        constexpr uint32_t kNumCombineChunkUint4 = kNumCombineChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumCombineChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumCombineChunks == 0, "Hidden must be divisible by combine chunks");
        DG_STATIC_ASSERT(kNumCombineChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumCombineChunks <= SMEM_BEFORE_BARRIER_SIZE, "Too large hidden");
        DG_STATIC_ASSERT(kNumCombineChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned");
        DG_STATIC_ASSERT(kNumCombineChunkBytes % sizeof(uint4) == 0, "Combine chunk must be uint4-aligned");
        DG_STATIC_ASSERT(kNumCombineChunkUint4 % 32 == 0, "Combine chunk must be multiple of 32");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_DEVICE_ASSERT(kNumCombineChunkSlots * kNumEpilogueWarps * kNumCombineChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumCombineChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumCombineChunkBytes);
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumCombineChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumCombineChunkBytes;
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumCombineChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumCombineChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumCombineChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
