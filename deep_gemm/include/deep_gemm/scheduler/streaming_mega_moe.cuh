#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/streaming_mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

// BlockPhase is defined in mega_moe.cuh (same enum values)
// No redefinition here to avoid conflict when both headers are included

// Streaming scheduler: extends MegaMoEScheduler with per-chunk expert counts.
// The pool is reused between chunks (same addresses), but each chunk has its
// own arrival counts and src metadata in StreamingWorkspace.
// Usage: call init_chunk(chunk_idx) then for_each_block_in_chunk(func) for each chunk.
// The kernel code manages the chunk loop and inter-chunk synchronization.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumChunks,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct StreamingMegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    const layout::StreamingWorkspace& workspace;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current chunk and expert indices
    uint32_t current_chunk_idx = 0;
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Pre-cached per-expert token counts (filled during `init_chunk`)
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit StreamingMegaMoEScheduler(const layout::StreamingWorkspace& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        return math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }
            } else {
                if (fetch_next_l2_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    next_phase = BlockPhase::Linear1;
                }
            }
        }
        return {BlockPhase::None, 0, 0, 0};
    }

    // Initialize scheduler state for a specific chunk.
    // Must be called before for_each_block_in_chunk() for each chunk.
    CUTLASS_DEVICE void init_chunk(const uint32_t& chunk_idx) {
        current_chunk_idx = chunk_idx;

        // Wait for all expert counters to be finalized for this chunk
        fetch_expert_recv_count(chunk_idx);

        // Reset scheduler state for this chunk
        next_phase = BlockPhase::Linear1;
        set_expert_idx(0);
        block_idx = blockIdx.x;
    }

    // Per-chunk expert recv count fetching
    CUTLASS_DEVICE void fetch_expert_recv_count(const uint32_t& chunk_idx) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(chunk_idx, expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    // Iterate over all blocks for the current chunk (set by init_chunk).
    // The kernel code manages the chunk loop and inter-chunk synchronization.
    template <typename Func>
    CUTLASS_DEVICE void for_each_block_in_chunk(Func&& func) {
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }
};

} // namespace deep_gemm::sched