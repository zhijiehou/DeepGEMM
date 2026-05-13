#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

// Streaming workspace for MegaMoE kernel with barrier compression
// Phase 1: sequential chunk processing with per-chunk workspace copies
// All chunks' dispatch-send happens before a single NVLink barrier,
// then chunks are processed sequentially (pool buffer reused between chunks).
// Each chunk has its own set of arrival counts, src metadata, etc.
struct StreamingWorkspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_chunks;
    uint32_t chunk_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;  // per chunk: num_ranks * chunk_tokens_per_rank

    // Pool capacity per chunk (chunk-sized, reused between chunks)
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // For both grid barrier and NVLink barrier
    static constexpr uint64_t kNumBarrierSignalBytes = 32;
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    CUTLASS_HOST_DEVICE
    StreamingWorkspace(void* base,
                       const uint32_t& num_ranks,
                       const uint32_t& num_experts,
                       const uint32_t& num_chunks,
                       const uint32_t& chunk_tokens_per_rank,
                       const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_chunks(num_chunks), chunk_tokens_per_rank(chunk_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * chunk_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, chunk_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    // Offset calculations for workspace layout sections
    CUTLASS_HOST_DEVICE uint64_t get_expert_send_counts_bytes() const {
        return num_chunks * num_experts * sizeof(uint64_t);
    }

    CUTLASS_HOST_DEVICE uint64_t get_expert_recv_counts_bytes() const {
        return num_chunks * num_ranks * num_experts_per_rank * sizeof(uint64_t);
    }

    CUTLASS_HOST_DEVICE uint64_t get_expert_recv_count_sums_bytes() const {
        return num_chunks * num_experts_per_rank * sizeof(uint64_t);
    }

    CUTLASS_HOST_DEVICE uint64_t get_l1_arrival_bytes() const {
        return num_chunks * math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE uint64_t get_l2_arrival_bytes() const {
        return num_chunks * num_max_pool_blocks * sizeof(uint64_t);
    }

    CUTLASS_HOST_DEVICE uint64_t get_src_token_topk_bytes() const {
        return num_chunks * num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);
    }

    CUTLASS_HOST_DEVICE uint64_t get_token_src_metadata_bytes() const {
        return num_chunks * num_max_pool_tokens * sizeof(TokenSrcMetadata);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier signals
        num_bytes += kNumBarrierSignalBytes;

        // Per-chunk expert send count
        num_bytes += get_expert_send_counts_bytes();

        // Per-chunk expert recv count (from all source ranks)
        num_bytes += get_expert_recv_counts_bytes();

        // Per-chunk expert recv count sum
        num_bytes += get_expert_recv_count_sums_bytes();

        // Per-chunk L1 arrival count
        num_bytes += get_l1_arrival_bytes();

        // Per-chunk L2 arrival mask
        num_bytes += get_l2_arrival_bytes();

        // Per-chunk dispatch pulling source token-topk
        num_bytes += get_src_token_topk_bytes();

        // Per-chunk token src metadata for combine
        num_bytes += get_token_src_metadata_bytes();

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync and NVLink barrier accessors (same layout as Workspace)
    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    // Per-chunk expert send count
    // Layout: [chunk_0: E entries | chunk_1: E entries | ... ]
    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& chunk_idx = 0, const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + chunk_idx * num_experts + expert_idx;
    }

    // Per-chunk expert recv count (from source rank for local expert)
    // Layout: [chunk_0: R*EPR entries | chunk_1: R*EPR entries | ... ]
    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& chunk_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        const auto offset = kNumBarrierSignalBytes + get_expert_send_counts_bytes();
        return math::advance_ptr<uint64_t>(base, offset) +
               chunk_idx * (num_ranks * num_experts_per_rank) + rank_idx * num_experts_per_rank + expert_idx;
    }

    // Per-chunk expert recv count sum (total received for local expert across all ranks)
    // Layout: [chunk_0: EPR entries | chunk_1: EPR entries | ... ]
    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& chunk_idx = 0, const uint32_t& expert_idx = 0) const {
        const auto offset = kNumBarrierSignalBytes + get_expert_send_counts_bytes() + get_expert_recv_counts_bytes();
        return math::advance_ptr<uint64_t>(base, offset) + chunk_idx * num_experts_per_rank + expert_idx;
    }

    // Per-chunk L1 arrival count
    // Layout: [chunk_0: align(PB,2) entries | chunk_1: align(PB,2) entries | ... ]
    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& chunk_idx = 0, const uint32_t& pool_block_idx = 0) const {
        const auto offset = kNumBarrierSignalBytes + get_expert_send_counts_bytes() + get_expert_recv_counts_bytes() +
                            get_expert_recv_count_sums_bytes();
        const auto aligned_blocks = math::align(num_max_pool_blocks, 2u);
        return reinterpret_cast<uint32_t*>(math::advance_ptr(base, offset)) +
               chunk_idx * aligned_blocks + pool_block_idx;
    }

    // Per-chunk L2 arrival mask
    // Layout: [chunk_0: PB entries | chunk_1: PB entries | ... ]
    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& chunk_idx = 0, const uint32_t& pool_block_idx = 0) const {
        const auto offset = kNumBarrierSignalBytes + get_expert_send_counts_bytes() + get_expert_recv_counts_bytes() +
                            get_expert_recv_count_sums_bytes() + get_l1_arrival_bytes();
        return reinterpret_cast<uint64_t*>(math::advance_ptr(base, offset)) +
               chunk_idx * num_max_pool_blocks + pool_block_idx;
    }

    // Per-chunk src token topk idx (for dispatch pulling)
    // Layout: [chunk_0: EPR*R*RTE entries | chunk_1: EPR*R*RTE entries | ... ]
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& chunk_idx = 0,
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto offset = kNumBarrierSignalBytes + get_expert_send_counts_bytes() + get_expert_recv_counts_bytes() +
                            get_expert_recv_count_sums_bytes() + get_l1_arrival_bytes() + get_l2_arrival_bytes();
        return reinterpret_cast<uint32_t*>(math::advance_ptr(base, offset)) +
            chunk_idx * (num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // Per-chunk token src metadata (for combine)
    // Layout: [chunk_0: PT entries | chunk_1: PT entries | ... ]
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& chunk_idx = 0, const uint32_t& pool_token_idx = 0) const {
        const auto offset = kNumBarrierSignalBytes + get_expert_send_counts_bytes() + get_expert_recv_counts_bytes() +
                            get_expert_recv_count_sums_bytes() + get_l1_arrival_bytes() + get_l2_arrival_bytes() +
                            get_src_token_topk_bytes();
        return reinterpret_cast<TokenSrcMetadata*>(math::advance_ptr(base, offset)) + chunk_idx * num_max_pool_tokens + pool_token_idx;
    }
};

} // namespace deep_gemm::layout