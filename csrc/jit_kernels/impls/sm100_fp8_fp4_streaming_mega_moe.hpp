#pragma once

#include <cstring>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/streaming_mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP8FP4StreamingMegaMoERuntime final : public LaunchRuntime<SM100FP8FP4StreamingMegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_chunks, chunk_tokens_per_rank;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        int chunk_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_streaming_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_streaming_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.num_chunks, args.chunk_tokens_per_rank,
    args.launch_args.grid_dim.first, args.num_ranks,
    [&]() { uint32_t b; std::memcpy(&b, &args.activation_clamp, sizeof(b)); return fmt::format("{}u", b); }(),
    args.fast_math ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.chunk_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf
        ));
    }
};

} // namespace deep_gemm