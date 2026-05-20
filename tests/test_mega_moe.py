import argparse
import math
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        # Check SF requirements
        assert hidden % 128 == 0
        assert intermediate_hidden % 128 == 0
        assert l1_weights.shape[2] % 128 == 0 and l2_weights.shape[2] % 128 == 0

        # Cast inputs to FP8 with per-32 UE8M0 SF
        x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

        # Cast grouped BF16 weights to FP4 with MN-major SF
        # TODO: merge with `cast_fp8_fp4_with_major`
        def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_groups, n, k = bf16_weights.shape
            w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
            w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
            for i in range(num_groups):
                w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
            w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
            return w, w_sf

        l1_weights = cast_grouped_weights_to_fp4(l1_weights)
        l2_weights = cast_grouped_weights_to_fp4(l2_weights)
        transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def run_fused():
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        # noinspection PyTypeChecker
        deep_gemm.fp8_fp4_mega_moe(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=True,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
    ) if is_legacy_loaded else None

    def run_baseline():
        recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
            num_experts=num_experts, expert_alignment=alignment,
            do_cpu_sync=False, do_handle_copy=False,
            do_expand=True, use_tma_aligned_col_major_sf=True,
        )
        n = recv_x[0].size(0)
        l1_y = torch.empty((n, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        # noinspection PyCallingNonCallable
        l1_y = tilelang_ops.swiglu_apply_weight_to_fp8(
            x=l1_y,
            topk_weights=recv_topk_weights,
            avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
            num_per_channels=32,
            use_col_major_scales=True,
            round_scale=True,
            ue8m0_scale=True,
            output_bf16=False,
            clamp_value=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # Check correctness (must be bitwise identical)
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            for fused_result, baseline_result in zip(run_fused(), run_baseline()):
                assert torch.equal(fused_result, baseline_result)
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    t_fused = bench_kineto(
        run_fused, 'mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights (FP4 packed = 0.5 bytes) + activations (FP8 = 1 byte) + output (BF16 = 2 bytes)
    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1 # NOTES minus 1 to exclude "-1"
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2 +   # L1 weights (FP4)
        num_touched_experts * hidden * intermediate_hidden // 2 +       # L2 weights (FP4)
        num_recv_tokens * hidden +                                      # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                         # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                         # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                    # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x legacy')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()



# Test chunked MegaMoE correctness
def test_chunked(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42)
    random.seed(42)

    # Settings — use same max_tokens as baseline for fair comparison
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    chunk_size = args.chunk_size
    aligned_chunk_size = deep_gemm.utils.math.align(chunk_size, deep_gemm._C.get_token_alignment_for_mega_moe())
    total_tokens = args.num_tokens if args.num_tokens > 0 else num_max_tokens_per_rank
    assert total_tokens <= num_max_tokens_per_rank, \
        f'total_tokens ({total_tokens}) must <= num_max_tokens_per_rank ({num_max_tokens_per_rank})'

    dist_print('Chunked Test Config:', once_in_node=True)
    dist_print(f' > Total tokens: {total_tokens}', once_in_node=True)
    dist_print(f' > Max tokens per rank: {num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Chunk size: {chunk_size} (aligned: {aligned_chunk_size})', once_in_node=True)
    dist_print(f' > Num chunks: {math.ceil(total_tokens / aligned_chunk_size)}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(once_in_node=True)

    # Allocate two symmetric buffers:
    # 1) full-sized buffer for non-chunked reference (same capacity as baseline test)
    # 2) chunk-sized buffer for chunked mode
    buffer_full = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )
    buffer_chunked = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        chunk_tokens=chunk_size
    )

    # Log buffer sizes for verification
    dist_print('Buffer sizes:', once_in_node=True)
    dist_print(f' > Baseline buffer (full): {buffer_full.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(f' > Chunked buffer: {buffer_chunked.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(f' > Memory saving: {(buffer_full.buffer.nbytes - buffer_chunked.buffer.nbytes) / 2 ** 30:.3f} GiB ({1 - buffer_chunked.buffer.nbytes / buffer_full.buffer.nbytes:.1%})', once_in_node=True)
    dist_print(f' > Chunked buffer capacity: {buffer_chunked.num_max_tokens_per_rank} tokens', once_in_node=True)
    dist_print(once_in_node=True)

    # Verify chunked buffer capacity equals aligned chunk_size
    from deep_gemm.utils.math import align as math_align
    expected_capacity = math_align(chunk_size, deep_gemm._C.get_token_alignment_for_mega_moe())
    assert buffer_chunked.num_max_tokens_per_rank == expected_capacity,         f'Chunked buffer capacity {buffer_chunked.num_max_tokens_per_rank} != expected {expected_capacity}'
    dist_print(f' > Buffer capacity verification: PASSED ({buffer_chunked.num_max_tokens_per_rank} == {expected_capacity})', once_in_node=True)

    # Create inputs
    x_bf16 = torch.randn((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((total_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    # Check SF requirements
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0

    # Cast inputs to FP8
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    # Cast weights to FP4
    def cast_grouped_weights_to_fp4(bf16_weights):
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    l1_weights = cast_grouped_weights_to_fp4(l1_weights_bf16)
    l2_weights = cast_grouped_weights_to_fp4(l2_weights_bf16)
    transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Run non-chunked reference
    buffer_full.x[:total_tokens].copy_(x_fp8[0])
    buffer_full.x_sf[:total_tokens].copy_(x_fp8[1])
    buffer_full.topk_idx[:total_tokens].copy_(topk_idx)
    buffer_full.topk_weights[:total_tokens].copy_(topk_weights)

    y_ref = torch.empty((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_mega_moe(
        y_ref,
        transformed_l1_weights, transformed_l2_weights,
        buffer_full,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math)
    )
    dist_print('Non-chunked reference run: DONE', once_in_node=True)

    # Run chunked mode
    y_chunked = torch.empty((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_chunked_mega_moe(
        y_chunked,
        x_fp8,
        topk_idx,
        topk_weights,
        transformed_l1_weights, transformed_l2_weights,
        buffer_chunked,
        chunk_size=aligned_chunk_size,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math)
    )
    dist_print('Chunked mode run: DONE', once_in_node=True)

    # Compare outputs
    max_diff = (y_ref - y_chunked).abs().max().item()
    mean_diff = (y_ref - y_chunked).abs().mean().item()
    dist_print(f'Output comparison:', once_in_node=True)
    dist_print(f' > Max absolute diff: {max_diff:.6f}', once_in_node=True)
    dist_print(f' > Mean absolute diff: {mean_diff:.6f}', once_in_node=True)

    # Verification: error < 1e-3
    assert max_diff < 1e-3, f'Chunked vs non-chunked max diff {max_diff} >= 1e-3'
    dist_print(f' > Correctness test: PASSED (max_diff={max_diff:.6f} < 1e-3)', once_in_node=True)

    # Count local received tokens (same method as baseline test)
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark chunked mode
    # NOTES: chunked internally loops over N chunks, each calling fp8_fp4_mega_moe.
    # bench_kineto measures per-invocation average of 'mega_moe' kernel.
    # Total time for all tokens = t_per_chunk × num_chunks.
    num_chunks_actual = math.ceil(total_tokens / aligned_chunk_size)

    def run_chunked_bench():
        deep_gemm.fp8_fp4_chunked_mega_moe(
            y_chunked,
            x_fp8,
            topk_idx,
            topk_weights,
            transformed_l1_weights, transformed_l2_weights,
            buffer_chunked,
            chunk_size=aligned_chunk_size,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        return y_chunked

    # Warmup first
    for _ in range(5):
        run_chunked_bench()
    torch.cuda.synchronize()

    # Measure time with CUDA events — each iteration timed separately
    num_bench_iters = 30
    times_ms = []
    for _ in range(num_bench_iters):
        torch.cuda._sleep(int(2e7))  # ~10ms, eliminate CPU launch overhead imbalance
        dist.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run_chunked_bench()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    t_chunked_total = sum(times_ms) / len(times_ms) / 1e3  # average total time in seconds

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    # Use total time (all chunks) for fair comparison with baseline
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_chunked_total)

    # HBM bytes: weights (FP4 packed = 0.5 bytes) + activations (FP8 = 1 byte) + output (BF16 = 2 bytes)
    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2 +   # L1 weights (FP4)
        num_touched_experts * hidden * intermediate_hidden // 2 +       # L2 weights (FP4)
        num_recv_tokens * hidden +                                      # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                         # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                         # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                    # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_chunked_total)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_chunked_total)

    # Combine reduction (serial) time approximation
    t_reduction = total_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_chunked_total / (t_chunked_total - t_reduction)
    dist_print('Chunked Performance:')
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{num_chunks_actual} chunks | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_chunked_total * 1e6:4.0f} us total, '
               f'reduction: {t_reduction * 1e6:4.1f} us')

    # Clean up
    dist.barrier()
    buffer_full.destroy()
    buffer_chunked.destroy()
    dist.destroy_process_group()




# Test normal kernel chunk MegaMoE (Plan B: C++ host-driven loop)
def test_normal_kernel(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42)
    random.seed(42)

    # Settings — same as test_chunked
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    chunk_size = args.chunk_size
    aligned_chunk_size = deep_gemm.utils.math.align(chunk_size, deep_gemm._C.get_token_alignment_for_mega_moe())
    total_tokens = args.num_tokens if args.num_tokens > 0 else num_max_tokens_per_rank
    assert total_tokens <= num_max_tokens_per_rank, \
        f'total_tokens ({total_tokens}) must <= num_max_tokens_per_rank ({num_max_tokens_per_rank})'

    dist_print('Normal-kernel-chunk Test Config:', once_in_node=True)
    dist_print(f' > Total tokens: {total_tokens}', once_in_node=True)
    dist_print(f' > Max tokens per rank: {num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Chunk size: {chunk_size} (aligned: {aligned_chunk_size})', once_in_node=True)
    dist_print(f' > Num chunks: {math.ceil(total_tokens / aligned_chunk_size)}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(once_in_node=True)

    # Allocate buffers: a full-sized one for the baseline reference, and a
    # chunk-sized one for the normal-kernel-chunk path.
    buffer_full = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )
    buffer_normal = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        chunk_tokens=chunk_size
    )

    dist_print('Buffer sizes:', once_in_node=True)
    dist_print(f' > Baseline buffer (full): {buffer_full.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(f' > Normal-kernel-chunk buffer: {buffer_normal.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(f' > Memory saving: {(buffer_full.buffer.nbytes - buffer_normal.buffer.nbytes) / 2 ** 30:.3f} GiB ({1 - buffer_normal.buffer.nbytes / buffer_full.buffer.nbytes:.1%})', once_in_node=True)
    dist_print(f' > Normal-kernel-chunk buffer capacity: {buffer_normal.num_max_tokens_per_rank} tokens', once_in_node=True)
    dist_print(once_in_node=True)

    from deep_gemm.utils.math import align as math_align
    expected_capacity = math_align(chunk_size, deep_gemm._C.get_token_alignment_for_mega_moe())
    assert buffer_normal.num_max_tokens_per_rank == expected_capacity, \
        f'Normal-kernel-chunk buffer capacity {buffer_normal.num_max_tokens_per_rank} != expected {expected_capacity}'
    dist_print(f' > Buffer capacity verification: PASSED ({buffer_normal.num_max_tokens_per_rank} == {expected_capacity})', once_in_node=True)

    # Create inputs (identical to test_chunked)
    x_bf16 = torch.randn((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((total_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0

    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    def cast_grouped_weights_to_fp4(bf16_weights):
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    l1_weights = cast_grouped_weights_to_fp4(l1_weights_bf16)
    l2_weights = cast_grouped_weights_to_fp4(l2_weights_bf16)
    transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Reference: non-chunked baseline
    buffer_full.x[:total_tokens].copy_(x_fp8[0])
    buffer_full.x_sf[:total_tokens].copy_(x_fp8[1])
    buffer_full.topk_idx[:total_tokens].copy_(topk_idx)
    buffer_full.topk_weights[:total_tokens].copy_(topk_weights)

    y_ref = torch.empty((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_mega_moe(
        y_ref,
        transformed_l1_weights, transformed_l2_weights,
        buffer_full,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math)
    )
    dist_print('Non-chunked reference run: DONE', once_in_node=True)

    # Run normal-kernel-chunk mode
    y_normal = torch.empty((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_normal_kernel_chunk_mega_moe(
        y_normal,
        x_fp8,
        topk_idx,
        topk_weights,
        transformed_l1_weights, transformed_l2_weights,
        buffer_normal,
        chunk_size=aligned_chunk_size,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math)
    )
    dist_print('Normal-kernel-chunk mode run: DONE', once_in_node=True)

    max_diff = (y_ref - y_normal).abs().max().item()
    mean_diff = (y_ref - y_normal).abs().mean().item()
    dist_print(f'Output comparison:', once_in_node=True)
    dist_print(f' > Max absolute diff: {max_diff:.6f}', once_in_node=True)
    dist_print(f' > Mean absolute diff: {mean_diff:.6f}', once_in_node=True)

    assert max_diff < 1e-3, f'Normal-kernel-chunk vs baseline max diff {max_diff} >= 1e-3'
    dist_print(f' > Correctness test: PASSED (max_diff={max_diff:.6f} < 1e-3)', once_in_node=True)

    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    num_chunks_actual = math.ceil(total_tokens / aligned_chunk_size)

    def run_normal_bench():
        deep_gemm.fp8_fp4_normal_kernel_chunk_mega_moe(
            y_normal,
            x_fp8,
            topk_idx,
            topk_weights,
            transformed_l1_weights, transformed_l2_weights,
            buffer_normal,
            chunk_size=aligned_chunk_size,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        return y_normal

    for _ in range(5):
        run_normal_bench()
    torch.cuda.synchronize()

    num_bench_iters = 30
    times_ms = []
    for _ in range(num_bench_iters):
        torch.cuda._sleep(int(2e7))
        dist.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run_normal_bench()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    t_normal_total = sum(times_ms) / len(times_ms) / 1e3

    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_normal_total)

    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2 +
        num_touched_experts * hidden * intermediate_hidden // 2 +
        num_recv_tokens * hidden +
        num_recv_tokens * intermediate_hidden +
        num_recv_tokens * intermediate_hidden +
        num_recv_tokens * hidden * 2
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_normal_total)
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_normal_total)
    t_reduction = total_tokens * hidden * 2 * (1 + num_topk) / 6.5e12
    approx_factor = t_normal_total / (t_normal_total - t_reduction)

    dist_print('Normal-kernel-chunk Performance:')
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{num_chunks_actual} chunks | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_normal_total * 1e6:4.0f} us total, '
               f'reduction: {t_reduction * 1e6:4.1f} us')

    dist.barrier()
    buffer_full.destroy()
    buffer_normal.destroy()
    dist.destroy_process_group()


# Test streaming MegaMoE correctness and performance
def test_streaming(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42)
    random.seed(42)

    # Settings — use same max_tokens as baseline for fair comparison
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    chunk_size = args.chunk_size
    # Align chunk_size to block sizes for streaming
    aligned_chunk_size = deep_gemm.utils.math.align(chunk_size, deep_gemm._C.get_token_alignment_for_mega_moe())
    # Streaming kernel requires total_tokens == num_chunks * aligned_chunk_size exactly.
    # Compute from max_tokens_per_rank: num_chunks = ceil(max/aligned_cs), then total = num_chunks * aligned_cs
    num_chunks = math.ceil(num_max_tokens_per_rank / aligned_chunk_size)
    total_tokens = num_chunks * aligned_chunk_size

    dist_print('Streaming Test Config:', once_in_node=True)
    dist_print(f' > Total tokens: {total_tokens}', once_in_node=True)
    dist_print(f' > Max tokens per rank: {num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Chunk size: {chunk_size} (aligned: {aligned_chunk_size})', once_in_node=True)
    dist_print(f' > Num chunks: {num_chunks}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(once_in_node=True)

    # Allocate symmetric buffers:
    # 1) full-sized buffer for non-streaming reference (use total_tokens since chunk
    #    alignment may cause total_tokens > num_max_tokens_per_rank)
    # 2) streaming buffer for streaming mode
    buffer_full = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        total_tokens, num_topk,
        hidden, intermediate_hidden
    )
    buffer_streaming = deep_gemm.get_streaming_symm_buffer_for_mega_moe(
        group, num_experts,
        num_chunks, aligned_chunk_size, num_topk,
        hidden, intermediate_hidden
    )

    # Log buffer sizes
    dist_print('Buffer sizes:', once_in_node=True)
    dist_print(f' > Baseline buffer (full): {buffer_full.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(f' > Streaming buffer: {buffer_streaming.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(f' > Memory saving: {(buffer_full.buffer.nbytes - buffer_streaming.buffer.nbytes) / 2 ** 30:.3f} GiB ({1 - buffer_streaming.buffer.nbytes / buffer_full.buffer.nbytes:.1%})', once_in_node=True)
    dist_print(once_in_node=True)

    # Create inputs
    x_bf16 = torch.randn((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((total_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    # Check SF requirements
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0

    # Cast inputs to FP8
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    # Cast weights to FP4
    def cast_grouped_weights_to_fp4(bf16_weights):
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    l1_weights = cast_grouped_weights_to_fp4(l1_weights_bf16)
    l2_weights = cast_grouped_weights_to_fp4(l2_weights_bf16)
    transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Run non-streaming reference
    buffer_full.x[:total_tokens].copy_(x_fp8[0])
    buffer_full.x_sf[:total_tokens].copy_(x_fp8[1])
    buffer_full.topk_idx[:total_tokens].copy_(topk_idx)
    buffer_full.topk_weights[:total_tokens].copy_(topk_weights)

    y_ref = torch.empty((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_mega_moe(
        y_ref,
        transformed_l1_weights, transformed_l2_weights,
        buffer_full,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math)
    )
    dist_print('Non-streaming reference run: DONE', once_in_node=True)

    # Run streaming mode
    y_streaming = torch.empty((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_streaming_mega_moe(
        y_streaming,
        x_fp8,
        topk_idx,
        topk_weights,
        transformed_l1_weights, transformed_l2_weights,
        buffer_streaming,
        chunk_size=aligned_chunk_size,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math)
    )
    dist_print('Streaming mode run: DONE', once_in_node=True)

    # Compare outputs
    max_diff = (y_ref - y_streaming).abs().max().item()
    mean_diff = (y_ref - y_streaming).abs().mean().item()
    dist_print(f'Output comparison:', once_in_node=True)
    dist_print(f' > Max absolute diff: {max_diff:.6f}', once_in_node=True)
    dist_print(f' > Mean absolute diff: {mean_diff:.6f}', once_in_node=True)

    # Verification: error < 1e-2 for streaming (slightly relaxed due to potential
    # different accumulation order across chunks)
    assert max_diff < 1e-2, f'Streaming vs non-streaming max diff {max_diff} >= 1e-2'
    dist_print(f' > Correctness test: PASSED (max_diff={max_diff:.6f} < 1e-2)', once_in_node=True)

    # Count local received tokens (same method as baseline test)
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark streaming mode
    def run_streaming_bench():
        deep_gemm.fp8_fp4_streaming_mega_moe(
            y_streaming,
            x_fp8,
            topk_idx,
            topk_weights,
            transformed_l1_weights, transformed_l2_weights,
            buffer_streaming,
            chunk_size=aligned_chunk_size,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        return y_streaming

    # Warmup
    for _ in range(5):
        run_streaming_bench()
    torch.cuda.synchronize()

    # Measure time with CUDA events — each iteration timed separately
    num_bench_iters = 30
    times_ms = []
    for _ in range(num_bench_iters):
        torch.cuda._sleep(int(2e7))  # ~10ms, eliminate CPU launch overhead imbalance
        dist.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run_streaming_bench()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    t_streaming = sum(times_ms) / len(times_ms) / 1e3  # average total time in seconds

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_streaming)

    # HBM bytes: weights (FP4 packed = 0.5 bytes) + activations (FP8 = 1 byte) + output (BF16 = 2 bytes)
    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2 +   # L1 weights (FP4)
        num_touched_experts * hidden * intermediate_hidden // 2 +       # L2 weights (FP4)
        num_recv_tokens * hidden +                                      # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                         # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                         # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                    # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_streaming)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_streaming)

    # Combine reduction (serial) time approximation
    t_reduction = total_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_streaming / (t_streaming - t_reduction)
    dist_print('Streaming Performance:')
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{num_chunks} chunks | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_streaming * 1e6:4.0f} us total, '
               f'reduction: {t_reduction * 1e6:4.1f} us')

    # Clean up
    dist.barrier()
    buffer_full.destroy()
    buffer_streaming.destroy()
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')

    # Chunked/streaming test settings
    parser.add_argument('--chunk-size', type=int, default=4096, help='Chunk size for chunked/streaming MoE test')
    parser.add_argument('--test-chunked', action='store_true', help='Run chunked MoE correctness test')
    parser.add_argument('--test-streaming', action='store_true', help='Run streaming MoE correctness test')
    parser.add_argument('--test-normal-kernel', action='store_true', help='Run normal-kernel-chunk MoE correctness test (Plan B: C++ host-driven loop)')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        if args.test_chunked:
            test_chunked(args.local_rank_idx, args.num_processes, args)
        elif args.test_streaming:
            test_streaming(args.local_rank_idx, args.num_processes, args)
        elif args.test_normal_kernel:
            test_normal_kernel(args.local_rank_idx, args.num_processes, args)
        else:
            test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        if args.test_chunked:
            torch.multiprocessing.spawn(test_chunked, args=(num_processes, args), nprocs=num_processes)
        elif args.test_streaming:
            torch.multiprocessing.spawn(test_streaming, args=(num_processes, args), nprocs=num_processes)
        elif args.test_normal_kernel:
            torch.multiprocessing.spawn(test_normal_kernel, args=(num_processes, args), nprocs=num_processes)
        else:
            torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
