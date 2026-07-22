"""Specialized vector-quantization search operators."""

from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is not shipped with every PyTorch backend.
    triton = None
    tl = None


class _SearchConfig(NamedTuple):
    block_m: int
    block_n: int
    exact_block: int
    coarse_warps: int = 8
    exact_warps: int = 8


_ACCELERATED_L2_CONFIGS = {
    # Product VQ commonly splits the 64d Mapping feature into two 32d groups.
    # Retain four coarse candidates per tile and replay the final near-tie in
    # ordered FP32 arithmetic for this newly covered shape.
    32: _SearchConfig(32, 256, 64, 4, 4),
    # The 64d path refines each tile's top two candidates in the coarse
    # program. A smaller query tile retains occupancy after the extra FP32
    # loads and reductions; four warps are sufficient for the scalar reduction.
    64: _SearchConfig(32, 256, 128, 4, 4),
    96: _SearchConfig(32, 128, 32),
    128: _SearchConfig(16, 256, 64),
}
_ACCELERATED_COSINE_CONFIGS = {
    32: _SearchConfig(32, 256, 64, 4, 4),
    64: _SearchConfig(32, 256, 64, 4, 4),
}


if triton is not None:

    @triton.jit
    def _ema_codebook_kernel(
        embed_ptr,
        embed_avg_ptr,
        cluster_size_ptr,
        cluster_total_ptr,
        codebook_size: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        norm_eps: tl.constexpr,
        use_cosine_sim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        code = tl.program_id(0)
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < dim
        cluster_total = tl.load(cluster_total_ptr)
        cluster_size = tl.load(cluster_size_ptr + code)
        smoothed_size = cluster_total * (cluster_size + eps) / (
            cluster_total + codebook_size * eps
        )
        embed_avg = tl.load(
            embed_avg_ptr + code * dim + dims,
            mask=dim_mask,
            other=0.0,
        )
        normalized = embed_avg / smoothed_size
        if use_cosine_sim:
            norm = tl.sqrt(tl.sum(normalized * normalized, axis=0))
            # Preserve PyTorch clamp_min's NaN propagation semantics. Triton's
            # maximum ignores NaN by default on some supported versions.
            safe_norm = tl.where(norm < norm_eps, norm_eps, norm)
            normalized = normalized / safe_norm
        tl.store(embed_ptr + code * dim + dims, normalized, mask=dim_mask)

    @triton.jit
    def _perplexity_partial_kernel(
        cluster_size_ptr,
        partial_ptr,
        num_assignments,
        NUM_CODES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUM_CODES
        counts = tl.load(cluster_size_ptr + offsets, mask=mask, other=0).to(tl.float32)
        probabilities = counts / num_assignments
        entropy_terms = probabilities * tl.log(probabilities + 1e-10)
        tl.store(partial_ptr + block, tl.sum(entropy_terms, axis=0))

    @triton.jit
    def _perplexity_finalize_kernel(
        partial_ptr,
        output_ptr,
        NUM_PARTIALS: tl.constexpr,
        NUM_CODES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        partials = tl.load(
            partial_ptr + offsets,
            mask=offsets < NUM_PARTIALS,
            other=0.0,
        )
        perplexity = tl.exp(-tl.sum(partials, axis=0))
        tl.store(output_ptr, perplexity)
        tl.store(output_ptr + 1, perplexity / NUM_CODES)

    @triton.jit
    def _coarse_l2_topk_kernel(
        x_ptr,
        codebook_ptr,
        x_norm_ptr,
        codebook_norm_ptr,
        candidates_ptr,
        num_queries,
        NUM_TILES: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        TOP_K: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        code_block = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        code_offsets = code_block * BLOCK_N + tl.arange(0, BLOCK_N)
        query_mask = query_offsets < num_queries
        if DIM == 96:
            # Keep both Tensor-Core products dense instead of padding K to 128.
            dim0 = tl.arange(0, 64)
            dim1 = 64 + tl.arange(0, 32)
            x0 = tl.load(
                x_ptr + query_offsets[:, None] * DIM + dim0[None, :],
                mask=query_mask[:, None],
                other=0.0,
            ).to(tl.float16)
            x1 = tl.load(
                x_ptr + query_offsets[:, None] * DIM + dim1[None, :],
                mask=query_mask[:, None],
                other=0.0,
            ).to(tl.float16)
            codebook0 = tl.load(
                codebook_ptr + code_offsets[:, None] * DIM + dim0[None, :]
            ).to(tl.float16)
            codebook1 = tl.load(
                codebook_ptr + code_offsets[:, None] * DIM + dim1[None, :]
            ).to(tl.float16)
            dots = tl.dot(x0, tl.trans(codebook0), out_dtype=tl.float32)
            dots += tl.dot(x1, tl.trans(codebook1), out_dtype=tl.float32)
        else:
            dim_offsets = tl.arange(0, BLOCK_D)
            dim_mask = dim_offsets < DIM
            x = tl.load(
                x_ptr + query_offsets[:, None] * DIM + dim_offsets[None, :],
                mask=query_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float16)
            codebook = tl.load(
                codebook_ptr + code_offsets[:, None] * DIM + dim_offsets[None, :],
                mask=dim_mask[None, :],
                other=0.0,
            ).to(tl.float16)
            dots = tl.dot(x, tl.trans(codebook), out_dtype=tl.float32)
        x_norm = tl.load(x_norm_ptr + query_offsets, mask=query_mask, other=0.0)
        codebook_norm = tl.load(codebook_norm_ptr + code_offsets)
        dists = x_norm[:, None] + codebook_norm[None, :] - 2.0 * dots
        dists = tl.where(query_mask[:, None], dists, float("inf"))

        columns = tl.arange(0, BLOCK_N)[None, :]
        first = tl.argmin(dists, axis=1, tie_break_left=True)
        dists = tl.where(columns == first[:, None], float("inf"), dists)
        second = tl.argmin(dists, axis=1, tie_break_left=True)
        if TOP_K >= 3:
            dists = tl.where(columns == second[:, None], float("inf"), dists)
            third = tl.argmin(dists, axis=1, tie_break_left=True)
        if TOP_K == 4:
            dists = tl.where(columns == third[:, None], float("inf"), dists)
            fourth = tl.argmin(dists, axis=1, tie_break_left=True)

        output_base = (query_offsets * NUM_TILES + code_block) * TOP_K
        tl.store(
            candidates_ptr + output_base,
            code_block * BLOCK_N + first,
            mask=query_mask,
        )
        tl.store(
            candidates_ptr + output_base + 1,
            code_block * BLOCK_N + second,
            mask=query_mask,
        )
        if TOP_K >= 3:
            tl.store(
                candidates_ptr + output_base + 2,
                code_block * BLOCK_N + third,
                mask=query_mask,
            )
        if TOP_K == 4:
            tl.store(
                candidates_ptr + output_base + 3,
                code_block * BLOCK_N + fourth,
                mask=query_mask,
            )


    @triton.jit
    def _coarse_l2_top2_refined_kernel(
        x_ptr,
        codebook_ptr,
        coarse_x_ptr,
        coarse_codebook_ptr,
        x_norm_ptr,
        codebook_norm_ptr,
        tile_code_ptr,
        tile_dist_ptr,
        num_queries,
        NUM_TILES: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        code_block = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        code_offsets = code_block * BLOCK_N + tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, DIM)
        query_mask = query_offsets < num_queries

        x_coarse = tl.load(
            coarse_x_ptr + query_offsets[:, None] * DIM + dim_offsets[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        codebook_coarse = tl.load(
            coarse_codebook_ptr + code_offsets[:, None] * DIM + dim_offsets[None, :]
        )
        dots = tl.dot(x_coarse, tl.trans(codebook_coarse), out_dtype=tl.float32)
        x_norm = tl.load(x_norm_ptr + query_offsets, mask=query_mask, other=0.0)
        codebook_norm = tl.load(codebook_norm_ptr + code_offsets)
        dists = x_norm[:, None] + codebook_norm[None, :] - 2.0 * dots
        dists = tl.where(query_mask[:, None], dists, float("inf"))

        columns = tl.arange(0, BLOCK_N)[None, :]
        first = tl.argmin(dists, axis=1, tie_break_left=True)
        dists = tl.where(columns == first[:, None], float("inf"), dists)
        second = tl.argmin(dists, axis=1, tie_break_left=True)
        first_code = code_block * BLOCK_N + first
        second_code = code_block * BLOCK_N + second

        # Refine the two coarse winners while the query/codebook tile is still
        # parallel. The following kernel then reduces one exact scalar winner
        # per tile instead of reloading 512 full code vectors per query.
        x = tl.load(
            x_ptr + query_offsets[:, None] * DIM + dim_offsets[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        first_value = tl.load(
            codebook_ptr + first_code[:, None] * DIM + dim_offsets[None, :]
        )
        second_value = tl.load(
            codebook_ptr + second_code[:, None] * DIM + dim_offsets[None, :]
        )
        first_delta = first_value - x
        second_delta = second_value - x
        first_exact = tl.sum(first_delta * first_delta, axis=1)
        second_exact = tl.sum(second_delta * second_delta, axis=1)
        take_second = (second_exact < first_exact) | (
            (second_exact == first_exact) & (second_code < first_code)
        )
        best_code = tl.where(take_second, second_code, first_code)
        best_dist = tl.where(take_second, second_exact, first_exact)

        output_offsets = query_offsets * NUM_TILES + code_block
        tl.store(tile_code_ptr + output_offsets, best_code, mask=query_mask)
        tl.store(tile_dist_ptr + output_offsets, best_dist, mask=query_mask)


    @triton.jit
    def _reduce_l2_tile_winners_kernel(
        tile_code_ptr,
        tile_dist_ptr,
        output_ptr,
        NUM_TILES: tl.constexpr,
    ):
        query = tl.program_id(0)
        tile_offsets = tl.arange(0, NUM_TILES)
        offsets = query * NUM_TILES + tile_offsets
        codes = tl.load(tile_code_ptr + offsets)
        dists = tl.load(tile_dist_ptr + offsets)
        best_dist = tl.min(dists, axis=0)
        best_code = tl.min(
            tl.where(dists == best_dist, codes, 2147483647),
            axis=0,
        )
        tl.store(output_ptr + query, best_code)


    @triton.jit
    def _coarse_cosine_top3_refined_kernel(
        x_ptr,
        codebook_ptr,
        coarse_x_ptr,
        coarse_codebook_ptr,
        tile_code_ptr,
        tile_dist_ptr,
        num_queries,
        NUM_TILES: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        code_block = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        code_offsets = code_block * BLOCK_N + tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, DIM)
        query_mask = query_offsets < num_queries

        coarse_x = tl.load(
            coarse_x_ptr + query_offsets[:, None] * DIM + dim_offsets[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        coarse_codebook = tl.load(
            coarse_codebook_ptr + code_offsets[:, None] * DIM + dim_offsets[None, :]
        )
        coarse_dists = -tl.dot(coarse_x, tl.trans(coarse_codebook), out_dtype=tl.float32)
        coarse_dists = tl.where(query_mask[:, None], coarse_dists, float("inf"))

        columns = tl.arange(0, BLOCK_N)[None, :]
        first = tl.argmin(coarse_dists, axis=1, tie_break_left=True)
        coarse_dists = tl.where(columns == first[:, None], float("inf"), coarse_dists)
        second = tl.argmin(coarse_dists, axis=1, tie_break_left=True)
        coarse_dists = tl.where(columns == second[:, None], float("inf"), coarse_dists)
        third = tl.argmin(coarse_dists, axis=1, tie_break_left=True)
        first_code = code_block * BLOCK_N + first
        second_code = code_block * BLOCK_N + second
        third_code = code_block * BLOCK_N + third

        # Refine while codebook tiles are distributed across the grid.  The
        # following kernel only reduces three FP32 scalars per tile, avoiding
        # a serial reload of all 768 candidate vectors for every query.
        x = tl.load(
            x_ptr + query_offsets[:, None] * DIM + dim_offsets[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        first_value = tl.load(
            codebook_ptr + first_code[:, None] * DIM + dim_offsets[None, :]
        )
        second_value = tl.load(
            codebook_ptr + second_code[:, None] * DIM + dim_offsets[None, :]
        )
        third_value = tl.load(
            codebook_ptr + third_code[:, None] * DIM + dim_offsets[None, :]
        )
        first_exact = -tl.sum(first_value * x, axis=1)
        second_exact = -tl.sum(second_value * x, axis=1)
        third_exact = -tl.sum(third_value * x, axis=1)

        output_base = (query_offsets * NUM_TILES + code_block) * 3
        tl.store(tile_code_ptr + output_base, first_code, mask=query_mask)
        tl.store(tile_code_ptr + output_base + 1, second_code, mask=query_mask)
        tl.store(tile_code_ptr + output_base + 2, third_code, mask=query_mask)
        tl.store(tile_dist_ptr + output_base, first_exact, mask=query_mask)
        tl.store(tile_dist_ptr + output_base + 1, second_exact, mask=query_mask)
        tl.store(tile_dist_ptr + output_base + 2, third_exact, mask=query_mask)


    @triton.jit
    def _reduce_cosine_tile_candidates_kernel(
        x_ptr,
        codebook_ptr,
        tile_code_ptr,
        tile_dist_ptr,
        output_ptr,
        NUM_CANDIDATES: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        query = tl.program_id(0)
        candidate_offsets = tl.arange(0, BLOCK_C)
        candidate_mask = candidate_offsets < NUM_CANDIDATES
        codes = tl.load(
            tile_code_ptr + query * NUM_CANDIDATES + candidate_offsets,
            mask=candidate_mask,
            other=2147483647,
        )
        dists = tl.load(
            tile_dist_ptr + query * NUM_CANDIDATES + candidate_offsets,
            mask=candidate_mask,
            other=float("inf"),
        )
        # Keep the final codebook reads in bounds even if an upstream numerical
        # failure produces only NaN scores. Masked lanes and NaNs rank last, so
        # the deterministic minimum valid code remains a safe diagnostic result.
        dists = tl.where(dists == dists, dists, float("inf"))

        best0_dist = tl.min(dists, axis=0)
        best0_code = tl.min(
            tl.where(dists == best0_dist, codes, 2147483647),
            axis=0,
        )
        dists = tl.where(codes == best0_code, float("inf"), dists)
        best1_dist = tl.min(dists, axis=0)
        best1_code = tl.min(
            tl.where(dists == best1_dist, codes, 2147483647),
            axis=0,
        )
        dists = tl.where(codes == best1_code, float("inf"), dists)
        best2_dist = tl.min(dists, axis=0)
        best2_code = tl.min(
            tl.where(dists == best2_dist, codes, 2147483647),
            axis=0,
        )

        # Match KeOps' left-to-right FP32 FMA reduction for the only candidates
        # close enough for vector-reduction order to alter their final ranking.
        score0 = 0.0
        score1 = 0.0
        score2 = 0.0
        for dim in tl.static_range(0, DIM):
            x_value = tl.load(x_ptr + query * DIM + dim)
            value0 = tl.load(codebook_ptr + best0_code * DIM + dim)
            value1 = tl.load(codebook_ptr + best1_code * DIM + dim)
            value2 = tl.load(codebook_ptr + best2_code * DIM + dim)
            score0 = tl.fma(value0, x_value, score0)
            score1 = tl.fma(value1, x_value, score1)
            score2 = tl.fma(value2, x_value, score2)

        best_score = score0
        best_code = best0_code
        update = (score1 > best_score) | ((score1 == best_score) & (best1_code < best_code))
        best_score = tl.where(update, score1, best_score)
        best_code = tl.where(update, best1_code, best_code)
        update = (score2 > best_score) | ((score2 == best_score) & (best2_code < best_code))
        best_code = tl.where(update, best2_code, best_code)
        tl.store(output_ptr + query, best_code)


    @triton.jit
    def _exact_l2_candidates_kernel(
        x_ptr,
        codebook_ptr,
        candidates_ptr,
        output_ptr,
        NUM_CANDIDATES: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ORDERED_FINAL: tl.constexpr,
    ):
        query = tl.program_id(0)
        if DIM == 96:
            dim0 = tl.arange(0, 64)
            dim1 = 64 + tl.arange(0, 32)
            x0 = tl.load(x_ptr + query * DIM + dim0)
            x1 = tl.load(x_ptr + query * DIM + dim1)
        else:
            dim_offsets = tl.arange(0, BLOCK_D)
            dim_mask = dim_offsets < DIM
            x = tl.load(
                x_ptr + query * DIM + dim_offsets,
                mask=dim_mask,
                other=0.0,
            )
        best_dist = float("inf")
        best_idx = 0
        if ORDERED_FINAL:
            # Reduction-tree ordering can reverse sub-ULP L2 near-ties. Keep
            # the best two tree-reduced candidates so the final scalar replay
            # can reproduce the observed KeOps ordered-FP32 near-tie result.
            second_dist = float("inf")
            second_idx = 0
        for start in tl.static_range(0, NUM_CANDIDATES, BLOCK_C):
            candidate_offsets = start + tl.arange(0, BLOCK_C)
            candidate_idx = tl.load(
                candidates_ptr + query * NUM_CANDIDATES + candidate_offsets
            )
            if DIM == 96:
                candidate0 = tl.load(
                    codebook_ptr + candidate_idx[:, None] * DIM + dim0[None, :]
                )
                candidate1 = tl.load(
                    codebook_ptr + candidate_idx[:, None] * DIM + dim1[None, :]
                )
                difference0 = candidate0 - x0[None, :]
                difference1 = candidate1 - x1[None, :]
                dists = tl.sum(difference0 * difference0, axis=1)
                dists += tl.sum(difference1 * difference1, axis=1)
            else:
                candidate = tl.load(
                    codebook_ptr + candidate_idx[:, None] * DIM + dim_offsets[None, :],
                    mask=dim_mask[None, :],
                    other=0.0,
                )
                difference = candidate - x[None, :]
                dists = tl.sum(difference * difference, axis=1)
            local_dist = tl.min(dists, axis=0)
            local_code = tl.min(
                tl.where(dists == local_dist, candidate_idx, 2147483647),
                axis=0,
            )
            if ORDERED_FINAL:
                remaining = tl.where(candidate_idx == local_code, float("inf"), dists)
                local_second_dist = tl.min(remaining, axis=0)
                local_second_code = tl.min(
                    tl.where(
                        remaining == local_second_dist,
                        candidate_idx,
                        2147483647,
                    ),
                    axis=0,
                )

                update = (local_dist < best_dist) | (
                    (local_dist == best_dist) & (local_code < best_idx)
                )
                update_second = (~update) & (
                    (local_dist < second_dist)
                    | ((local_dist == second_dist) & (local_code < second_idx))
                )
                second_dist = tl.where(
                    update,
                    best_dist,
                    tl.where(update_second, local_dist, second_dist),
                )
                second_idx = tl.where(
                    update,
                    best_idx,
                    tl.where(update_second, local_code, second_idx),
                )
                best_dist = tl.where(update, local_dist, best_dist)
                best_idx = tl.where(update, local_code, best_idx)

                update = (local_second_dist < best_dist) | (
                    (local_second_dist == best_dist) & (local_second_code < best_idx)
                )
                update_second = (~update) & (
                    (local_second_dist < second_dist)
                    | (
                        (local_second_dist == second_dist)
                        & (local_second_code < second_idx)
                    )
                )
                second_dist = tl.where(
                    update,
                    best_dist,
                    tl.where(update_second, local_second_dist, second_dist),
                )
                second_idx = tl.where(
                    update,
                    best_idx,
                    tl.where(update_second, local_second_code, second_idx),
                )
                best_dist = tl.where(update, local_second_dist, best_dist)
                best_idx = tl.where(update, local_second_code, best_idx)
            else:
                update = (local_dist < best_dist) | (
                    (local_dist == best_dist) & (local_code < best_idx)
                )
                best_dist = tl.where(update, local_dist, best_dist)
                best_idx = tl.where(update, local_code, best_idx)
        if ORDERED_FINAL:
            best_score = 0.0
            second_score = 0.0
            for dim in tl.static_range(0, DIM):
                x_value = tl.load(x_ptr + query * DIM + dim)
                best_value = tl.load(codebook_ptr + best_idx * DIM + dim)
                second_value = tl.load(codebook_ptr + second_idx * DIM + dim)
                best_delta = best_value - x_value
                second_delta = second_value - x_value
                best_score = tl.fma(best_delta, best_delta, best_score)
                second_score = tl.fma(second_delta, second_delta, second_score)
            take_second = (second_score < best_score) | (
                (second_score == best_score) & (second_idx < best_idx)
            )
            best_idx = tl.where(take_second, second_idx, best_idx)
        tl.store(output_ptr + query, best_idx)


def supports_accelerated_l2_argmin(x: torch.Tensor, codebook: torch.Tensor) -> bool:
    """Return whether the validated Tensor-Core search specialization applies."""
    if triton is None or not x.is_cuda or not codebook.is_cuda:
        return False
    if x.device != codebook.device:
        return False
    if x.dtype != torch.float32 or codebook.dtype != torch.float32:
        return False
    if not x.is_contiguous() or not codebook.is_contiguous():
        return False
    if x.ndim != 2 or codebook.ndim != 2 or x.shape[1] != codebook.shape[1]:
        return False
    # The default 16k Mix9sVQ table and 32d VQ search cross over well
    # below their smallest measured training shape. Keep the established
    # threshold for wider embeddings with the larger table, whose extra code
    # tiles change the launch/occupancy balance.
    min_queries = (
        4096 if codebook.shape[0] == 16384 or x.shape[1] == 32 else 16384
    )
    if not min_queries <= x.shape[0] <= 65536:
        return False
    if x.shape[1] not in _ACCELERATED_L2_CONFIGS or codebook.shape[0] not in (16384, 65536):
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 8


@torch.no_grad()
def accelerated_perplexity_stats(
    cluster_size: torch.Tensor,
    num_assignments: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return fused CUDA VQ statistics for a caller-supplied exact count.

    These monitoring-only values use a split FP32 reduction, so their low bits
    can differ from the portable eager reduction. Unsupported layouts retain
    that reference path through a ``None`` result.
    """
    supported = (
        triton is not None
        and cluster_size.is_cuda
        and cluster_size.dtype == torch.int64
        and cluster_size.is_contiguous()
        and cluster_size.ndim == 1
        and cluster_size.shape[0] in (16384, 65536)
        and num_assignments > 0
    )
    if not supported:
        return None

    block, num_warps = (256, 2) if cluster_size.shape[0] == 16384 else (512, 2)
    num_partials = triton.cdiv(cluster_size.shape[0], block)
    partials = torch.empty((num_partials,), dtype=torch.float32, device=cluster_size.device)
    output = torch.empty((2,), dtype=torch.float32, device=cluster_size.device)
    _perplexity_partial_kernel[(num_partials,)](
        cluster_size,
        partials,
        num_assignments,
        NUM_CODES=cluster_size.shape[0],
        BLOCK=block,
        num_warps=num_warps,
    )
    _perplexity_finalize_kernel[(1,)](
        partials,
        output,
        NUM_PARTIALS=num_partials,
        NUM_CODES=cluster_size.shape[0],
        BLOCK=triton.next_power_of_2(num_partials),
        num_warps=1,
    )
    return output[0], output[1]


@torch.no_grad()
def _update_ema_codebook_(
    embed: torch.Tensor,
    embed_avg: torch.Tensor,
    cluster_size: torch.Tensor,
    use_cosine_sim: bool,
    eps: float = 1e-5,
) -> bool:
    """Rebuild an internal no-grad EMA codebook without large temporaries."""
    supported = (
        triton is not None
        and embed.is_cuda
        and embed_avg.device == embed.device
        and cluster_size.device == embed.device
        and embed.dtype == torch.float32
        and embed_avg.dtype == torch.float32
        and cluster_size.dtype == torch.float32
        and embed.is_contiguous()
        and embed_avg.is_contiguous()
        and cluster_size.is_contiguous()
        and embed.ndim == 2
        and embed.shape[0] > 0
        and embed_avg.shape == embed.shape
        and cluster_size.shape == (embed.shape[0],)
        and 1 <= embed.shape[1] <= 256
    )
    if not supported:
        return False

    cluster_total = cluster_size.sum()
    block_d = triton.next_power_of_2(embed.shape[1])
    _ema_codebook_kernel[(embed.shape[0],)](
        embed,
        embed_avg,
        cluster_size,
        cluster_total,
        embed.shape[0],
        embed.shape[1],
        eps,
        1e-7,
        use_cosine_sim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return True


def supports_accelerated_cosine_argmin(x: torch.Tensor, codebook: torch.Tensor) -> bool:
    """Return whether the validated Tensor-Core cosine specialization applies."""
    if triton is None or not x.is_cuda or not codebook.is_cuda:
        return False
    if x.device != codebook.device:
        return False
    if x.dtype != torch.float32 or codebook.dtype != torch.float32:
        return False
    if not x.is_contiguous() or not codebook.is_contiguous():
        return False
    if x.ndim != 2 or codebook.ndim != 2 or x.shape[1] != codebook.shape[1]:
        return False
    # Both validated table sizes cross over by the smallest training shape.
    if not 4096 <= x.shape[0] <= 65536:
        return False
    if x.shape[1] not in _ACCELERATED_COSINE_CONFIGS or codebook.shape[0] not in (
        16384,
        65536,
    ):
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 8


@torch.no_grad()
def accelerated_l2_argmin(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor | None:
    """Find L2-nearest codes with an FP16 coarse shortlist and exact FP32 refinement.

    Returns ``None`` outside the measured CUDA specialization so callers can
    retain an exact portable fallback.
    """
    if not supports_accelerated_l2_argmin(x, codebook):
        return None

    block_d = triton.next_power_of_2(x.shape[1])
    config = _ACCELERATED_L2_CONFIGS[x.shape[1]]
    block_m, block_n, exact_block, coarse_warps, exact_warps = config
    num_tiles = triton.cdiv(codebook.shape[0], block_n)
    # The 64d specialization retains the exact winner with two candidates per
    # tile on the production codebooks and the randomized validation corpus.
    # The 32d Product-VQ path keeps four candidates because its independent
    # sub-codebooks create more global near-ties. Wider embeddings keep the
    # third candidate: 128d near-code stress can need it after FP16 projection.
    if x.shape[1] == 32:
        top_k = 4
    elif x.shape[1] == 64:
        top_k = 2
    else:
        top_k = 3
    num_candidates = num_tiles * top_k
    x_norm = x.square().sum(dim=1)
    codebook_norm = codebook.square().sum(dim=1)
    # Coarse Tensor-Core products already round both operands to FP16 inside
    # the kernel. Materialize that representation once instead of reloading
    # and converting FP32 values for every query/codebook tile. Exact
    # refinement below deliberately retains the original FP32 tensors.
    coarse_x = x.to(torch.float16)
    coarse_codebook = codebook.to(torch.float16)
    if x.shape[1] == 64:
        tile_codes = torch.empty(
            (x.shape[0], num_tiles),
            dtype=torch.int32,
            device=x.device,
        )
        tile_distances = torch.empty_like(tile_codes, dtype=torch.float32)
        _coarse_l2_top2_refined_kernel[(triton.cdiv(x.shape[0], block_m), num_tiles)](
            x,
            codebook,
            coarse_x,
            coarse_codebook,
            x_norm,
            codebook_norm,
            tile_codes,
            tile_distances,
            x.shape[0],
            NUM_TILES=num_tiles,
            DIM=x.shape[1],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=coarse_warps,
        )
        output = torch.empty((x.shape[0],), dtype=torch.int64, device=x.device)
        _reduce_l2_tile_winners_kernel[(x.shape[0],)](
            tile_codes,
            tile_distances,
            output,
            NUM_TILES=num_tiles,
            num_warps=exact_warps,
        )
        return output

    candidates = torch.empty(
        (x.shape[0], num_candidates),
        dtype=torch.int32,
        device=x.device,
    )
    _coarse_l2_topk_kernel[(triton.cdiv(x.shape[0], block_m), num_tiles)](
        coarse_x,
        coarse_codebook,
        x_norm,
        codebook_norm,
        candidates,
        x.shape[0],
        NUM_TILES=num_tiles,
        DIM=x.shape[1],
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        TOP_K=top_k,
        num_warps=coarse_warps,
    )
    output = torch.empty((x.shape[0],), dtype=torch.int64, device=x.device)
    _exact_l2_candidates_kernel[(x.shape[0],)](
        x,
        codebook,
        candidates,
        output,
        NUM_CANDIDATES=num_candidates,
        DIM=x.shape[1],
        BLOCK_D=block_d,
        BLOCK_C=exact_block,
        ORDERED_FINAL=x.shape[1] == 32,
        num_warps=exact_warps,
    )
    return output


@torch.no_grad()
def accelerated_cosine_argmin(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor | None:
    """Find cosine-nearest codes with FP16 coarse top-3 and exact FP32 refinement.

    Returns ``None`` outside the measured CUDA specialization so callers can
    retain an exact portable fallback.
    """
    if not supports_accelerated_cosine_argmin(x, codebook):
        return None

    config = _ACCELERATED_COSINE_CONFIGS[x.shape[1]]
    block_m, block_n, _, coarse_warps, exact_warps = config
    num_tiles = triton.cdiv(codebook.shape[0], block_n)
    num_candidates = num_tiles * 3
    tile_candidates = torch.empty(
        (x.shape[0], num_candidates),
        dtype=torch.int32,
        device=x.device,
    )
    tile_distances = torch.empty_like(tile_candidates, dtype=torch.float32)
    # Match the FP16 coarse arithmetic performed by the kernel while avoiding
    # a redundant FP32 load/conversion for every query/codebook tile.
    coarse_x = x.to(torch.float16)
    coarse_codebook = codebook.to(torch.float16)
    _coarse_cosine_top3_refined_kernel[(triton.cdiv(x.shape[0], block_m), num_tiles)](
        x,
        codebook,
        coarse_x,
        coarse_codebook,
        tile_candidates,
        tile_distances,
        x.shape[0],
        NUM_TILES=num_tiles,
        DIM=x.shape[1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=coarse_warps,
    )
    output = torch.empty((x.shape[0],), dtype=torch.int64, device=x.device)
    _reduce_cosine_tile_candidates_kernel[(x.shape[0],)](
        x,
        codebook,
        tile_candidates,
        tile_distances,
        output,
        NUM_CANDIDATES=num_candidates,
        DIM=x.shape[1],
        BLOCK_C=triton.next_power_of_2(num_candidates),
        num_warps=exact_warps,
    )
    return output
