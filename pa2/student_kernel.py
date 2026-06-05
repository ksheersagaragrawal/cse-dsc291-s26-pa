KERNEL_CONFIGS = [
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 4}
]


@triton.jit
def matmul_add_relu_kernel_fp16(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Step 1: Tile: Assignment
    #
    # Each kernel instance is mapped to a tile in the output matrix C.
    # Compute the starting indices (m_start, n_start) for this tile.
    # -------------------------------------------------------------------------
    # TODO: Compute the tile indices using program_id(0) for M and program_id(1) for N.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # -------------------------------------------------------------------------
    # Step 2: Register Tiling
    # -------------------------------------------------------------------------
    # TODO: Initialize the accumulator "acc" with zeros (dtype: float16 or float32).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Step 3: Shared Memory Tiling & Cooperative Fetching.
    # Compute pointers to the sub-tiles of A and B that are needed to compute
    # the current C tile. The offsets here serve to load BLOCK_M x BLOCK_K
    # and BLOCK_K x BLOCK_N blocks from A and B respectively.
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

        a_tile = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Step 4: Add C and Apply ReLU to the accumulator
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_tile = tl.load(c_ptrs, mask=mask, other=0.0)
    out = tl.maximum(acc + c_tile, 0.0)

    # -------------------------------------------------------------------------
    # Step 5: Write Cache / Epilogue Fusion: Write the computed tile to D.
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    d_ptrs = d_ptr + (offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn)
    tl.store(d_ptrs, out, mask=mask)
