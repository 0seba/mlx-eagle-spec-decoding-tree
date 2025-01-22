import mlx.core as mx


def get_block_dims(dim0, dim1, dim2, pow2=10):
    """
    Calculates block dimensions based on input dimensions and a power of 2 limit.

    Args:
    dim0: The first dimension.
    dim1: The second dimension.
    dim2: The third dimension.
    pow2: The maximum sum of the powers of 2 for the dimensions (default: 10).

    Returns:
    A tuple representing the block dimensions (size0, size1, size2).
    """
    pows = [0, 0, 0]
    sum_pows = 0
    while True:
        presum = sum_pows

        # Check and increment powers for each dimension
        if dim0 >= (1 << (pows[0] + 1)):
            pows[0] += 1
            sum_pows += 1
        if sum_pows == pow2:
            break

        if dim1 >= (1 << (pows[1] + 1)):
            pows[1] += 1
            sum_pows += 1
        if sum_pows == pow2:
            break

        if dim2 >= (1 << (pows[2] + 1)):
            pows[2] += 1
            sum_pows += 1

        # Break if no change or maximum sum reached
        if sum_pows == presum or sum_pows == pow2:
            break

    return (1 << pows[0], 1 << pows[1], 1 << pows[2])


@mx.custom_function
def rope_positions(
    inp: mx.array,
    offsets: mx.array,
    inv_freqs: mx.array,
    scaling_factor: float,
    verbose=False,
):
    traditional = False
    forward = True
    N = 4
    mat_size = inp.shape[-1] * inp.shape[-2]
    n_batch = inp.size // mat_size
    n_per_thread = 4
    source = """
        // size_t batch_stride = in_strides[0] * in_strides[1];
        size_t batch_stride = in_strides[1];
        size_t head_size = in_shape[1];

        // float inv_freq = 1.0 / (inv_freqs[inv_freqs_strides[0] * thread_position_in_grid.x]);
        float inv_freq = (inv_freqs[inv_freqs_strides[0] * thread_position_in_grid.x]);
        // int offset = offsets[thread_position_in_grid.y];
        float L = scaling_factor * inv_freq;

        // Compute the input and output indices
        size_t in_index_1, in_index_2;
        size_t out_index_1, out_index_2;
        if (traditional) {
            out_index_1 = 2 * thread_position_in_grid.x * in_strides[in_ndim-1] + thread_position_in_grid.y * in_strides[in_ndim-2] +
                N * thread_position_in_grid.z * batch_stride;
            out_index_2 = out_index_1 + 1;
            in_index_1 =
                2 * thread_position_in_grid.x * in_strides[in_ndim-1] + thread_position_in_grid.y * in_strides[in_ndim-2] + N * thread_position_in_grid.z * batch_stride;
            in_index_2 = in_index_1 + in_strides[in_ndim-1];
        } else {
            out_index_1 = thread_position_in_grid.x * in_strides[in_ndim-1] + thread_position_in_grid.y * in_strides[in_ndim-2] +
                N * thread_position_in_grid.z * batch_stride;
            out_index_2 = out_index_1 + threads_per_grid.x * in_strides[in_ndim-1];
            in_index_1 =
                thread_position_in_grid.x * in_strides[in_ndim-1] + thread_position_in_grid.y * in_strides[in_ndim-2] + N * thread_position_in_grid.z * batch_stride;
            in_index_2 = in_index_1 + threads_per_grid.x * in_strides[in_ndim-1];
        }

        int offset;
        int current_batch_index = thread_position_in_grid.z * N;
        for (int i = 0; i < N && thread_position_in_grid.z * N + i < n_batch; ++i) {

            // thread_position_in_grid.y is the sequence element, corresponding stride should be 1 I think
            // to get the batch index, we multiply the current thread_position_in_grid.z by N, this gives us the number of
            // processed heads, which we have to divide by the number of heads per batch element. and finally we multiply
            // by the offsets stride

            offset = offsets[thread_position_in_grid.y + (current_batch_index + i) / head_size * offsets_strides[0]];
            float theta = L * static_cast<float>(offset);

            // Read and write the output
            float costheta = metal::fast::cos(theta);
            float sintheta = metal::fast::sin(theta);

            float x1 = static_cast<float>(in[in_index_1]);
            float x2 = static_cast<float>(in[in_index_2]);
            float rx1;
            float rx2;
            if (forward) {
            rx1 = x1 * costheta - x2 * sintheta;
            rx2 = x1 * sintheta + x2 * costheta;
            } else {
            rx1 = x2 * sintheta + x1 * costheta;
            rx2 = x2 * costheta - x1 * sintheta;
            }
            out[out_index_1] = static_cast<T>(rx1);
            out[out_index_2] = static_cast<T>(rx2);
            in_index_1 += batch_stride;
            in_index_2 += batch_stride;
            out_index_1 += batch_stride;
            out_index_2 += batch_stride;
        } 
    """

    dim0 = inp.shape[-1] // 2
    dim1 = inp.shape[-2]
    dim2 = (n_batch + n_per_thread - 1) // n_per_thread

    group_dims = get_block_dims(dim0, dim1, dim2)
    grid_dims = (dim0, dim1, dim2)

    kernel = mx.fast.metal_kernel(
        name="rope_array_positions",
        input_names=["in", "offsets", "inv_freqs", "scaling_factor", "n_batch"],
        output_names=["out"],
        source=source,
    )
    outputs = kernel(
        inputs=[inp, offsets, inv_freqs, scaling_factor, n_batch],
        template=[
            ("T", inp.dtype),
            ("traditional", traditional),
            ("forward", forward),
            ("N", N),
        ],
        grid=grid_dims,
        threadgroup=group_dims,
        output_shapes=[inp.shape],
        output_dtypes=[inp.dtype],
        verbose=verbose,
    )
    return outputs[0]


if __name__ == "__main__":
    from mlx.nn import RoPE
    from mlx_lm.models.rope_utils import Llama3RoPE

    mx.random.seed(42)

    B = 34
    L = 249
    NH = 37
    DIM = 64
    OFFSET = 467
    rope_theta = 500000.0
    rope_traditional = False
    max_position_embeddings = 131072
    rope_scaling = {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }

    llama3_rope = Llama3RoPE(
        DIM, max_position_embeddings, rope_traditional, rope_theta, rope_scaling
    )

    # x = mx.random.normal((1, 3, 3, 6))
    x = mx.random.normal((B, NH, L, DIM))
    # x = mx.repeat(x, L, 2)
    x_bf16 = x.astype(mx.bfloat16)

    permutations = mx.stack(
        [mx.random.permutation(mx.arange(L, dtype=mx.int32)) for _ in range(B)], axis=0
    )  # + OFFSET
    reverse_order = mx.argsort(permutations, axis=1)
    reordered_x = mx.take_along_axis(x, reverse_order[:, None, :, None], axis=2)
    reordered_x_bf16 = reordered_x.astype(mx.bfloat16)

    # Llama3 RoPE tests
    xr_arrayed = rope_positions(
        x,
        # mx.take_along_axis(x, permutations[:, None, :, None], axis=2),
        permutations + OFFSET,
        1 / llama3_rope._freqs[: DIM // 2],
        1.0,
    )
    xr_fast = mx.fast.rope(
        # x,
        reordered_x,
        DIM,
        traditional=False,
        base=None,
        scale=1.0,
        offset=OFFSET,
        freqs=llama3_rope._freqs[: DIM // 2],
    )
    assert mx.equal(
        xr_arrayed,
        # mx.take_along_axis(xr_arrayed, permutations[:, None, :, None], axis=2),
        # xr_fast[],
        mx.take_along_axis(xr_fast, permutations[:, None, :, None], axis=2),
    ).all()

    xr_arrayed = rope_positions(
        x_bf16, permutations + OFFSET, 1 / llama3_rope._freqs[: DIM // 2], 1.0
    )
    xr_fast = mx.fast.rope(
        # x_bf16,
        # mx.take_along_axis(x_bf16, permutations[:, None, :, None], axis=2),
        reordered_x_bf16,
        DIM,
        traditional=False,
        base=None,
        scale=1.0,
        offset=OFFSET,
        freqs=llama3_rope._freqs[: DIM // 2],
    )
    assert mx.equal(
        xr_arrayed,
        # xr_fast,
        mx.take_along_axis(xr_fast, permutations[:, None, :, None], axis=2),
    ).all()

    regular_rope = RoPE(DIM, traditional=rope_traditional, base=rope_theta)
    # _regular_freqs = rope_theta ** (mx.arange(0, DIM, 2) / DIM)
    regular_freqs = mx.array(2, dtype=mx.uint32) ** (
        -mx.log2(rope_theta) * mx.arange(0, DIM, 2) / DIM
    )

    xr_arrayed = rope_positions(
        x, permutations + OFFSET, inv_freqs=regular_freqs, scaling_factor=1.0
    )
    xr_regular = regular_rope(
        # x,
        # mx.take_along_axis(x, permutations[:, None, :, None], axis=2),
        reordered_x,
        OFFSET,
    )
    # assert mx.equal(xr_arrayed, mx.take_along_axis(xr_regular, permutations[:, None, :, None], axis=2)).all()
    # Freqs calculation is not exactly the same, which I think may cause small mismatch
    # mx.fast.rope uses metal::exp2 which is not directly usable from python api
    assert mx.allclose(
        xr_arrayed,
        # xr_regular,
        mx.take_along_axis(xr_regular, permutations[:, None, :, None], axis=2),
        atol=1e-10,
        rtol=1e-10,
    )

    xr_arrayed = rope_positions(
        x_bf16,
        permutations + OFFSET,
        inv_freqs=regular_freqs,
        scaling_factor=1.0,
    )
    xr_regular = regular_rope(
        reordered_x_bf16,
        # mx.take_along_axis(x_bf16, permutations[:, None, :, None], axis=2),
        # x_bf16,
        OFFSET,
    )
    # assert mx.equal(xr_arrayed, mx.take_along_axis(xr_regular, permutations[:, None, :, None], axis=2)).all()
    # # Freqs calculation is not exactly the same, which I think may cause small mismatch
    assert mx.allclose(
        xr_arrayed,
        # xr_regular,
        mx.take_along_axis(xr_regular, permutations[:, None, :, None], axis=2),
        atol=1e-10,
        rtol=1e-10,
    )

    # mx.metal.stop_capture()

    # Timings
    import time

    fast_rope_times = []
    for _ in range(13):
        x = mx.random.normal((B, NH, L, DIM))
        x_bf16 = x.astype(mx.bfloat16)
        mx.eval((x, x_bf16))
        start = time.perf_counter()
        mx.eval(
            mx.fast.rope(
                x_bf16,
                DIM,
                traditional=False,
                base=None,
                scale=1.0,
                offset=OFFSET,
                freqs=llama3_rope._freqs,
            )
        )
        fast_rope_times.append(time.perf_counter() - start)

    arrayed_rope_times = []
    inv_freqs = 1 / llama3_rope._freqs
    offsets = permutations + OFFSET
    for _ in range(13):
        x = mx.random.normal((B, NH, L, DIM))
        x_bf16 = x.astype(mx.bfloat16)
        mx.eval((x, x_bf16))
        start = time.perf_counter()
        mx.eval(
            rope_positions(
                x_bf16,
                offsets,
                inv_freqs=inv_freqs,
                scaling_factor=1.0,
            )
        )
        arrayed_rope_times.append(time.perf_counter() - start)

    print("Fast RoPE time:", sum(fast_rope_times[3:]) / 10 * 1000, "ms")
    print("Custom Kernel RoPE time:", sum(arrayed_rope_times[3:]) / 10 * 1000, "ms")

    trace_file = "mlx_trace.gputrace"

    x = mx.random.normal((B, NH, L, DIM))
    x_bf16 = x.astype(mx.bfloat16)
    mx.eval((x, x_bf16))

    mx.metal.start_capture(trace_file)
    mx.eval(
        mx.fast.rope(
            x_bf16,
            DIM,
            traditional=False,
            base=None,
            scale=1.0,
            offset=OFFSET,
            freqs=llama3_rope._freqs,
        )
    )
    mx.eval(
        rope_positions(
            x_bf16,
            offsets,
            inv_freqs=inv_freqs,
            scaling_factor=1.0,
        )
    )

    mx.metal.stop_capture()
