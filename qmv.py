import os
import math
import mlx.core as mx
import argparse
import time
import shutil

from utils import print_2d, washout, bench_print, round_cast


# FILE_PATH = "kernels/qmv/1.metal"
# FILE_PATH = "kernels/qmv/1b.metal"
FILE_PATH = "kernels/qmv/1b_1.metal"
# FILE_PATH = "kernels/qmv/1b_2.metal"
# FILE_PATH = "kernels/qmv/1b_swizzled_contiguous.metal"
# FILE_PATH = "kernels/qmv/2.metal"
HELPERS_PATH = "kernels/qmv/helpers.metal"

with open("kernels/mfa_async_ulong.metal", "r") as f:
    # with open("kernels/mfa_async.metal", "r") as f:
    async_source = f.read()
with open("kernels/qmv/helpers.metal", "r") as f:
    helpers_source = f.read()
headers = [
    # "#include <metal_stdlib>",
    "#include <metal_compute>",
    "#include <metal_simdgroup_matrix>",
    "using namespace metal;",
    async_source,
    helpers_source,
    """
inline half uint4_to_float16_shift(thread ushort x) {
    half xcast = as_type<half>(ushort(ushort(0x4000) | x));
    // return x ? xcast : 0.0h;
    return xcast;
}
    """,
]
with open(FILE_PATH, "r") as f:
    source = f.read()


def qmv(
    x,
    w,
    s,
    b,
    group_size,
    kernel,
    d_dtype=mx.float16,
    verbose=False,
    debug=False,
    simd_groups=2,
    rows_per_simd=4,
):
    *B, K = x.shape
    M = w.shape[0]
    # z_threads = min(24_576 // 32, M // 8)
    y_threads = max(1, M // rows_per_simd)
    thread_group_y = min(simd_groups, y_threads)

    outputs = kernel(
        inputs=[mx.flatten(x, 0, -2), w, s, b, group_size],
        # template=[],
        grid=(
            32,
            y_threads,
            1,
            # 1,
        ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        threadgroup=(32, thread_group_y, 1),
        output_shapes=[(*B, M), (K, M)] if debug else [(*B, M)],
        output_dtypes=[d_dtype, mx.float16] if debug else [d_dtype],
        verbose=verbose,
    )

    return outputs


def qmv_quad(
    x,
    w,
    s,
    b,
    group_size,
    d_dtype=mx.float16,
    verbose=False,
    load_type=mx.int32,
    load_size=1,
    shmem_load_type=mx.int32,
    shmem_load_size=1,
    debug=False,
):
    *B, K = x.shape
    M = w.shape[0]
    # z_threads = min(24_576 // 32, M // 8)
    z_threads = max(1, M)
    thread_group_z = min(16, z_threads)

    WIGHTS_PER_ELEMENT = {
        mx.int32: 4,
        mx.int64: 8,
        # these aren't accesible from mlx, maybe with header macro
        # "ulong2": 16,
        # "ulong4": 32,
    }

    kernel = mx.fast.metal_kernel(
        name="qmv",
        input_names=["X", "W", "scales", "biases", "group_size"],
        output_names=["result", "debugW"] if debug else ["result"],
        source=source,
        header="\n".join(headers) + "\n\n",
        # atomic_outputs=True,
    )

    outputs = kernel(
        inputs=[x, w, s, b, group_size],
        # template=[],
        grid=(
            4,
            1,
            z_threads,
            # 1,
        ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        threadgroup=(4, 1, thread_group_z),
        output_shapes=[(*B, M), (K, M)] if debug else [*B, M],
        output_dtypes=[d_dtype, mx.float16] if debug else [d_dtype],
        verbose=verbose,
    )

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matrix multiplication with optional metal capture"
    )
    parser.add_argument(
        "--enable-capture",
        "-c",
        default=False,
        action="store_true",
        help="Enable metal capture",
    )
    parser.add_argument(
        "--capture-name",
        # "-cn",
        default="traces/qmv.gputrace",
    )
    parser.add_argument(
        "--rm-capture",
        "-r",
        default=False,
        action="store_true",
        help="Delete capture with the same name if exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
        help="Print generated kernels",
    )
    parser.add_argument("--simd-groups", "-s", type=int, default=4)
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-K", type=int, default=256)
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("--benchmark", "-b", action="store_true", default=False)
    parser.add_argument("--model-benchmark", "-mb", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    mx.random.seed(42)
    dtype = mx.float16
    N = args.N
    K = args.K
    M = args.M
    scale = 0.2
    if args.slice:
        x = mx.random.normal(shape=(N, args.slice), scale=scale, dtype=dtype)
    else:
        x = mx.random.normal(shape=(N, K), scale=scale, dtype=dtype)
    w = mx.random.normal(shape=(M, K), scale=scale, dtype=dtype)
    wq, s, b = mx.quantize(w)
    if args.slice:
        w = mx.dequantize(wq, s, b)[:, : args.slice]
    else:
        w = mx.dequantize(wq, s, b)
    # for i in range(w.shape[0]):
    #     print(round_cast(w[i], 3).tolist())
    #     print("-" * 20)
    _s = mx.repeat(s, 64, 1)
    _b = mx.repeat(b, 64, 1)
    wiq = round_cast((w - _b) / _s)
    if args.slice:
        wq = wq[:, : args.slice]

    print("S")
    print(s)
    # print_2d(s, 3)
    print("B")
    print(b)
    # print_2d(b, 3)

    print("W")
    print(w)
    # print_2d(w[:, :32], 3)
    print("-" * 20, "X", "-" * 20)
    print(x)
    # print_2d(x[:, :32], 3)
    print("-" * 20)

    mx.eval(w, wiq)
    washout()

    target = []
    for i in range(args.N):
        target.append(mx.quantized_matmul(x[i], wq, s, b))
    target = mx.stack(target)
    mx.eval(target)

    x0 = x[[0]]
    mx.eval(x0)
    kernel_vector = mx.fast.metal_kernel(
        name="qmv_vector",
        input_names=["X", "W", "scales", "biases", "group_size"],
        output_names=["result", "debugW"] if args.debug else ["result"],
        source=source,
        header="\n".join(headers + ["#define BATCH_SIZE 1"]) + "\n\n",
        # atomic_outputs=True,
    )
    kernel_batched = mx.fast.metal_kernel(
        name="qmv_batch",
        input_names=["X", "W", "scales", "biases", "group_size"],
        output_names=["result", "debugW"] if args.debug else ["result"],
        source=source,
        header="\n".join(headers + ["#define BATCH_SIZE " + str(args.N)]) + "\n\n",
        # atomic_outputs=True,
    )

    # warmup?
    for _ in range(5):
        qmv(
            x,
            wq,
            s,
            b,
            group_size=args.slice if args.slice else 64,
            kernel=kernel_batched,
            verbose=args.verbose,
            debug=args.debug,
            simd_groups=args.simd_groups,
        )
        # qmv(
        #     # o = qmv_quad(
        #     x0,
        #     wq,
        #     s,
        #     b,
        #     group_size=args.slice if args.slice else 64,
        #     kernel=kernel_vector,
        #     verbose=args.verbose,
        #     debug=args.debug,
        #     simd_groups=args.simd_groups,
        # )
    washout()

    if args.enable_capture:
        # Make sure that the path trace_file does not already exist.
        if args.rm_capture and os.path.exists(args.capture_name):
            # capture is a director
            shutil.rmtree(args.capture_name)
        mx.metal.start_capture(args.capture_name)

    # if args.N > 1:
    #     target_batch = mx.quantized_matmul(x, wq, s, b)
    #     mx.eval(target_batch)
    #     washout()
    # for batch size 1 tracing
    washout()
    target_x0 = mx.quantized_matmul(x0, wq, s, b)
    mx.eval(target_x0)
    washout()

    # if args.slice:
    #     target = mx.matmul(x, w.T)
    # else:
    #     target = mx.quantized_matmul(x, wq, s, b)

    # print("-" * 20, "TARGET")
    # print(target)
    # print("-" * 20)

    # print("-" * 20, "SCALES")
    # print_2d(s, 3)
    # print("-" * 20, "BIASES")
    # print_2d(b, 3)
    # print("-" * 20)

    o = qmv(
        # o = qmv_quad(
        x,
        wq,
        s,
        b,
        group_size=args.slice if args.slice else 64,
        kernel=kernel_batched,
        verbose=args.verbose,
        debug=args.debug,
        simd_groups=args.simd_groups,
    )
    mx.eval(o)
    washout()

    # o1 = qmv(
    #     # o = qmv_quad(
    #     x0,
    #     wq,
    #     s,
    #     b,
    #     group_size=args.slice if args.slice else 64,
    #     kernel=kernel_vector,
    #     verbose=args.verbose,
    #     debug=args.debug,
    #     simd_groups=args.simd_groups,
    # )
    # mx.eval(o1)
    mx.synchronize()

    if args.enable_capture:
        mx.metal.stop_capture()

    if len(o) > 1:
        _wt = w.T
        # _wt = wiq.T
        # print(_wt.shape)
        print("----")
        for i in range(_wt.shape[0]):
            print(round_to_list(_wt[i], 5))
            print(round_to_list(o[1][i], 5))
            print((_wt[i] - o[1][i]).tolist())
            print("-" * 20)
        # print(_wt)
        print("----")
        print(_wt - o[1])
        print("----")
        print((_wt != o[1]).sum())
        print("----")
        # print(_wt[:6, :6])
        # print(o[1][:6, :6])
        # print(_wt[:6, :6] - o[1][:6, :6])
        # print(_wt[:6, :6] - o[1][:6, :6])
        print("----")
    # for r1, r2 in zip(o[1].tolist(), w.T.tolist()):
    #     print("----")
    #     print(r1)
    #     print(r2)
    #     print((mx.array(r1) - mx.array(r2)).tolist())
    # print("----")
    # print(outs[(dtype_in, dtype_out)][0])

    # for i in range(target.shape[0]):
    #     print(target[i].tolist())
    #     print(o[0][i].tolist())
    #     print((target[i] - o[0][i]).tolist())
    #     print("-" * 20)
    # print(o[0].shape)
    # print(o[0])
    # print(target)
    print("----- TARGET -----")
    print(target)
    print("----- RESULT -----")
    print(o)
    print("DIFF")
    print((o[0] - target))
    print((o[0] != target).sum())
    # print(o[1])

    if args.benchmark:
        from functools import partial

        washout()

        custom_method = qmv
        custom_method_vector = partial(
            custom_method, w=wq, s=s, b=b, group_size=64, kernel=kernel_vector
        )
        compiled_custom_vector = mx.compile(custom_method_vector)
        custom_method_batched = partial(
            custom_method, w=wq, s=s, b=b, group_size=64, kernel=kernel_batched
        )
        compiled_custom_batched = mx.compile(custom_method_batched)

        bench_print(compiled_custom_batched, f"Custom compiled batch size {args.N}", x)
        print("-" * 20)
        # bench_print(compiled_custom_vector, f"Custom compiled batch size 1", x0)
        # print("-" * 20)
        bench_print(custom_method_batched, f"Custom Not-compiled batch size {args.N}", x)
        print("-" * 20)
        # bench_print(custom_method_vector, f"Custom Not-compiled batch size 1", x0)
        # print("-" * 20)

        mlx_qmm = lambda x: mx.quantized_matmul(x, wq, s, b)
        compiled_mlx = mx.compile(mlx_qmm)
        bench_print(compiled_mlx, f"MLX compiled batch size {args.N}", x)
        print("-" * 20)
        bench_print(compiled_mlx, f"MLX compiled batch size 1", x0)
        print("-" * 20)
        bench_print(mlx_qmm, f"MLX Not-compiled batch size {args.N}", x)
        print("-" * 20)
        bench_print(mlx_qmm, f"MLX Not-compiled batch size 1", x0)
        print("-" * 20)

    if args.model_benchmark:
        from mlx_model_fwd_bench import bench
        from mlx_lm.utils import load_model

        class NewQuantizedLinear(QuantizedLinear):
            def __init__(self, qlayer: QuantizedLinear):
                Module.__init__(
                    self
                )  # Initialize from Module instead of QuantizedLinear

                # Quantization config
                self.group_size = qlayer.group_size
                self.bits = qlayer.bits

                # Initialize the quantized weight
                self.weight, self.scales, self.biases = (
                    qlayer.weight,
                    qlayer.scales,
                    qlayer.biases,
                )

                if "bias" in qlayer:
                    self.bias = qlayer.bias

                # Freeze this model's parameters
                self.freeze()

            def __call__(self, x):
                if x.shape[1] != 1:
                    return super().__call__(x)
                x = qmv(x, self.weight, self.scales, self.biases, self.group_size)[0]
                if "bias" in self:
                    x = x + self.bias
                return x

        def patch_quantized(model: Module):
            def _maybe_quantize(path, m):
                if isinstance(m, QuantizedLinear):
                    return NewQuantizedLinear(m)
                else:
                    return m

            leaves = model.leaf_modules()
            leaves = tree_map_with_path(
                _maybe_quantize, leaves, is_leaf=Module.is_module
            )
            model.update_modules(leaves)

        model, _ = load_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        tps_mlx_bs1 = bench(model, 1)
        tps_mlx_bs_ = bench(model, args.N)

        # custom_method = qmv
        # custom_method = partial(custom_method, group_size=64)
        # compiled_custom = mx.compile(custom_method)

        patch_quantized(model)
