import os
import math
import mlx.core as mx
import argparse
import time
import shutil


FILE_PATH = "kernels/qmv/1.metal"
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
    """""",
]
with open(FILE_PATH, "r") as f:
    source = f.read()



def qmv(
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
    K = x.shape[-1]
    M = w.shape[0]
    z_threads = min(24_576 // 32, M // 8)
    thread_group_z = min(8, z_threads)

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
            32,
            1,
            z_threads,
            # 1,
        ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        threadgroup=(32, 1, thread_group_z),
        output_shapes=[(x.shape[0], M), (K, M)] if debug else [(x.shape[0], M)],
        output_dtypes=[d_dtype, mx.float16] if debug else [d_dtype],
        verbose=verbose,
    )

    return outputs


def round_cast(x, *args, **kwargs):
    return mx.round(x.astype(mx.float32), *args, **kwargs)

def round_to_list(x, decimals):
    return [round(v, decimals) for v in x.tolist()]

def print_2d(x, decimals):
    for row in x:
        print('  '.join("{: 1.{prec}f}".format(v, prec=decimals) for v in row))

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
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-K", type=int, default=256)
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("--benchmark", "-b", action="store_true", default=False)
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
    # print("W")
    # print_2d(w[:, :32], 3)
    # print("-" * 20, "X", "-" * 20)
    # print_2d(x[:, :32], 3)
    # print("-" * 20)

    mx.synchronize()

    if args.enable_capture:
        # Make sure that the path trace_file does not already exist.
        if args.rm_capture and os.path.exists(args.capture_name):
            # capture is a director
            shutil.rmtree(args.capture_name)
        mx.metal.start_capture(args.capture_name)

    # print(wiq[:6, :6])
    # print(w[:6, :6])
    # print(x[:6, :6])

    mx.eval(mx.quantized_matmul(x[0], wq, s, b))

    mx.synchronize()

    if args.slice:
        target = mx.matmul(x, w.T)
    else:
        target = mx.quantized_matmul(x, wq, s, b)

    mx.eval(target)
    mx.synchronize()
    # print("-" * 20, "TARGET")
    # print(target)
    # print("-" * 20)

    # print("-" * 20, "SCALES")
    # print_2d(s, 3)
    # print("-" * 20, "BIASES")
    # print_2d(b, 3)
    # print("-" * 20)

    o = qmv(
        x,
        wq,
        s,
        b,
        group_size=args.slice if args.slice else 64,
        verbose=args.verbose,
        debug=args.debug,
    )

    # print(s.shape)
    # print(s)
    # print(b.shape)
    # print(b)
    # print(mx.dequantize(wq, s, b))
    # print()
    # print(wq)
    # print(o[-1])
    # print((mx.dequantize(wq, s, b) - o[1]))
    # print((o[1] == mx.dequantize(wq, s, b)).all())
    # for row in o[1].tolist():
    #     print(row)
    # _s = mx.repeat(s, 64, 1)
    # _b = mx.repeat(b, 64, 1)
    # for r1, r2 in zip(o[1].tolist(), (round_cast((mx.dequantize(wq)).tolist()):

    # w = round_cast((w - mx.repeat(b, sK, 1)) / mx.repeat(s, sK, 1))

    # print(round_cast(o[1]))
    # print(round_cast(w))

    # print(round_cast(o[1]).tolist())
    # print(round_cast(w).tolist())
    mx.synchronize()

    print(o)

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
    print((o[0] - target))
    print((o[0] != target).sum())
    # print(o[1])


    if args.enable_capture:
        mx.metal.stop_capture()

    compiled_mm = mx.compile(qmv)
    if args.benchmark:
        times = []
        mx.synchronize()
        for i in range(13):
            tic = time.perf_counter()
            # mx.eval(qmv(x, w, s, b, 64))
            mx.eval(compiled_mm(x, w, s, b, 64))
            toc = time.perf_counter()
            times.append(toc - tic)
            mx.metal.clear_cache()
            mx.eval(mx.full((10 * 1024 ** 2,), 1.0) + mx.full(10 * 1024 ** 2, 2.0)) # clear cache, each of this is 40MB

        times = times[3:]
        print(f"Average time: {sum(times) / len(times)}")
        print(f"Median time: {sorted(times)[len(times) // 2]}")

        mx.synchronize()
        compiled_mm = mx.compile(mx.quantized_matmul)
        mlx_matmul_times = []
        for _ in range(13):
            tic = time.perf_counter()
            mx.eval(compiled_mm(x, wq, s, b))
            toc = time.perf_counter()
            mlx_matmul_times.append(toc - tic)
            mx.metal.clear_cache()
            mx.eval(mx.full((10 * 1024 ** 2,), 1.0) + mx.full(10 * 1024 ** 2, 2.0)) # clear cache, each of this is 40MB

        times = mlx_matmul_times[3:]
        print(f"Average time MLX mm: {sum(times) / len(times)}")
        print(f"Median time MLX mm: {sorted(times)[len(times) // 2]}")

    pass
