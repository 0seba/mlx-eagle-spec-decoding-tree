import os
import math
import mlx.core as mx
import argparse
import time
import shutil

from numpy import mat

from swizzle import swizzle
from utils import print_2d, washout, bench_print

import mlx.nn as nn
from mlx_lm.models.llama import  LlamaModel, ModelArgs, Model

# FILE_PATH = "kernels/simd_qmm.metal"
# FILE_PATH = "kernels/simd_qmm_gpt.metal"
# FILE_PATH = "kernels/progression/1.metal"
# FILE_PATH = "kernels/progression/2_async_load.metal"
# FILE_PATH = "kernels/progression/3.metal"
# FILE_PATH = "kernels/progression/5_7_1.metal"
FILE_PATH = "kernels/progression/simd_no_swizzle_cast.metal"

with open("kernels/mfa_async_ulong.metal", "r") as f:
    # with open("kernels/mfa_async.metal", "r") as f:
    async_source = f.read()
headers = [
    # "#include <metal_stdlib>",
    "#include <metal_compute>",
    "#include <metal_simdgroup_matrix>",
    "using namespace metal;",
    async_source,
    """
static constant half cast_int4[16] = {
    0.0h, 1.0h, 2.0h, 3.0h,
    4.0h, 5.0h, 6.0h, 7.0h,
    8.0h, 9.0h, 10.0h, 11.0h,
    12.0h, 13.0h, 14.0h, 15.0h
};

static constant ushort _float16_from_uint4[16] = {
    0x0000, // 0    ->  0.0
    0x3C00, // 1    ->  1.0
    0x4000, // 2    ->  2.0
    0x4200, // 3    ->  3.0
    0x4400, // 4    ->  4.0
    0x4580, // 5    ->  5.0 (approx)
    0x4600, // 6    ->  6.0
    0x4680, // 7    ->  7.0 (approx)
    0x4700, // 8    ->  8.0
    0x4780, // 9    ->  9.0 (approx)
    0x4800, // 10   -> 10.0
    0x4880, // 11   -> 11.0 (approx)
    0x4900, // 12   -> 12.0
    0x4980, // 13   -> 13.0 (approx)
    0x4A00, // 14   -> 14.0
    0x4A80  // 15   -> 15.0 (approx)
};

inline half uint4_to_float16_shift(thread ushort x) {
    half xcast = as_type<half>(ushort(ushort(0x4000) | x));
    // return x ? xcast : 0.0h;
    return xcast;
}
    """,
]
with open(FILE_PATH, "r") as f:
    source = f.read()


def floor_pow_2(n):
    return 2 ** (math.floor(math.log2(n)))


def floor_multiple_2(n):
    return min(2 * (n // 2), 1)


def matmul(
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
    simds_per_threadgroup=4,
    debug=False,
):
    *B, K = x.shape
    M = w.shape[0]
    # z_threads = min(24_576 // 32, M // 8)
    z_threads = M // 8 # seems that Metal somehow handles more than 24_576 threads
    thread_group_z = min(simds_per_threadgroup, z_threads)

    WIGHTS_PER_ELEMENT = {
        mx.int32: 4,
        mx.int64: 8,
        # these aren't accesible from mlx, maybe with header macro
        # "ulong2": 16,
        # "ulong4": 32,
    }

    kernel = mx.fast.metal_kernel(
        name="matmul",
        input_names=["X", "W", "scales", "biases", "group_size"],
        output_names=["result", "debugW", "debugX"] if debug else ["result"],
        source=source,
        header="\n".join(headers + [f"#define SIMDGROUPS_PER_THREADGROUP {simds_per_threadgroup}"]) + "\n\n",
        # atomic_outputs=True,
    )

    outputs = kernel(
        inputs=[mx.flatten(x, 0, -2), w, s, b, group_size],
        template=[
            # ("LOAD_T", load_type),
            # ("LOAD_SIZE", load_size),
            # ("WIGHTS_PER_ELEMENT", WIGHTS_PER_ELEMENT[load_type]),
            # ("SIMDGROUPS_PER_THREADGROUP", thread_group_z),
            # ("SHMEM_LOAD_TYPE", shmem_load_type),
            # ("SHMEM_WIGHTS_PER_ELEMENT", WIGHTS_PER_ELEMENT[shmem_load_type]),
            # ("SHMEM_LOAD_SIZE", shmem_load_size),
        ],
        # grid=(
        #     32,
        #     1,
        #     1,
        # ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        # threadgroup=(32, 1, 1),
        grid=(
            32,
            1,
            z_threads,
            # 1,
        ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        threadgroup=(32, 1, thread_group_z),
        output_shapes=[(*B, M), (K, M), (K, *B)] if debug else [(*B, M)],
        output_dtypes=[d_dtype, mx.float16, mx.float16] if debug else [d_dtype],
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
        default="traces/qmm.gputrace",
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
    parser.add_argument("-K", type=int, default=64)
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("--benchmark", "-b", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--num_simds", type=int, default=4)
    parser.add_argument("--with-model", action="store_true", default=False)
    parser.add_argument("--swizzle", action="store_true", default=False)
    args = parser.parse_args()

    mx.random.seed(42)
    dtype = mx.float16
    N = 8
    K = args.K
    M = args.M

    if args.slice:
        x = mx.random.normal(shape=(N, args.slice), scale=1.0, dtype=dtype)
    else:
        x = mx.random.normal(shape=(N, K), scale=1.0, dtype=dtype)
    w = mx.random.normal(shape=(M, K), scale=1.0, dtype=dtype)
    wq, s, b = mx.quantize(w)
    wq6, s6, b6 = mx.quantize(w, bits=6)
    if args.swizzle:
        wq_swizzled = swizzle(wq)[0]
    else:
        wq_swizzled = wq
    mx.eval(wq_swizzled, wq6, s6, b6)
    if args.slice:
        w = mx.dequantize(wq, s, b)[:, : args.slice]
    else:
        w = mx.dequantize(wq, s, b)

    _s = mx.repeat(s, 64, 1)
    _b = mx.repeat(b, 64, 1)
    wiq = mx.round((w - _b) / _s)
    if args.slice:
        wq = wq[:, : args.slice]

    mx.eval(wiq)
    mx.synchronize()

    if args.with_model:
        model_args = ModelArgs(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=4,
            intermediate_size=8192,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128_256,
            tie_word_embeddings=False,
            rms_norm_eps=1e-5,
        )
        model_4bits = LlamaModel(model_args)
        nn.quantize(model_4bits)
        mx.eval(model_4bits)
        model_6bits = LlamaModel(model_args)
        nn.quantize(model_6bits, bits=6)
        mx.eval(model_6bits)
        model_input = mx.array([[54_321]], dtype=mx.int32)
        mx.eval(model_input)

    if args.slice:
        target = mx.matmul(x, w.T)
    else:
        target = mx.quantized_matmul(x, wq, s, b)
    mx.eval(target)

    if args.enable_capture:
        # Make sure that the path trace_file does not already exist.
        if args.rm_capture and os.path.exists(args.capture_name):
            # capture is a director
            shutil.rmtree(args.capture_name)
        mx.metal.start_capture(args.capture_name)

    # print(wq_swizzled[:6, :6])
    # print(wiq[:6, :6])
    print("----- W -----")
    print(w[:6, :6])
    print(w)
    if args.debug:
        print_2d(w.T, 3)
    print("----- X -----")
    print(x[:6, :6])
    print(x)
    if args.debug:
        print_2d(x.T, 3)
    print("----- Target -----")
    print(target[:6, :6])
    print(target)


    x0 = x[[0]]
    mx.eval(x0)
    washout()
    mx.eval(mx.quantized_matmul(x0, wq, s, b))
    washout()
    mx.eval(mx.quantized_matmul(x, wq, s, b))
    washout()
    mx.eval(mx.quantized_matmul(x0, wq6, s6, b6, bits=6))
    washout()


    o = matmul(
        x,
        # wq,
        wq_swizzled,
        s,
        b,
        # group_size=sK,
        group_size=args.slice if args.slice else 64,
        d_dtype=mx.float16,
        verbose=args.verbose,
        debug=args.debug,
        simds_per_threadgroup=args.num_simds,
    )
    mx.eval(o)
    washout()

    if args.with_model:
        mx.eval(model_4bits(model_input))
        washout()
        mx.eval(model_6bits(model_input))
        washout()

        # print(model_4bits)
        # print(model_6bits)

    if args.enable_capture:
        mx.metal.stop_capture()

    print("----- Ouput -----")
    print(o[0][:6, :6])
    print(o)

    if len(o) > 1 and args.debug:
        print("Recon W")
        print_2d(o[1], 3)
        print("Recon X")
        print_2d(o[2], 3)

    print("----- Difference -----")
    print((o[0] - target))
    print((o[0] != target).sum())

    mx.synchronize()


    if args.benchmark:
        from functools import partial

        washout()

        custom_method = matmul
        custom_method_batched = partial(
            custom_method, w=wq, s=s, b=b, group_size=64,
        )
        compiled_custom_batched = mx.compile(custom_method_batched)

        print(FILE_PATH)

        bench_print(custom_method_batched, f"Custom 4-bits Not-compiled batch size {N}", x)
        print("-" * 20)
        bench_print(compiled_custom_batched, f"Custom 4-bits compiled batch size {N}", x)
        print("-" * 20)

        mlx_qmm = lambda x: mx.quantized_matmul(x, wq, s, b)
        compiled_mlx = mx.compile(mlx_qmm)
        mlx_qmm6 = lambda x: mx.quantized_matmul(x, wq6, s6, b6, bits=6)
        compiled_mlx6 = mx.compile(mlx_qmm6)
        bench_print(compiled_mlx, f"MLX 4-bits compiled batch size {N}", x)
        print("-" * 20)
        bench_print(compiled_mlx, f"MLX 4-bits compiled batch size 1", x0)
        print("-" * 20)
        bench_print(mlx_qmm, f"MLX 4-bits Not-compiled batch size {N}", x)
        print("-" * 20)
        bench_print(mlx_qmm, f"MLX 4-bits Not-compiled batch size 1", x0)
        print("-" * 20)

        bench_print(compiled_mlx6, f"MLX 6-bits compiled batch size {N}", x)
        print("-" * 20)
        bench_print(compiled_mlx6, f"MLX 6-bits compiled batch size 1", x0)
        print("-" * 20)
        bench_print(mlx_qmm6, f"MLX 6-bits Not-compiled batch size {N}", x)
        print("-" * 20)
        bench_print(mlx_qmm, f"MLX 6-bits Not-compiled batch size 1", x0)
        print("-" * 20)