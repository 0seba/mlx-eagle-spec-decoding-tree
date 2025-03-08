import os
import shutil
import math
import mlx.core as mx
import argparse
import time

FILE_PATH = "kernels/swizzle.metal"



headers = [
    # "#include <metal_stdlib>",
    "#include <metal_compute>",
    "#include <metal_simdgroup_matrix>",
    "using namespace metal;",
]

with open(FILE_PATH, "r") as f:
    source = f.read()
kernel = mx.fast.metal_kernel(
    name="swizzle",
    input_names=["Q"],
    # output_names=["Q_swizzled", "debugQ"],
    output_names=["Q_swizzled"],
    source=source,
    header="\n".join(headers) + "\n\n",
    # atomic_outputs=True,
)

with open("kernels/unpack_swizzled.metal", "r") as f:
    unpack_source = f.read()
unpack_kernel = mx.fast.metal_kernel(
    name="unpack",
    input_names=["Qswizzled"],
    output_names=["Q"],
    source=unpack_source,
    header="\n".join(headers) + "\n\n",
)

def swizzle(Q, verbose=False):
    return kernel(
        inputs=[Q],
        grid=(
            32,
            1,
            Q.shape[0] // 8,
        ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        threadgroup=(32, 1, 1),
        # output_shapes=[Q.shape, (Q.shape[1] * 8, Q.shape[0])],
        # output_dtypes=[Q.dtype, mx.float16],
        output_shapes=[Q.shape],
        output_dtypes=[Q.dtype],
        verbose=verbose,
    )

def unpack(Qswizzled, verbose=False):
    return unpack_kernel(
        inputs=[Qswizzled],
        # grid=(
        #     32,
        #     1,
        #     1,
        # ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        # threadgroup=(32, 1, 1),
        grid=(
            32,
            1,
            1,
        ),  # Max number of threads is 24,576, this means every I have to make every threadgroup compute more of the output columns
        threadgroup=(32, 1, 1),
        output_shapes=[(Qswizzled.shape[1] * 8, Qswizzled.shape[0])],
        output_dtypes=[mx.float16],
        verbose=verbose,
    )

if __name__ == "__main__":
    import argparse

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
        default="swizzle_trace.gputrace",
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
    args = parser.parse_args()


    mx.random.seed(42)
    w = mx.random.normal(shape=(args.M, args.K), scale=1.0, dtype=mx.float16)

    wq, s, b = mx.quantize(w)
    w = mx.dequantize(wq, s, b)
    _s = mx.repeat(s, 64, 1)
    _b = mx.repeat(b, 64, 1)
    wiq = mx.round((w - _b) / _s).T

    trace_file = args.capture_name
    if args.enable_capture:
        # Delete capture with the same name if exists
        if args.rm_capture and os.path.exists(trace_file):
            # capture is a director
            shutil.rmtree(trace_file)
        mx.metal.start_capture(trace_file)

    # print(wiq)

    print(*wiq.tolist(), sep="\n")
    print("-" * 20)

    swizzled = swizzle(wq)
    print(swizzled[0])

    # print(*swizzled[1].astype(mx.int16).tolist(), sep="\n")

    unpacked = unpack(swizzled[0])[0]
    print(*unpacked.tolist(), sep="\n")

    print("-" * 20)
    print(*(unpacked - wiq).tolist(), sep="\n")

    if args.enable_capture:
        mx.metal.stop_capture()
