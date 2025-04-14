import time
import copy
from functools import reduce

import mlx.core as mx

# import mlx_lm

from mlx_lm.models import cache
from mlx.nn import QuantizedLinear, Module
from mlx.utils import tree_map_with_path

# model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-bf16")
# model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
# model, tokenizer = mlx_lm.load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")


def bench(model, seqlen):
    prompt_cache = cache.make_prompt_cache(model)
    prompt = mx.array([[100] * 256])
    # warmup ?
    logits = model(prompt, cache=prompt_cache)
    mx.eval(logits)

    # print("Seqlen | Toks/s | Fwd Time (ms)")
    # print("------ | ------ | -------------")
    inp = mx.array([[100] * seqlen])
    tic = time.perf_counter()
    its = 25
    for _ in range(its):
        logits = model(inp, cache=copy.deepcopy(prompt_cache))
        mx.eval(logits)
    toc = time.perf_counter()
    s = (toc - tic) / its
    tps = seqlen / s
    ms = 1000 * s

    return tps

    print(f"{seqlen} | {tps:.3f} | {ms:.3f}")


def patch_model(model, kernel):
    class NewQuantizedLinear(QuantizedLinear):
        def __init__(self, qlayer: QuantizedLinear):
            Module.__init__(self)  # Initialize from Module instead of QuantizedLinear

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

        def __call__(self, x: mx.array):
            B = reduce(lambda x, y: x * y, x.shape[:-1])
            if B == 8 or B == 16 or B == 24 or B == 32:
                assert (x.shape[-1] % 128) == 0
                x = kernel(x, self.weight, self.scales, self.biases, self.group_size)
                if "bias" in self:
                    x = x + self.bias
                return x
            return super().__call__(x)

    def patch_quantized(model: Module):
        def _maybe_quantize(path, m):
            if isinstance(m, QuantizedLinear):
                return NewQuantizedLinear(m)
            else:
                return m

        leaves = model.leaf_modules()
        leaves = tree_map_with_path(_maybe_quantize, leaves, is_leaf=Module.is_module)
        model.update_modules(leaves)

    patch_quantized(model)
