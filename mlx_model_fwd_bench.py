import time
import copy

import mlx.core as mx
# import mlx_lm

from mlx_lm.models import cache

#model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-bf16")
# model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
#model, tokenizer = mlx_lm.load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")


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