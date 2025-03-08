# Copyright © 2023-2024 Apple Inc.

import json
import sys


import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.quantized import QuantizedLinear
from mlx.utils import tree_map_with_path

from mlx_lm.models.cache import QuantizedKVCache, load_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import generate, load
from mlx_lm.generate import (
    setup_arg_parser,
)
DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
from qmv import qmv

class NewQuantizedLinear(QuantizedLinear):
    def __init__(self, qlayer: QuantizedLinear):
        Module.__init__(self)  # Initialize from Module instead of QuantizedLinear
        
        # Quantization config
        self.group_size = qlayer.group_size
        self.bits = qlayer.bits

        # Initialize the quantized weight
        self.weight, self.scales, self.biases = qlayer.weight, qlayer.scales, qlayer.biases

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
    leaves = tree_map_with_path(_maybe_quantize, leaves, is_leaf=Module.is_module)
    model.update_modules(leaves)

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            if args.kv_bits is not None and args.kv_bits != prompt_cache[0].bits:
                raise ValueError(
                    "--kv-bits does not match the kv cache loaded from --prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"])
    )
    tokenizer_config["trust_remote_code"] = True

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )
    patch_quantized(model)

    for eos_token in args.extra_eos_token:
        tokenizer.add_eos_token(eos_token)

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
    elif using_cache:
        tokenizer.chat_template = json.loads(metadata["chat_template"])

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    if not args.ignore_chat_template and tokenizer.chat_template is not None:
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        # has_prefill = args.prefill_response is not None
        has_prefill = False
        if has_prefill:
            messages.append({"role": "assistant", "content": args.prefill_response})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
            **template_kwargs,
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            messages[-1]["content"] = "<query>"
            test_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=has_prefill,
                add_generation_prompt=not has_prefill,
            )
            prompt = prompt[test_prompt.index("<query>") :]
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt = tokenizer.encode(prompt)

    if args.draft_model is not None:
        draft_model, draft_tokenizer = load(args.draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match model tokenizer.")
    else:
        draft_model = None
    sampler = make_sampler(args.temp, args.top_p, args.min_p, args.min_tokens_to_keep)
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
        prompt_cache=prompt_cache if using_cache else None,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        draft_model=draft_model,
        num_draft_tokens=args.num_draft_tokens,
    )
    if not args.verbose:
        print(response)


if __name__ == "__main__":
    main()