import math
import argparse
from typing import List, Union, Optional

# import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.utils import wired_limit

from mlx_utils import load
from mlx_llama import Model as LlamaModel
from mlx_llama import Model as Qwen2Model
from utils import Performer
from mlx_model_fwd_bench import patch_model
from qmm_kernel import matmul as qmm

# from mlx_lm.models.llama import Model as LlamaModel

performer = Performer(2)


def topk(x, k, axis, stable=False):
    # Use argpartition to find the indices of the top k elements along the specified axis
    # We negate x because we want for the first ocurrance to be selected
    if stable:
        partitioned = mx.argsort(-x, axis=axis)
    else:
        partitioned = mx.argpartition(
            x, -k, axis=axis
        )  # I'm unsure if argpartition has stable algorithm

    # Select the indices of the top k elements
    slicing_obj = [slice(None)] * x.ndim
    if stable:
        slicing_obj[axis] = slice(None, k)
    else:
        slicing_obj[axis] = slice(-k, None)
    # slicing_obj[axis] = slice(-k, None)
    topk_indices = partitioned[tuple(slicing_obj)]

    # Gather the top k values using advanced indexing
    topk_values = mx.take_along_axis(x, topk_indices, axis=axis)

    # Now sort the top k values and corresponding indices
    # Since we want highest to lowest, we sort in descending order
    # if stable:
    #     sort_order = mx.argsort(-topk_values, axis=axis, kind="stable")
    # else:
    #     sort_order = mx.flip(
    #         mx.argsort(topk_values, axis=axis, kind="stable"), axis=axis
    #     )
    sort_order = mx.argsort(-topk_values, axis=axis)
    topk_values = mx.take_along_axis(topk_values, sort_order, axis=axis)
    topk_indices = mx.take_along_axis(topk_indices, sort_order, axis=axis)

    return topk_indices, topk_values


def search_sorted(arr: mx.array, target: mx.array, left=True):
    """
    Vectorized binary search implementation similar to numpy's searchsorted.

    Args:
        arr: Input array to search in (must be sorted)
        target: Values to search for
        left: If True, returns leftmost suitable index. If False, returns rightmost

    Returns:
        Array of indices where elements should be inserted to maintain order
    """
    # Initialize low and high arrays matching target shape
    low = mx.zeros_like(target)
    high = mx.full(target.shape, arr.size, dtype=mx.uint32)
    num_steps = int(math.ceil(math.log2(arr.size)))

    for _ in range(num_steps):
        mid = (low + high) // 2

        # Get values at mid indices
        mid_values = arr[mid]

        if left:
            # For left=True, update boundaries based on < and >= comparisons
            low = mx.where(mid_values < target, mid + 1, low)
            high = mx.where(mid_values >= target, mid, high)
        else:
            # For left=False, update boundaries based on <= and > comparisons
            low = mx.where(mid_values <= target, mid + 1, low)
            high = mx.where(mid_values > target, mid, high)

    return low


# @mx.compile
def process_generate_tree_iteration(
    logits: mx.array,
    current_scores: mx.array,
    k: int = 8,
    tokenizer=None,
):
    # lse = mx.logsumexp(logits, -1, keepdims=True)
    logits = mx.softmax(logits, axis=-1).log()
    topk_index, topk_p = topk(logits, k, -1)
    # topk_p -= lse
    cu_scores = topk_p + current_scores
    topk_cs_index, topk_cs_p = topk(cu_scores.flatten(), k, -1)
    out_ids = mx.floor_divide(topk_cs_index, k)

    next_input_ids = mx.expand_dims(topk_index.flatten()[topk_cs_index], 0)
    next_scores = mx.expand_dims(topk_cs_p, (0, -1))
    # mask = curent_mask_pointer[out_ids]

    return next_input_ids, next_scores, out_ids, topk_index, cu_scores, topk_cs_index


def generate_draft_tree(
    topk_index: mx.array,
    topk_p: mx.array,
    draft_model: Union[Qwen2Model, LlamaModel],
    cache: List[KVCache],
    # offset: int,
    depth=5,
    k=8,
    tokenizer=None,
):
    """
    Args:
        topk_index (mx.array): After base model verification, next token is sampled from logits,
            that token is forwarded through draft model and these are the topk argnums for it's logits
        topk_p (mx.array): Draft model topk argnums logits
        draft_model (nn.Module): _description_
        cache (List[KVCache]): _description_
        offset (int): Number of current token
        depth (int, optional): Tree depth. Defaults to 5.
        k (int, optional): Top-K. Defaults to 8.
    """
    offset = cache[0].offset
    current_input_ids = topk_index  # .reshape(1, 1)
    scores = topk_p  # .reshape(1, k, 1)

    mask = mx.full((1, 1, k, offset + depth * k), -mx.inf, dtype=mx.bfloat16)
    mask[..., :offset] = 0

    eye = mx.expand_dims(mx.where(mx.eye(k, dtype=mx.bool_), 0, -mx.inf), (0, 1))
    saved_topk_indices = [topk_index[:, None]]
    saved_cu_scores = [scores.transpose(0, 2, 1)]
    # saved_topk_cs_indices = [mx.arange(0, k)]
    parents_list = [mx.zeros((1,), dtype=mx.uint32)]

    topk_cs_index = mx.arange(k, dtype=mx.uint32)

    phrases_debugger = current_input_ids[0, :, None]
    # print(tokenizer._tokenizer.batch_decode(np.array(phrases_debugger)))

    for i in range(depth):
        bias1 = k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + k**2 * bias2 + bias1
        parents = topk_cs_index + bias
        parents_list.append(parents)

        # mask[..., range(k), range(offset + i * k, offset + (i + 1) * k)] = 0
        # mx.put_along_axis(
        #     mask,
        #     mx.arange(offset + i * k, offset + (i + 1) * k).reshape(1, 1, k, 1),
        #     mx.zeros((1, 1, k, 1)),
        #     axis=3,
        # )
        mask[..., offset + i * k : offset + (i + 1) * k] = eye
        logits = draft_model(
            current_input_ids,
            cache,
            position_ids=offset + i,
            flat_rope=True,
            mask=mask[..., : offset + (i + 1) * k],
        )

        (
            new_input_ids,
            scores,
            out_ids,
            topk_index,
            cu_scores,
            topk_cs_index,
        ) = process_generate_tree_iteration(
            logits,
            scores,
            tokenizer=tokenizer,
            k=k,
        )

        phrases_debugger = phrases_debugger[out_ids]
        phrases_debugger = mx.concatenate(
            (phrases_debugger, new_input_ids[0, :, None]), axis=1
        )
        current_input_ids = new_input_ids

        saved_topk_indices.append(topk_index)
        saved_cu_scores.append(cu_scores)

        # print(tokenizer._tokenizer.batch_decode(np.array(phrases_debugger)))

        # saved_topk_cs_indices.append(topk_cs_index)

        mask = mask[..., out_ids, :]

    # print()
    # print(tokenizer._tokenizer.batch_decode(np.array(phrases_debugger)))
    # print()

    return parents_list, saved_topk_indices, saved_cu_scores


# it's compilable
def process_draft_tree(
    scores_list: List[mx.array],
    ss_token: List[mx.array],
    parents_list: List[mx.array],
    sampled_token_id: mx.array,
    k: int,
    total_tokens: int,
):
    scores_list = mx.concatenate(scores_list, axis=1).flatten()
    ss_token = mx.concatenate(ss_token, axis=1).flatten()
    parents_list = mx.concatenate(parents_list, axis=0)  # .flatten()

    top_scores_index, _ = topk(scores_list, k=total_tokens, stable=True, axis=-1)
    top_scores_index = mx.sort(top_scores_index)

    draft_tokens = ss_token[top_scores_index]
    draft_tokens = mx.concatenate([sampled_token_id[0], draft_tokens])

    draft_parents = parents_list[mx.floor_divide(top_scores_index, k)]
    mask_index = search_sorted(top_scores_index, draft_parents - 1)
    mask_index = mx.where(draft_parents == 0, -1, mask_index) + 1

    tree_mask = mx.eye(total_tokens + 1, dtype=mx.bool_)
    # tree_mask[..., 0] = True

    for i in range(total_tokens):
        tree_mask[i + 1] = tree_mask[i + 1] + tree_mask[mask_index[i]]

    tree_position_ids = mx.sum(tree_mask, 1) - 1
    # tree_mask = tree_mask.astype(mx.bfloat16)
    draft_tokens = mx.expand_dims(draft_tokens, 0)
    tree_mask = mx.where(tree_mask, mx.array(0, dtype=mx.bfloat16), -mx.inf)

    return tree_position_ids, mask_index, draft_tokens, tree_mask


# current implementation of this method is not compilable
def post_process_draft_tree(
    tree_position_ids: mx.array, total_tokens: int, mask_index: mx.array
) -> mx.array:
    """
    Post-process a draft tree by constructing retrieval indices matrix.

    Args:
        tree_position_ids: MLX array of tree position IDs
        total_tokens: Total number of tokens
        mask_index: MLX array of mask indices

    Returns:
        MLX array of retrieval indices with shape [leaf_num, max_depth]
    """
    # Convert to Python types for processing
    max_depth = tree_position_ids.max().item() + 1
    noleaf_index = set(mask_index.tolist())
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num
    tree_position_ids_list = tree_position_ids.tolist()
    mask_index_list = mask_index.tolist()

    # Initialize retrieve_indices as 2D list with -1s
    retrieve_indices = [[-1] * max_depth for _ in range(leaf_num)]

    rid = 0
    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = tree_position_ids_list[i]

            # Fill in indices from depth down to 0
            for j in range(depth, -1, -1):
                retrieve_indices[rid][j] = cid
                # Update cid using mask_index_list
                cid = mask_index_list[-1] if cid == 0 else mask_index_list[cid - 1]
            rid += 1

    # Convert back to MLX array
    # First flatten the 2D list and then reshape
    flat_indices = [item for sublist in retrieve_indices for item in sublist]
    return mx.array(flat_indices).reshape(leaf_num, max_depth)

    # Commented out logits processor section as in original:
    # if logits_processor is not None:
    #     maxitem = total_tokens + 5
    #
    #     def custom_sort(lst):
    #         # sort_keys=[len(list)]
    #         sort_keys = []
    #         for i in range(len(lst)):
    #             sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
    #         return sort_keys
    #
    #     retrieve_indices = sorted(retrieve_indices, key=custom_sort)


def evaluate_posterior_greedy(
    logits,
    candidates,
):
    # Find the tokens that match the maximum logits for each position in the sequence
    posterior_mask = (candidates[:, 1:] == mx.argmax(logits[:, :-1], axis=-1)).astype(
        mx.int32
    )
    candidates_accept_length = (mx.cumprod(posterior_mask, axis=1)).sum(axis=1)
    accept_length = candidates_accept_length.max()
    best_candidate = mx.where(
        accept_length == 0,
        mx.array(0, dtype=mx.int32),
        mx.argmax(candidates_accept_length).astype(mx.int32),
    )
    return best_candidate, accept_length, logits[best_candidate, accept_length]


def tree_decoding(
    token_ids: mx.array,
    tree_mask: mx.array,
    tree_position_ids: mx.array,
    base_model: Union[LlamaModel, Qwen2Model],
    cache: List[KVCache],
) -> mx.array:
    offset = cache[0].offset
    mask = mx.zeros((1, 1, tree_mask.shape[0], offset), dtype=mx.bfloat16)
    mask = mx.concatenate((mask, mx.expand_dims(tree_mask, (0, 1))), axis=-1)
    # tree_position_ids += offse # inplace update breaks other methods
    logits = base_model(
        token_ids,
        cache,
        position_ids=tree_position_ids[None] + offset,
        mask=mask,
    )
    return logits


def accept_candidate(
    cache: List[KVCache],
    accept_indices: mx.array,
    accept_length: int,
    offset: int,
):
    """
    Updates the cache with the adequate offset and accepted indices.

    Args:
        cache: The KVCache associated with the model.
        accept_length: The number of tokens to accept from the candidates.
        offset: The current offset in the cache.

    Returns:
        None
    """
    if accept_length > 0:
        # accept_indices = retrieve_indices[accept_length - 1]
        for c in cache:
            c.keys[:, :, offset + 1 : offset + accept_length + 1] = c.keys[
                :, :, offset + accept_indices
            ]
            c.values[:, :, offset + 1 : offset + accept_length + 1] = c.values[
                :, :, offset + accept_indices
            ]
    for c in cache:
        c.offset = offset + accept_length + 1


def spec_step(
    y: mx.array,
    # offset: int,
    base_model: Union[LlamaModel, Qwen2Model],
    draft_model: Union[LlamaModel, Qwen2Model],
    base_cache: List[KVCache],
    draft_cache: List[KVCache],
    offset: int,
    accept_length: int,
    k: int = 8,
    depth: int = 5,
    verify_num_tokens=39,
    tokenizer=None,
):
    base_offset = base_cache[0].offset
    draft_offset = draft_cache[0].offset + 1 + accept_length

    # mx.eval(y, *draft_cache)
    # performer.tic("init")

    _y = y
    if accept_length > 0:
        pad_size = 8 - accept_length - 1
        # pad to 8
        y = mx.concatenate((y, mx.zeros((1, pad_size), dtype=mx.int32)), axis=-1)
        draft_logits = draft_model(y, cache=draft_cache)[:, [accept_length]]
        for cache in draft_cache:
            cache.offset = draft_offset
    else:
        draft_logits = draft_model(y, cache=draft_cache)[:, [accept_length]]
    # draft_logits = mx.softmax(draft_logits, axis=-1).log()
    lse = draft_logits.logsumexp(axis=-1)
    topk_index, topk_p = topk(mx.squeeze(draft_logits, 1), k, -1)
    topk_p = mx.expand_dims(topk_p - lse, -1)

    # mx.eval(topk_index, topk_p, *draft_cache)
    # performer.toc("init")
    # performer.tic("generate_tree")

    out_parents, out_tokens, out_scores = generate_draft_tree(
        topk_index=topk_index,
        topk_p=topk_p,
        draft_model=draft_model,
        cache=draft_cache,
        # offset=offset + accept_length + 1,
        depth=depth,
        k=k,
        tokenizer=tokenizer,
    )
    # mx.eval(out_parents, out_tokens, out_scores)
    # performer.toc("generate_tree")
    # performer.tic("process_draft_tree")

    tree_position_ids, mask_index, draft_tokens, tree_mask = process_draft_tree(
        scores_list=out_scores,
        ss_token=out_tokens,
        parents_list=out_parents,
        sampled_token_id=_y[:, [-1]],
        k=k,
        total_tokens=verify_num_tokens,
    )
    # mx.eval(tree_position_ids, mask_index, draft_tokens, tree_mask)
    # performer.toc("process_draft_tree")
    # performer.tic("tree_decoding")

    base_logits = tree_decoding(
        token_ids=draft_tokens,
        tree_mask=tree_mask,
        base_model=base_model,
        cache=base_cache,
        tree_position_ids=tree_position_ids,
    )

    # mx.eval(base_logits)
    # performer.toc("tree_decoding")
    # performer.tic("post_process_draft_tree")

    return (
        tree_position_ids,
        mask_index,
        draft_tokens,
        base_logits,
        base_offset,
        draft_offset,
    )

    retrieve_indices = post_process_draft_tree(
        tree_position_ids=tree_position_ids,
        total_tokens=verify_num_tokens,
        mask_index=mask_index,
    )
    # mx.eval(retrieve_indices)
    # performer.toc("post_process_draft_tree")
    # performer.tic("evaluate_posterior_greedy")

    candidates = draft_tokens[0, retrieve_indices]
    # print("\n\nDraft candidates:")
    # for c in candidates:
    #     print(tokenizer.decode(c.tolist()).replace("\n", "\\n"))
    # print("-" * 20)
    (best_candidates, new_accept_length, sample_p) = evaluate_posterior_greedy(
        base_logits[0, retrieve_indices], candidates
    )

    new_accept_length = new_accept_length.item()
    accept_indices = retrieve_indices[best_candidates, 1 : 1 + new_accept_length]
    accepted_tokens = candidates[best_candidates, 1 : 1 + new_accept_length]

    mx.eval(accept_indices, accepted_tokens, sample_p)
    performer.toc("evaluate_posterior_greedy")
    performer.tic("accept_candidate")

    accept_candidate(base_cache, accept_indices, new_accept_length, base_offset)
    performer.toc("accept_candidate")

    # print(draft_offset)
    # print(base_offset)
    # print(new_accept_length)
    for c in draft_cache:
        c.offset = draft_offset

    return (accepted_tokens, new_accept_length, sample_p)


def generate_step(
    prompt: mx.array,
    base_model: Union[LlamaModel, Qwen2Model],
    draft_model: Union[LlamaModel, Qwen2Model],
    prefill_step_size: int = 512,
    k=8,
    depth=5,
    verify_num_tokens=39,
    max_tokens=128,
    tokenizer=None,  # For debugging
    N=20,
):
    y = prompt
    tokens = None

    base_cache = make_prompt_cache(base_model)
    draft_cache = make_prompt_cache(draft_model)
    offset = y.size

    while y.size > prefill_step_size:
        draft_model(y[:prefill_step_size][None], cache=draft_cache)
        base_model(y[:prefill_step_size][None], cache=base_cache)
        mx.eval([c.state for c in base_cache])
        mx.eval([c.state for c in draft_cache])
        y = y[prefill_step_size:]
        mx.metal.clear_cache()

    draft_model(y[None], cache=draft_cache)
    mx.async_eval([c.state for c in draft_cache])

    base_logits = base_model(y[None], cache=base_cache)
    y = mx.argmax(base_logits[:, -1], -1, keepdims=True)
    mx.eval(y)

    # print(tokenizer.decode(y.item()), end="", flush=True)

    n = 0
    accept_length = 0
    performer.tic("total")
    # accept_lengths = []
    # N = 50
    for i in range(N):
        if n >= max_tokens:
            break

        # performer.tic("spec_step")
        # accepted_tokens, accept_length, sample_p = spec_step(
        # schedule async execution of operations graph that does not have any break
        (
            tree_position_ids,
            mask_index,
            draft_tokens,
            base_logits,
            base_offset,
            draft_offset,
        ) = spec_step(
            y,
            base_model=base_model,
            draft_model=draft_model,
            base_cache=base_cache,
            draft_cache=draft_cache,
            offset=offset,
            accept_length=accept_length,
            k=k,
            depth=depth,
            verify_num_tokens=verify_num_tokens,
            tokenizer=tokenizer,
        )
        mx.async_eval(tree_position_ids, mask_index, draft_tokens, base_logits)

        print(tokenizer.decode(y[0].tolist()), end="", flush=True)

        # the following operations have cpu synchronization so they are executed after the print
        retrieve_indices = post_process_draft_tree(
            tree_position_ids=tree_position_ids,
            total_tokens=verify_num_tokens,
            mask_index=mask_index,
        )

        candidates = draft_tokens[0, retrieve_indices]
        (best_candidates, accept_length, sample_p) = evaluate_posterior_greedy(
            base_logits[0, retrieve_indices], candidates
        )

        accept_length = accept_length.item()
        accept_indices = retrieve_indices[best_candidates, 1 : 1 + accept_length]
        accepted_tokens = candidates[best_candidates, 1 : 1 + accept_length]

        accept_candidate(base_cache, accept_indices, accept_length, base_offset)

        for c in draft_cache:
            c.offset = draft_offset

        next_token_id = sample_p.argmax(-1, keepdims=True)
        y = mx.expand_dims(mx.concatenate((accepted_tokens, next_token_id), 0), 0)

        n += 1 + accept_length

    performer.toc("total")
    print()
    performer.print_all_averages_ms(rounding_digits=3)
    print("Generated Tokens", n)
    print("Mean Tokens Per Step:", n / (i + 1))
    print("Tokens Per Second:", round(n / performer.counters["total"][-1], 1))


def stream_generate():
    pass


class WiredLimitModel(nn.Module):
    def __init__(self, draft_model, base_model):
        super().__init__()
        self.draft_model = draft_model
        self.base_model = base_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using draft and base models with specified parameters.")

    # Model arguments
    parser.add_argument(
        "--draft-model",
        type=str,
        # default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Name of the draft model to use."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        # default="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="Name of the base model to use."
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The user prompt for the model."
    )

    # Generation parameters
    # parser.add_argument(
    #     "--k",
    #     type=int,
    #     default=8,
    #     help="Value of k for generation."
    # )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Value of depth for generation."
    )
    parser.add_argument(
        "--verify-num-tokens",
        type=int,
        default=15,
        help="Number of tokens to verify, has to be 7, 15, 23 or 31."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--max-generation-steps",
        "-N",
        type=int,
        default=300,
        help="Number of steps to generate."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # model_name = "mlx-community/SmolLM2-1.7B-Instruct"
    # model_name = "mlx-community/Llama-3.2-1B-Instruct-bf16"
    # draft_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    # draft_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    # draft_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    # draft_name = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    # draft_name = "mlx-community/Qwen2.5-1B-Instruct-4bit"
    # base_name = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    # base_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    # base_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    # base_name = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"
    # base_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # model, tokenizer = mlx_lm.load(model_name, tokenizer_config={})
    args = parse_args()
    draft_name = args.draft_model
    base_name = args.base_model

    draft_model, tokenizer = load(draft_name, tokenizer_config={})
    base_model, _ = load(base_name, tokenizer_config={})

    def kernel(*args):
        return qmm(*args)[0]

    patch_model(draft_model, kernel)
    patch_model(base_model, kernel)

    use_default_chat_template = True
    if use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    # prompt = "Tell me a joke poem about Harry Potter"
    # prompt = "How many positive whole-number divisors does 196 have?"
    prompt = args.prompt

    system_prompt = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt = mx.array(
        prompt
        if isinstance(prompt, list)
        else tokenizer.encode(prompt, add_special_tokens=False)
    )
    print(tokenizer.decode(prompt.tolist()))

    generation_stream = mx.new_stream(mx.default_device())

    wired_model = WiredLimitModel(draft_model, base_model)
    with wired_limit(wired_model, [generation_stream]):
        generate_step(
            prompt=prompt,
            base_model=base_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            # k=args.k,
            k=8,
            depth=args.depth,
            verify_num_tokens=args.verify_num_tokens,
            max_tokens=args.max_tokens,
            prefill_step_size=512,
            N=args.max_generation_steps,
        )
