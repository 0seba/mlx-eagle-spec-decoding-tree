import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

from mlx_utils import load
from mlx_eagle import tree_decoding, evaluate_posterior_greedy, accept_candidate

from mlx_model_fwd_bench import patch_model
from qmm_kernel import matmul as qmm

# Default unmodified chat template
# CHAT_TEMPLATE = (
#     '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n',
# )


def test_greedy_decode_and_accept():
    # model_name = "mlx-community/SmolLM2-1.7B-Instruct"
    model_name = "mlx-community/Llama-3.2-1B-Instruct-bf16"
    # model, tokenizer = mlx_lm.load(model_name, tokenizer_config={})
    model, tokenizer = load(model_name, tokenizer_config={})

    def kernel(*args):
        return qmm(*args)[0]

    patch_model(model, kernel)

    use_default_chat_template = True
    if use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    messages = [{"role": "user", "content": "Tell me a joke poem about Harry Potter"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        date_string="03 Jan 2025",
        tokenize=False,
    )
    input_ids = mx.array(
        prompt
        if isinstance(prompt, list)
        else tokenizer.encode(prompt, add_special_tokens=False)
    )

    vanilla_cache = make_prompt_cache(model)
    spec_cache = make_prompt_cache(model)

    y = mx.array(input_ids)
    prefill_step_size = 256
    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], cache=vanilla_cache)
        model(y[:prefill_step_size][None], cache=spec_cache)
        mx.eval([c.state for c in vanilla_cache])
        mx.eval([c.state for c in spec_cache])
        y = y[prefill_step_size:]
        mx.metal.clear_cache()

    model(y[None], cache=spec_cache)
    logits = model(y[None], cache=vanilla_cache)

    mx.eval([c.state for c in vanilla_cache])
    mx.eval([c.state for c in spec_cache])

    # y = logits[:, [-1]].argmax(-1)
    # print(y, tokenizer.decode(y.item()))
    # for _ in range(10):
    #     # y = draft_tokens[:, retrieve_indices[0]]
    #     logits = model(y, cache=vanilla_cache)
    #     y = logits[:, [-1]].argmax(-1)
    #     mx.eval(y)
    #     print(y, tokenizer.decode(y.item()))

    draft = ["Here", "'s", " a", " Harry", " joke", " Potter", " poem", " joke"]
    draft_tokens = mx.array(
        tokenizer._tokenizer(
            draft, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
    ).T
    retrieve_indices = mx.array([[0, 1, 2, 4, 6, -1], [0, 1, 2, 3, 5, 7]])
    correct_draft_index = 1

    y = draft_tokens[:, retrieve_indices[correct_draft_index]]
    logits = model(y, cache=vanilla_cache)
    next_tokens = logits[0].argmax(-1)
    target_tokens = mx.concatenate(
        (
            draft_tokens[0, retrieve_indices[correct_draft_index]][1:],
            mx.array(
                tokenizer._tokenizer([" poem"], add_special_tokens=False)["input_ids"]
            )[:, 0],
        )
    )
    assert mx.all(next_tokens == target_tokens)
    new_offset = vanilla_cache[0].offset

    tree_mask = mx.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 1],
        ],
        # dtype=mx.bfloat16,
        # dtype=mx.float32,
        dtype=mx.bool_,
    ) # .astype(mx.bfloat16)
    tree_mask = mx.where(tree_mask, mx.array(0.0, dtype=mx.bfloat16), mx.array(-mx.inf, dtype=mx.bfloat16))
    tree_position_ids = mx.array([0, 1, 2, 3, 3, 4, 4, 5])
    offset = spec_cache[0].offset
    tree_logits = tree_decoding(
        draft_tokens, tree_mask, tree_position_ids, model, spec_cache
    )

    candidates = draft_tokens[0, retrieve_indices]
    (best_candidates, new_accept_length, sample_p) = evaluate_posterior_greedy(
        tree_logits[0, retrieve_indices], candidates
    )
    new_accept_length = new_accept_length.item()
    assert sample_p.argmax().item() == 33894
    assert new_accept_length == 5
    assert best_candidates.item() == correct_draft_index

    # accept_indices = retrieve_indices[best_candidates.item()]
    accept_indices = mx.take(retrieve_indices, best_candidates, axis=0)
    accept_indices = accept_indices[1: 1 + new_accept_length]
    # accept_indices = mx.slice(accept_indices, 1 : 1 + new_accept_length)
    accept_candidate(
        spec_cache, accept_indices, new_accept_length, offset)

    for gt_cache_i, spec_cache_i in zip(vanilla_cache, spec_cache):
        assert gt_cache_i.offset == spec_cache_i.offset

    mx.eval([c.state for c in vanilla_cache])
    mx.eval([c.state for c in spec_cache])

    new_offset = vanilla_cache[0].offset
    # test individually each position for fine grained detail of the issue
    # separate loop for keys and values for further detail
    for _offset in range(new_offset):
        for i, (gt_cache_i, spec_cache_i) in enumerate(zip(vanilla_cache, spec_cache)):

            # assert mx.allclose(
            #     gt_cache_i.keys[:, :, :new_offset], spec_cache_i.keys[:, :, :new_offset]
            # )
            # assert mx.allclose(
            #     gt_cache_i.values[:, :, :new_offset], spec_cache_i.values[:, :, :new_offset]
            # )
            # assert mx.allclose(
            #     gt_cache_i.keys[:, :, _offset], spec_cache_i.keys[:, :, _offset]
            # )
            assert mx.allclose(
                gt_cache_i.values[:, :, _offset], spec_cache_i.values[:, :, _offset]
            )
    for _offset in range(new_offset):
        for i, (gt_cache_i, spec_cache_i) in enumerate(zip(vanilla_cache, spec_cache)):

            # assert mx.allclose(
            #     gt_cache_i.keys[:, :, :new_offset], spec_cache_i.keys[:, :, :new_offset]
            # )
            # assert mx.allclose(
            #     gt_cache_i.values[:, :, :new_offset], spec_cache_i.values[:, :, :new_offset]
            # )
            assert mx.allclose(
                gt_cache_i.keys[:, :, _offset], spec_cache_i.keys[:, :, _offset]
            )
            # assert mx.allclose(
            #     gt_cache_i.values[:, :, _offset], spec_cache_i.values[:, :, _offset]
            # )

        # print((gt_cache_i.keys[:, :6, offset:offset + 6] != spec_cache_i.keys[:, :6, offset + accept_indices]).sum(-1))
        # print((gt_cache_i.values[:, :6, offset:offset + 6] != spec_cache_i.values[:, :6, offset + accept_indices]).sum(-1))

        # print(i)

    print("test_greedy_decode_and_accept success")

if __name__ == "__main__":
    test_greedy_decode_and_accept()
