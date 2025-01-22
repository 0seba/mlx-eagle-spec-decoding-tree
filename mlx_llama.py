# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.llama import LlamaModel
from mlx_lm.models.rope_utils import initialize_rope

from mlx_arrayed_rope_kernel import rope_positions

# from mlx_rope import LlamaRotaryEmbedding, apply_rotary_pos_emb


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            if not "factor" in self.rope_scaling:
                raise ValueError(f"rope_scaling must contain 'factor'")
            rope_type = self.rope_scaling.get("type") or self.rope_scaling.get(
                "rope_type"
            )
            if rope_type is None:
                raise ValueError(
                    f"rope_scaling must contain either 'type' or 'rope_type'"
                )
            if rope_type not in ["linear", "dynamic", "llama3"]:
                raise ValueError(
                    "rope_scaling 'type' currently only supports 'linear', 'dynamic' or 'llama3'"
                )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        elif args.model_type == "qwen2":
            attention_bias = True
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.fast_rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

        # self.slow_rope = LlamaRotaryEmbedding(
        #     args.rope_theta,
        #     head_dim,
        #     args.rope_scaling,
        #     max_position_embeddings=args.max_position_embeddings,
        # )

        if args.rope_scaling is not None:
            rope_type = args.rope_scaling.get("type") or args.rope_scaling.get(
                "rope_type", "default"
            )
        else:
            rope_type = "default"

        if rope_type in ["default", "linear"]:
            self._scaling_factor = (
                1 / args.rope_scaling["factor"] if rope_type == "linear" else 1.0
            )
            self._inv_freqs = mx.array(2, dtype=mx.uint32) ** (
                -mx.log2(args.rope_theta)
                * mx.arange(0, head_dim, 2)
                / head_dim
            )
        else:
            self._inv_freqs = 1 / self.fast_rope._freqs
            self._scaling_factor = 1.0
        # self.rope = nn.RoPE(
        #     args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        # )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[Union[mx.array, int]] = None,
        flat_rope=False,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            qshape = queries.shape
            kshape = keys.shape
            if flat_rope:
                queries = queries.reshape(qshape[0], -1, 1, qshape[-1])
                keys = keys.reshape(kshape[0], -1, 1, kshape[-1])

            if type(position_ids) is mx.array:
                # cos, sin = self.slow_rope(queries, position_ids)
                # queries, keys = apply_rotary_pos_emb(
                #     queries, keys, cos, sin, position_ids
                # )
                queries = rope_positions(
                    queries,
                    position_ids,
                    self._inv_freqs,
                    scaling_factor=self._scaling_factor,
                )
                keys = rope_positions(
                    keys,
                    position_ids,
                    self._inv_freqs,
                    scaling_factor=self._scaling_factor,
                )
            else:
                queries = self.fast_rope(
                    queries,
                    offset=cache.offset if position_ids is None else position_ids,
                    # flat_rope=flat_rope,
                )
                keys = self.fast_rope(
                    keys,
                    offset=cache.offset if position_ids is None else position_ids,
                    # flat_rope=flat_rope,
                )

            if flat_rope:
                queries = queries.reshape(qshape)
                keys = keys.reshape(kshape)

            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # return values

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        # return output

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[Union[mx.array, int]] = None,
        flat_rope=False,
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            position_ids=position_ids,
            flat_rope=flat_rope,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[Union[mx.array, int]] = None,
        flat_rope=False,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)
        else:
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c, position_ids=position_ids, flat_rope=flat_rope)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[Union[mx.array, int]] = None,
        flat_rope=False,
    ):
        out = self.model(
            inputs, cache, mask=mask, position_ids=position_ids, flat_rope=flat_rope
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers
