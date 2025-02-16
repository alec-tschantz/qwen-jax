import torch

import equinox as eqx
from jax import Array, numpy as jnp

from .model import (
    Embedding,
    Linear,
    RotaryEmbedding,
    RMSNorm,
    Attention,
    Dense,
    DecoderLayer,
    QwenModel,
)


def torch_to_jax(tensor: torch.Tensor) -> Array:
    return jnp.array(tensor.detach().numpy())


def from_hf(hf_model: torch.nn.Module) -> QwenModel:
    cfg = hf_model.config

    embed = Embedding(weight=torch_to_jax(hf_model.model.embed_tokens.weight))

    final_norm = RMSNorm(
        weight=torch_to_jax(hf_model.model.norm.weight),
        eps=hf_model.config.rms_norm_eps,
    )

    hidden_size = hf_model.config.hidden_size
    num_heads = hf_model.config.num_attention_heads
    dim = hidden_size // num_heads
    inv_freq = 1.0 / (
        hf_model.config.rope_theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim)
    )
    rot_emb = RotaryEmbedding(inv_freq=inv_freq)

    layers_out = []
    for i, hf_layer in enumerate(hf_model.model.layers):

        q_proj = Linear(
            weight=torch_to_jax(hf_layer.self_attn.q_proj.weight),
            bias=torch_to_jax(hf_layer.self_attn.q_proj.bias),
        )
        k_proj = Linear(
            weight=torch_to_jax(hf_layer.self_attn.k_proj.weight),
            bias=torch_to_jax(hf_layer.self_attn.k_proj.bias),
        )
        v_proj = Linear(
            weight=torch_to_jax(hf_layer.self_attn.v_proj.weight),
            bias=torch_to_jax(hf_layer.self_attn.v_proj.bias),
        )
        o_proj = Linear(
            weight=torch_to_jax(hf_layer.self_attn.o_proj.weight),
            bias=None,
        )
        attn_struct = Attention(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            num_heads=hf_model.config.num_attention_heads,
            num_key_value_heads=hf_model.config.num_key_value_heads,
            head_dim=(
                hf_model.config.hidden_size // hf_model.config.num_attention_heads
            ),
        )

        mlp_struct = Dense(
            gate_proj=Linear(
                weight=torch_to_jax(hf_layer.mlp.gate_proj.weight), bias=None
            ),
            up_proj=Linear(weight=torch_to_jax(hf_layer.mlp.up_proj.weight), bias=None),
            down_proj=Linear(
                weight=torch_to_jax(hf_layer.mlp.down_proj.weight), bias=None
            ),
        )

        in_ln = RMSNorm(
            weight=torch_to_jax(hf_layer.input_layernorm.weight),
            eps=hf_model.config.rms_norm_eps,
        )
        post_ln = RMSNorm(
            weight=torch_to_jax(hf_layer.post_attention_layernorm.weight),
            eps=hf_model.config.rms_norm_eps,
        )
        layers_out.append(
            DecoderLayer(
                self_attn=attn_struct,
                mlp=mlp_struct,
                input_layernorm=in_ln,
                post_attention_layernorm=post_ln,
            )
        )

    lm_head = Linear(weight=torch_to_jax(hf_model.lm_head.weight), bias=None)
    return QwenModel(
        embed_tokens=embed,
        layers=layers_out,
        norm=final_norm,
        rotary_emb=rot_emb,
        lm_head=lm_head,
    )
