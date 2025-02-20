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

    emb_dim = hf_model.config.hidden_size // hf_model.config.num_attention_heads
    rot_emb = RotaryEmbedding(theta=hf_model.config.rope_theta, dim=emb_dim)

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


# def init(
#     key,
#     input_dim=9,
#     output_dim=9,
#     hidden_size=256,
#     num_layers=4,
#     num_heads=4,
#     num_key_value_heads=4,
#     rope_theta=5000.0,
#     rms_norm_eps=1e-5,
#     dropout=0.05,
# ) -> QwenModel:
#     keys = jr.split(key, 3 + num_layers)

#     inp_proj = init_linear(keys[0], input_dim, hidden_size)
#     final_norm = init_rms_norm(hidden_size, rms_norm_eps)
#     head_dim = hidden_size // num_heads
#     rot_emb = init_rotary_embedding(head_dim, rope_theta)

#     layers = [
#         init_decoder_layer(
#             keys[i + 1],
#             hidden_size,
#             num_heads,
#             num_key_value_heads,
#             rms_norm_eps,
#             dropout,
#         )
#         for i in range(num_layers)
#     ]

#     out_proj = init_linear(keys[-1], hidden_size, output_dim, bias=True)

#     embed_dropout_in = eqx.nn.Dropout(p=dropout, inference=False)
#     embed_dropout_out = eqx.nn.Dropout(p=dropout, inference=False)

#     return QwenModel(
#         input_proj=inp_proj,
#         layers=layers,
#         norm=final_norm,
#         rotary_emb=rot_emb,
#         output_proj=out_proj,
#         embed_dropout_in=embed_dropout_in,
#         embed_dropout_out=embed_dropout_out,
#     )


# def init_linear(
#     key: jr.PRNGKey, in_dim: int, out_dim: int, bias: bool = True
# ) -> Linear:
#     k1, k2 = jr.split(key)
#     weight = jr.normal(k1, (out_dim, in_dim)) * jnp.sqrt(2.0 / (in_dim + out_dim))
#     b = jr.normal(k2, (out_dim,)) * 0.01 if bias else None
#     return Linear(weight=weight, bias=b)


# def init_rms_norm(hidden_dim: int, eps: float) -> RMSNorm:
#     weight = jnp.ones((hidden_dim,))
#     return RMSNorm(weight=weight, eps=eps)


# def init_rotary_embedding(head_dim: int, rope_theta: float) -> RotaryEmbedding:
#     return RotaryEmbedding(dim=head_dim, theta=rope_theta)


# def init_attention(
#     key: jr.PRNGKey,
#     hidden_size: int,
#     num_heads: int,
#     num_key_value_heads: int,
#     dropout: float,
# ) -> Attention:
#     head_dim = hidden_size // num_heads
#     k1, k2, k3, k4 = jr.split(key, 4)
#     q_proj = init_linear(k1, hidden_size, hidden_size)
#     k_proj = init_linear(k2, hidden_size, hidden_size)
#     v_proj = init_linear(k3, hidden_size, hidden_size)
#     o_proj = init_linear(k4, hidden_size, hidden_size, bias=False)

#     attn_dropout = eqx.nn.Dropout(p=dropout, inference=False)

#     return Attention(
#         q_proj=q_proj,
#         k_proj=k_proj,
#         v_proj=v_proj,
#         o_proj=o_proj,
#         num_heads=num_heads,
#         head_dim=head_dim,
#         num_key_value_heads=num_key_value_heads,
#         attn_dropout=attn_dropout,
#     )


# def init_dense(key: jr.PRNGKey, hidden_size: int) -> Dense:
#     k1, k2, k3 = jr.split(key, 3)
#     gate_proj = init_linear(k1, hidden_size, hidden_size * 2, bias=False)
#     up_proj = init_linear(k2, hidden_size, hidden_size * 2, bias=False)
#     down_proj = init_linear(k3, hidden_size * 2, hidden_size, bias=False)
#     return Dense(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)


# def init_decoder_layer(
#     key: jr.PRNGKey,
#     hidden_size: int,
#     num_heads: int,
#     num_key_value_heads: int,
#     rms_norm_eps: float,
#     dropout: float,
# ) -> DecoderLayer:
#     k1, k2, k3, k4 = jr.split(key, 4)
#     attn = init_attention(k1, hidden_size, num_heads, num_key_value_heads, dropout)
#     mlp = init_dense(k2, hidden_size)
#     in_ln = init_rms_norm(hidden_size, rms_norm_eps)
#     post_ln = init_rms_norm(hidden_size, rms_norm_eps)
#     residual_dropout = eqx.nn.Dropout(p=dropout, inference=False)

#     return DecoderLayer(
#         self_attn=attn,
#         mlp=mlp,
#         input_layernorm=in_ln,
#         post_attention_layernorm=post_ln,
#         residual_dropout=residual_dropout,
#     )
