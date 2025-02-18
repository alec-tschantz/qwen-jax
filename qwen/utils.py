import equinox as eqx
from jax import Array, numpy as jnp, random as jr

from .model import (
    Linear,
    RotaryEmbedding,
    RMSNorm,
    Attention,
    Dense,
    DecoderLayer,
    QwenModel,
)


def init(
    key,
    input_dim=15,
    output_dim=15,
    hidden_size=256,
    num_layers=2,
    num_heads=4,
    num_key_value_heads=4,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
) -> QwenModel:
    keys = jr.split(key, 3 + num_layers)
    inp_proj = init_linear(keys[0], input_dim, hidden_size)
    final_norm = init_rms_norm(hidden_size, rms_norm_eps)
    head_dim = hidden_size // num_heads
    rot_emb = init_rotary_embedding(head_dim, rope_theta)
    layers = [
        init_decoder_layer(
            keys[i + 1],
            hidden_size,
            num_heads,
            num_key_value_heads,
            rms_norm_eps,
        )
        for i in range(num_layers)
    ]
    out_proj = init_linear(keys[-1], hidden_size, output_dim, bias=True)
    return QwenModel(inp_proj, layers, final_norm, rot_emb, out_proj)


def init_linear(
    key: jr.PRNGKey, in_dim: int, out_dim: int, bias: bool = True
) -> Linear:
    k1, k2 = jr.split(key)
    weight = jr.normal(k1, (out_dim, in_dim)) * jnp.sqrt(2.0 / (in_dim + out_dim))
    b = jr.normal(k2, (out_dim,)) * 0.01 if bias else None
    return Linear(weight=weight, bias=b)


def init_rms_norm(hidden_dim: int, eps: float) -> RMSNorm:
    weight = jnp.ones((hidden_dim,))
    return RMSNorm(weight=weight, eps=eps)


def init_rotary_embedding(head_dim: int, rope_theta: float) -> RotaryEmbedding:
    return RotaryEmbedding(dim=head_dim, theta=rope_theta)


def init_attention(
    key: jr.PRNGKey, hidden_size: int, num_heads: int, num_key_value_heads: int
) -> Attention:
    head_dim = hidden_size // num_heads
    k1, k2, k3, k4 = jr.split(key, 4)
    q_proj = init_linear(k1, hidden_size, hidden_size)
    k_proj = init_linear(k2, hidden_size, hidden_size)
    v_proj = init_linear(k3, hidden_size, hidden_size)
    o_proj = init_linear(k4, hidden_size, hidden_size, bias=False)
    return Attention(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        num_heads=num_heads,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
    )


def init_dense(key: jr.PRNGKey, hidden_size: int) -> Dense:
    k1, k2, k3 = jr.split(key, 3)
    gate_proj = init_linear(k1, hidden_size, hidden_size * 2, bias=False)
    up_proj = init_linear(k2, hidden_size, hidden_size * 2, bias=False)
    down_proj = init_linear(k3, hidden_size * 2, hidden_size, bias=False)
    return Dense(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)


def init_decoder_layer(
    key: jr.PRNGKey,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    rms_norm_eps: float,
) -> DecoderLayer:
    k1, k2, k3, k4 = jr.split(key, 4)
    attn = init_attention(k1, hidden_size, num_heads, num_key_value_heads)
    mlp = init_dense(k2, hidden_size)
    in_ln = init_rms_norm(hidden_size, rms_norm_eps)
    post_ln = init_rms_norm(hidden_size, rms_norm_eps)
    return DecoderLayer(
        self_attn=attn, mlp=mlp, input_layernorm=in_ln, post_attention_layernorm=post_ln
    )
