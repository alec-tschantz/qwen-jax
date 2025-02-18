from typing import Optional, List
from dataclasses import dataclass

import jax
import equinox as eqx

from jax import Array, numpy as jnp, lax, nn, random


class Embedding(eqx.Module):
    weight: Array


class Linear(eqx.Module):
    weight: Array
    bias: Optional[Array]


class RotaryEmbedding(eqx.Module):
    dim: int
    theta: float


class RMSNorm(eqx.Module):
    weight: Array
    eps: float


class Attention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    num_heads: int
    head_dim: int
    num_key_value_heads: int
    attn_dropout: eqx.nn.Dropout

    
class Dense(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear


class DecoderLayer(eqx.Module):
    self_attn: Attention
    mlp: Dense
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    residual_dropout: eqx.nn.Dropout


class QwenModel(eqx.Module):
    input_proj: Linear
    layers: List[DecoderLayer]
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    output_proj: Linear
    
    embed_dropout_in: eqx.nn.Dropout
    embed_dropout_out: eqx.nn.Dropout


def forward_linear(l: Linear, x: Array) -> Array:
    y = jnp.dot(x, l.weight.T)
    return y + l.bias if l.bias is not None else y


def forward_rotary_embedding(
    r: RotaryEmbedding, hidden: Array, position_ids: Array
) -> tuple[Array, Array]:
    b, s, _ = hidden.shape
    inv_freq = 1.0 / (r.theta ** (jnp.arange(0, r.dim, 2) / r.dim))
    freqs = position_ids.reshape(b, s, 1) * inv_freq[None, None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def forward_rms_norm(r: RMSNorm, hidden: Array) -> Array:
    variance = jnp.mean(hidden**2, axis=-1, keepdims=True)
    x = hidden * lax.rsqrt(variance + r.eps)
    return r.weight * x


def forward_attention(
    a: Attention,
    hidden: Array,
    cos: Array,
    sin: Array,
    attention_mask: Optional[Array],
    *,
    key: random.PRNGKey,
    inference: bool
) -> Array:

    b, seqlen, _ = hidden.shape

    def rotate_half(u: Array) -> Array:
        u1, u2 = jnp.split(u, 2, axis=-1)
        return jnp.concatenate((-u2, u1), axis=-1)

    def apply_rotary_pos_emb(q: Array, k: Array, c: Array, s: Array) -> tuple[Array, Array]:
        c = jnp.expand_dims(c, axis=1)  
        s = jnp.expand_dims(s, axis=1)  
        q_ = (q * c) + (rotate_half(q) * s)
        k_ = (k * c) + (rotate_half(k) * s)
        return q_, k_

    # Projections
    q = forward_linear(a.q_proj, hidden)
    k = forward_linear(a.k_proj, hidden)
    v = forward_linear(a.v_proj, hidden)

    # Reshape for multi-head
    q = q.reshape(b, seqlen, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, seqlen, a.num_key_value_heads, a.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, seqlen, a.num_key_value_heads, a.head_dim).transpose(0, 2, 1, 3)

    # Rotary embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Expand K, V if num_key_value_heads < num_heads
    if a.num_key_value_heads != a.num_heads:
        factor = a.num_heads // a.num_key_value_heads
        k = jnp.repeat(k, repeats=factor, axis=1)
        v = jnp.repeat(v, repeats=factor, axis=1)

    # Attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(a.head_dim)

    # Causal and optional attention mask
    causal_mask = jnp.tril(jnp.ones((seqlen, seqlen)))
    causal_mask = causal_mask[None, None, :, :]
    if attention_mask is not None:
        attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
        mask = jnp.minimum(causal_mask, attention_mask)
    else:
        mask = causal_mask

    scores = jnp.where(mask == 0, float("-inf"), scores)

    # Softmax + dropout on attention probabilities
    probs = nn.softmax(scores, axis=-1)
    # Use subkey for attention dropout
    attn_key, _ = random.split(key, 2)
    probs = a.attn_dropout(probs, key=attn_key, inference=inference)

    # Weighted sum
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
    out = out.transpose(0, 2, 1, 3).reshape(b, seqlen, -1)

    # Final output projection
    return forward_linear(a.o_proj, out)


def forward_mlp(m: Dense, x: Array) -> Array:
    gx = forward_linear(m.gate_proj, x)
    ux = forward_linear(m.up_proj, x)
    return forward_linear(m.down_proj, nn.silu(gx) * ux)


def forward_decoder(
    d: DecoderLayer,
    hidden: Array,
    cos: Array,
    sin: Array,
    attention_mask: Optional[Array],
    *,
    key: random.PRNGKey,
    inference: bool
) -> Array:
    # Self-attention block
    residual = hidden
    hidden = forward_rms_norm(d.input_layernorm, hidden)

    attn_key, residual_key, subkey = random.split(key, 3)
    hidden = forward_attention(
        d.self_attn, hidden, cos, sin, attention_mask, key=attn_key, inference=inference
    )
    hidden = residual + hidden

    # MLP block
    residual = hidden
    hidden = forward_rms_norm(d.post_attention_layernorm, hidden)
    hidden = forward_mlp(d.mlp, hidden)
    hidden = d.residual_dropout(hidden, key=residual_key, inference=inference)
    hidden = residual + hidden
    return hidden


def forward(
    model: QwenModel,
    x: Array,
    attention_mask: Optional[Array] = None,
    position_ids: Optional[Array] = None,
    *,
    key: random.PRNGKey,
    inference: bool = False
) -> Array:
    b, s, _ = x.shape
    if position_ids is None:
        position_ids = jnp.tile(jnp.arange(s)[None, :], (b, 1))

    # Split the key for embedding in/out dropout vs attention/layer dropouts
    key_in, key_layers, key_out = random.split(key, 3)

    # --- Embedding (input) + dropout
    hidden = forward_linear(model.input_proj, x)
    hidden = model.embed_dropout_in(hidden, key=key_in, inference=inference)

    # --- Rotary embeddings
    cos, sin = forward_rotary_embedding(model.rotary_emb, hidden, position_ids)

    # --- Decoder layers
    layer_key_seq = random.split(key_layers, len(model.layers))
    for layer, layer_key in zip(model.layers, layer_key_seq):
        hidden = forward_decoder(
            layer,
            hidden,
            cos,
            sin,
            attention_mask,
            key=layer_key,
            inference=inference,
        )

    # --- Final RMSNorm
    hidden = forward_rms_norm(model.norm, hidden)

    # --- Output projection + dropout
    logits = forward_linear(model.output_proj, hidden)
    logits = model.embed_dropout_out(logits, key=key_out, inference=inference)
    return logits
