from typing import Optional, List
from dataclasses import dataclass

import equinox as eqx
from jax import Array, numpy as jnp, lax, nn


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


class Dense(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear


class DecoderLayer(eqx.Module):
    self_attn: Attention
    mlp: Dense
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm


class QwenModel(eqx.Module):
    embed_obj1: Embedding
    embed_obj2: Embedding
    embed_obj3: Embedding
    embed_action: Embedding
    embed_proj: Linear
    layers: List[DecoderLayer]
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    lm_head: Linear


def forward_embedding(e: Embedding, x: Array) -> Array:
    return jnp.take(e.weight, x, axis=0)


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
) -> Array:
    b, seqlen, _ = hidden.shape

    def rotate_half(u: Array) -> Array:
        u1, u2 = jnp.split(u, 2, axis=-1)
        return jnp.concatenate((-u2, u1), axis=-1)

    def apply_rotary_pos_emb(
        q: Array, k: Array, c: Array, s: Array
    ) -> tuple[Array, Array]:
        c = jnp.expand_dims(c, axis=1)
        s = jnp.expand_dims(s, axis=1)
        q_ = (q * c) + (rotate_half(q) * s)
        k_ = (k * c) + (rotate_half(k) * s)
        return q_, k_

    q = forward_linear(a.q_proj, hidden)
    k = forward_linear(a.k_proj, hidden)
    v = forward_linear(a.v_proj, hidden)

    q = q.reshape(b, seqlen, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, seqlen, a.num_key_value_heads, a.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, seqlen, a.num_key_value_heads, a.head_dim).transpose(0, 2, 1, 3)

    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if a.num_key_value_heads != a.num_heads:
        factor = a.num_heads // a.num_key_value_heads
        k = jnp.repeat(k, repeats=factor, axis=1)
        v = jnp.repeat(v, repeats=factor, axis=1)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(a.head_dim)

    cmask = jnp.tril(jnp.ones((seqlen, seqlen)))
    cmask = cmask[None, None, :, :]
    if attention_mask is not None:
        attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
        mask = jnp.minimum(cmask, attention_mask)
    else:
        mask = cmask

    scores = jnp.where(mask == 0, float("-inf"), scores)
    probs = nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
    out = out.transpose(0, 2, 1, 3).reshape(b, seqlen, -1)

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
) -> Array:
    residual = hidden
    hidden = forward_rms_norm(d.input_layernorm, hidden)
    hidden = forward_attention(d.self_attn, hidden, cos, sin, attention_mask)
    hidden = residual + hidden

    residual = hidden
    hidden = forward_rms_norm(d.post_attention_layernorm, hidden)
    hidden = forward_mlp(d.mlp, hidden)
    hidden = residual + hidden
    return hidden


def forward(
    model: QwenModel,
    obj1_ids: Array,
    obj2_ids: Array,
    obj3_ids: Array,
    action_ids: Array,
    attention_mask: Optional[Array] = None,
    position_ids: Optional[Array] = None,
) -> Array:
    b, s = obj1_ids.shape
    if position_ids is None:
        position_ids = jnp.tile(jnp.arange(s)[None, :], (b, 1))

    emb1 = forward_embedding(model.embed_obj1, obj1_ids)
    emb2 = forward_embedding(model.embed_obj2, obj2_ids)
    emb3 = forward_embedding(model.embed_obj3, obj3_ids)
    embA = forward_embedding(model.embed_action, action_ids)

    combined = jnp.concatenate([emb1, emb2, emb3, embA], axis=-1)
    hidden = forward_linear(model.embed_proj, combined)

    cos, sin = forward_rotary_embedding(model.rotary_emb, hidden, position_ids)
    for layer in model.layers:
        hidden = forward_decoder(layer, hidden, cos, sin, attention_mask)
    hidden = forward_rms_norm(model.norm, hidden)
    return forward_linear(model.lm_head, hidden)


def generate(
    model: QwenModel,
    obj1_tokens: Array,
    obj2_tokens: Array,
    obj3_tokens: Array,
    action_tokens: Array,
    max_tokens: int,
) -> tuple[Array, Array, Array]:
    for i in range(max_tokens):
        T = obj1_tokens.shape[1]
        current_action_tokens = action_tokens[:, :T]

        logits = forward(
            model, obj1_tokens, obj2_tokens, obj3_tokens, current_action_tokens
        )
        last_logits = logits[:, -1, :]
        logits_obj1, logits_obj2, logits_obj3 = jnp.split(last_logits, 3, axis=-1)

        next_obj1 = jnp.argmax(logits_obj1, axis=-1)
        next_obj2 = jnp.argmax(logits_obj2, axis=-1)
        next_obj3 = jnp.argmax(logits_obj3, axis=-1)

        obj1_tokens = jnp.concatenate([obj1_tokens, next_obj1[:, None]], axis=1)
        obj2_tokens = jnp.concatenate([obj2_tokens, next_obj2[:, None]], axis=1)
        obj3_tokens = jnp.concatenate([obj3_tokens, next_obj3[:, None]], axis=1)

    return obj1_tokens, obj2_tokens, obj3_tokens
