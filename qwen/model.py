import equinox as eqx
from jax import Array, numpy as jnp, random as jr, lax, nn


class QwenConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.hidden_size = kwargs.get("hidden_size", 896)
        self.intermediate_size = kwargs.get("intermediate_size", 4864)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.num_attention_heads = kwargs.get("num_attention_heads", 14)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.use_cache = kwargs.get("use_cache", True)
        self.head_dim = self.hidden_size // self.num_attention_heads


class QwenEmbedding(eqx.Module):
    weight: Array

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = jr.normal(jr.PRNGKey(0), (num_embeddings, embedding_dim))

    def __call__(self, x: Array) -> Array:
        return jnp.take(self.weight, x, axis=0)


class QwenLinear(eqx.Module):
    weight: Array
    bias: Array | None

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        self.weight = jr.normal(jr.PRNGKey(0), (out_features, in_features))
        self.bias = jr.normal(jr.PRNGKey(1), (out_features,)) if bias else None

    def __call__(self, x: Array) -> Array:
        y = jnp.dot(x, self.weight.T)
        return y + self.bias if self.bias is not None else y


class QwenRotaryEmbedding(eqx.Module):
    inv_freq: Array
    max_seq_len_cached: int
    rope_theta: float

    def __init__(self, config: QwenConfig):
        dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len_cached = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        inv_freq = 1.0 / (
            self.rope_theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim)
        )
        self.inv_freq = inv_freq

    def __call__(
        self, hidden_states: Array, position_ids: Array
    ) -> tuple[Array, Array]:
        batch, seq_len, _ = hidden_states.shape
        t = position_ids.astype(jnp.float32).reshape(batch, seq_len, 1)
        inv_freq_ = self.inv_freq[None, None, :]
        freqs = t * inv_freq_

        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos.astype(hidden_states.dtype), sin.astype(hidden_states.dtype)


class QwenRMSNorm(eqx.Module):
    weight: Array
    eps: float

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        self.weight = jnp.ones(hidden_size)
        self.eps = eps

    def __call__(self, hidden_states: Array) -> Array:
        variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * lax.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class QwenAttention(eqx.Module):
    q_proj: QwenLinear
    k_proj: QwenLinear
    v_proj: QwenLinear
    o_proj: QwenLinear
    num_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int

    def __init__(self, config: QwenConfig):
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = QwenLinear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = QwenLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = QwenLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = QwenLinear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def __call__(
        self,
        hidden_states: Array,
        cos: Array,
        sin: Array,
        attention_mask: Array | None = None,
    ) -> Array:

        def rotate_half(u: Array) -> Array:
            u1, u2 = jnp.split(u, 2, axis=-1)
            return jnp.concatenate((-u2, u1), axis=-1)

        def apply_rotary_pos_emb(q: Array, k: Array, cos_: Array, sin_: Array):
            cos_ = jnp.expand_dims(cos_, axis=1)
            sin_ = jnp.expand_dims(sin_, axis=1)
            q_embed = (q * cos_) + (rotate_half(q) * sin_)
            k_embed = (k * cos_) + (rotate_half(k) * sin_)
            return q_embed, k_embed

        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_key_value_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_key_value_heads != self.num_heads:
            factor = self.num_heads // self.num_key_value_heads
            k = jnp.repeat(k, repeats=factor, axis=1)
            v = jnp.repeat(v, repeats=factor, axis=1)

        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(self.head_dim)

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask[None, None, :, :]

        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
            combined_mask = jnp.minimum(causal_mask, attention_mask)
        else:
            combined_mask = causal_mask

        attn_scores = jnp.where(combined_mask == 0, float("-inf"), attn_scores)
        attn_probs = nn.softmax(attn_scores, axis=-1)

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


class QwenMLP(eqx.Module):
    gate_proj: QwenLinear
    up_proj: QwenLinear
    down_proj: QwenLinear

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_proj = QwenLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = QwenLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = QwenLinear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: Array) -> Array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(eqx.Module):
    self_attn: QwenAttention
    mlp: QwenMLP
    input_layernorm: QwenRMSNorm
    post_attention_layernorm: QwenRMSNorm

    def __init__(self, config: QwenConfig):
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: Array,
        cos: Array,
        sin: Array,
        attention_mask: Array | None = None,
    ) -> Array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QwenModel(eqx.Module):
    embed_tokens: QwenEmbedding
    layers: list[QwenDecoderLayer]
    norm: QwenRMSNorm
    rotary_emb: QwenRotaryEmbedding

    def __init__(self, config: QwenConfig):
        self.embed_tokens = QwenEmbedding(config.vocab_size, config.hidden_size)
        self.layers = [
            QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = QwenRotaryEmbedding(config)

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
    ) -> Array:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class QwenForCausalLM(eqx.Module):
    model: QwenModel
    lm_head: QwenLinear

    def __init__(self, config: QwenConfig) -> None:
        self.model = QwenModel(config)
        self.lm_head = QwenLinear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
    ) -> Array:
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        return logits
