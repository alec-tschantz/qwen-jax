import torch

import equinox as eqx
from jax import Array, numpy as jnp

from .model import QwenForCausalLM


def torch_to_jax(tensor: torch.Tensor) -> Array:
    return jnp.array(tensor.detach().numpy())


def convert_hf(hf_model: torch.nn.Module, model: QwenForCausalLM) -> QwenForCausalLM:
    model = eqx.tree_at(
        lambda t: t.model.embed_tokens.weight,
        model,
        torch_to_jax(hf_model.model.embed_tokens.weight),
    )

    model = eqx.tree_at(
        lambda t: t.model.norm.weight,
        model,
        torch_to_jax(hf_model.model.norm.weight),
    )

    model = eqx.tree_at(
        lambda t: t.lm_head.weight,
        model,
        torch_to_jax(hf_model.lm_head.weight),
    )

    for i, layer in enumerate(model.model.layers):
        hf_layer = hf_model.model.layers[i]

        model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.q_proj.weight,
            model,
            torch_to_jax(hf_layer.self_attn.q_proj.weight),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.q_proj.bias,
            model,
            torch_to_jax(hf_layer.self_attn.q_proj.bias),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.k_proj.weight,
            model,
            torch_to_jax(hf_layer.self_attn.k_proj.weight),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.k_proj.bias,
            model,
            torch_to_jax(hf_layer.self_attn.k_proj.bias),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.v_proj.weight,
            model,
            torch_to_jax(hf_layer.self_attn.v_proj.weight),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.v_proj.bias,
            model,
            torch_to_jax(hf_layer.self_attn.v_proj.bias),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.o_proj.weight,
            model,
            torch_to_jax(hf_layer.self_attn.o_proj.weight),
        )

        model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.gate_proj.weight,
            model,
            torch_to_jax(hf_layer.mlp.gate_proj.weight),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.up_proj.weight,
            model,
            torch_to_jax(hf_layer.mlp.up_proj.weight),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.down_proj.weight,
            model,
            torch_to_jax(hf_layer.mlp.down_proj.weight),
        )

        model = eqx.tree_at(
            lambda t: t.model.layers[i].input_layernorm.weight,
            model,
            torch_to_jax(hf_layer.input_layernorm.weight),
        )
        model = eqx.tree_at(
            lambda t: t.model.layers[i].post_attention_layernorm.weight,
            model,
            torch_to_jax(hf_layer.post_attention_layernorm.weight),
        )

    return model
