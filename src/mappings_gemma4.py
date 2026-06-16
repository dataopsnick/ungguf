"""Shared tensor name mappings, constants, and helpers for Gemma 4 architecture.

Used by both the converter (gguf_to_safetensors_gemma4.py) and the
verifier (verify_conversion_gemma4.py) to ensure mapping consistency.

Gemma 4 uses the Gemma 2-style architecture with sliding window / global attention 
interleaving, and adds native multimodal capabilities (vision and audio encoders).
It features both Dense and Mixture-of-Experts (MoE) variants.

Key features:
  - 4x RMSNorms per layer (pre/post attention and pre/post FFN).
  - Norm weights stored as (w + 1.0) in GGUF -> needs subtraction.
  - Multimodal tensors (vision, audio) and buffers (RoPE) are missing from GGUF text models.
  - MoE variants include shared experts and routed experts.
"""

from __future__ import annotations

import re
import torch

# --- Global tensor mappings ---
GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

LAYER_SUFFIX_MAP = {
    # Gemma 4 uses 4x RMSNorms per layer
    "attn_norm.weight": "input_layernorm.weight",
    "attn_post_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "ffn_post_norm.weight": "post_feedforward_layernorm.weight",
    
    # Attention projections
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
}

DENSE_MLP_MAP = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

MOE_SHARED_MAP = {
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}

MOE_GATE_MAP = {
    "ffn_gate_inp.weight": "mlp.gate.weight",
}

EXPERT_SUFFIXES = frozenset({
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
})

EXPERT_HF_MAP = {
    "ffn_gate_exps.weight": "gate_proj.weight",
    "ffn_up_exps.weight": "up_proj.weight",
    "ffn_down_exps.weight": "down_proj.weight",
}

MISSING_PREFIXES = ("vision", "audio", "inv_freq", "rope")


def gguf_name_to_hf(gguf_name: str) -> str | None:
    """Map a GGUF tensor name to its HuggingFace equivalent for Gemma 4.
    
    Returns None for expert tensors (ffn_*_exps) which are handled separately 
    if MoE stacking splitting is required.
    """
    if gguf_name in GLOBAL_MAP:
        return GLOBAL_MAP[gguf_name]
    
    m = re.match(r"blk\.(\d+)\.(.*)", gguf_name)
    if m:
        layer_num, suffix = m.group(1), m.group(2)
        
        if suffix in EXPERT_SUFFIXES:
            return None
        if suffix in LAYER_SUFFIX_MAP:
            return f"model.layers.{layer_num}.{LAYER_SUFFIX_MAP[suffix]}"
        if suffix in DENSE_MLP_MAP:
            return f"model.layers.{layer_num}.{DENSE_MLP_MAP[suffix]}"
        if suffix in MOE_SHARED_MAP:
            return f"model.layers.{layer_num}.{MOE_SHARED_MAP[suffix]}"
        if suffix in MOE_GATE_MAP:
            return f"model.layers.{layer_num}.{MOE_GATE_MAP[suffix]}"
            
    return None


def is_missing_tensor(hf_name: str) -> bool:
    """Check if a tensor belongs to multimodal encoders or buffers (absent from GGUF)."""
    lower = hf_name.lower()
    return any(p in lower for p in MISSING_PREFIXES)


def apply_conventions(hf_name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Apply Gemma 4-specific value conventions to a tensor.
    
    GGUF (llama.cpp) stores Gemma RMSNorm weights centered around 1.0 (w + 1.0).
    HuggingFace computes Gemma2RMSNorm as x * (1.0 + weight), so it expects 
    weights centered around 0.0. We subtract 1.0 to restore the HF convention.
    """
    if "norm.weight" in hf_name or "layernorm.weight" in hf_name:
        orig_dtype = tensor.dtype
        tensor = (tensor.float() - 1.0).to(orig_dtype)
    return tensor
