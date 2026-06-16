"""Unit tests for mappings_gemma4.py — Gemma 4 tensor name mapping and conventions.

Covers:
  - GLOBAL_MAP / LAYER_SUFFIX_MAP / DENSE_MLP_MAP / MOE_*_MAP name mapping
  - EXPERT_SUFFIXES handling (returns None from gguf_name_to_hf)
  - is_missing_tensor() for multimodal and buffer exclusion
  - apply_conventions() for RMSNorm -1.0 subtraction
"""

from __future__ import annotations

import pytest
import torch

from mappings_gemma4 import (
    DENSE_MLP_MAP,
    EXPERT_SUFFIXES,
    GLOBAL_MAP,
    LAYER_SUFFIX_MAP,
    MISSING_PREFIXES,
    MOE_GATE_MAP,
    MOE_SHARED_MAP,
    apply_conventions,
    gguf_name_to_hf,
    is_missing_tensor,
)


class TestGlobalMap:
    """Tests for the top-level (non-layer) tensor mappings."""

    @pytest.mark.parametrize(
        "gguf_name, expected_hf",
        list(GLOBAL_MAP.items()),
        ids=list(GLOBAL_MAP.keys()),
    )
    def test_global_map_entry(self, gguf_name: str, expected_hf: str):
        assert gguf_name_to_hf(gguf_name) == expected_hf

    def test_global_map_completeness(self):
        for gguf_name in GLOBAL_MAP:
            result = gguf_name_to_hf(gguf_name)
            assert result is not None, f"GLOBAL_MAP entry {gguf_name} returned None"


class TestLayerMaps:
    """Tests for per-layer tensor mappings (blk.N.*) across dense and MoE variants."""

    @pytest.mark.parametrize("layer", [0, 15, 47])
    @pytest.mark.parametrize(
        "suffix, expected_suffix",
        list(LAYER_SUFFIX_MAP.items()),
        ids=list(LAYER_SUFFIX_MAP.keys()),
    )
    def test_layer_suffix_map(self, layer: int, suffix: str, expected_suffix: str):
        gguf_name = f"blk.{layer}.{suffix}"
        expected = f"model.layers.{layer}.{expected_suffix}"
        assert gguf_name_to_hf(gguf_name) == expected

    @pytest.mark.parametrize("layer", [0, 42])
    @pytest.mark.parametrize(
        "suffix, expected_suffix",
        list(DENSE_MLP_MAP.items()),
        ids=list(DENSE_MLP_MAP.keys()),
    )
    def test_dense_mlp_map(self, layer: int, suffix: str, expected_suffix: str):
        gguf_name = f"blk.{layer}.{suffix}"
        expected = f"model.layers.{layer}.{expected_suffix}"
        assert gguf_name_to_hf(gguf_name) == expected

    @pytest.mark.parametrize("layer", [1, 29])
    @pytest.mark.parametrize(
        "suffix, expected_suffix",
        list(MOE_SHARED_MAP.items()),
        ids=list(MOE_SHARED_MAP.keys()),
    )
    def test_moe_shared_map(self, layer: int, suffix: str, expected_suffix: str):
        gguf_name = f"blk.{layer}.{suffix}"
        expected = f"model.layers.{layer}.{expected_suffix}"
        assert gguf_name_to_hf(gguf_name) == expected

    @pytest.mark.parametrize("layer", [1, 29])
    @pytest.mark.parametrize(
        "suffix, expected_suffix",
        list(MOE_GATE_MAP.items()),
        ids=list(MOE_GATE_MAP.keys()),
    )
    def test_moe_gate_map(self, layer: int, suffix: str, expected_suffix: str):
        gguf_name = f"blk.{layer}.{suffix}"
        expected = f"model.layers.{layer}.{expected_suffix}"
        assert gguf_name_to_hf(gguf_name) == expected


class TestEdgeCasesAndExperts:
    """Tests for unknown names and expert tensors."""

    def test_unknown_global_name(self):
        assert gguf_name_to_hf("totally_fake_tensor") is None

    def test_unknown_layer_suffix(self):
        assert gguf_name_to_hf("blk.0.unknown.weight") is None

    def test_empty_string(self):
        assert gguf_name_to_hf("") is None

    def test_expert_suffix_returns_none(self):
        """Expert tensors are handled by the converter splitting logic, so mapping is None."""
        for suffix in EXPERT_SUFFIXES:
            assert gguf_name_to_hf(f"blk.0.{suffix}") is None

    def test_no_duplicate_hf_targets(self):
        """Verify 1:1 mapping mapping holds for standard tensors."""
        all_hf: list[str] = list(GLOBAL_MAP.values())
        for layer in range(2):
            for suffix_map in [LAYER_SUFFIX_MAP, DENSE_MLP_MAP, MOE_SHARED_MAP, MOE_GATE_MAP]:
                all_hf.extend(f"model.layers.{layer}.{hf_suffix}" for hf_suffix in suffix_map.values())
        
        assert len(all_hf) == len(set(all_hf)), "Duplicate HF target names found across maps"


class TestIsMissingTensor:
    """Tests for multimodal and buffer exclusions."""

    def test_visual_prefix(self):
        assert is_missing_tensor("model.vision_tower.vision_model.embeddings.weight") is True

    def test_audio_prefix(self):
        assert is_missing_tensor("model.audio_tower.encoder.layers.0.weight") is True

    def test_inv_freq_rope_buffers(self):
        assert is_missing_tensor("model.layers.0.self_attn.rotary_emb.inv_freq") is True
        assert is_missing_tensor("rope.freqs") is True

    def test_language_model_not_missing(self):
        assert is_missing_tensor("model.layers.0.self_attn.q_proj.weight") is False
        assert is_missing_tensor("lm_head.weight") is False

    def test_empty_string_not_missing(self):
        assert is_missing_tensor("") is False

    def test_case_insensitive(self):
        assert is_missing_tensor("MODEL.VISION_TOWER") is True


class TestApplyConventions:
    """Tests for Gemma 4 specific conventions."""

    def test_norm_weight_subtracts_one(self):
        """Gemma stores norm weights as w+1.0; HF expects w."""
        norm_names = [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
        ]
        
        for hf_name in norm_names:
            tensor = torch.ones(64, dtype=torch.bfloat16) * 3.5
            result = apply_conventions(hf_name, tensor)
            # 3.5 - 1.0 = 2.5
            assert torch.allclose(result.float(), torch.full((64,), 2.5)), f"Failed for {hf_name}"
            assert result.dtype == torch.bfloat16

    def test_non_norm_passthrough(self):
        """Non-norm tensors should pass through unchanged."""
        hf_name = "model.layers.0.self_attn.q_proj.weight"
        tensor = torch.randn(32, 16, dtype=torch.bfloat16)
        result = apply_conventions(hf_name, tensor)
        assert torch.equal(result, tensor)
        assert result.dtype == torch.bfloat16
