"""Verify GGUF-to-safetensors conversion correctness for Gemma 4.

Self-contained verifier for the Gemma 4 architecture (Dense + MoE).
Applies the -1.0 RMSNorm value convention, splits expert 3D tensors (for MoE variants),
verifies copied multimodal tensors (vision/audio), and compares every tensor against 
the converted safetensors bit-for-bit.

Usage (inside Docker):
    python3 verify_conversion_gemma4.py \
        --gguf /input/model.gguf \
        --converted /converted/ \
        --reference /ref/ \
        --output /results/gemma4_verification.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader

from common import decode_gguf_tensor, load_converted_tensors, load_reference_shapes
from mappings_gemma4 import (
    EXPERT_HF_MAP,
    EXPERT_SUFFIXES,
    apply_conventions,
    gguf_name_to_hf,
    is_missing_tensor,
)

_MAX_DISPLAY_MISMATCH = 50
_MAX_DISPLAY_OTHER = 20


def compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
    gguf_label: str,
    hf_name: str,
    quant_type: str = "",
) -> dict[str, Any]:
    if list(expected.shape) != list(actual.shape):
        return {
            "gguf_name": gguf_label,
            "hf_name": hf_name,
            "error": f"shape mismatch: expected {list(expected.shape)} vs actual {list(actual.shape)}",
            "quant_type": quant_type,
        }

    dtype_note = None
    if expected.dtype != actual.dtype:
        dtype_note = f"dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        match = torch.equal(expected.float(), actual.float())
    else:
        match = torch.equal(expected, actual)

    if match:
        entry: dict[str, Any] = {
            "gguf_name": gguf_label,
            "hf_name": hf_name,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "quant_type": quant_type,
            "status": "MATCH",
        }
        if dtype_note:
            entry["dtype_note"] = dtype_note
        return entry

    diff = (expected.float() - actual.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    num_different = (diff > 0).sum().item()
    total_elements = diff.numel()

    entry = {
        "gguf_name": gguf_label,
        "hf_name": hf_name,
        "shape": list(actual.shape),
        "dtype_expected": str(expected.dtype),
        "dtype_actual": str(actual.dtype),
        "quant_type": quant_type,
        "status": "MISMATCH",
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "num_different_elements": num_different,
        "total_elements": total_elements,
        "pct_different": round(100.0 * num_different / total_elements, 4),
    }
    if dtype_note:
        entry["dtype_note"] = dtype_note

    if num_different > 0:
        flat_diff = diff.flatten()
        worst_idx = int(flat_diff.argmax().item())
        entry["sample_worst"] = {
            "index": worst_idx,
            "expected": expected.flatten()[worst_idx].float().item(),
            "actual": actual.flatten()[worst_idx].float().item(),
        }
        diff_indices = (flat_diff > 0).nonzero(as_tuple=True)[0][:5]
        entry["sample_diffs"] = [
            {
                "index": int(idx.item()),
                "expected": expected.flatten()[int(idx.item())].float().item(),
                "actual": actual.flatten()[int(idx.item())].float().item(),
            }
            for idx in diff_indices
        ]

    return entry


def verify(
    gguf_path: str,
    conv_dir: str,
    ref_dir: str,
    output_path: str | None,
    keep_fp16: bool = False,
):
    print(f"Loading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    gguf_tensors = reader.tensors
    print(f"  GGUF tensors: {len(gguf_tensors)}")

    qtypes: dict[str, int] = {}
    for t in gguf_tensors:
        qt = str(t.tensor_type)
        qtypes[qt] = qtypes.get(qt, 0) + 1
    print(f"  Quant types: {qtypes}")

    print(f"\nLoading reference shapes: {ref_dir}")
    ref_shapes = load_reference_shapes(ref_dir)
    print(f"  Reference tensors: {len(ref_shapes)}")

    print(f"\nLoading converted safetensors: {conv_dir}")
    conv_tensors = load_converted_tensors(conv_dir)
    print(f"  Converted tensors: {len(conv_tensors)}")

    gguf_by_name = {t.name: t for t in gguf_tensors}

    results: dict[str, Any] = {
        "gguf_path": gguf_path,
        "converted_dir": conv_dir,
        "reference_dir": ref_dir,
        "gguf_tensor_count": len(gguf_tensors),
        "converted_tensor_count": len(conv_tensors),
        "quant_types": qtypes,
        "matched": [],
        "mismatched": [],
        "unmapped_gguf": [],
        "missing_in_converted": [],
        "extra_in_converted": [],
        "copied_matched": [],
        "copied_mismatched": [],
    }

    verified_hf_names = set()

    # Phase 1: 1:1 mapped standard tensors
    print("\nPhase 1: Verifying standard 1:1 mapped tensors...")
    phase1_count = 0

    for gt in gguf_tensors:
        hf_name = gguf_name_to_hf(gt.name)
        
        if hf_name is None:
            # Might be an expert tensor, defer to Phase 2
            m = re.match(r"blk\.(\d+)\.(.*)", gt.name)
            if m and m.group(2) in EXPERT_SUFFIXES:
                continue
                
            results["unmapped_gguf"].append(
                {
                    "gguf_name": gt.name,
                    "shape": [int(x) for x in gt.shape],
                    "reason": "no mapping rule",
                }
            )
            continue

        if hf_name not in ref_shapes:
            results["unmapped_gguf"].append(
                {
                    "gguf_name": gt.name,
                    "shape": [int(x) for x in gt.shape],
                    "reason": f"mapped to {hf_name} but not in reference",
                }
            )
            continue

        verified_hf_names.add(hf_name)

        if hf_name not in conv_tensors:
            results["missing_in_converted"].append(
                {
                    "gguf_name": gt.name,
                    "hf_name": hf_name,
                    "shape": [int(x) for x in gt.shape],
                }
            )
            continue

        try:
            decoded = decode_gguf_tensor(gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True)
        except Exception as e:
            results["mismatched"].append(
                {
                    "gguf_name": gt.name,
                    "hf_name": hf_name,
                    "error": f"GGUF decode failed: {e}",
                }
            )
            continue

        target_shape = ref_shapes.get(hf_name)
        if target_shape and list(decoded.shape) != target_shape:
            if decoded.numel() == int(np.prod(target_shape)):
                decoded = decoded.reshape(target_shape).contiguous()
            else:
                results["mismatched"].append(
                    {
                        "gguf_name": gt.name,
                        "hf_name": hf_name,
                        "error": f"element count mismatch: GGUF {decoded.numel()} vs ref {int(np.prod(target_shape))}",
                    }
                )
                continue

        # Apply Gemma 4 conventions (RMSNorm w + 1.0 -> w)
        expected = apply_conventions(hf_name, decoded)
        actual = conv_tensors[hf_name]
        
        result = compare_tensors(expected, actual, gt.name, hf_name, str(gt.tensor_type))

        if result.get("status") == "MATCH":
            results["matched"].append(result)
        else:
            results["mismatched"].append(result)

        phase1_count += 1

    print(f"  Verified {phase1_count} direct 1:1 tensors")

    # Phase 2: Expert tensor splitting (for Gemma 4 MoE variants)
    print("\nPhase 2: Verifying MoE expert tensor splits...")
    phase2_count = 0
    phase2_experts = 0

    expert_tensors = []
    for gt in gguf_tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", gt.name)
        if m and m.group(2) in EXPERT_SUFFIXES:
            expert_tensors.append((gt.name, int(m.group(1)), m.group(2)))

    for gguf_name, layer_num, suffix in expert_tensors:
        gt = gguf_by_name[gguf_name]
        hf_proj_name = EXPERT_HF_MAP[suffix]

        try:
            expert_3d = decode_gguf_tensor(
                gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True
            )
            num_experts = expert_3d.shape[0]

            for expert_i in range(num_experts):
                hf_name = f"model.layers.{layer_num}.mlp.experts.{expert_i}.{hf_proj_name}"
                verified_hf_names.add(hf_name)

                if hf_name not in conv_tensors:
                    results["missing_in_converted"].append(
                        {
                            "gguf_name": gguf_name,
                            "hf_name": hf_name,
                            "expert_index": expert_i,
                        }
                    )
                    continue

                expert_slice = expert_3d[expert_i]

                target_shape = ref_shapes.get(hf_name)
                if target_shape and expert_slice.numel() == int(np.prod(target_shape)):
                    expert_slice = expert_slice.reshape(target_shape).contiguous()

                # Apply conventions to expert slice
                expected = apply_conventions(hf_name, expert_slice)
                actual = conv_tensors[hf_name]
                
                result = compare_tensors(
                    expected,
                    actual,
                    f"{gguf_name}[..., {expert_i}]",
                    hf_name,
                    str(gt.tensor_type),
                )

                if result.get("status") == "MATCH":
                    results["matched"].append(result)
                else:
                    results["mismatched"].append(result)

                phase2_experts += 1

        except Exception as e:
            results["mismatched"].append(
                {
                    "gguf_name": gguf_name,
                    "hf_name": f"model.layers.{layer_num}.mlp.experts.*.{hf_proj_name}",
                    "error": f"expert split failed: {e}",
                }
            )

        phase2_count += 1

    if phase2_count > 0:
        print(f"  Verified {phase2_count} expert 3D tensors -> {phase2_experts} individual experts")
    else:
        print("  No expert tensors found (Dense model detected).")

    # Phase 3: Copied Multimodal & Tied Tensors
    print("\nPhase 3: Verifying copied multimodal/buffer tensors & tied embeddings...")
    phase3_copied = 0
    ref_tensors = load_converted_tensors(ref_dir)
    
    # Check tied lm_head
    if "lm_head.weight" in conv_tensors and "lm_head.weight" not in verified_hf_names:
        if "model.embed_tokens.weight" in conv_tensors:
            actual = conv_tensors["lm_head.weight"]
            expected = conv_tensors["model.embed_tokens.weight"]
            result = compare_tensors(expected, actual, "tied(model.embed_tokens.weight)", "lm_head.weight", "tied")
            
            if result.get("status") == "MATCH":
                results["copied_matched"].append(result)
            else:
                results["copied_mismatched"].append(result)
                
            verified_hf_names.add("lm_head.weight")
            phase3_copied += 1

    for hf_name in list(conv_tensors.keys()):
        if hf_name in verified_hf_names:
            continue
            
        if is_missing_tensor(hf_name):
            if hf_name not in ref_tensors:
                # Buffer like inv_freq may safely be generated dynamically
                verified_hf_names.add(hf_name)
                continue
                
            actual = conv_tensors[hf_name]
            ref = ref_tensors[hf_name]
            result = compare_tensors(ref, actual, "copied_from_ref", hf_name, "copied")
            
            if result.get("status") == "MATCH":
                results["copied_matched"].append(result)
            else:
                results["copied_mismatched"].append(result)
                
            verified_hf_names.add(hf_name)
            phase3_copied += 1

    print(f"  Verified {phase3_copied} copied/tied tensors")

    # Phase 4: Check for extra/missing tensors in converted
    for hf_name in conv_tensors:
        if hf_name not in verified_hf_names:
            results["extra_in_converted"].append(
                {
                    "hf_name": hf_name,
                    "shape": list(conv_tensors[hf_name].shape),
                    "dtype": str(conv_tensors[hf_name].dtype),
                }
            )
            
    for hf_name in ref_shapes:
        if hf_name not in conv_tensors:
            # ignore dynamically generated buffers like inv_freq
            if "inv_freq" not in hf_name and "rope" not in hf_name:
                results["missing_in_converted"].append({"hf_name": hf_name})

    summary = {
        "total_gguf_tensors": len(gguf_tensors),
        "total_converted_tensors": len(conv_tensors),
        "mapped_and_matched": len(results["matched"]),
        "mapped_and_mismatched": len(results["mismatched"]),
        "copied_matched": len(results["copied_matched"]),
        "copied_mismatched": len(results["copied_mismatched"]),
        "unmapped_gguf": len(results["unmapped_gguf"]),
        "missing_in_converted": len(results["missing_in_converted"]),
        "extra_in_converted": len(results["extra_in_converted"]),
        "conversion_correct": (
            len(results["mismatched"]) == 0 and 
            len(results["missing_in_converted"]) == 0 and
            len(results["copied_mismatched"]) == 0
        ),
    }
    results["summary"] = summary

    print(f"\n{'=' * 60}")
    print("GEMMA 4 VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  GGUF tensors:           {summary['total_gguf_tensors']}")
    print(f"  Converted tensors:      {summary['total_converted_tensors']}")
    print(f"  GGUF Matched:           {summary['mapped_and_matched']}")
    print(f"  GGUF MISMATCHED:        {summary['mapped_and_mismatched']}")
    print(f"  Copied Matched:         {summary['copied_matched']}")
    print(f"  Copied MISMATCHED:      {summary['copied_mismatched']}")
    print(f"  Unmapped GGUF:          {summary['unmapped_gguf']}")
    print(f"  Missing in converted:   {summary['missing_in_converted']}")
    print(f"  Extra in converted:     {summary['extra_in_converted']}")
    print(f"  CONVERSION CORRECT:     {'YES' if summary['conversion_correct'] else '*** NO ***'}")
    print(f"{'=' * 60}")

    if results["mismatched"]:
        print(f"\nMISMATCHED GGUF TENSORS ({len(results['mismatched'])}):")
        for m in results["mismatched"][:_MAX_DISPLAY_MISMATCH]:
            print(f"  {m.get('hf_name', m.get('gguf_name', '???'))}")
            if "error" in m:
                print(f"    Error: {m['error']}")
            else:
                print(
                    f"    max_diff={m.get('max_abs_diff', '?'):.6e}, "
                    f"mean_diff={m.get('mean_abs_diff', '?'):.6e}"
                )
                print(
                    f"    {m.get('num_different_elements', '?')}/{m.get('total_elements', '?')} "
                    f"elements differ ({m.get('pct_different', '?')}%)"
                )
                if "sample_worst" in m:
                    s = m["sample_worst"]
                    print(f"    worst: expected={s['expected']:.8f}, actual={s['actual']:.8f}")
        if len(results["mismatched"]) > _MAX_DISPLAY_MISMATCH:
            print(f"  ... and {len(results['mismatched']) - _MAX_DISPLAY_MISMATCH} more")

    if results["copied_mismatched"]:
        print(f"\nMISMATCHED COPIED TENSORS ({len(results['copied_mismatched'])}):")
        for m in results["copied_mismatched"][:_MAX_DISPLAY_MISMATCH]:
            print(f"  {m.get('hf_name')}")
            print(
                f"    max_diff={m.get('max_abs_diff', '?'):.6e}, "
                f"mean_diff={m.get('mean_abs_diff', '?'):.6e}"
            )
            
    if results["unmapped_gguf"]:
        print(f"\nUNMAPPED GGUF TENSORS ({len(results['unmapped_gguf'])}):")
        for u in results["unmapped_gguf"]:
            print(f"  {u['gguf_name']}  shape={u.get('shape')}  reason={u.get('reason')}")

    if results["missing_in_converted"]:
        print(f"\nMISSING IN CONVERTED ({len(results['missing_in_converted'])}):")
        for m in results["missing_in_converted"][:_MAX_DISPLAY_OTHER]:
            print(f"  {m['hf_name']}")

    if results["extra_in_converted"]:
        print(f"\nEXTRA IN CONVERTED ({len(results['extra_in_converted'])}):")
        for entry in results["extra_in_converted"][:_MAX_DISPLAY_OTHER]:
            print(f"  {entry['hf_name']}  shape={entry['shape']}")

    if output_path:
        output_results = dict(results)
        output_results["matched"] = [
            {
                "gguf_name": m["gguf_name"],
                "hf_name": m["hf_name"],
                "quant_type": m.get("quant_type", ""),
            }
            for m in results["matched"]
        ]
        output_results["copied_matched"] = [
            {
                "gguf_name": m["gguf_name"],
                "hf_name": m["hf_name"],
            }
            for m in results["copied_matched"]
        ]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return summary["conversion_correct"]


def main():
    parser = argparse.ArgumentParser(
        description="Verify GGUF-to-safetensors conversion for Gemma 4 (Dense + MoE)"
    )
    parser.add_argument("--gguf", required=True, help="Path to original GGUF file")
    parser.add_argument(
        "--converted", required=True, help="Path to converted safetensors directory"
    )
    parser.add_argument("--reference", required=True, help="Path to reference HF model directory")
    parser.add_argument("--output", help="Path to save JSON results")
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Expect float16 tensors preserved (not converted to bfloat16)",
    )
    args = parser.parse_args()

    ok = verify(args.gguf, args.converted, args.reference, args.output, keep_fp16=args.keep_fp16)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
