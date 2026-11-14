#!/usr/bin/env python3
"""Full-caption SVAR Heads Guided Attention Intervention on paired LLaVA images.

Implements Eq. (6) of Jiang et al., CVPR 2025, in the installed Transformers
LlamaAttention API. No detector score or test annotation is used for decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from scripts.generation_time_paper_transfer_seed43 import _baseline_rows
from scripts.generation_time_sar_ems_seed43 import (
    PROMPT, ROOT, _append_jsonl, _chair_module, _generation_kwargs, _inputs,
)
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config

OUT = ROOT / "outputs/generation_time_svar_official_seed43/llava_1_5_7b/test_100"
CONFIG = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
ALPHA = 0.5
VARIANTS = {"paper_layers_5_18": tuple(range(5, 19)),
            "code_range_5_18": tuple(range(5, 18))}


@contextmanager
def heads_guided_attention(model, *, layers: tuple[int, ...],
                           image_start: int, image_end: int, alpha: float):
    """Temporary modern-HF port of the author's modify_attention.py rule."""
    decoder_layers = model.language_model.layers
    originals = {}
    calls = {i: 0 for i in layers}

    def guided_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_values.update(key, value, self.layer_idx, cache_kwargs)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :key.shape[-2]]
        if image_end > scores.shape[-1]:
            raise RuntimeError("Image-token span exceeds attention key length")
        # Author code changes the last query row at each forward, before softmax.
        visual = scores[:, :, -1, image_start:image_end]
        scores[:, :, -1, image_start:image_end] = (
            visual + alpha * visual.abs().mean(dim=1, keepdim=True)
        )
        weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = self.o_proj(output.reshape(*input_shape, -1).contiguous())
        calls[self.layer_idx] += 1
        return output, weights

    try:
        for i in layers:
            module = decoder_layers[i].self_attn
            originals[i] = module.forward
            module.forward = types.MethodType(guided_forward, module)
        yield calls
    finally:
        for i, forward in originals.items():
            decoder_layers[i].self_attn.forward = forward


@torch.inference_mode()
def generate_one(wrapper, image: Image.Image, layers: tuple[int, ...], alpha: float):
    inputs = _inputs(wrapper, image, "llava_1_5_7b")
    image_start, image_end = wrapper._find_visual_token_range(inputs, wrapper._image_token_id())
    prompt_len = int(inputs["input_ids"].shape[1])
    with heads_guided_attention(wrapper.model, layers=layers, image_start=image_start,
                                image_end=image_end, alpha=alpha) as calls:
        output = wrapper.model.generate(
            **inputs, **_generation_kwargs(wrapper, "llava_1_5_7b"),
        )
    ids = [int(v) for v in output[0, prompt_len:].tolist()]
    if not all(calls.values()):
        raise RuntimeError("Some guided layers were not invoked")
    return ids, wrapper.tokenizer.decode(ids, skip_special_tokens=True), {
        "image_start": image_start, "image_end": image_end,
        "guided_layer_calls": calls,
    }


def generate(limit: int, device: str, variants: list[str]) -> None:
    torch.set_num_threads(1)
    image_ids, baseline = _baseline_rows(limit)
    config = load_config(str(CONFIG))
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {name: OUT / f"{name}.jsonl" for name in variants}
    done = {name: {int(json.loads(x)["image_id"]) for x in path.read_text().splitlines()}
            if path.exists() else set() for name, path in paths.items()}
    wrapper = build_model("llava_1_5_7b", get_extraction_model_cfg(config, "llava_1_5_7b"), device=device)
    wrapper.model.eval().requires_grad_(False)
    for ordinal, image_id in enumerate(image_ids, 1):
        todo = [name for name in variants if image_id not in done[name]]
        if not todo:
            continue
        with Image.open(coco / f"COCO_val2014_{image_id:012d}.jpg") as opened:
            image = opened.convert("RGB")
        for name in todo:
            start = time.perf_counter()
            ids, caption, details = generate_one(wrapper, image, VARIANTS[name], ALPHA)
            row = {"image_id": image_id, "method": name, "alpha": ALPHA,
                   "layers": VARIANTS[name], "generated_text": caption,
                   "response_token_ids": ids, "details": details,
                   "seconds": time.perf_counter() - start,
                   "changed": ids != baseline[image_id]["response_token_ids"]}
            _append_jsonl(paths[name], row)
            done[name].add(image_id)
            print(f"[{ordinal}/{len(image_ids)}] {name}: changed={row['changed']} "
                  f"seconds={row['seconds']:.2f}", flush=True)
    (OUT / "run_config.json").write_text(json.dumps({
        "seed": 43, "images": len(image_ids), "model": "llava_1_5_7b",
        "prompt": PROMPT, "alpha": ALPHA,
        "variants": VARIANTS,
        "paper": "Jiang et al. CVPR 2025 Eq. (6)",
        "official_code": "https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations/blob/1467fe07fefd4f15ed1ac3560455dff5b53dbc08/modify_attention.py",
    }, indent=2))


def evaluate(limit: int) -> None:
    image_ids, baseline = _baseline_rows(limit)
    config = load_config(str(CONFIG))
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=OUT.parent.parent / "chair_cache.pkl",
    )
    variants = {"greedy": {i: baseline[i]["generated_text"] for i in image_ids}}
    generation = {}
    for name in VARIANTS:
        path = OUT / f"{name}.jsonl"
        rows = {int(r["image_id"]): r for r in
                (json.loads(x) for x in path.read_text().splitlines() if x.strip())}
        if any(i not in rows for i in image_ids):
            raise ValueError(f"Incomplete {name}")
        generation[name] = rows
        variants[name] = {i: rows[i]["generated_text"] for i in image_ids}
    cache = {}
    def chair_row(i, caption):
        key = (i, caption)
        if key not in cache:
            cache[key] = evaluator.compute_chair_token(i, caption)
        return cache[key]
    per_image = {name: [chair_row(i, caps[i]) for i in image_ids]
                 for name, caps in variants.items()}
    draws = np.random.default_rng(43).integers(0, len(image_ids), size=(5000, len(image_ids)))
    summary = {"seed": 43, "images": limit, "methods": {}}
    base = per_image["greedy"]
    base_s = np.array([r["metrics"]["CHAIRs"] for r in base])
    base_h = np.array([len(r["mscoco_hallucinated_words"]) for r in base])
    base_m = np.array([len(r["object_mentions"]) for r in base])
    with (OUT / "chair_per_image.jsonl").open("w", encoding="utf-8") as handle:
        for k, i in enumerate(image_ids):
            row = {"image_id": i}
            for name in variants:
                result = per_image[name][k]
                row[name] = {"chair_s": result["metrics"]["CHAIRs"],
                             "hallucinated_mentions": len(result["mscoco_hallucinated_words"]),
                             "all_mentions": len(result["object_mentions"]),
                             "recall": result["metrics"]["Recall"],
                             "precision": result["metrics"]["Precision"],
                             "f1": result["metrics"]["F1"]}
            handle.write(json.dumps(row) + "\n")
    for name, rows in per_image.items():
        s = np.array([r["metrics"]["CHAIRs"] for r in rows])
        h = np.array([len(r["mscoco_hallucinated_words"]) for r in rows])
        m = np.array([len(r["object_mentions"]) for r in rows])
        result = {"chair_s": float(s.mean()),
                  "chair_i": float(h.sum() / max(1, m.sum())),
                  "object_recall": float(np.mean([r["metrics"]["Recall"] for r in rows])),
                  "object_precision": float(np.mean([r["metrics"]["Precision"] for r in rows])),
                  "object_f1": float(np.mean([r["metrics"]["F1"] for r in rows])),
                  "true_mentions_per_image": float(np.mean([len(r["mscoco_real_words"]) for r in rows])),
                  "words_per_image": float(np.mean([len(r["words"]) for r in rows])),
                  "hallucinated_mentions": int(h.sum()), "all_mentions": int(m.sum())}
        if name != "greedy":
            result["changed_images"] = sum(generation[name][i]["changed"] for i in image_ids)
            result["seconds_per_image"] = float(np.mean([generation[name][i]["seconds"] for i in image_ids]))
            result["fixed_hallucinated_images"] = int(((base_s == 1) & (s == 0)).sum())
            result["introduced_hallucinated_images"] = int(((base_s == 0) & (s == 1)).sum())
            delta_s = s[draws].mean(1) - base_s[draws].mean(1)
            delta_i = h[draws].sum(1) / np.maximum(1, m[draws].sum(1)) - base_h[draws].sum(1) / np.maximum(1, base_m[draws].sum(1))
            result["delta_chair_s_ci95"] = np.quantile(delta_s, [0.025, 0.975]).tolist()
            result["delta_chair_i_ci95"] = np.quantile(delta_i, [0.025, 0.975]).tolist()
            base_recall = np.array([r["metrics"]["Recall"] for r in base])
            recall = np.array([r["metrics"]["Recall"] for r in rows])
            result["delta_object_recall_ci95"] = np.quantile(
                recall[draws].mean(1) - base_recall[draws].mean(1), [0.025, 0.975]
            ).tolist()
        summary["methods"][name] = result
    (OUT / "chair_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def smoke(device: str) -> None:
    torch.set_num_threads(1)
    ids, baseline = _baseline_rows(100)
    image_id = ids[0]
    config = load_config(str(CONFIG))
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    wrapper = build_model("llava_1_5_7b", get_extraction_model_cfg(config, "llava_1_5_7b"), device=device)
    wrapper.model.eval().requires_grad_(False)
    with Image.open(coco / f"COCO_val2014_{image_id:012d}.jpg") as opened:
        image = opened.convert("RGB")
    current = wrapper.generate(image, prompt=PROMPT)
    expected = baseline[image_id]["response_token_ids"]
    if list(map(int, current.response_token_ids)) != expected:
        raise RuntimeError("Unguided model differs from paired baseline")
    guided_ids, _, details = generate_one(wrapper, image, VARIANTS["paper_layers_5_18"], 0.0)
    if guided_ids != expected:
        raise RuntimeError("Zero-alpha port differs from unguided LLaVA")
    guided_ids, _, _ = generate_one(wrapper, image, VARIANTS["paper_layers_5_18"], ALPHA)
    print(json.dumps({"image_id": image_id, "zero_alpha_matches": True,
                      "image_start": details["image_start"],
                      "image_end": details["image_end"],
                      "alpha_05_changed": guided_ids != expected}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "generate", "evaluate"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=list(VARIANTS))
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.device)
    elif args.command == "generate":
        generate(args.limit, args.device, args.variants)
    else:
        evaluate(args.limit)
