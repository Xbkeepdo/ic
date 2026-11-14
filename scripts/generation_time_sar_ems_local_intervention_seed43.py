#!/usr/bin/env python3
"""One-position SAR patch attention and EMS-gated FFN visual-write pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.logs_risk_f_log import compute_logs_risk_f_log_batch
from models import build_model
from models.dgst_capture import (
    apply_decoder_final_norm,
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
    run_forward_with_dgst_captures,
    target_probabilities_multi,
)
from models.llava_wrapper import _append_prefix_token_ids
from scripts.generation_time_sar_ems_seed43 import (
    PROMPT, ROOT, _chair_module, _decode,
    _generation_kwargs, _inputs, _mention_matcher,
)
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


MODEL = "llava_1_5_7b"
SEED = 43
LIMIT = 100
THRESHOLD = 0.4013161063194275
METHODS = ("sar_patch_attention", "ems_ffn_visual", "joint")
OUT = ROOT / "outputs/generation_time_sar_ems_local_seed43/llava_1_5_7b/test_100"
OUT_ALWAYS = ROOT / "outputs/generation_time_sar_ems_local_always_seed43/llava_1_5_7b/test_100"
OUT_FIXED = ROOT / "outputs/generation_time_sar_ems_local_fixed_seed43/llava_1_5_7b/test_100"
BASELINE = ROOT / "outputs/generation_time_sar_ems_seed43_v2/llava_1_5_7b/test/greedy.jsonl"
CLOSED_ROWS = ROOT / "outputs/generation_time_closed_loop_tau0p4013161_seed43/llava_1_5_7b/test_100/rows"
SPLIT = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45/splits/seed43.json"
FEATURES = ROOT / "outputs/log_s_mechanism_ablation_811/llava_1_5_7b/compact_features.npz"
CONFIG = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
ATTENTION_LAYERS = tuple(range(5, 19))  # Inclusive 5..18, as in the paper.
FFN_POOL = tuple(range(18, 26))
MAP_LAYERS = tuple(range(5, 26))
PATCH_TOP_K = 16
MAX_FFN_LAYERS = 3
MAX_PATCH_MULTIPLIER = 1.7
MAX_FFN_NORM_FRACTION = 0.15
INTERVENTION_WINDOW = 12
FIXED_FFN_NORM_FRACTION = 0.2


def read_rows(path: Path) -> dict[int, dict]:
    return {int(row["image_id"]): row for row in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def training_stats() -> dict:
    train_ids = set(map(int, json.loads(SPLIT.read_text())["train"]))
    with np.load(FEATURES, allow_pickle=False) as data:
        train = np.isin(data["image_id"], list(train_ids))
        labels = data["label"][train]
        sar = np.asarray(data["sar"][train], dtype=np.float64)
        ems = np.asarray(data["ems"][train], dtype=np.float64)
    sar_gap = np.maximum(0, sar[labels == 1].mean(0) - sar[labels == 0].mean(0))
    ems_gap = np.maximum(0, ems[labels == 0].mean(0) - ems[labels == 1].mean(0))
    ems_real = ems[labels == 1]
    return {
        "sar_gap": sar_gap,
        "ems_gap": ems_gap,
        "ems_real_q90": np.quantile(ems_real, 0.9, axis=0),
        "ems_real_iqr": np.quantile(ems_real, 0.75, axis=0) - np.quantile(ems_real, 0.25, axis=0),
        "train_tokens": int(train.sum()),
    }


def object_token_ids(tokenizer, matcher) -> list[tuple[str, int]]:
    """Use single-token COCO object names, excluding function words."""
    names = sorted(set(matcher.inverse_synonym_dict.values()))
    result = []
    for name in names:
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        if len(token_ids) == 1:
            result.append((name, int(token_ids[0])))
    if len(result) < 20:
        raise RuntimeError(f"Only {len(result)} single-token COCO object names")
    return result


@torch.inference_mode()
def plan_intervention(wrapper, image: Image.Image, prefix: list[int], trigger: dict,
                      object_tokens: list[tuple[str, int]], stats: dict,
                      always_ffn: bool = False,
                      require_greedy_trigger: bool = True) -> dict:
    prompt_inputs = _inputs(wrapper, image, MODEL)
    full_inputs = _append_prefix_token_ids(
        prompt_inputs, prefix_token_ids=prefix, device=wrapper.device, dtype=torch.float16)
    visual_start, visual_end = wrapper._find_visual_token_range(
        full_inputs, wrapper._image_token_id())
    predicted_position = int(full_inputs["input_ids"].shape[1]) - 1
    out, captures = run_forward_with_dgst_captures(
        wrapper.model, output_hidden_states=False, retain_attention_updates=True,
        attention_query_positions=[-1], capture_device=None,
        attention_query_chunk_size=wrapper.dgst_attention_query_chunk_size,
        record_model_attentions=False, **full_inputs)
    if require_greedy_trigger and int(out.logits[0, -1].argmax()) != int(trigger["token_id"]):
        raise RuntimeError("Cached trigger token disagrees with unmodified model")
    features = compute_logs_risk_f_log_batch(
        model=wrapper.model, captures=captures, visual_start=visual_start,
        visual_end=visual_end, target_token_ids=[int(trigger["token_id"])],
        prediction_positions=[predicted_position], tau=0.03, transport_top_k=32,
        semantic_chunk_size=64, ot_solver="emd")[0]
    ems = np.asarray(features["logs_risk_per_layer"], dtype=np.float64)
    sar = np.asarray(features["f_log_per_layer"], dtype=np.float64)
    excess = (ems - stats["ems_real_q90"]) / np.maximum(stats["ems_real_iqr"], 1e-6)
    eligible_ffn = FFN_POOL if always_ffn else tuple(l for l in FFN_POOL if excess[l] > 0)
    selected_ffn = sorted(eligible_ffn, key=lambda l: excess[l], reverse=True)[:MAX_FFN_LAYERS]
    final_norm = resolve_decoder_final_norm(wrapper.model)
    output_head = resolve_output_embedding_layer(wrapper.model)
    categories = [(name, token_id) for name, token_id in object_tokens
                  if name != trigger["canonical"]]
    target_ids = [token_id for _, token_id in categories]
    vocab_size = int(output_head.weight.shape[0])
    patch_boost: dict[int, torch.Tensor] = {}
    patch_mask: dict[int, torch.Tensor] = {}
    patch_peaks = {}
    for layer in MAP_LAYERS:
        visual = captures[layer]["h_prev"][0, visual_start:visual_end].float()
        normalized = apply_decoder_final_norm(final_norm, visual)
        probabilities = target_probabilities_multi(
            output_layer=output_head, states=normalized,
            target_token_ids=target_ids, chunk_size=64)
        semantic = torch.log1p(vocab_size * probabilities.float()).amax(dim=1)
        if semantic.numel() == 576:
            semantic = F.avg_pool2d(semantic.reshape(1, 1, 24, 24), 3, 1, 1).flatten()
        k = min(PATCH_TOP_K, semantic.numel())
        values, indices = semantic.topk(k)
        mask = torch.zeros_like(semantic, dtype=torch.bool)
        mask[indices] = True
        boost = torch.zeros_like(semantic)
        boost[indices] = (values - values.min()) / (values.max() - values.min() + 1e-6)
        patch_mask[layer] = mask
        patch_boost[layer] = boost
        patch_peaks[layer] = float(values.max().item())
    del out, captures
    return {
        "visual_start": visual_start, "visual_end": visual_end,
        "patch_boost": patch_boost, "patch_mask": patch_mask,
        "selected_ffn": selected_ffn, "ems": ems.tolist(), "sar": sar.tolist(),
        "ems_excess": excess.tolist(), "patch_peaks": patch_peaks,
    }


@contextmanager
def local_intervention(model, plan: dict, stats: dict, mode: str, trigger_index: int,
                       fixed_coefficients: bool = False):
    layers = resolve_decoder_layers(model)
    selected_ffn = set(plan["selected_ffn"]) if mode in {"ems_ffn_visual", "joint"} else set()
    patched = set(ATTENTION_LAYERS if mode in {"sar_patch_attention", "joint"} else ()) | selected_ffn
    originals = {}
    handles = []
    calls = {layer: 0 for layer in patched}
    visual_writes: dict[int, torch.Tensor] = {}
    applied = {"attention": 0, "ffn": 0}
    visual_start, visual_end = plan["visual_start"], plan["visual_end"]
    sar_max = max(float(stats["sar_gap"][l]) for l in ATTENTION_LAYERS)
    ems_max = max(float(stats["ems_gap"][l]) for l in FFN_POOL)

    def guided_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, cache_position=None, **kwargs):
        layer = self.layer_idx
        step = calls[layer]
        calls[layer] += 1
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(
                key, value, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :key.shape[-2]]
        if visual_end > scores.shape[-1]:
            raise RuntimeError("Visual span exceeds attention key length")
        in_window = trigger_index <= step < trigger_index + INTERVENTION_WINDOW
        if in_window and layer in ATTENTION_LAYERS and mode != "ems_ffn_visual":
            alpha = (MAX_PATCH_MULTIPLIER if fixed_coefficients else
                     1 + (MAX_PATCH_MULTIPLIER - 1) * float(stats["sar_gap"][layer]) / (sar_max + 1e-9))
            map_values = plan["patch_mask"][layer] if fixed_coefficients else plan["patch_boost"][layer]
            boost = map_values.to(device=scores.device, dtype=scores.dtype)
            scores[:, :, -1, visual_start:visual_end] += math.log(alpha) * boost[None, None, :]
            applied["attention"] += 1
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if in_window and layer in selected_ffn:
            mask = plan["patch_mask"][layer].to(device=weights.device)
            visual_weights = weights[:, :, -1, visual_start:visual_end] * mask[None, None, :]
            visual_values = value[:, :, visual_start:visual_end, :]
            per_head = (visual_weights.unsqueeze(-1) * visual_values).sum(dim=2)
            visual_writes[layer] = F.linear(
                per_head.reshape(hidden_states.shape[0], -1), self.o_proj.weight, bias=None)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = self.o_proj(output.reshape(*input_shape, -1).contiguous())
        return output, weights

    def ffn_hook(layer: int):
        def hook(_module, _args, output):
            if not trigger_index <= calls[layer] - 1 < trigger_index + INTERVENTION_WINDOW:
                return output
            visual = visual_writes.pop(layer, None)
            if visual is None:
                raise RuntimeError(f"Missing visual attention write at layer {layer}")
            fraction = (FIXED_FFN_NORM_FRACTION if fixed_coefficients else
                        MAX_FFN_NORM_FRACTION * float(stats["ems_gap"][layer]) / (ems_max + 1e-9))
            branch = output[:, -1, :]
            norm = visual.float().norm(dim=-1, keepdim=True)
            if float(norm.max()) < 1e-8:
                return output
            delta = visual.float() / norm.clamp_min(1e-8) * branch.float().norm(dim=-1, keepdim=True) * fraction
            changed = output.clone()
            changed[:, -1, :] += delta.to(dtype=output.dtype)
            applied["ffn"] += 1
            return changed
        return hook

    try:
        for layer in patched:
            module = layers[layer].self_attn
            originals[layer] = module.forward
            module.forward = types.MethodType(guided_forward, module)
        for layer in selected_ffn:
            handles.append(layers[layer].mlp.register_forward_hook(ffn_hook(layer)))
        yield applied
    finally:
        for handle in handles:
            handle.remove()
        for layer, forward in originals.items():
            layers[layer].self_attn.forward = forward


@torch.inference_mode()
def generate_variant(wrapper, image: Image.Image, plan: dict, stats: dict,
                     mode: str, trigger: dict, original: dict,
                     fixed_coefficients: bool = False) -> dict:
    inputs = _inputs(wrapper, image, MODEL)
    prompt_len = int(inputs["input_ids"].shape[1])
    start = time.perf_counter()
    with local_intervention(wrapper.model, plan, stats, mode,
                            int(trigger["token_index"]),
                            fixed_coefficients=fixed_coefficients) as applied:
        generated = wrapper.model.generate(
            **inputs, **_generation_kwargs(wrapper, MODEL))
    ids = [int(value) for value in generated[0, prompt_len:].tolist()]
    index = int(trigger["token_index"])
    if ids[:index] != list(map(int, original["response_token_ids"][:index])):
        raise RuntimeError("Intervention changed the prefix before the trigger")
    if mode in {"sar_patch_attention", "joint"} and applied["attention"] < len(ATTENTION_LAYERS):
        raise RuntimeError(f"Expected attention intervention in {len(ATTENTION_LAYERS)} layers: {applied}")
    if mode in {"ems_ffn_visual", "joint"} and plan["selected_ffn"] and applied["ffn"] < len(plan["selected_ffn"]):
        raise RuntimeError(f"Expected FFN intervention in {len(plan['selected_ffn'])} layers: {applied}")
    return {"generated_text": _decode(wrapper, ids, MODEL),
            "response_token_ids": ids, "seconds": time.perf_counter() - start,
            "applied": applied, "changed": ids != original["response_token_ids"]}


def generate(device: str, start_index: int, end_index: int, image_id: int | None,
             always_ffn: bool = False, fixed_coefficients: bool = False) -> None:
    torch.set_num_threads(1)
    config = load_config(str(CONFIG))
    ids = sorted(map(int, json.loads(SPLIT.read_text())["test"]))[:LIMIT]
    ids = ids[start_index:end_index]
    if image_id is not None:
        if image_id not in ids:
            raise ValueError("Image outside selected 100-image cohort/range")
        ids = [image_id]
    baseline = read_rows(BASELINE)
    always_ffn = always_ffn or fixed_coefficients
    output_dir = OUT_FIXED if fixed_coefficients else OUT_ALWAYS if always_ffn else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = training_stats()
    matcher = _mention_matcher()
    wrapper = build_model(MODEL, get_extraction_model_cfg(config, MODEL), device=device)
    wrapper.model.eval().requires_grad_(False)
    categories = object_token_ids(wrapper.tokenizer, matcher)
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    for number, iid in enumerate(ids, 1):
        path = output_dir / f"{iid}.json"
        if path.exists():
            continue
        original = baseline[iid]
        closed = json.loads((CLOSED_ROWS / f"{iid}.json").read_text())
        first = closed["actions"][0] if closed["actions"] else None
        trigger = ({"canonical": first["word"], "surface": first["surface"],
                    "token_index": first["token_index"],
                    "token_id": first["banned_first_id"],
                    "hall_score": first["score"]} if first else None)
        if trigger is not None and original["response_token_ids"][trigger["token_index"]] != trigger["token_id"]:
            raise RuntimeError("Closed-loop trigger is not aligned with greedy tokens")
        row = {"image_id": iid, "threshold": THRESHOLD, "window": INTERVENTION_WINDOW,
               "ffn_gate_mode": "always_top3" if always_ffn else "above_real_q90_top3",
               "coefficient_mode": "fixed_alpha1p7_ffn_norm0p2" if fixed_coefficients else "training_gap_scaled",
               "trigger": trigger,
               "methods": {}}
        if trigger is not None:
            with Image.open(coco / f"COCO_val2014_{iid:012d}.jpg") as opened:
                image = opened.convert("RGB")
            started = time.perf_counter()
            plan = plan_intervention(
                wrapper, image, list(map(int, original["response_token_ids"][:trigger["token_index"]])),
                trigger, categories, stats, always_ffn=always_ffn)
            row["planning_seconds"] = time.perf_counter() - started
            row["plan"] = {key: value for key, value in plan.items()
                           if key not in {"patch_boost", "patch_mask"}}
            for mode in METHODS:
                row["methods"][mode] = generate_variant(
                    wrapper, image, plan, stats, mode, trigger, original,
                    fixed_coefficients=fixed_coefficients)
        else:
            row["planning_seconds"] = 0.0
            for mode in METHODS:
                row["methods"][mode] = {
                    "generated_text": original["generated_text"],
                    "response_token_ids": original["response_token_ids"],
                    "seconds": 0.0, "applied": {"attention": 0, "ffn": 0},
                    "changed": False}
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        print(f"[{number}/{len(ids)}] image={iid} trigger={trigger is not None} "
              f"ffn={row.get('plan', {}).get('selected_ffn', [])} "
              f"changed={[row['methods'][m]['changed'] for m in METHODS]} "
              f"seconds={row['planning_seconds'] + sum(row['methods'][m]['seconds'] for m in METHODS):.1f}",
              flush=True)


def evaluate(always_ffn: bool = False, fixed_coefficients: bool = False) -> None:
    always_ffn = always_ffn or fixed_coefficients
    output_dir = OUT_FIXED if fixed_coefficients else OUT_ALWAYS if always_ffn else OUT
    config = load_config(str(CONFIG))
    ids = sorted(map(int, json.loads(SPLIT.read_text())["test"]))[:LIMIT]
    baseline = read_rows(BASELINE)
    rows = {int(path.stem): json.loads(path.read_text()) for path in output_dir.glob("*.json")
            if path.stem.isdigit()}
    if set(ids) - rows.keys():
        raise RuntimeError(f"Missing {len(set(ids) - rows.keys())} images")
    if always_ffn:
        for iid in ids:
            row = rows[iid]
            if row["trigger"] is None:
                continue
            if len(row["plan"]["selected_ffn"]) != MAX_FFN_LAYERS:
                raise RuntimeError(f"{iid}: FFN layers were skipped after detection")
            applied = row["methods"]["joint"]["applied"]
            if applied["attention"] < len(ATTENTION_LAYERS) or applied["ffn"] < MAX_FFN_LAYERS:
                raise RuntimeError(f"{iid}: joint branch did not modify attention and FFN")
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=ROOT / "outputs/generation_time_svar_official_seed43/chair_cache.pkl")
    variants = {"greedy": {iid: baseline[iid]["generated_text"] for iid in ids}}
    for mode in METHODS:
        variants[mode] = {iid: rows[iid]["methods"][mode]["generated_text"] for iid in ids}
    evaluated = {mode: [evaluator.compute_chair_token(iid, captions[iid]) for iid in ids]
                 for mode, captions in variants.items()}
    metrics = {}
    for mode, items in evaluated.items():
        hall = sum(len(item["mscoco_hallucinated_words"]) for item in items)
        mentions = sum(len(item["object_mentions"]) for item in items)
        tp = sum(len(set(item["mscoco_gt_words"]) & set(item["mscoco_generated_words"]))
                 for item in items)
        gt = sum(len(set(item["mscoco_gt_words"])) for item in items)
        pred = sum(len(set(item["mscoco_generated_words"])) for item in items)
        metrics[mode] = {
            "CHAIRs": float(np.mean([item["metrics"]["CHAIRs"] for item in items])),
            "CHAIRi": hall / mentions if mentions else 0.0,
            "hallucinated_mentions": hall, "object_mentions": mentions,
            "macro_object_recall": float(np.mean([item["metrics"]["Recall"] for item in items])),
            "macro_object_F1": float(np.mean([item["metrics"]["F1"] for item in items])),
            "paper_style_micro_object_F1": 2 * tp / (gt + pred) if gt + pred else 0.0,
            "paper_style_counts": {"TP": tp, "GT": gt, "Pred": pred},
            "words_per_image": float(np.mean([len(item["words"]) for item in items])),
        }
        if mode != "greedy":
            metrics[mode]["changed_images"] = sum(rows[iid]["methods"][mode]["changed"] for iid in ids)
            metrics[mode]["generation_seconds"] = sum(rows[iid]["methods"][mode]["seconds"] for iid in ids)
            metrics[mode]["fixed_images"] = sum(
                b["metrics"]["CHAIRs"] and not r["metrics"]["CHAIRs"]
                for b, r in zip(evaluated["greedy"], items))
            metrics[mode]["introduced_images"] = sum(
                not b["metrics"]["CHAIRs"] and r["metrics"]["CHAIRs"]
                for b, r in zip(evaluated["greedy"], items))
    result = {"seed": SEED, "images": LIMIT, "threshold": THRESHOLD,
              "ffn_gate_mode": "always_top3" if always_ffn else "above_real_q90_top3",
              "coefficient_mode": "fixed_alpha1p7_ffn_norm0p2" if fixed_coefficients else "training_gap_scaled",
              "triggered_images": sum(rows[iid]["trigger"] is not None for iid in ids),
              "planning_seconds": sum(rows[iid]["planning_seconds"] for iid in ids),
              "methods": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=LIMIT)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--always-ffn", action="store_true")
    parser.add_argument("--fixed-coefficients", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.device, args.start_index, args.end_index, args.image_id,
                 always_ffn=args.always_ffn,
                 fixed_coefficients=args.fixed_coefficients)
    else:
        evaluate(always_ffn=args.always_ffn,
                 fixed_coefficients=args.fixed_coefficients)
