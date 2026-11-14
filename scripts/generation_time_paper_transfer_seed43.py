#!/usr/bin/env python3
"""Paired LLaVA pilot: detector-gated single-token transfers of VCD/DeCo/ICD.

Only the baseline caption and detector scores determine intervention locations.
COCO annotations are read by the separate evaluate command.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.baselines import SVARMLP
from features.baseline import compute_svar_features
from models import build_model
from models.base_wrapper import AttentionRequirement, ExtractionRequirements
from models.llava_wrapper import _append_prefix_token_ids
from scripts.generation_time_sar_ems_seed43 import (
    PROBES, PROMPT, ROOT, RiskScorer, _append_jsonl, _chair_module,
    _decode, _generation_kwargs, _inputs, _read_json,
)
from scripts.run_endac_baselines_811_dual_threshold import key_of, load_baseline_records
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config

OUT = ROOT / "outputs/generation_time_paper_transfer_seed43"
GREEDY = ROOT / "outputs/generation_time_sar_ems_seed43_v2/llava_1_5_7b/test/greedy.jsonl"
METHODS = ("vcd", "deco", "icd")
GATES = ("sar_ems", "svar")
NEGATIVE_PROMPT = "You are a confused objects detector to provide a fuzzy overview or impression of the image. " + PROMPT
SVAR_REQUIREMENTS = ExtractionRequirements(
    attention=AttentionRequirement.PER_HEAD, logits=False,
    token_hidden_states=False, patch_hidden_states=False,
    visual_layout=False, dgst_capture=False,
)


def validation_threshold(folder: Path) -> float:
    with np.load(folder / "scores.npz") as data:
        scores = np.asarray(data["val_hall_score"], dtype=np.float64)
        labels = np.asarray(data["val_label"], dtype=np.int8)
    candidates = np.unique(scores)
    valid = [float(t) for t in candidates if np.mean(labels[scores >= t] == 0) >= 0.8]
    if not valid:
        raise ValueError(f"No 80%-precision threshold: {folder}")
    return min(valid)


class SVARRiskScorer:
    def __init__(self):
        folder = PROBES / "llava_1_5_7b/seed43/svar"
        checkpoint = torch.load(folder / "checkpoint/svar.pt", map_location="cpu", weights_only=False)
        self.net = SVARMLP(checkpoint["input_dim"], checkpoint["hidden_dim"])
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        self.layer_start = int(checkpoint["layer_start"])
        self.layer_end = int(checkpoint["layer_end_exclusive"])
        self.trigger_threshold = validation_threshold(folder)
        split = _read_json(PROBES / "splits/seed43.json")
        test_ids = set(split["test"])
        records = [r for r in load_baseline_records("llava_1_5_7b") if int(r["image_id"]) in test_ids]
        with np.load(folder / "scores.npz") as data:
            scores = np.asarray(data["test_hall_score"], dtype=np.float64)
        if len(records) != len(scores):
            raise ValueError("SVAR score and record counts disagree")
        self.cached = {key_of(r): float(score) for r, score in zip(records, scores)}
        self.cache_hits = 0
        self.recomputed = 0

    @torch.inference_mode()
    def score(self, wrapper, image: Image.Image, baseline_ids: list[int], image_id: int,
              mention: dict) -> float:
        index, token = int(mention["token_index"]), int(mention["token_id"])
        key = (image_id, index, token)
        if key in self.cached:
            self.cache_hits += 1
            return self.cached[key]
        output = wrapper.extract_token_features(
            image=image, prefix_token_ids=baseline_ids[:index],
            response_token_idx=index, target_token_id=token,
            cfg_dgst_t=None, prompt=PROMPT, requirements=SVAR_REQUIREMENTS,
        )
        features = compute_svar_features(
            output.text_to_patch_attn,
            layer_start=self.layer_start, layer_end=self.layer_end,
        ).vector
        with torch.inference_mode():
            score = torch.softmax(self.net(torch.from_numpy(features)[None]), -1)[0, 1].item()
        self.recomputed += 1
        return float(score)


def _forward_inputs(base: dict, current_ids: torch.Tensor, pixel_values=None) -> dict:
    values = dict(base)
    values["input_ids"] = current_ids
    values["attention_mask"] = torch.ones_like(current_ids)
    if pixel_values is not None:
        values["pixel_values"] = pixel_values
    values.pop("position_ids", None)
    values.pop("cache_position", None)
    return values


def vcd_noisy_pixels(clean: torch.Tensor, image_id: int) -> torch.Tensor:
    # Official VCD sigmoid beta schedule and noise_step=500, with a fixed per-image seed.
    beta = torch.sigmoid(torch.linspace(-6, 6, 1000, dtype=torch.float64)) * (0.005 - 0.00001) + 0.00001
    alpha_bar = torch.prod(1.0 - beta[:501]).item()
    generator = torch.Generator(device="cpu").manual_seed(43 + image_id)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean.device)
    return (math.sqrt(alpha_bar) * clean.float() + math.sqrt(1 - alpha_bar) * noise).to(clean.dtype)


class PaperStep(LogitsProcessor):
    def __init__(self, wrapper, image: Image.Image, image_id: int, base_inputs: dict,
                 index: int, target_id: int, method: str):
        self.wrapper = wrapper
        self.image = image
        self.base_inputs = base_inputs
        self.index = index
        self.target_id = target_id
        self.method = method
        self.prompt_len = int(base_inputs["input_ids"].shape[1])
        self.fired = False
        self.selected_token = None
        self.extra = {}
        if method == "vcd":
            self.noisy = vcd_noisy_pixels(base_inputs["pixel_values"], image_id)
        if method == "icd":
            self.negative_prompt_inputs = wrapper.processor(
                text=wrapper._format_prompt(NEGATIVE_PROMPT), images=image,
                return_tensors="pt",
            ).to(wrapper.device, torch.float16)

    @torch.inference_mode()
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] - self.prompt_len != self.index:
            return scores
        self.fired = True
        clean = scores[0].float()
        if int(clean.argmax()) != self.target_id:
            raise RuntimeError("Current greedy prefix differs from saved baseline")
        if self.method == "vcd":
            extra_inputs = _forward_inputs(self.base_inputs, input_ids, self.noisy)
            negative = self.wrapper.model(**extra_inputs, use_cache=False, return_dict=True).logits[0, -1].float()
            alpha, beta = 1.0, 0.1
            adjusted = (1 + alpha) * clean - alpha * negative
            adjusted = adjusted.masked_fill(clean < clean.max() + math.log(beta), -torch.inf)
            self.extra = {"alpha": alpha, "beta": beta, "noise_step": 500}
        elif self.method == "icd":
            prefix = input_ids[0, self.prompt_len:].tolist()
            negative_inputs = _append_prefix_token_ids(
                self.negative_prompt_inputs, prefix_token_ids=prefix,
                device=self.wrapper.device, dtype=torch.float16,
            )
            negative = self.wrapper.model(**negative_inputs, use_cache=False, return_dict=True).logits[0, -1].float()
            alpha, beta = 0.5, 0.1
            adjusted = (1 + alpha) * clean - alpha * negative
            adjusted = adjusted.masked_fill(clean < clean.max() + math.log(beta), -torch.inf)
            self.extra = {"alpha": alpha, "beta": beta, "negative_prompt": NEGATIVE_PROMPT}
        elif self.method == "deco":
            extra_inputs = _forward_inputs(self.base_inputs, input_ids)
            output = self.wrapper.model(**extra_inputs, use_cache=False,
                                        return_dict=True, output_hidden_states=True)
            # HF hidden_states[0] is embedding; hidden_states[l] is decoder layer l.
            hidden = output.hidden_states
            model = self.wrapper.model
            probs = clean.softmax(-1)
            candidate_probs, candidate_ids = probs.topk(20)
            cumulative = candidate_probs.cumsum(-1)
            cutoff = int(torch.searchsorted(cumulative, 0.9, right=False).item()) + 1
            candidate_ids = candidate_ids[:min(cutoff, 20)]
            best_layer, best_prob, selected_logits = None, -1.0, None
            for layer in range(20, 29):
                # Official DeCo projects the selected early exit with final norm and lm_head.
                premature = model.lm_head(model.language_model.norm(hidden[layer][:, -1:, :]))[0, -1].float()
                probability = premature.softmax(-1)[candidate_ids].max().item()
                if probability > best_prob:
                    best_layer, best_prob, selected_logits = layer, probability, premature
            adjusted = clean + 0.6 * best_prob * selected_logits
            mask = torch.ones_like(adjusted, dtype=torch.bool)
            mask[candidate_ids] = False
            adjusted = adjusted.masked_fill(mask, -torch.inf)
            self.extra = {"alpha": 0.6, "top_p": 0.9, "top_k": 20,
                          "selected_layer": best_layer, "premature_max_probability": best_prob,
                          "candidate_count": int(len(candidate_ids))}
        else:
            raise ValueError(self.method)
        if not torch.isfinite(adjusted).any():
            raise RuntimeError("Paper intervention excluded every token")
        self.selected_token = int(adjusted.argmax())
        return adjusted[None].to(scores.dtype)


@torch.inference_mode()
def intervene(wrapper, image: Image.Image, image_id: int, trigger: dict,
              method: str) -> tuple[list[int], dict]:
    inputs = _inputs(wrapper, image, "llava_1_5_7b")
    prompt_len = int(inputs["input_ids"].shape[1])
    processor = PaperStep(wrapper, image, image_id, inputs,
                          int(trigger["token_index"]), int(trigger["token_id"]), method)
    generated = wrapper.model.generate(
        **inputs, **_generation_kwargs(wrapper, "llava_1_5_7b"),
        logits_processor=LogitsProcessorList([processor]),
    )
    if not processor.fired:
        raise RuntimeError("Target index never reached")
    ids = [int(v) for v in generated[0, prompt_len:].tolist()]
    if ids[processor.index] != processor.selected_token:
        raise RuntimeError("Selected token and generated token disagree")
    return ids, {"selected_token": processor.selected_token, **processor.extra}


def _baseline_rows(limit: int) -> tuple[list[int], dict[int, dict]]:
    image_ids = sorted(_read_json(PROBES / "splits/seed43.json")["test"])[:limit]
    rows = {int(row["image_id"]): row for row in
            (json.loads(line) for line in GREEDY.read_text().splitlines() if line.strip())}
    if any(i not in rows for i in image_ids):
        raise ValueError("Incomplete greedy baseline")
    return image_ids, rows


def generate(limit: int, device: str, end_offset: int | None) -> None:
    torch.set_num_threads(1)
    config = load_config(str(ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"))
    image_ids, baseline = _baseline_rows(limit)
    sar = RiskScorer("llava_1_5_7b")
    svar = SVARRiskScorer()
    wrapper = build_model("llava_1_5_7b", get_extraction_model_cfg(config, "llava_1_5_7b"), device=device)
    wrapper.model.eval().requires_grad_(False)
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    folder = OUT / "llava_1_5_7b/test_100"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {(gate, method): folder / f"{gate}_{method}.jsonl" for gate in GATES for method in METHODS}
    done = {(gate, method): {int(json.loads(line)["image_id"]) for line in path.read_text().splitlines()}
            if path.exists() else set() for (gate, method), path in paths.items()}
    for ordinal, image_id in enumerate(image_ids[:end_offset], 1):
        todo = [(gate, method) for gate in GATES for method in METHODS
                if image_id not in done[(gate, method)]]
        if not todo:
            continue
        with Image.open(coco / f"COCO_val2014_{image_id:012d}.jpg") as opened:
            image = opened.convert("RGB")
        prior = baseline[image_id]
        baseline_ids = [int(i) for i in prior["response_token_ids"]]
        mentions = [dict(m) for m in prior["mentions"]]
        for mention in mentions:
            mention["svar_score"] = svar.score(wrapper, image, baseline_ids, image_id, mention)
        triggers = {
            "sar_ems": next((m for m in mentions if m["hall_score"] >= sar.trigger_threshold), None),
            "svar": next((m for m in mentions if m["svar_score"] >= svar.trigger_threshold), None),
        }
        for gate, method in todo:
            trigger = triggers[gate]
            started = time.perf_counter()
            ids, details = baseline_ids, {}
            if trigger is not None:
                ids, details = intervene(wrapper, image, image_id, trigger, method)
                if ids[:int(trigger["token_index"])] != baseline_ids[:int(trigger["token_index"])]:
                    raise RuntimeError("Intervention changed the pre-trigger prefix")
            row = {"image_id": image_id, "method": method, "gate": gate,
                   "generated_text": _decode(wrapper, ids, "llava_1_5_7b"),
                   "response_token_ids": ids, "trigger": trigger,
                   "details": details, "seconds": time.perf_counter() - started}
            _append_jsonl(paths[(gate, method)], row)
            done[(gate, method)].add(image_id)
            print(f"[{ordinal}/{len(image_ids)}] {gate}_{method}: trigger={trigger is not None} "
                  f"changed={ids != baseline_ids}", flush=True)
    (folder / "run_config.json").write_text(json.dumps({
        "seed": 43, "limit": limit, "model": "llava_1_5_7b", "prompt": PROMPT,
        "sar_ems_threshold": sar.trigger_threshold, "svar_threshold": svar.trigger_threshold,
        "svar_cache_hits": svar.cache_hits, "svar_recomputed": svar.recomputed,
        "methods": METHODS, "gates": GATES, "negative_prompt": NEGATIVE_PROMPT,
    }, indent=2))


def evaluate(limit: int) -> None:
    config = load_config(str(ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"))
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=OUT / "chair_cache.pkl",
    )
    image_ids, baseline = _baseline_rows(limit)
    folder = OUT / "llava_1_5_7b/test_100"
    variants = {"greedy": {i: baseline[i]["generated_text"] for i in image_ids}}
    generation = {}
    for gate in GATES:
        for method in METHODS:
            name = f"{gate}_{method}"
            path = folder / f"{name}.jsonl"
            rows = {int(row["image_id"]): row for row in
                    (json.loads(line) for line in path.read_text().splitlines() if line.strip())}
            if any(i not in rows for i in image_ids):
                raise ValueError(f"Incomplete variant {name}")
            generation[name] = rows
            variants[name] = {i: rows[i]["generated_text"] for i in image_ids}
    evaluated = {}
    def chair_row(image_id: int, caption: str):
        key = (image_id, caption)
        if key not in evaluated:
            evaluated[key] = evaluator.compute_chair_token(image_id, caption)
        return evaluated[key]
    per_image = {name: [chair_row(i, captions[i]) for i in image_ids]
                 for name, captions in variants.items()}
    rng = np.random.default_rng(43)
    draws = rng.integers(0, len(image_ids), size=(5000, len(image_ids)))
    summary = {"seed": 43, "images": limit, "methods": {}}
    base = per_image["greedy"]
    base_s = np.array([r["metrics"]["CHAIRs"] for r in base])
    base_h = np.array([len(r["mscoco_hallucinated_words"]) for r in base])
    base_m = np.array([len(r["object_mentions"]) for r in base])
    for name, rows in per_image.items():
        s = np.array([r["metrics"]["CHAIRs"] for r in rows])
        h = np.array([len(r["mscoco_hallucinated_words"]) for r in rows])
        m = np.array([len(r["object_mentions"]) for r in rows])
        result = {
            "chair_s": float(s.mean()), "chair_i": float(h.sum() / max(1, m.sum())),
            "object_recall": float(np.mean([r["metrics"]["Recall"] for r in rows])),
            "true_mentions_per_image": float(np.mean([len(r["mscoco_real_words"]) for r in rows])),
            "all_mentions_per_image": float(m.mean()),
            "words_per_image": float(np.mean([len(r["words"]) for r in rows])),
            "hallucinated_mentions": int(h.sum()), "all_mentions": int(m.sum()),
        }
        if name != "greedy":
            source = generation[name]
            result["triggered_images"] = sum(source[i]["trigger"] is not None for i in image_ids)
            result["changed_images"] = sum(source[i]["generated_text"] != baseline[i]["generated_text"] for i in image_ids)
            result["fixed_hallucinated_images"] = int(((base_s == 1) & (s == 0)).sum())
            result["introduced_hallucinated_images"] = int(((base_s == 0) & (s == 1)).sum())
            result["seconds_per_triggered_image"] = float(np.mean([
                source[i]["seconds"] for i in image_ids if source[i]["trigger"] is not None]))
            delta_s = s[draws].mean(1) - base_s[draws].mean(1)
            delta_i = h[draws].sum(1) / np.maximum(1, m[draws].sum(1)) - base_h[draws].sum(1) / np.maximum(1, base_m[draws].sum(1))
            result["delta_chair_s_ci95"] = np.quantile(delta_s, [0.025, 0.975]).tolist()
            result["delta_chair_i_ci95"] = np.quantile(delta_i, [0.025, 0.975]).tolist()
        summary["methods"][name] = result
    gate_audits = {}
    for gate in GATES:
        source = generation[f"{gate}_vcd"]
        triggered = [i for i in image_ids if source[i]["trigger"] is not None]
        gate_audits[gate] = {
            "triggered_ids": triggered,
            "triggered_baseline_hallucinated_images": sum(
                bool(base[image_ids.index(i)]["metrics"]["CHAIRs"]) for i in triggered),
            "triggered_object_hallucinations": sum(
                source[i]["trigger"]["canonical"] not in set(base[image_ids.index(i)]["mscoco_gt_words"])
                for i in triggered),
        }
    summary["gates"] = gate_audits
    summary["gates"]["overlap_images"] = len(set(gate_audits["sar_ems"]["triggered_ids"]) & set(gate_audits["svar"]["triggered_ids"]))
    per_image_path = folder / "chair_per_image.jsonl"
    with per_image_path.open("w", encoding="utf-8") as handle:
        for position, image_id in enumerate(image_ids):
            row = {"image_id": image_id}
            for name in variants:
                item = per_image[name][position]
                row[name] = {"chair_s": item["metrics"]["CHAIRs"],
                             "hallucinated_mentions": len(item["mscoco_hallucinated_words"]),
                             "all_mentions": len(item["object_mentions"]),
                             "recall": item["metrics"]["Recall"]}
            handle.write(json.dumps(row) + "\n")
    (folder / "chair_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--end-offset", type=int)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.limit, args.device, args.end_offset)
    else:
        evaluate(args.limit)
