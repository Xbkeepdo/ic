#!/usr/bin/env python3
"""One-seed, paired COCO generation interventions guided by SAR+EMS.

The decoder reads only caption text, token IDs, and detector scores.  COCO
annotations are loaded only by the separate ``evaluate`` command.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from models.base_wrapper import AttentionRequirement, ExtractionRequirements
from scripts.train_torch_probe_feature_sets import DGSTStyleProbe
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config
from utils.token_alignment import build_response_token_offsets, token_indices_for_char_span


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
OUT = ROOT / "outputs/generation_time_sar_ems_seed43_v2"
SEED = 43
PROMPT = "Describe this image."
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
METHODS = ("risk_beam", "risk_vcd", "risk_rollback")
CFG_DGST = {
    "feature_output_profile": "logs_risk_f_log_hpre",
    "target_gate_mode": "logs_risk_f_log",
    "tau": 0.03,
    "transport_top_k": 32,
    "ot_solver": "emd",
    "semantic_chunk_size": 64,
    "compute_prompt_cafe": False,
    "compute_ffn_injection_features": False,
}
REQUIREMENTS = ExtractionRequirements(
    attention=AttentionRequirement.NONE,
    logits=False,
    token_hidden_states=False,
    patch_hidden_states=False,
    visual_layout=False,
    dgst_capture=True,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chair_module():
    if "coco_chair" in sys.modules:
        return sys.modules["coco_chair"]
    path = ROOT / "coco-labeling/coco_chair.py"
    spec = importlib.util.spec_from_file_location("coco_chair", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["coco_chair"] = module
    spec.loader.exec_module(module)
    return module


def _mention_matcher():
    chair = _chair_module()

    class MentionMatcher(chair.CHAIR):
        def get_annotations(self) -> None:
            pass

    # CHAIR's shared object lexicon and tokenizer, with no image annotations.
    return MentionMatcher()


def _mentions(matcher, tokenizer, ids: list[int], caption: str) -> list[dict]:
    offsets = build_response_token_offsets(tokenizer, ids, caption)
    found = []
    for mention in matcher.caption_to_mentions(caption):
        indices = token_indices_for_char_span(
            offsets, mention["char_start"], mention["char_end"]
        )
        if not indices:
            continue
        found.append({
            "canonical": mention["canonical_name"],
            "surface": mention["surface"],
            "token_index": int(indices[0]),
            "token_id": int(ids[indices[0]]),
        })
    return found


class RiskScorer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        folder = PROBES / model_name / f"seed{SEED}" / "sar_ems"
        config = _read_json(folder / "config.json")
        self.net = DGSTStyleProbe(
            int(config["input_dim"]), config["hidden_sizes"],
            float(config["dropout"]), bool(config["batch_norm"]),
        )
        self.net.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        with np.load(folder / "input_scaler.npz") as saved:
            self.mean = np.asarray(saved["mean"], dtype=np.float64)
            self.scale = np.asarray(saved["scale"], dtype=np.float64)
        with np.load(folder / "scores.npz") as saved:
            val_hall = np.asarray(saved["val_hall_score"], dtype=np.float64)
            val_label = np.asarray(saved["val_label"], dtype=np.int8)
        # Most permissive threshold with >=80% Hall precision on validation.
        candidates = np.unique(val_hall)
        valid = [float(t) for t in candidates if (val_label[val_hall >= t] == 0).mean() >= 0.8]
        if not valid:
            raise ValueError(f"{model_name}: validation Hall precision never reaches 80%")
        self.trigger_threshold = min(valid)
        split = _read_json(PROBES / "splits" / f"seed{SEED}.json")
        self.ids_by_split = {key: [int(value) for value in split[key]] for key in ("val", "test")}

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(((matrix - self.mean) / self.scale).astype(np.float32))
        with torch.inference_mode():
            return (1.0 - torch.sigmoid(self.net(x)).flatten()).numpy()

    def score_features(self, outputs: list[Any]) -> list[float]:
        vectors = []
        for output in outputs:
            item = output.dgst_t_result
            if item is None:
                raise ValueError("SAR+EMS extractor returned no result")
            vectors.append(np.concatenate((
                np.asarray(item["f_log_per_layer"], dtype=np.float32),
                np.asarray(item["logs_risk_per_layer"], dtype=np.float32),
            )))
        return self.score_matrix(np.stack(vectors)).astype(float).tolist()


def _decode(wrapper, ids: list[int], model_name: str) -> str:
    if model_name == "qwen3_vl_8b":
        return wrapper.processor.decode(ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).strip()
    return wrapper.tokenizer.decode(ids, skip_special_tokens=True)


def _inputs(wrapper, image: Image.Image, model_name: str):
    if model_name == "qwen3_vl_8b":
        return wrapper._prompt_inputs(image, PROMPT).to(wrapper.device)
    return wrapper.processor(text=wrapper._format_prompt(PROMPT), images=image,
                             return_tensors="pt").to(wrapper.device, torch.float16)


def _generation_kwargs(wrapper, model_name: str) -> dict:
    if model_name == "qwen3_vl_8b":
        return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                "max_new_tokens": wrapper.generation_max_new_tokens}
    pad = wrapper.tokenizer.pad_token_id or wrapper.tokenizer.eos_token_id
    return {"do_sample": False, "temperature": wrapper.cfg["temperature"],
            "top_p": wrapper.cfg["top_p"], "max_new_tokens": wrapper.generation_max_new_tokens,
            "pad_token_id": pad}


class OneStepIntervention(LogitsProcessor):
    def __init__(self, wrapper, model_name: str, image: Image.Image,
                 prompt_len: int, index: int, target_id: int, kind: str):
        self.wrapper = wrapper
        self.model_name = model_name
        self.image = image
        self.prompt_len = prompt_len
        self.index = index
        self.target_id = target_id
        self.kind = kind
        self.fired = False
        self.clean_top_id = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        position = int(input_ids.shape[1]) - self.prompt_len
        if position != self.index:
            return scores
        self.fired = True
        self.clean_top_id = int(scores[0].argmax().item())
        if self.clean_top_id != self.target_id:
            raise RuntimeError("Cached baseline prefix disagrees with current greedy generation")
        if self.kind == "rollback":
            scores[:, self.target_id] = -float("inf")
            return scores
        if self.kind != "vcd":
            raise ValueError(self.kind)
        prefix = input_ids[0, self.prompt_len:].tolist()
        blurred = self.image.filter(ImageFilter.GaussianBlur(radius=8.0))
        output = self.wrapper.extract_token_features(
            image=blurred, prefix_token_ids=prefix,
            response_token_idx=position, cfg_dgst_t=None, prompt=PROMPT,
            requirements=ExtractionRequirements(
                attention=AttentionRequirement.NONE, logits=True,
                token_hidden_states=False, patch_hidden_states=False,
                visual_layout=False, dgst_capture=False,
            ),
        )
        negative = output.token_logits.to(device=scores.device, dtype=scores.dtype)
        if negative.shape != scores[0].shape:
            raise ValueError("VCD negative branch vocabulary mismatch")
        # Preserve the original model's plausible candidates, as in VCD.
        top = scores[0].topk(min(20, scores.shape[-1])).indices
        adjusted = scores.clone()
        adjusted[0, top] += scores[0, top] - negative[top]
        return adjusted


@torch.inference_mode()
def _intervene(wrapper, model_name: str, image: Image.Image,
               trigger: dict, kind: str) -> tuple[list[int], dict]:
    inputs = _inputs(wrapper, image, model_name)
    prompt_len = int(inputs["input_ids"].shape[1])
    processor = OneStepIntervention(
        wrapper, model_name, image, prompt_len,
        int(trigger["token_index"]), int(trigger["token_id"]), kind,
    )
    kwargs = _generation_kwargs(wrapper, model_name)
    output = wrapper.model.generate(
        **inputs, **kwargs, logits_processor=LogitsProcessorList([processor]),
    )
    ids = [int(v) for v in output[0, prompt_len:].tolist()]
    if not processor.fired:
        raise RuntimeError("Intervention position was never reached")
    return ids, {"fired": processor.fired, "clean_top_id": processor.clean_top_id}


@torch.inference_mode()
def _beam_candidates(wrapper, model_name: str, image: Image.Image) -> list[tuple[list[int], float]]:
    inputs = _inputs(wrapper, image, model_name)
    prompt_len = int(inputs["input_ids"].shape[1])
    kwargs = _generation_kwargs(wrapper, model_name)
    output = wrapper.model.generate(
        **inputs, **kwargs, num_beams=3, num_return_sequences=3,
        return_dict_in_generate=True, output_scores=True,
    )
    return [
        ([int(v) for v in sequence[prompt_len:].tolist()], float(score))
        for sequence, score in zip(output.sequences, output.sequences_scores)
    ]


@torch.inference_mode()
def _score_caption(wrapper, scorer: RiskScorer, matcher,
                   image: Image.Image, ids: list[int], caption: str) -> tuple[list[dict], float]:
    mentions = _mentions(matcher, wrapper.tokenizer, ids, caption)
    seen = set()
    unique = []
    for mention in mentions:
        if mention["canonical"] not in seen:
            unique.append(mention)
            seen.add(mention["canonical"])
    if not unique:
        return [], 1.0
    outputs = wrapper.extract_token_features_batch(
        image=image, response_token_ids=ids,
        response_token_indices=[item["token_index"] for item in unique],
        target_token_ids=[item["token_id"] for item in unique],
        cfg_dgst_t=CFG_DGST, prompt=PROMPT, requirements=REQUIREMENTS,
    )
    risks = scorer.score_features(outputs)
    for item, risk in zip(unique, risks):
        item["hall_score"] = risk
    return unique, max(risks)


def _risk_beam(wrapper, scorer: RiskScorer, matcher, image: Image.Image,
               baseline: dict) -> tuple[list[int], dict]:
    candidates = _beam_candidates(wrapper, scorer.model_name, image)
    base_count = len(baseline["mentions"])
    base_length = len(baseline["ids"])
    rows = []
    for ids, sequence_score in candidates:
        caption = _decode(wrapper, ids, scorer.model_name)
        mentions, max_risk = _score_caption(wrapper, scorer, matcher, image, ids, caption)
        acceptable = (len(mentions) >= max(1, base_count - 1)
                      and len(ids) >= 0.8 * base_length)
        rows.append({"ids": ids, "caption": caption, "max_risk": max_risk,
                     "mentions": mentions, "sequence_score": sequence_score,
                     "acceptable": acceptable})
    accepted = [row for row in rows if row["acceptable"]]
    if not accepted:
        return baseline["ids"], {"beam_candidates": rows, "fallback": "no_acceptable_candidate"}
    chosen = min(accepted, key=lambda row: (row["max_risk"], -row["sequence_score"]))
    return chosen["ids"], {"beam_candidates": rows, "selected_max_risk": chosen["max_risk"]}


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def generate(model_name: str, split: str, limit: int | None,
             methods: list[str], device: str,
             shard_index: int = 0, num_shards: int = 1,
             start_offset: int = 0, end_offset: int | None = None) -> None:
    if model_name not in MODELS or split not in ("val", "test"):
        raise ValueError("Unsupported model or split")
    torch.set_num_threads(1)
    scorer = RiskScorer(model_name)
    matcher = _mention_matcher()
    config = load_config(str(ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"))
    wrapper = build_model(model_name, get_extraction_model_cfg(config, model_name), device=device)
    wrapper.model.eval().requires_grad_(False)
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index/count")
    if start_offset < 0 or (end_offset is not None and end_offset < start_offset):
        raise ValueError("Invalid shard offset range")
    image_ids = sorted(scorer.ids_by_split[split])[:limit][shard_index::num_shards]
    image_ids = image_ids[start_offset:end_offset]
    output = OUT / model_name / split
    paths = {method: output / f"{method}.jsonl" for method in methods}
    greedy_path = output / "greedy.jsonl"
    greedy_rows = {
        int(row["image_id"]): row
        for row in (json.loads(line) for line in greedy_path.read_text().splitlines() if line.strip())
    } if greedy_path.exists() else {}
    done = {method: {int(json.loads(line)["image_id"]) for line in path.read_text().splitlines()}
            if path.exists() else set() for method, path in paths.items()}
    audit = {"seed": SEED, "split": split, "model": model_name,
             "trigger_threshold": scorer.trigger_threshold,
             "methods": methods, "count": len(image_ids),
             "shard_index": shard_index, "num_shards": num_shards,
             "start_offset": start_offset, "end_offset": end_offset}
    output.mkdir(parents=True, exist_ok=True)
    range_label = f"{start_offset}_{end_offset if end_offset is not None else 'end'}"
    (output / f"run_config_shard{shard_index}_{range_label}.json").write_text(json.dumps(audit, indent=2))
    for ordinal, image_id in enumerate(image_ids, 1):
        todo = [method for method in methods if image_id not in done[method]]
        if not todo:
            continue
        image_path = coco / f"COCO_val2014_{image_id:012d}.jpg"
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if image_id in greedy_rows:
            prior = greedy_rows[image_id]
            baseline_ids = [int(v) for v in prior["response_token_ids"]]
            baseline_caption = str(prior["generated_text"])
            mentions = prior["mentions"]
        else:
            started = time.perf_counter()
            generated = wrapper.generate(image, prompt=PROMPT)
            baseline_ids = [int(v) for v in generated.response_token_ids]
            baseline_caption = str(generated.generated_text)
            mentions, _ = _score_caption(
                wrapper, scorer, matcher, image, baseline_ids, baseline_caption
            )
            prior = {"image_id": image_id, "generated_text": baseline_caption,
                     "response_token_ids": baseline_ids, "mentions": mentions,
                     "seconds": time.perf_counter() - started}
            _append_jsonl(greedy_path, prior)
            greedy_rows[image_id] = prior
        risky = [m for m in mentions if m["hall_score"] >= scorer.trigger_threshold]
        trigger = min(risky, key=lambda m: m["token_index"]) if risky else None
        baseline = {"ids": baseline_ids, "caption": baseline_caption, "mentions": mentions}
        for method in todo:
            started = time.perf_counter()
            details: dict = {}
            ids = baseline_ids
            if trigger is not None:
                if method == "risk_beam":
                    ids, details = _risk_beam(wrapper, scorer, matcher, image, baseline)
                elif method == "risk_vcd":
                    ids, details = _intervene(wrapper, model_name, image, trigger, "vcd")
                elif method == "risk_rollback":
                    ids, details = _intervene(wrapper, model_name, image, trigger, "rollback")
                else:
                    raise ValueError(method)
            caption = _decode(wrapper, ids, model_name)
            row = {"image_id": image_id, "method": method, "seed": SEED,
                   "generated_text": caption, "response_token_ids": ids,
                   "baseline_text": baseline_caption, "trigger": trigger,
                   "baseline_mentions": mentions, "details": details,
                   "seconds": time.perf_counter() - started}
            _append_jsonl(paths[method], row)
            print(f"[{model_name} {split} {ordinal}/{len(image_ids)}] {method}: "
                  f"trigger={trigger is not None} changed={ids != baseline_ids}", flush=True)


def evaluate(model_name: str, split: str, limit: int | None) -> None:
    if model_name not in MODELS or split not in ("val", "test"):
        raise ValueError("Unsupported model or split")
    config = load_config(str(ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"))
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=OUT / "chair_cache.pkl",
    )
    split_ids = sorted(_read_json(PROBES / "splits" / f"seed{SEED}.json")[split])[:limit]
    folder = OUT / model_name / split
    greedy_path = folder / "greedy.jsonl"
    if not greedy_path.exists():
        raise FileNotFoundError(greedy_path)
    baseline = {int(row["image_id"]): row["generated_text"]
                for row in (json.loads(line) for line in greedy_path.read_text().splitlines() if line.strip())}
    if any(image_id not in baseline for image_id in split_ids):
        raise ValueError("Greedy baseline missing paired images")
    variants: dict[str, dict[int, str]] = {"greedy": baseline}
    generation_rows: dict[str, dict[int, dict]] = {}
    for method in METHODS:
        path = folder / f"{method}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        indexed_rows = {int(row["image_id"]): row for row in rows}
        indexed = {image_id: row["generated_text"] for image_id, row in indexed_rows.items()}
        if any(image_id not in indexed for image_id in split_ids):
            raise ValueError(f"{method}: missing paired images")
        variants[method] = indexed
        generation_rows[method] = indexed_rows
    if "risk_beam" in generation_rows:
        variants["beam_top"] = {}
        for image_id in split_ids:
            row = generation_rows["risk_beam"][image_id]
            candidates = row["details"].get("beam_candidates", [])
            variants["beam_top"][image_id] = (
                max(candidates, key=lambda x: x["sequence_score"])["caption"]
                if candidates else baseline[image_id]
            )
    per_image: dict[str, dict[int, dict]] = {}
    summary = {"model": model_name, "split": split, "seed": SEED,
               "images": len(split_ids), "methods": {}}
    for method, captions in variants.items():
        per_image[method] = {
            image_id: evaluator.compute_chair_token(image_id, captions[image_id])
            for image_id in split_ids
        }
        rows = list(per_image[method].values())
        total_mentions = sum(len(row["object_mentions"]) for row in rows)
        hall_mentions = sum(len(row["mscoco_hallucinated_words"]) for row in rows)
        summary["methods"][method] = {
            "chair_s": np.mean([row["metrics"]["CHAIRs"] for row in rows]).item(),
            "chair_i": hall_mentions / total_mentions if total_mentions else math.nan,
            "object_recall": np.mean([row["metrics"]["Recall"] for row in rows]).item(),
            "true_mentions_per_image": np.mean([len(row["mscoco_real_words"]) for row in rows]).item(),
            "all_mentions_per_image": total_mentions / len(rows),
            "words_per_image": np.mean([len(row["words"]) for row in rows]).item(),
        }
        if method in generation_rows:
            source_rows = [generation_rows[method][image_id] for image_id in split_ids]
            summary["methods"][method].update({
                "triggered_images": sum(row["trigger"] is not None for row in source_rows),
                "changed_images": sum(row["generated_text"] != baseline[row["image_id"]]
                                      for row in source_rows),
                "intervention_seconds_per_image": np.mean([row["seconds"] for row in source_rows]).item(),
            })
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(split_ids), size=(2000, len(split_ids)))
    base_rows = [per_image["greedy"][image_id] for image_id in split_ids]
    base_chairs = np.asarray([row["metrics"]["CHAIRs"] for row in base_rows])
    base_hall = np.asarray([len(row["mscoco_hallucinated_words"]) for row in base_rows])
    base_mentions = np.asarray([len(row["object_mentions"]) for row in base_rows])
    for method in variants:
        if method == "greedy":
            continue
        rows = [per_image[method][image_id] for image_id in split_ids]
        chairs = np.asarray([row["metrics"]["CHAIRs"] for row in rows])
        hall = np.asarray([len(row["mscoco_hallucinated_words"]) for row in rows])
        mentions = np.asarray([len(row["object_mentions"]) for row in rows])
        delta_s = chairs[draws].mean(axis=1) - base_chairs[draws].mean(axis=1)
        delta_i = (hall[draws].sum(axis=1) / np.maximum(1, mentions[draws].sum(axis=1))
                   - base_hall[draws].sum(axis=1) / np.maximum(1, base_mentions[draws].sum(axis=1)))
        summary["methods"][method]["paired_delta_chair_s_ci95"] = np.quantile(delta_s, [0.025, 0.975]).tolist()
        summary["methods"][method]["paired_delta_chair_i_ci95"] = np.quantile(delta_i, [0.025, 0.975]).tolist()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "chair_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--end-offset", type=int)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.model, args.split, args.limit, args.methods, args.device,
                 args.shard_index, args.num_shards,
                 args.start_offset, args.end_offset)
    else:
        evaluate(args.model, args.split, args.limit)


if __name__ == "__main__":
    main()
