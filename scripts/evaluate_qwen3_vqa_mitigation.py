#!/usr/bin/env python3
"""Paired Qwen3-VL mitigation evaluation on the fixed POPE and AMBER test splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import copy
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.qa_benchmark import atomic_write_json, label_answer, load_jsonl, normalize_yes_no
from features.llava_decoding_evidence import lexical_nouns
from features.qa_extractor import qa_prompt
from features.qwen3_sar_ems_attention import Qwen3LateAttention, VQA_OBJECT_VARIANT
from models import build_model
from models.base_wrapper import PromptTargetRequest
from models.prompt_target import resolve_prompt_target_alignment, truncate_multimodal_inputs
from scripts.generation_time_sar_ems_seed43 import _append_jsonl, _generation_kwargs, _read_json
from utils.config_utils import get_extraction_model_cfg, load_config

MODEL = "qwen3_vl_8b"
CONFIG = ROOT / "configs/model_configs_qa_sam_svar_pope_amber_server.yaml"
DATA = Path("/root/rivermind-data/dataset/qa_sam_svar_base")
OUT = ROOT / "outputs/qa_benchmarks/qwen3_vqa_sadi_mitigation"
DATASETS = ("pope", "amber_discriminative")
VARIANTS = ("greedy", "sadi", "sadi_sar_ems_attention", VQA_OBJECT_VARIANT)


def test_questions(data_root, dataset):
    folder = data_root / dataset
    split = _read_json(folder / "image_splits.json")
    test_images = set(map(int, split["test"]))
    grouped = defaultdict(list)
    keys = set()
    for row in load_jsonl(folder / "questions.jsonl"):
        image_id = int(row["image_id"])
        if image_id in test_images:
            if row["key"] in keys:
                raise ValueError(f"Duplicate {dataset} question {row['key']}")
            if normalize_yes_no(row["gt_answer"]) is None:
                raise ValueError(f"Missing binary GT for {row['key']}")
            keys.add(row["key"])
            grouped[image_id].append(row)
    if test_images != grouped.keys():
        raise RuntimeError(f"Incomplete {dataset} test image split")
    return dict(sorted(grouped.items()))


def fork_cache(cache):
    # Dynamic layers append by concatenation, so each branch can share the original tensors.
    branch = copy(cache)
    branch.layers = [copy(layer) for layer in cache.layers]
    return branch


def cached_answer(model, suffix, prefix, rope_delta, eos, limit):
    cache = fork_cache(prefix)
    tokens = []
    for _ in range(limit):
        start = cache.get_seq_length()
        position = torch.arange(start, start + suffix.shape[1], device=suffix.device)
        position_ids = (position[None, :] + rope_delta).unsqueeze(0).expand(3, -1, -1)
        result = model(input_ids=suffix, position_ids=position_ids, cache_position=position,
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        suffix = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = int(suffix.item())
        tokens.append(token)
        if token in eos:
            break
    return tokens


def align_object(wrapper, inputs, prompt, row, visual_span):
    if row.get("object_span_status") != "found":
        return None
    request = PromptTargetRequest(
        prompt=prompt, target_text=row["object_surface"],
        target_char_start=row["object_char_start"], target_char_end=row["object_char_end"],
    )
    result = resolve_prompt_target_alignment(
        tokenizer=wrapper.tokenizer, full_input_ids=inputs["input_ids"][0].tolist(),
        request=request, image_token_id=wrapper._image_token_id,
        visual_token_count=visual_span[1] - visual_span[0],
    )
    return result.target_token_id


def decoded_answer(wrapper, tokens):
    return wrapper.processor.decode(tokens, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config["hf_name"] = str(args.model_root)
    config["max_new_tokens"] = 8
    config["max_pixels"] = args.max_pixels
    wrapper = build_model(MODEL, config, device=args.device)
    model = wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer, word_start="Ġ")
    if not nouns:
        raise RuntimeError("The Qwen3 WordNet noun gate has no eligible tokens")
    yes_no = [wrapper.tokenizer.encode(text, add_special_tokens=False) for text in ("Yes", "No")]
    if any(len(ids) != 1 for ids in yes_no):
        raise RuntimeError("Expected one canonical Qwen3 token each for Yes and No")
    yes_no_ids = tuple(ids[0] for ids in yes_no)
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    limit = wrapper.generation_max_new_tokens
    if limit != 8:
        raise ValueError(f"Expected historic VQA 8-token cap, got {limit}")
    native_kwargs = _generation_kwargs(wrapper, MODEL)

    for dataset in args.datasets:
        groups = list(test_questions(args.data_root, dataset).items())[args.shard::args.shards]
        if args.smoke:
            groups = groups[:2]
        filename = args.output_root / f"{dataset}.shard{args.shard}.jsonl"
        existing = load_jsonl(filename) if filename.exists() else []
        if len({row["key"] for row in existing}) != len(existing):
            raise ValueError(f"Duplicate saved questions in {filename}")
        done = {row["key"] for row in existing}
        if not done.issubset({row["key"] for _, questions in groups for row in questions}) and not args.smoke:
            raise ValueError(f"Existing rows do not belong to {dataset}/{args.shard}")
        print(f"[{dataset}] {len(groups)} images, {sum(len(rows) for _, rows in groups)} questions, "
              f"max_new_tokens={limit}, yes_no={yes_no_ids}", flush=True)
        for index, (image_id, questions) in enumerate(groups, 1):
            pending = [row for row in questions if args.smoke or row["key"] not in done]
            if not pending:
                continue
            if args.smoke:
                pending = pending[:2]
            with Image.open(pending[0]["image_path"]) as image:
                image = image.convert("RGB")
            shared_prefix = prefix_cache = rope_delta = controller = None
            started = time.perf_counter()
            for row in pending:
                prompt = qa_prompt(MODEL, row["question"])
                inputs = wrapper._prompt_inputs(image, prompt).to(wrapper.device)
                input_ids = inputs["input_ids"]
                if not inputs["attention_mask"].all().item():
                    raise ValueError("Expected one unpadded question per model forward")
                span = wrapper._find_vision_token_range(input_ids[0])
                if controller is None:
                    prefix_length = span[1] + 1
                    if int(input_ids[0, span[1]]) != wrapper._vision_end_id:
                        raise ValueError("Question prefix does not end with Qwen3's image boundary")
                    shared_prefix = input_ids[:, :prefix_length].clone()
                    controller = Qwen3LateAttention(model, span, nouns, "greedy", yes_no_ids=yes_no_ids)
                    prefix_inputs = truncate_multimodal_inputs(inputs, tokenized_end=prefix_length)
                    with controller.capture_visual_prefix():
                        prefix = model(**prefix_inputs, use_cache=True, logits_to_keep=1)
                    prefix_cache = prefix.past_key_values
                    rope_delta = prefix.rope_deltas.detach().clone()
                    controller.use_visual_prefix_cache(prefix_cache)
                if (span != (controller.start, controller.end)
                        or not torch.equal(shared_prefix, input_ids[:, :prefix_length])):
                    raise ValueError(f"Incompatible shared image prefix for {row['key']}")

                object_token = align_object(wrapper, inputs, prompt, row, span)
                outputs = {}
                for name in VARIANTS:
                    controller.reset(name, object_token=object_token if name == VQA_OBJECT_VARIANT else None)
                    with controller.capture():
                        tokens = cached_answer(model, input_ids[:, prefix_length:], prefix_cache,
                                               rope_delta, eos, limit)
                    text = decoded_answer(wrapper, tokens)
                    stats = controller.statistics(tokens) if name.startswith("sadi_sar_ems") else {}
                    if args.smoke:
                        native = Qwen3LateAttention(model, span, nouns, name,
                                                   object_token=object_token if name == VQA_OBJECT_VARIANT else None,
                                                   yes_no_ids=yes_no_ids)
                        with native.capture():
                            response = model.generate(**inputs, **native_kwargs)
                        native_tokens = response[0, input_ids.shape[1]:].tolist()
                        if normalize_yes_no(text) != normalize_yes_no(decoded_answer(wrapper, native_tokens)):
                            raise RuntimeError(f"Cached vs native answer mismatch for {row['key']} {name}: "
                                               f"{tokens} vs {native_tokens}")
                        native_stats = native.statistics(native_tokens) if name.startswith("sadi_sar_ems") else {}
                        if native_stats.get("guided_steps") != stats.get("guided_steps"):
                            raise RuntimeError(f"Cached vs native trigger mismatch for {row['key']} {name}")
                        print(f"[smoke] {row['key']} {name}: {text!r} "
                              f"tokens_equal={tokens == native_tokens} guided={stats.get('guided_steps', 0)}",
                              flush=True)
                    outputs[name] = {
                        "answer": normalize_yes_no(text), "text": text, "response_token_ids": tokens,
                        **stats,
                    }
                if not args.smoke:
                    _append_jsonl(filename, {
                        "key": row["key"], "image_id": image_id, "gt_answer": normalize_yes_no(row["gt_answer"]),
                        "object_token_id": object_token, "methods": outputs,
                    })
            print(f"[{dataset}:{args.shard}:{index}/{len(groups)}] image={image_id} "
                  f"questions={len(pending)} seconds={time.perf_counter()-started:.2f}", flush=True)


def summarize(args):
    summary = {}
    for dataset in args.datasets:
        grouped = test_questions(args.data_root, dataset)
        expected = {r["key"]: r for group in grouped.values() for r in group}
        all_rows = [row for path in sorted(args.output_root.glob(f"{dataset}.shard*.jsonl"))
                    for row in load_jsonl(path)]
        if len(all_rows) != len(expected) or {r["key"] for r in all_rows} != expected.keys():
            raise ValueError(f"Expected exactly {len(expected)} distinct paired questions for {dataset}")
        for row in all_rows:
            original = expected[row["key"]]
            if (row["gt_answer"] != normalize_yes_no(original["gt_answer"])
                    or row["image_id"] != original["image_id"] or set(row["methods"]) != set(VARIANTS)):
                raise ValueError(f"Question or variants differ from the fixed test split: {row['key']}")

        truth = np.array([r["gt_answer"] == "yes" for r in all_rows])
        image_to_index = {image: index for index, image in enumerate(grouped)}
        images = np.array([image_to_index[r["image_id"]] for r in all_rows])
        samples = np.random.default_rng(42).integers(0, len(grouped), size=(5000, len(grouped)))
        bootstrap = {}
        methods = {}
        for name in VARIANTS:
            responses = [r["methods"][name] for r in all_rows]
            pred = np.array([r["answer"] or "" for r in responses])
            correct = (truth & (pred == "yes")) | (~truth & (pred == "no"))
            counters = Counter(label_answer(r["answer"], row["gt_answer"])[1]
                               for r, row in zip(responses, all_rows, strict=True))
            tp, tn, fp, fn = (counters[k] for k in ("correct_yes", "correct_no", "false_positive", "false_negative"))
            positives, negatives = int(truth.sum()), int((~truth).sum())
            methods[name] = {
                "counts": {key: counters[key] for key in
                           ("correct_yes", "correct_no", "false_positive", "false_negative", "invalid")},
                "accuracy": (tp + tn) / len(all_rows),
                "yes_f1": 2 * tp / (tp + fp + positives) if positives else None,
                "false_yes_rate": fp / negatives,
                "false_no_rate": fn / positives if positives else None,
                "guided_questions": sum(int(response.get("guided_steps", 0) > 0) for response in responses),
                "guided_steps": sum(response.get("guided_steps", 0) for response in responses),
                "paired_vs_sadi": Counter(
                    f"{row['gt_answer']}:{row['methods']['sadi']['answer']}->{response['answer']}"
                    for row, response in zip(all_rows, responses, strict=True)
                    if row["methods"]["sadi"]["answer"] != response["answer"]
                ),
            }
            image_counts = lambda mask: np.bincount(images, weights=mask, minlength=len(grouped))[samples].sum(axis=1)
            bootstrap[name] = {
                "accuracy": image_counts(correct) / image_counts(np.ones_like(truth)),
                "false_yes_rate": image_counts(~truth & (pred == "yes")) / image_counts(~truth),
            }
        for name, estimates in bootstrap.items():
            methods[name]["paired_vs_sadi_ci95"] = {
                metric: np.quantile(values - bootstrap["sadi"][metric], [0.025, 0.975]).tolist()
                for metric, values in estimates.items()
            }
        summary[dataset] = {"questions": len(all_rows), "images": len(grouped),
                            "gt_yes": int(truth.sum()), "gt_no": int((~truth).sum()),
                            "object_alignment_count": sum(row["object_token_id"] is not None for row in all_rows),
                            "methods": methods}
    atomic_write_json(args.output_root / "summary.json", summary)
    print(args.output_root / "summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--model-root", type=Path, default=Path("/root/rivermind-data/models/qwen3_vl_8b"))
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.max_pixels <= 0:
        parser.error("Expected a valid shard and positive image pixel budget")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (summarize if args.summarize else run)(args)


if __name__ == "__main__":
    main()
