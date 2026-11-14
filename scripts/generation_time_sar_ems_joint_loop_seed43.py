#!/usr/bin/env python3
"""Joint mean-absolute-logit/attention-output steering: one versus 32 rollbacks."""

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
import torch.nn.functional as F
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from features.logs_risk_f_log import compute_logs_risk_f_log_batch
from models.dgst_capture import (
    apply_decoder_final_norm, pre_token_prediction_positions,
    resolve_decoder_final_norm, resolve_decoder_layers,
    resolve_output_embedding_layer, resolve_prompt_positions,
    run_forward_with_dgst_captures, target_probabilities_multi,
)
from models.llava_wrapper import _append_prefix_token_ids
from scripts.generation_time_closed_loop_tau05_seed43 import score_all_mentions
from scripts.generation_time_sar_ems_local_intervention_seed43 import (
    BASELINE, CONFIG, FFN_POOL, LIMIT, MAP_LAYERS, MAX_FFN_LAYERS,
    MODEL, PATCH_TOP_K, SEED, SPLIT,
    _chair_module, _decode, _generation_kwargs, _inputs, _mention_matcher,
    object_token_ids, plan_intervention, read_rows, training_stats,
)
from scripts.generation_time_sar_ems_seed43 import (
    PROMPT, RiskScorer, ROOT, _mentions,
)
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


THRESHOLD = 0.4013161063194275
MAX_ROUNDS = 32
WINDOW = 12
ATTENTION_LAYERS = tuple(range(5, 19))
ATTENTION_MEAN_FACTOR = 0.7
FFN_ATTENTION_FACTOR = 0.4
OUT = ROOT / "outputs/generation_time_sar_ems_joint_loop_absmean_seed43/llava_1_5_7b/test_100"
OUT_PREFIX = ROOT / "outputs/generation_time_sar_ems_joint_loop_prefix_seed43/llava_1_5_7b/test_100"
OUT_FUSED = ROOT / "outputs/generation_time_sar_ems_joint_loop_fused_seed43/llava_1_5_7b/test_100"
OUT_FUSED_64 = ROOT / "outputs/generation_time_sar_ems_joint_loop_fused_64_seed43/llava_1_5_7b/test_100"
OUT_FUSED_64_VIS_FFN = ROOT / "outputs/generation_time_sar_ems_joint_loop_fused_64_vis_ffn_seed43/llava_1_5_7b/test_100"


@torch.inference_mode()
def score_and_plan(wrapper, scorer, matcher, image: Image.Image, ids: list[int],
                   object_tokens, stats):
    """Score every mention and plan the first trigger from one causal forward."""
    caption = _decode(wrapper, ids, MODEL)
    mentions = _mentions(matcher, wrapper.tokenizer, ids, caption)
    if not mentions:
        return caption, mentions, None, None
    prompt_inputs = _inputs(wrapper, image, MODEL)
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    full_inputs = _append_prefix_token_ids(
        prompt_inputs, prefix_token_ids=ids, device=wrapper.device,
        dtype=torch.float16)
    full_ids = full_inputs["input_ids"][0].tolist()
    image_token_id = wrapper._image_token_id()
    visual_start, visual_end = wrapper._find_visual_token_range(full_inputs, image_token_id)
    prompt_positions = resolve_prompt_positions(
        full_input_ids=full_ids, prompt_tokenized_length=prompt_length,
        image_token_id=image_token_id, visual_start=visual_start,
        visual_end=visual_end)
    prediction_positions = pre_token_prediction_positions(
        full_input_ids=full_ids, prompt_tokenized_length=prompt_length,
        response_token_indices=[mention["token_index"] for mention in mentions],
        image_token_id=image_token_id,
        visual_token_count=visual_end - visual_start,
        prompt_positions=prompt_positions)
    out, captures = run_forward_with_dgst_captures(
        wrapper.model, output_hidden_states=False,
        retain_attention_updates=True,
        attention_query_positions=prediction_positions,
        capture_device=wrapper.dgst_capture_device,
        attention_query_chunk_size=wrapper.dgst_attention_query_chunk_size,
        record_model_attentions=False, **full_inputs)
    features = compute_logs_risk_f_log_batch(
        model=wrapper.model, captures=captures,
        visual_start=visual_start, visual_end=visual_end,
        target_token_ids=[mention["token_id"] for mention in mentions],
        prediction_positions=prediction_positions, tau=0.03,
        transport_top_k=32, semantic_chunk_size=64, ot_solver="emd")
    vectors = np.stack([
        np.concatenate((np.asarray(item["f_log_per_layer"], dtype=np.float32),
                        np.asarray(item["logs_risk_per_layer"], dtype=np.float32)))
        for item in features])
    risks = scorer.score_matrix(vectors)
    for mention, risk in zip(mentions, risks):
        mention["hall_score"] = float(risk)
    offset = next((index for index, mention in enumerate(mentions)
                   if mention["hall_score"] >= THRESHOLD), None)
    if offset is None:
        del out, captures
        return caption, mentions, None, None
    trigger = mentions[offset]
    ems = np.asarray(features[offset]["logs_risk_per_layer"], dtype=np.float64)
    excess = (ems - stats["ems_real_q90"]) / np.maximum(stats["ems_real_iqr"], 1e-6)
    selected_ffn = sorted(FFN_POOL, key=lambda layer: excess[layer], reverse=True)[:MAX_FFN_LAYERS]
    final_norm = resolve_decoder_final_norm(wrapper.model)
    output_head = resolve_output_embedding_layer(wrapper.model)
    categories = [(name, token_id) for name, token_id in object_tokens
                  if name != trigger["canonical"]]
    target_ids = [token_id for _, token_id in categories]
    vocab_size = int(output_head.weight.shape[0])
    patch_mask = {}
    for layer in MAP_LAYERS:
        visual = captures[layer]["h_prev"][0, visual_start:visual_end].float()
        normalized = apply_decoder_final_norm(final_norm, visual)
        probabilities = target_probabilities_multi(
            output_layer=output_head, states=normalized,
            target_token_ids=target_ids, chunk_size=64)
        semantic = torch.log1p(vocab_size * probabilities.float()).amax(dim=1)
        if semantic.numel() == 576:
            semantic = F.avg_pool2d(semantic.reshape(1, 1, 24, 24), 3, 1, 1).flatten()
        indices = semantic.topk(min(PATCH_TOP_K, semantic.numel())).indices
        mask = torch.zeros_like(semantic, dtype=torch.bool)
        mask[indices] = True
        patch_mask[layer] = mask
    del out, captures
    return caption, mentions, trigger, {
        "visual_start": visual_start, "visual_end": visual_end,
        "selected_ffn": selected_ffn, "patch_mask": patch_mask,
    }


class ForcePrefixBans(LogitsProcessor):
    """Replay the current caption up to the rollback point, then honor prefix bans."""

    def __init__(self, prompt_length: int, current: list[int], rollback_index: int,
                 bans: dict[tuple[int, ...], set[int]]):
        self.prompt_length = prompt_length
        self.current = current
        self.rollback_index = rollback_index
        self.bans = bans
        self.ban_applied = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        prefix = tuple(int(x) for x in input_ids[0, self.prompt_length:].tolist())
        if len(prefix) < self.rollback_index:
            expected = self.current[len(prefix)]
            if expected in self.bans.get(prefix, set()):
                raise RuntimeError("Forced prefix contains a previously banned token")
            forced = torch.full_like(scores, -torch.inf)
            forced[:, expected] = scores[:, expected]
            return forced
        forbidden = self.bans.get(prefix)
        if not forbidden:
            return scores
        changed = scores.clone()
        changed[:, list(forbidden)] = -torch.inf
        if not torch.isfinite(changed).any():
            raise RuntimeError("All logits banned at a prefix")
        self.ban_applied += 1
        return changed


@contextmanager
def joint_intervention(model, plan: dict, trigger_index: int,
                       ffn_visual_only: bool = False):
    layers = resolve_decoder_layers(model)
    selected_ffn = set(plan["selected_ffn"])
    patched = set(ATTENTION_LAYERS) | selected_ffn
    calls = {layer: 0 for layer in patched}
    originals = {}
    handles = []
    attention_outputs: dict[int, torch.Tensor] = {}
    applied = {"attention": 0, "ffn": 0, "mean_abs_logit_sum": None,
               "raw_mean_logit_sum": None, "mean_logit_count": 0,
               "negative_raw_mean_count": None,
               "ffn_visual_mass_sum": 0.0, "ffn_visual_norm_sum": 0.0,
               "ffn_full_attention_norm_sum": 0.0, "ffn_output_norm_sum": 0.0,
               "ffn_added_norm_sum": 0.0}
    visual_start, visual_end = plan["visual_start"], plan["visual_end"]

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
        in_window = trigger_index <= step < trigger_index + WINDOW
        if in_window and layer in ATTENTION_LAYERS:
            visual_scores = scores[:, :, -1, visual_start:visual_end]
            raw_mean_logit = visual_scores.float().mean(dim=-1, keepdim=True)
            mean_abs_logit = visual_scores.float().abs().mean(dim=-1, keepdim=True)
            patch_mask = plan["patch_mask"][layer].to(device=scores.device, dtype=scores.dtype)
            visual_scores += (ATTENTION_MEAN_FACTOR * mean_abs_logit.to(scores.dtype)
                              * patch_mask[None, None, :])
            applied["attention"] += 1
            total = mean_abs_logit.sum().detach()
            raw_total = raw_mean_logit.sum().detach()
            negative = (raw_mean_logit < 0).sum().detach()
            applied["mean_abs_logit_sum"] = (total if applied["mean_abs_logit_sum"] is None
                                              else applied["mean_abs_logit_sum"] + total)
            applied["raw_mean_logit_sum"] = (raw_total if applied["raw_mean_logit_sum"] is None
                                              else applied["raw_mean_logit_sum"] + raw_total)
            applied["negative_raw_mean_count"] = (negative if applied["negative_raw_mean_count"] is None
                                                   else applied["negative_raw_mean_count"] + negative)
            applied["mean_logit_count"] += mean_abs_logit.numel()
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = self.o_proj(output.reshape(*input_shape, -1).contiguous())
        if in_window and layer in selected_ffn:
            if ffn_visual_only:
                visual_weights = weights[:, :, -1, visual_start:visual_end]
                visual_values = value[:, :, visual_start:visual_end, :]
                visual_heads = torch.matmul(
                    visual_weights.unsqueeze(-2), visual_values).squeeze(-2)
                visual_vector = F.linear(
                    visual_heads.contiguous().reshape(input_shape[0], -1),
                    self.o_proj.weight, bias=None)
                attention_outputs[layer] = (visual_vector, output[:, -1, :])
                applied["ffn_visual_mass_sum"] += float(visual_weights.float().sum(dim=-1).mean().item())
            else:
                attention_outputs[layer] = output[:, -1, :]
        return output, weights

    def make_ffn_hook(layer: int):
        def hook(_module, _args, output):
            if not trigger_index <= calls[layer] - 1 < trigger_index + WINDOW:
                return output
            attention = attention_outputs.pop(layer, None)
            if attention is None:
                raise RuntimeError(f"Missing attention output at FFN layer {layer}")
            changed = output.clone()
            if ffn_visual_only:
                visual, full_attention = attention
                ffn_norm = output[:, -1, :].float().norm(dim=-1, keepdim=True)
                visual_norm = visual.float().norm(dim=-1, keepdim=True)
                addition = (FFN_ATTENTION_FACTOR * ffn_norm / visual_norm.clamp_min(1e-6)
                            * visual.float()).to(output.dtype)
                changed[:, -1, :] += addition
                applied["ffn_visual_norm_sum"] += float(visual_norm.mean().item())
                applied["ffn_full_attention_norm_sum"] += float(full_attention.float().norm(dim=-1).mean().item())
                applied["ffn_output_norm_sum"] += float(ffn_norm.mean().item())
                applied["ffn_added_norm_sum"] += float(addition.float().norm(dim=-1).mean().item())
            else:
                changed[:, -1, :] += FFN_ATTENTION_FACTOR * attention.to(output.dtype)
            applied["ffn"] += 1
            return changed
        return hook

    try:
        for layer in patched:
            module = layers[layer].self_attn
            originals[layer] = module.forward
            module.forward = types.MethodType(guided_forward, module)
        for layer in selected_ffn:
            handles.append(layers[layer].mlp.register_forward_hook(make_ffn_hook(layer)))
        yield applied
    finally:
        for handle in handles:
            handle.remove()
        for layer, forward in originals.items():
            layers[layer].self_attn.forward = forward


@torch.inference_mode()
def intervene_once(wrapper, image: Image.Image, ids: list[int], trigger: dict,
                   bans: dict[tuple[int, ...], set[int]], object_tokens, stats,
                   prefix_prefill: bool = False, plan: dict | None = None,
                   ffn_visual_only: bool = False) -> tuple[list[int], dict]:
    index = int(trigger["token_index"])
    token = int(ids[index])
    if token != int(trigger["token_id"]):
        raise RuntimeError("Trigger token is not aligned with current caption")
    prefix = tuple(ids[:index])
    forbidden = bans.setdefault(prefix, set())
    if token in forbidden:
        raise RuntimeError("Current caption contains a token banned at its own prefix")
    forbidden.add(token)
    started = time.perf_counter()
    if plan is None:
        plan = plan_intervention(
            wrapper, image, ids[:index], trigger, object_tokens, stats,
            always_ffn=True, require_greedy_trigger=False)
    plan_seconds = time.perf_counter() - started
    if len(plan["selected_ffn"]) != 3:
        raise RuntimeError("Joint intervention did not select three FFN layers")
    inputs = _inputs(wrapper, image, MODEL)
    prompt_length = int(inputs["input_ids"].shape[1])
    generation_kwargs = _generation_kwargs(wrapper, MODEL)
    if prefix_prefill:
        inputs = _append_prefix_token_ids(
            inputs, prefix_token_ids=ids[:index], device=wrapper.device,
            dtype=torch.float16)
        generation_kwargs["max_new_tokens"] -= index
        if generation_kwargs["max_new_tokens"] < 1:
            raise RuntimeError("No generation budget remains at rollback position")
    processor = ForcePrefixBans(prompt_length, ids, index, bans)
    generated_at = time.perf_counter()
    with joint_intervention(wrapper.model, plan, 0 if prefix_prefill else index,
                            ffn_visual_only=ffn_visual_only) as applied:
        generated = wrapper.model.generate(
            **inputs, **generation_kwargs,
            logits_processor=LogitsProcessorList([processor]))
    new_ids = [int(x) for x in generated[0, prompt_length:].tolist()]
    if new_ids[:index] != ids[:index] or processor.ban_applied < 1:
        raise RuntimeError("Rollback failed to preserve the prefix or apply the ban")
    if len(new_ids) <= index or new_ids[index] == token:
        raise RuntimeError("Banned trigger token was generated again")
    if applied["attention"] < len(ATTENTION_LAYERS) or applied["ffn"] < 3:
        raise RuntimeError("Joint attention and FFN hooks did not fire")
    record = {
        "word": trigger["canonical"], "surface": trigger["surface"],
        "score": trigger["hall_score"], "token_index": index,
        "banned_first_id": token, "replacement_first_id": new_ids[index],
        "bans_at_prefix": len(forbidden), "selected_ffn": plan["selected_ffn"],
        "applied_attention": applied["attention"], "applied_ffn": applied["ffn"],
        "mean_abs_image_logit": float(applied["mean_abs_logit_sum"] / applied["mean_logit_count"]),
        "raw_mean_image_logit": float(applied["raw_mean_logit_sum"] / applied["mean_logit_count"]),
        "negative_raw_mean_fraction": float(applied["negative_raw_mean_count"] / applied["mean_logit_count"]),
        "plan_seconds": plan_seconds,
        "generation_seconds": time.perf_counter() - generated_at,
        "prefix_prefill": prefix_prefill,
        "ffn_visual_only": ffn_visual_only,
    }
    if ffn_visual_only:
        record.update({
            "mean_ffn_visual_mass": applied["ffn_visual_mass_sum"] / applied["ffn"],
            "mean_ffn_visual_norm": applied["ffn_visual_norm_sum"] / applied["ffn"],
            "mean_ffn_full_attention_norm": applied["ffn_full_attention_norm_sum"] / applied["ffn"],
            "mean_ffn_output_norm": applied["ffn_output_norm_sum"] / applied["ffn"],
            "mean_ffn_added_norm": applied["ffn_added_norm_sum"] / applied["ffn"],
        })
    return new_ids, record


@torch.inference_mode()
def generate_image(wrapper, scorer, matcher, image: Image.Image, image_id: int,
                   baseline: dict, object_tokens, stats,
                   prefix_prefill: bool = False,
                   fused_forward: bool = False,
                   max_rounds: int = MAX_ROUNDS,
                   ffn_visual_only: bool = False) -> dict:
    ids = [int(x) for x in baseline["response_token_ids"]]
    if fused_forward:
        initial_caption, mentions, trigger, plan = score_and_plan(
            wrapper, scorer, matcher, image, ids, object_tokens, stats)
    else:
        initial_caption, mentions = score_all_mentions(wrapper, scorer, matcher, image, ids)
        trigger, plan = None, None
    initial_risky = [m for m in mentions if m["hall_score"] >= THRESHOLD]
    bans: dict[tuple[int, ...], set[int]] = {}
    actions = []
    first_ids = ids
    first_caption = initial_caption
    first_risky = initial_risky
    for _ in range(max_rounds):
        if not fused_forward:
            trigger = next((m for m in mentions if m["hall_score"] >= THRESHOLD), None)
        if trigger is None:
            break
        ids, action = intervene_once(
            wrapper, image, ids, trigger, bans, object_tokens, stats,
            prefix_prefill=prefix_prefill, plan=plan,
            ffn_visual_only=ffn_visual_only)
        actions.append(action)
        if fused_forward:
            caption, mentions, trigger, plan = score_and_plan(
                wrapper, scorer, matcher, image, ids, object_tokens, stats)
        else:
            caption, mentions = score_all_mentions(wrapper, scorer, matcher, image, ids)
        if len(actions) == 1:
            first_ids = ids.copy()
            first_caption = caption
            first_risky = [m for m in mentions if m["hall_score"] >= THRESHOLD]
    final_caption = _decode(wrapper, ids, MODEL)
    final_risky = [m for m in mentions if m["hall_score"] >= THRESHOLD]
    return {
        "image_id": image_id, "initial_risky_count": len(initial_risky),
        "first_only": {"generated_text": first_caption, "response_token_ids": first_ids,
                       "remaining_risky_count": len(first_risky),
                       "changed": first_ids != baseline["response_token_ids"]},
        f"repeat_{max_rounds}": {"generated_text": final_caption, "response_token_ids": ids,
                                "remaining_risky_count": len(final_risky),
                                "stopping_reason": "clear" if not final_risky else "round_limit",
                                "rounds": len(actions),
                                "changed": ids != baseline["response_token_ids"]},
        "actions": actions,
    }


def generate(device: str, start_index: int, end_index: int, image_id: int | None,
             prefix_prefill: bool = False, fused_forward: bool = False,
             max_rounds: int = MAX_ROUNDS, ffn_visual_only: bool = False) -> None:
    prefix_prefill = prefix_prefill or fused_forward
    torch.set_num_threads(1)
    config = load_config(str(CONFIG))
    ids = sorted(map(int, json.loads(SPLIT.read_text())["test"]))[:LIMIT]
    ids = ids[start_index:end_index]
    if image_id is not None:
        if image_id not in ids:
            raise ValueError("Image outside selected 100-image cohort/range")
        ids = [image_id]
    if max_rounds != MAX_ROUNDS and not (max_rounds == 64 and fused_forward):
        raise ValueError("Only the fused mode supports a 64-round comparison")
    if ffn_visual_only and not (max_rounds == 64 and fused_forward):
        raise ValueError("Visual-only FFN requires --fused-forward --max-rounds 64")
    output_dir = (OUT_FUSED_64_VIS_FFN if ffn_visual_only else
                  (OUT_FUSED_64 if max_rounds == 64 else OUT_FUSED) if fused_forward else
                  OUT_PREFIX if prefix_prefill else OUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = read_rows(BASELINE)
    stats = training_stats()
    scorer = RiskScorer(MODEL)
    matcher = _mention_matcher()
    wrapper = build_model(MODEL, get_extraction_model_cfg(config, MODEL), device=device)
    wrapper.model.eval().requires_grad_(False)
    object_tokens = object_token_ids(wrapper.tokenizer, matcher)
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    for ordinal, iid in enumerate(ids, 1):
        path = output_dir / f"{iid}.json"
        if path.exists():
            continue
        started = time.perf_counter()
        with Image.open(coco / f"COCO_val2014_{iid:012d}.jpg") as opened:
            image = opened.convert("RGB")
        row = generate_image(
            wrapper, scorer, matcher, image, iid, baseline[iid], object_tokens,
            stats, prefix_prefill=prefix_prefill, fused_forward=fused_forward,
            max_rounds=max_rounds, ffn_visual_only=ffn_visual_only)
        row.update({"threshold": THRESHOLD, "max_rounds": max_rounds,
                    "window": WINDOW, "attention_mean_factor": ATTENTION_MEAN_FACTOR,
                    "ffn_attention_factor": FFN_ATTENTION_FACTOR,
                    "prefix_prefill": prefix_prefill,
                    "fused_forward": fused_forward,
                    "ffn_visual_only": ffn_visual_only,
                    "seconds": time.perf_counter() - started})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        print(f"[{ordinal}/{len(ids)}] image={iid} rounds={row[f'repeat_{max_rounds}']['rounds']} "
              f"remaining={row[f'repeat_{max_rounds}']['remaining_risky_count']} "
              f"first_remaining={row['first_only']['remaining_risky_count']} "
              f"seconds={row['seconds']:.1f}", flush=True)


def evaluate(prefix_prefill: bool = False, fused_forward: bool = False,
             max_rounds: int = MAX_ROUNDS, ffn_visual_only: bool = False) -> None:
    if max_rounds != MAX_ROUNDS and not (max_rounds == 64 and fused_forward):
        raise ValueError("Only the fused mode supports a 64-round comparison")
    if ffn_visual_only and not (max_rounds == 64 and fused_forward):
        raise ValueError("Visual-only FFN requires --fused-forward --max-rounds 64")
    output_dir = (OUT_FUSED_64_VIS_FFN if ffn_visual_only else
                  (OUT_FUSED_64 if max_rounds == 64 else OUT_FUSED) if fused_forward else
                  OUT_PREFIX if prefix_prefill else OUT)
    config = load_config(str(CONFIG))
    ids = sorted(map(int, json.loads(SPLIT.read_text())["test"]))[:LIMIT]
    baseline = read_rows(BASELINE)
    rows = {int(path.stem): json.loads(path.read_text()) for path in output_dir.glob("*.json")
            if path.stem.isdigit()}
    if set(ids) - rows.keys():
        raise RuntimeError(f"Missing {len(set(ids) - rows.keys())} images")
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=ROOT / "outputs/generation_time_svar_official_seed43/chair_cache.pkl")
    variants = {"greedy": {iid: baseline[iid]["generated_text"] for iid in ids}}
    repeat_key = f"repeat_{max_rounds}"
    for mode in ("first_only", repeat_key):
        variants[mode] = {iid: rows[iid][mode]["generated_text"] for iid in ids}
    evaluated = {mode: [evaluator.compute_chair_token(iid, texts[iid]) for iid in ids]
                 for mode, texts in variants.items()}
    metrics = {}
    for mode, items in evaluated.items():
        hallucinated = sum(len(item["mscoco_hallucinated_words"]) for item in items)
        mentions = sum(len(item["object_mentions"]) for item in items)
        tp = sum(len(set(item["mscoco_gt_words"]) & set(item["mscoco_generated_words"]))
                 for item in items)
        gt = sum(len(set(item["mscoco_gt_words"])) for item in items)
        pred = sum(len(set(item["mscoco_generated_words"])) for item in items)
        metrics[mode] = {
            "CHAIRs": float(np.mean([item["metrics"]["CHAIRs"] for item in items])),
            "CHAIRi": hallucinated / mentions if mentions else 0.0,
            "hallucinated_mentions": hallucinated, "object_mentions": mentions,
            "macro_object_recall": float(np.mean([item["metrics"]["Recall"] for item in items])),
            "paper_style_micro_object_F1": 2 * tp / (gt + pred),
            "paper_style_counts": {"TP": tp, "GT": gt, "Pred": pred},
            "words_per_image": float(np.mean([len(item["words"]) for item in items])),
        }
        if mode != "greedy":
            metrics[mode].update({
                "changed_images": sum(rows[iid][mode]["changed"] for iid in ids),
                "detector_clear_images": sum(rows[iid][mode]["remaining_risky_count"] == 0 for iid in ids),
                "fixed_hall_images": sum(b["metrics"]["CHAIRs"] and not v["metrics"]["CHAIRs"]
                                         for b, v in zip(evaluated["greedy"], items)),
                "introduced_hall_images": sum(not b["metrics"]["CHAIRs"] and v["metrics"]["CHAIRs"]
                                              for b, v in zip(evaluated["greedy"], items)),
            })
    result = {
        "seed": SEED, "images": LIMIT, "threshold": THRESHOLD,
        "prefix_prefill": prefix_prefill,
        "fused_forward": fused_forward,
        "ffn_visual_only": ffn_visual_only,
        "max_rounds": max_rounds,
        "initial_triggered_images": sum(rows[iid]["initial_risky_count"] > 0 for iid in ids),
        "first_only_risky_images": sum(rows[iid]["first_only"]["remaining_risky_count"] > 0 for iid in ids),
        "repeat_round_limit_images": [iid for iid in ids if rows[iid][repeat_key]["stopping_reason"] == "round_limit"],
        "total_repeat_rounds": sum(rows[iid][repeat_key]["rounds"] for iid in ids),
        "max_used_rounds": max(rows[iid][repeat_key]["rounds"] for iid in ids),
        "sum_image_seconds": sum(rows[iid]["seconds"] for iid in ids),
        "mean_abs_image_logit": float(np.mean([action["mean_abs_image_logit"] for iid in ids for action in rows[iid]["actions"]])),
        "raw_mean_image_logit": float(np.mean([action["raw_mean_image_logit"] for iid in ids for action in rows[iid]["actions"]])),
        "negative_raw_mean_fraction": float(np.mean([action["negative_raw_mean_fraction"] for iid in ids for action in rows[iid]["actions"]])),
        "methods": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=LIMIT)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--prefix-prefill", action="store_true")
    parser.add_argument("--fused-forward", action="store_true")
    parser.add_argument("--ffn-visual-only", action="store_true")
    parser.add_argument("--max-rounds", type=int, choices=(32, 64), default=MAX_ROUNDS)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.device, args.start_index, args.end_index, args.image_id,
                 prefix_prefill=args.prefix_prefill,
                 fused_forward=args.fused_forward,
                 max_rounds=args.max_rounds,
                 ffn_visual_only=args.ffn_visual_only)
    else:
        evaluate(prefix_prefill=args.prefix_prefill or args.fused_forward,
                 fused_forward=args.fused_forward,
                 max_rounds=args.max_rounds,
                 ffn_visual_only=args.ffn_visual_only)
