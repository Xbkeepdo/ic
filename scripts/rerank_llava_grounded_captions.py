#!/usr/bin/env python3
"""Score saved LLaVA captions with teacher-forced visual support and FFN writes."""

from __future__ import annotations

import argparse
import re
import sys
import time
from contextlib import ExitStack, nullcontext
from copy import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.llava_decoding_evidence import _PHYSICAL_NOUNS
from features.llava_caption_patch_drift import CaptionPatchDrift, FFN_LAYER, TAU, WRITE_LAYER
from models import build_model
from scripts.evaluate_generation_time_evidence_drift_llava import load_variant
from scripts.generation_time_evidence_drift_llava import (
    COCO_ROOT, CONFIG, MODEL, MODEL_ROOT, OUT, _prefill, _read_rows,
)
from scripts.generation_time_sar_ems_seed43 import PROMPT, _append_jsonl, _inputs, _read_json
from utils.config_utils import get_extraction_model_cfg, load_config


FFN_LAYERS = (20, 22, 26, 30)
WORD_PART = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)*\Z")
VERIFY_BATCH_SIZE = 4


@dataclass(frozen=True)
class NounSpan:
    word: str
    lemma: str
    start: int
    end: int


@lru_cache(maxsize=8192)
def physical_noun_lemma(word: str) -> str | None:
    from nltk.corpus import wordnet

    normalized = word.replace("-", "_")
    senses = wordnet.synsets(normalized, pos=wordnet.NOUN)
    if any(sense.lexname() in _PHYSICAL_NOUNS for sense in senses):
        return wordnet.morphy(normalized, pos=wordnet.NOUN) or normalized
    return None


def noun_spans(tokenizer, ids: tuple[int, ...]) -> list[NounSpan]:
    """Join SentencePiece pieces before checking the complete word in WordNet."""
    pieces = tokenizer.convert_ids_to_tokens(list(ids))
    special = set(tokenizer.all_special_ids)
    spans = []
    parts = []
    start = 0
    for index, piece in enumerate([*pieces, None]):
        text = piece.removeprefix("▁") if isinstance(piece, str) else ""
        is_word_part = index < len(ids) and ids[index] not in special and WORD_PART.fullmatch(text)
        if not is_word_part or piece.startswith("▁"):
            if parts:
                word = "".join(parts)
                lemma = physical_noun_lemma(word.lower())
                if lemma is not None:
                    spans.append(NounSpan(word, lemma, start, index))
                parts = []
        if is_word_part:
            if not parts:
                start = index
            parts.append(text)
    return spans


def first_noun_mentions(spans: list[NounSpan]) -> list[NounSpan]:
    by_lemma = {}
    for span in spans:
        by_lemma.setdefault(span.lemma, span)
    return list(by_lemma.values())


def batched_prefix(cache, batch_size: int):
    """Fork the shared prefix with batch views. DynamicLayer appends by concatenation."""
    branch = copy(cache)
    branch.layers = [copy(layer) for layer in cache.layers]
    for layer in branch.layers:
        layer.keys = layer.keys.expand(batch_size, -1, -1, -1)
        layer.values = layer.values.expand(batch_size, -1, -1, -1)
    return branch


class CaptionScorer:
    def __init__(self, wrapper, *, verify_nouns: bool = False, verify_patch_drift: bool = False):
        self.wrapper = wrapper
        self.decoder = wrapper.model.language_model
        self.lm_head = wrapper.model.lm_head
        self.text_key = None
        self.text_cache = None
        self.verify_nouns = verify_nouns
        self.patch_drift = CaptionPatchDrift(self.decoder, self.lm_head) if verify_patch_drift else None
        self.text_noun_margins: dict[str, float] = {}
        self.answer_token_ids: tuple[list[int], list[int]] | None = None
        if verify_nouns:
            import inflect

            self.inflect = inflect.engine()

    def _text_prefix(self, prompt, start: int, end: int):
        indexes = torch.arange(prompt.shape[1] - 1, device=prompt.device)
        keep = (indexes < start) | (indexes >= end)
        positions = indexes[keep].unsqueeze(0)
        text = prompt.index_select(1, positions[0])
        key = (tuple(text[0].tolist()), tuple(positions[0].tolist()))
        if key != self.text_key:
            # The removed patch block leaves a gap in RoPE positions. The
            # original text tokens keep the positions they have with the image.
            result = self.decoder(input_ids=text, attention_mask=torch.ones_like(text),
                                  position_ids=positions, use_cache=True)
            self.text_cache = result.past_key_values
            self.text_key = key
            self.text_noun_margins.clear()
        return self.text_cache

    def _decode_queries(self, cache, queries, lengths, positions):
        valid = torch.arange(queries.shape[1], device=queries.device)[None, :] < lengths[:, None]
        prefix_mask = torch.ones((queries.shape[0], cache.get_seq_length()),
                                 dtype=torch.bool, device=queries.device)
        return self.decoder(input_ids=queries, attention_mask=torch.cat((prefix_mask, valid), dim=1),
                            position_ids=positions, past_key_values=batched_prefix(cache, queries.shape[0]),
                            use_cache=True)

    def _score_tokens(self, cache, queries, lengths, positions, batch_rows, token_rows, targets):
        ffn_gains = {}
        middle = {}
        weights = self.lm_head.weight[targets].float()

        def capture(layer_index):
            def before(_module, args):
                middle[layer_index] = args[0][batch_rows, token_rows]

            def after(_module, _args, output):
                mid = middle.pop(layer_index)
                ffn = output[batch_rows, token_rows]
                added = self.decoder.norm(mid + ffn).float() - self.decoder.norm(mid).float()
                ffn_gains[layer_index] = (weights * added).sum(dim=1)

            return before, after

        with ExitStack() as stack:
            for layer_index in FFN_LAYERS:
                layer = self.decoder.layers[layer_index]
                before, after = capture(layer_index)
                stack.callback(layer.post_attention_layernorm.register_forward_pre_hook(before).remove)
                stack.callback(layer.mlp.register_forward_hook(after).remove)
            result = self._decode_queries(cache, queries, lengths, positions)
        states = result.last_hidden_state[batch_rows, token_rows]
        logits = self.lm_head(states).float()
        logp = logits.gather(1, targets[:, None])[:, 0] - torch.logsumexp(logits, dim=1)
        return torch.stack([logp, *(ffn_gains[layer] for layer in FFN_LAYERS)], dim=1).tolist()

    def _split_template(self, ids: list[int]) -> tuple[list[int], list[int]]:
        image_token = self.wrapper._image_token_id()
        if ids.count(image_token) != 1:
            raise ValueError("Expected one unexpanded image token in the LLaVA question")
        image_index = ids.index(image_token)
        return ids[:image_index], ids[image_index + 1:]

    def _find_answer_token_ids(self, formatted: str, prefix_ids: list[int]) -> tuple[list[int], list[int]]:
        yes = ("Yes", "yes", "YES")
        no = ("No", "no", "NO")
        forms = [f"{formatted}{space}{answer}" for answer in (*yes, *no) for space in (" ", "")]
        encodings = self.wrapper.tokenizer(forms, add_special_tokens=True).input_ids

        def matching_ids(rows):
            return sorted({ids[-1] for ids in rows if len(ids) == len(prefix_ids) + 1 and ids[:-1] == prefix_ids})

        yes_ids = matching_ids(encodings[:2 * len(yes)])
        no_ids = matching_ids(encodings[2 * len(yes):])
        if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
            raise ValueError("Could not align disjoint single-token yes/no continuations after ASSISTANT:")
        return yes_ids, no_ids

    def _binary_margins(self, cache, tails: dict[str, list[int]], visual_end: int, device) -> dict[str, float]:
        yes_ids, no_ids = self.answer_token_ids
        answer_ids = torch.tensor(yes_ids + no_ids, device=device)
        weight = self.lm_head.weight[answer_ids]
        bias = None if self.lm_head.bias is None else self.lm_head.bias[answer_ids]
        pad_id = self.wrapper.tokenizer.pad_token_id or 0
        results = {}
        nouns = list(tails)
        for first in range(0, len(nouns), VERIFY_BATCH_SIZE):
            batch = nouns[first:first + VERIFY_BATCH_SIZE]
            suffixes = [torch.tensor(tails[lemma], device=device) for lemma in batch]
            lengths = torch.tensor([len(tail) for tail in suffixes], device=device)
            queries = pad_sequence(suffixes, batch_first=True, padding_value=pad_id)
            positions = torch.arange(visual_end, visual_end + queries.shape[1], device=device)
            positions = positions[None, :].expand(len(batch), -1)
            result = self._decode_queries(cache, queries, lengths, positions)
            hidden = result.last_hidden_state[torch.arange(len(batch), device=device), lengths - 1]
            logits = torch.nn.functional.linear(hidden, weight, bias).float()
            yes_mass = logits[:, :len(yes_ids)].logsumexp(dim=1)
            no_mass = logits[:, len(yes_ids):].logsumexp(dim=1)
            results.update(zip(batch, (yes_mass - no_mass).tolist(), strict=True))
        return results

    def _verify_presence(self, prompt, start, end, image_cache, text_cache, lemmas) -> dict[str, dict]:
        tokenizer = self.wrapper.tokenizer
        format_prompt = self.wrapper._format_prompt
        questions = {
            lemma: f"Is there {self.inflect.a(lemma.replace('_', ' '))} in this image? Answer yes or no."
            for lemma in lemmas
        }
        formatted = [format_prompt(question) for question in questions.values()]
        encodings = tokenizer(formatted, add_special_tokens=True).input_ids
        text_before_image = prompt[0, :start].tolist()
        reference = tokenizer(format_prompt(PROMPT), add_special_tokens=True).input_ids
        before, after = self._split_template(reference)
        if before != text_before_image or after != prompt[0, end:].tolist():
            raise ValueError("Verification tokenizer and multimodal processor disagree on LLaVA prompt tokens")
        if self.answer_token_ids is None:
            self.answer_token_ids = self._find_answer_token_ids(formatted[0], encodings[0])

        tails = {}
        for lemma, ids in zip(questions, encodings, strict=True):
            before, suffix = self._split_template(ids)
            if before != text_before_image or not suffix:
                raise ValueError(f"Question for {lemma!r} does not share the caption's image context")
            tails[lemma] = suffix
        vision_prefix = batched_prefix(image_cache, 1)
        vision_prefix.crop(end)
        image_margins = self._binary_margins(vision_prefix, tails, end, prompt.device)
        missing = {lemma: tail for lemma, tail in tails.items() if lemma not in self.text_noun_margins}
        if missing:
            text_prefix = batched_prefix(text_cache, 1)
            text_prefix.crop(start)
            self.text_noun_margins.update(self._binary_margins(text_prefix, missing, end, prompt.device))
        return {
            lemma: {"question": question, "image_yes_minus_no": image_margins[lemma],
                    "text_yes_minus_no": self.text_noun_margins[lemma],
                    "image_margin_change": image_margins[lemma] - self.text_noun_margins[lemma]}
            for lemma, question in questions.items()
        }

    def score(self, image: Image.Image, captions: list[tuple[int, ...]]) -> dict[tuple[int, ...], dict]:
        words = {ids: noun_spans(self.wrapper.tokenizer, ids) for ids in dict.fromkeys(captions)}
        patch_only = self.patch_drift is not None
        mentions = {ids: first_noun_mentions(spans) for ids, spans in words.items()} if patch_only else words
        active = [ids for ids, spans in words.items() if spans]
        scored = {ids: summarize([], len(ids), patch_only=patch_only) for ids, spans in words.items() if not spans}
        if not active:
            return scored

        inputs = _inputs(self.wrapper, image, MODEL)
        prompt = inputs["input_ids"]
        start, end = self.wrapper._find_visual_token_range(inputs, self.wrapper._image_token_id())
        if not (0 <= start < end < prompt.shape[1]) or not bool(inputs["attention_mask"].all()):
            raise ValueError("Expected an unpadded LLaVA prompt containing a visual span and a text tail")
        prompt_context = self.patch_drift.capture_prompt(start, end) if patch_only else nullcontext()
        with prompt_context:
            image_cache = _prefill(self.wrapper, inputs)
        text_cache = self._text_prefix(prompt, start, end) if self.verify_nouns or not patch_only else None

        # Query zero is the last prompt token. At query j>=1 the input is
        # response[j-1], so logits at j predict the saved response[j].
        stops = [mentions[ids][-1].start + 1 if patch_only else mentions[ids][-1].end for ids in active]
        lengths = torch.tensor(stops, device=prompt.device)
        tails = [torch.cat((prompt[0, -1:], prompt.new_tensor(ids[:stop - 1])))
                 for ids, stop in zip(active, stops, strict=True)]
        pad_id = self.wrapper.tokenizer.pad_token_id or 0
        queries = pad_sequence(tails, batch_first=True, padding_value=pad_id)
        positions = torch.arange(prompt.shape[1] - 1, prompt.shape[1] - 1 + queries.shape[1],
                                 device=prompt.device)[None, :].expand(len(active), -1)
        token_requests = [
            (row, token, ids[token])
            for row, ids in enumerate(active)
            for word in mentions[ids]
            for token in range(word.start, word.start + 1 if patch_only else word.end)
        ]
        batch_rows, token_rows, targets = prompt.new_tensor(token_requests).T
        arguments = (queries, lengths, positions, batch_rows, token_rows, targets)
        if patch_only:
            with self.patch_drift.capture_tail(start, end, batch_rows, token_rows):
                self._decode_queries(image_cache, queries, lengths, positions)
            patch_scores = iter(self.patch_drift.score(image_cache, start, end, targets))
        else:
            with_image = iter(self._score_tokens(image_cache, *arguments))
            without_image = iter(self._score_tokens(text_cache, *arguments))
        verification = {}
        if self.verify_nouns:
            lemmas = sorted({word.lemma for spans in words.values() for word in spans})
            verification = self._verify_presence(prompt, start, end, image_cache, text_cache, lemmas)
        for ids in active:
            objects = []
            for word in mentions[ids]:
                if patch_only:
                    scored_word = {**asdict(word), "token_ids": list(ids[word.start:word.end]),
                                   "target_token_id": ids[word.start], "patch_drift": next(patch_scores)}
                else:
                    image_scores = [next(with_image) for _ in range(word.end - word.start)]
                    text_scores = [next(without_image) for _ in range(word.end - word.start)]
                    scored_word = score_word(ids, word, image_scores, text_scores)
                if verification:
                    scored_word["presence_verification"] = verification[word.lemma]
                objects.append(scored_word)
            scored[ids] = summarize(objects, len(ids), patch_only=patch_only, noun_count=len(words[ids]))
        return scored


def score_word(ids, word, image_scores, text_scores) -> dict:
    image_logp = [scores[0] for scores in image_scores]
    text_logp = [scores[0] for scores in text_scores]
    image_gains = [mean(scores[column] for scores in image_scores) for column in range(1, 1 + len(FFN_LAYERS))]
    text_gains = [mean(scores[column] for scores in text_scores) for column in range(1, 1 + len(FFN_LAYERS))]
    return {
        **asdict(word),
        "token_ids": list(ids[word.start:word.end]),
        "image_token_logp": image_logp,
        "text_token_logp": text_logp,
        "image_support": mean(image_logp) - mean(text_logp),
        "image_ffn_logit_gain": image_gains,
        "text_ffn_logit_gain": text_gains,
        "image_positive_ffn_gain": sum(max(0.0, value) for value in image_gains),
        "text_positive_ffn_gain": sum(max(0.0, value) for value in text_gains),
    }


def summarize(objects: list[dict], token_count: int, *, patch_only: bool = False,
              noun_count: int | None = None) -> dict:
    # Judge the first mention of each lemma. Repeated nouns become predictable
    # from the caption prefix itself and are reported separately as occurrences.
    first_mentions = {}
    for row in objects:
        first_mentions.setdefault(row["lemma"], row)
    summary = {
        "token_count": token_count,
        "noun_mentions": len(objects) if noun_count is None else noun_count,
        "unique_noun_lemmas": len(first_mentions),
        "objects": objects,
    }
    if patch_only:
        patches = [row["patch_drift"] for row in first_mentions.values()
                   if row["patch_drift"]["write_semantic_support"] is not None]
        support = sorted(patches, key=lambda row: row["write_semantic_support"])
        drift = sorted((row["ffn_hellinger2"] for row in patches), reverse=True)
        summary.update({
            "patch_drift_scored": len(patches),
            "minimum_patch_write_semantic_support": support[0]["write_semantic_support"] if support else None,
            "bottom3_patch_write_semantic_support": mean(row["write_semantic_support"] for row in support[:3])
            if support else None,
            "bottom3_support_patch_ffn_hellinger2": mean(row["ffn_hellinger2"] for row in support[:3])
            if support else None,
            "maximum_patch_ffn_hellinger2": drift[0] if drift else None,
            "top3_patch_ffn_hellinger2": mean(drift[:3]) if drift else None,
        })
    else:
        ordered = sorted(first_mentions.values(), key=lambda row: row["image_support"])
        bottom = ordered[:3]
        summary.update({
            "minimum_image_support": bottom[0]["image_support"] if bottom else None,
            "bottom3_image_support": mean(row["image_support"] for row in bottom) if bottom else None,
            "mean_image_support": mean(row["image_support"] for row in ordered) if ordered else None,
            "bottom3_image_positive_ffn_gain": mean(row["image_positive_ffn_gain"] for row in bottom)
            if bottom else None,
            "bottom3_text_positive_ffn_gain": mean(row["text_positive_ffn_gain"] for row in bottom)
            if bottom else None,
        })
    verification = [row["presence_verification"] for row in first_mentions.values() if "presence_verification" in row]
    if verification:
        for field in ("image_yes_minus_no", "image_margin_change"):
            margins = sorted(row[field] for row in verification)
            summary[f"minimum_verification_{field}"] = margins[0]
            summary[f"bottom3_verification_{field}"] = mean(margins[:3])
    return summary


@torch.inference_mode()
def run(args) -> Path:
    torch.set_num_threads(1)
    cohort_ids = list(map(int, _read_json(args.output_root / "cohort.json")[args.cohort]))
    if not cohort_ids or len(set(cohort_ids)) != len(cohort_ids):
        raise ValueError("Expected a nonempty cohort with unique image IDs")
    selected = cohort_ids[:args.limit]
    if args.image_ids:
        subset = set(map(int, _read_json(args.image_ids)))
        if not subset or not subset.issubset(selected):
            raise ValueError("The supplied image IDs must belong to the selected cohort")
        selected = [image_id for image_id in selected if image_id in subset]
    selected = selected[args.shard::args.shards]
    folder = args.output_root / args.cohort
    variants = args.variants
    rows = {name: load_variant(folder, name, cohort_ids, selected) for name in variants}
    verify_label = ".verify_nouns" if args.verify_nouns else ""
    patch_label = ".patch_drift" if args.verify_patch_drift else ""
    output = args.output or folder / f"grounding_scores.{'-'.join(variants)}{verify_label}{patch_label}.shard{args.shard}.jsonl"
    existing = _read_rows(output)
    protocol = {"model": MODEL, "model_root": str(args.model_root.resolve()), "prompt": PROMPT,
                "ffn_layers": list(FFN_LAYERS), "text_control": "remove_visual_keys_keep_rope_positions"}
    if args.verify_nouns:
        protocol["noun_verification"] = "llava_yes_no_first_token_with_image_and_without_image"
    if args.verify_patch_drift:
        protocol.pop("ffn_layers")
        if not args.verify_nouns:
            protocol.pop("text_control")
        protocol["patch_drift"] = {"visual_write_layer": WRITE_LAYER, "ffn_preference_layer": FFN_LAYER,
                                   "tau": TAU, "position": "first_subtoken_of_first_noun_lemma_mention"}
    pending = []
    for image_id in selected:
        if image_id not in existing:
            pending.append(image_id)
            continue
        old = existing[image_id]
        if old["protocol"] != protocol or list(old["captions"]) != variants or any(
            old["captions"][name]["response_token_ids"] != rows[name][image_id]["response_token_ids"]
            for name in variants
        ):
            raise ValueError(f"Saved scores for image {image_id} use different inputs or settings")
    if not pending:
        return output

    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config["hf_name"] = str(args.model_root)
    wrapper = build_model(MODEL, config, device=args.device)
    wrapper.model.eval().requires_grad_(False)
    scorer = CaptionScorer(wrapper, verify_nouns=args.verify_nouns, verify_patch_drift=args.verify_patch_drift)
    summary_key = "bottom3_patch_write_semantic_support" if args.verify_patch_drift else "bottom3_image_support"
    for ordinal, image_id in enumerate(pending, 1):
        started = time.perf_counter()
        ids = [tuple(map(int, rows[name][image_id]["response_token_ids"])) for name in variants]
        file = args.coco_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        with Image.open(file) as image:
            scores = scorer.score(image.convert("RGB"), ids)
        candidates = {
            name: {"generated_text": rows[name][image_id]["generated_text"], "response_token_ids": list(tokens),
                   **scores[tokens]}
            for name, tokens in zip(variants, ids, strict=True)
        }
        seconds = time.perf_counter() - started
        _append_jsonl(output, {"image_id": image_id, "protocol": protocol, "captions": candidates, "seconds": seconds})
        print(f"[{ordinal}/{len(pending)}] image={image_id} "
              f"bottom3={[scores[tokens][summary_key] for tokens in ids]} {seconds:.2f}s", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT, help="Folder containing cohort.json and variant shards")
    parser.add_argument("--output", type=Path, help="Optional output JSONL path")
    parser.add_argument("--coco-root", type=Path, default=COCO_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--cohort", default="pilot")
    parser.add_argument("--variants", nargs="+", default=["sadi", "svar"])
    parser.add_argument("--verify-nouns", action="store_true",
                        help="Also score a visual yes/no question for each distinct physical noun")
    parser.add_argument("--verify-patch-drift", action="store_true",
                        help="Score target-conditioned visual writes and later FFN preference across patches")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-ids", type=Path, help="JSON list restricting the selected cohort")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if len(args.variants) < 2 or len(set(args.variants)) != len(args.variants):
        parser.error("Select at least two distinct caption variants")
    if not 0 <= args.shard < args.shards or (args.limit is not None and args.limit < 1):
        parser.error("Invalid shard index or image limit")
    print(run(args))


if __name__ == "__main__":
    main()
