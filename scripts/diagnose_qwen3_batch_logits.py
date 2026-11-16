#!/usr/bin/env python3
"""Replay only a cached common prefix to inspect Qwen3 batch-sensitive logits.

Uses native generate/KV-cache paths and cached per-row tokens, never regenerates
complete captions. Run only after the main generation workers release a GPU.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.qa_benchmark import atomic_write_json
from features.llava_decoding_evidence import lexical_nouns
from features.qwen3_sar_ems_attention import LATE_VARIANTS, Qwen3LateAttention, VARIANTS
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from models import build_model
from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from utils.config_utils import get_extraction_model_cfg, load_config


def cached_rows(root: Path, cohort: str, variant: str) -> tuple[dict, dict]:
    """Keep source file and row order to recover the exact recorded batch."""
    by_id, files = {}, {}
    allowed = set(map(int, json.loads((root / 'cohort.json').read_text())[cohort]))
    for path in sorted((root / cohort).glob(f'{variant}.shard*.jsonl')):
        data = path.read_bytes()
        # Generation may still be appending elsewhere; use complete records only.
        rows = [json.loads(line) for line in data[:data.rfind(b'\n') + 1].splitlines() if line.strip()]
        files[path] = rows
        for index, row in enumerate(rows):
            image_id = int(row['image_id'])
            if image_id in by_id or image_id not in allowed or row['variant'] != variant or row['model'] != MODEL:
                raise ValueError(f'Duplicate/unexpected cached row: {path}, {image_id}')
            by_id[image_id] = (path, index, row)
    if not by_id:
        raise ValueError(f'No complete {variant} rows under {root / cohort}')
    return by_id, files


def recorded_group(index: dict, files: dict, image_id: int) -> list[dict]:
    path, position, row = index[image_id]
    if row['decoder'] == 'native_qwen3_generate':
        if row.get('batch_size', 1) != 1:
            raise ValueError('Single-image decoder has inconsistent batch metadata')
        return [row]
    if row['decoder'] != 'native_qwen3_generate_same_span_batch':
        raise ValueError(f"Unknown decoder {row['decoder']}")
    signature = (row['batch_size'], row['batch_wall_seconds'])
    rows = files[path]
    key = lambda r: (r.get('batch_size'), r.get('batch_wall_seconds'))
    begin, end = position, position + 1
    while begin and key(rows[begin - 1]) == signature:
        begin -= 1
    while end < len(rows) and key(rows[end]) == signature:
        end += 1
    group = rows[begin:end]
    if len(group) != int(row['batch_size']):
        raise ValueError('Cannot uniquely recover a complete batch from contiguous timing metadata; do not guess companions')
    if len({r['visual_token_count'] for r in group}) != 1:
        raise ValueError('Recovered group does not have a shared visual span size')
    return group


def first_difference(left: list[int], right: list[int]) -> int | None:
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)


def compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict:
    a, b = left.float(), right.float()
    if a.shape != b.shape:
        return {'shape_mismatch': [list(a.shape), list(b.shape)]}
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite] - b[finite]).abs()
    return {
        'shape': list(a.shape), 'equal': torch.equal(a, b),
        'nonfinite_mask_equal': torch.equal(torch.isfinite(a), torch.isfinite(b)),
        'finite_values': int(finite.sum()),
        'max_abs': float(delta.max()) if delta.numel() else None,
        'mean_abs': float(delta.mean()) if delta.numel() else None,
    }


def logit_summary(values: torch.Tensor, tokenizer) -> dict:
    logits = values.float()
    scores, ids = logits.topk(10)
    top1 = int(logits.argmax())
    tied = (logits == logits[top1]).nonzero(as_tuple=True)[0]
    return {
        # topk does not guarantee the same tie ordering as greedy argmax.
        'top1': top1, 'top1_text': tokenizer.decode([top1]),
        'top1_minus_top2': float(scores[0] - scores[1]),
        'max_logit': float(logits[top1]), 'max_tie_count': int(tied.numel()),
        'max_tie_token_ids': tied[:64].tolist(), 'topk_tie_order_unspecified': True,
        'top10': [{'id': int(i), 'text': tokenizer.decode([int(i)]), 'logit': float(s)}
                  for i, s in zip(ids, scores)],
    }


class PrefixProbe(LogitsProcessor):
    def __init__(self, records, target_row, prompt_len, step, eos, controller, native):
        self.records, self.target_row = records, target_row
        self.prompt_len, self.probe_step = prompt_len, step
        self.eos, self.controller, self.native = eos, controller, native
        self.forward_step = 0
        self.trace, self.tensors, self.guidance = [], {}, {}

    def save(self, name, value):
        self.tensors[name] = value.detach().float().cpu().clone()

    def model_output(self, _module, _args, output):
        if self.forward_step == self.probe_step:
            self.save('raw_logits', output.logits[self.target_row, -1])

    def prefill_input(self, _module, args, kwargs):
        if self.forward_step == 0:
            hidden = args[0] if args else kwargs['hidden_states']
            self.save('prefill.visual_input', hidden[self.target_row, self.controller.start:self.controller.end])
            self.save('prefill.last_text_input', hidden[self.target_row, -1])

    def layer_output(self, layer_index, kind):
        def capture(_module, _args, output):
            if self.forward_step == self.probe_step or self.forward_step == 0:
                tensor = output[0] if isinstance(output, tuple) else output
                prefix = '' if self.forward_step == self.probe_step else 'prefill.'
                self.save(f'{prefix}layer{layer_index}.{kind}', tensor[self.target_row, -1])
        return capture

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_len
        if step != self.forward_step:
            raise ValueError('Unexpected native generation step progression')
        target = self.records[self.target_row]
        actual_prefix = input_ids[self.target_row, self.prompt_len:].tolist()
        if actual_prefix != target['response_token_ids'][:step]:
            raise ValueError('Forced target prefix differs from the cached common prefix')
        natural = scores.argmax(dim=-1).tolist()
        self.trace.append({'step': step, 'natural_top1': natural,
                           'cached_token': [r['response_token_ids'][step] if step < len(r['response_token_ids']) else None for r in self.records]})
        if step == self.probe_step:
            self.save('processed_logits', scores[self.target_row])
            control = self.controller
            if control.variant in LATE_VARIANTS:
                attention = control.attention if self.native else control.attention[self.target_row]
                residual = control.ffn_residual if self.native else control.ffn_residual[self.target_row]
                self.save('sar_attention', attention)
                self.save('ffn_residual', residual)
                targets = control.targets if self.native else control.targets.get(self.target_row, [])
                self.guidance = {'target_at_probe': [token for index, token in targets if index == step]}
                budget = control.budget
                if self.native and budget is not None:
                    self.save('guidance_budget', budget)
                elif not self.native and self.target_row in control.budget_rows:
                    self.save('guidance_budget', budget[control.budget_rows.index(self.target_row)])
            self.forward_step += 1
            return scores
        self.forward_step += 1
        forced = torch.full_like(scores, -float('inf'))
        for row, record in enumerate(self.records):
            tokens = record['response_token_ids']
            token = tokens[step] if step < len(tokens) else self.eos
            forced[row, token] = 0
        return forced


class StopAfterProbe(StoppingCriteria):
    def __init__(self, total_length):
        self.total_length = total_length

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), input_ids.shape[1] >= self.total_length,
                          dtype=torch.bool, device=input_ids.device)


@torch.inference_mode()
def probe(wrapper, records, image_id, prepared, span, variant, nouns, step, native, capture_layers):
    if native and len(records) != 1:
        raise ValueError('Native single controller requires one image')
    row_index = [r['image_id'] for r in records].index(image_id)
    inputs = {key: torch.cat([prepared[r['image_id']][key] for r in records], dim=0).to(wrapper.device)
              for key in prepared[image_id]}
    controller = (Qwen3LateAttention if native else Qwen3BatchLateAttention)(wrapper.model, span, nouns, variant)
    eos = wrapper.model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    hook = PrefixProbe(records, row_index, inputs['input_ids'].shape[1], step, eos, controller, native)
    kwargs = _generation_kwargs(wrapper, MODEL)
    # Keep the original max length: shortening it could change forced-EOS or
    # other built-in logit processors at the diagnostic position.
    kwargs['logits_processor'] = LogitsProcessorList([hook])
    kwargs['stopping_criteria'] = StoppingCriteriaList([StopAfterProbe(inputs['input_ids'].shape[1] + step + 1)])
    with ExitStack() as stack:
        stack.enter_context(controller.capture())
        handle = wrapper.model.register_forward_hook(hook.model_output)
        stack.callback(handle.remove)
        handle = controller.decoder.layers[0].register_forward_pre_hook(hook.prefill_input, with_kwargs=True)
        stack.callback(handle.remove)
        for index in capture_layers:
            layer = controller.decoder.layers[index]
            for kind, module in (('attention_output', layer.self_attn), ('ffn_output', layer.mlp), ('layer_output', layer)):
                handle = module.register_forward_hook(hook.layer_output(index, kind))
                stack.callback(handle.remove)
        output = wrapper.model.generate(**inputs, **kwargs)
    if hook.forward_step != step + 1 or 'raw_logits' not in hook.tensors:
        raise ValueError('Native generation terminated before the probe position')
    target_tokens = output[row_index, inputs['input_ids'].shape[1]:].tolist()
    target = records[row_index]['response_token_ids']
    target_mismatches = [r['step'] for r in hook.trace[:step]
                         if r['natural_top1'][row_index] != r['cached_token'][row_index]]
    batch_mismatches = [
        {'step': r['step'], 'row': index, 'image_id': records[index]['image_id'],
         'cached_token': expected, 'unforced_token': r['natural_top1'][index]}
        for r in hook.trace[:step] for index, expected in enumerate(r['cached_token'])
        if expected is not None and r['natural_top1'][index] != expected
    ]
    summary = {
        'batch_image_ids': [r['image_id'] for r in records], 'target_row': row_index,
        'controller': type(controller).__name__, 'native_forward_steps': hook.forward_step,
        'common_prefix_matches': target_tokens[:step] == target[:step],
        'target_prior_unforced_mismatches': target_mismatches,
        'batch_prior_unforced_mismatches': batch_mismatches,
        'cached_token_at_probe': target[step], 'generated_token_at_probe': target_tokens[step],
        'probe_token_matches_cache': target_tokens[step] == target[step],
        'cached_branch_reproduced': not batch_mismatches and target_tokens[step] == target[step],
        'raw_logits': logit_summary(hook.tensors['raw_logits'], wrapper.tokenizer),
        'processed_logits': logit_summary(hook.tensors['processed_logits'], wrapper.tokenizer),
        'guidance': hook.guidance, 'per_step_trace': hook.trace,
    }
    return summary, hook.tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--comparison-root', type=Path, required=True)
    parser.add_argument('--reference-cohort', default='coco500')
    parser.add_argument('--comparison-cohort', default='coco500')
    parser.add_argument('--variant', choices=VARIANTS, default='svar')
    parser.add_argument('--image-id', type=int)
    parser.add_argument('--max-prefix-tokens', type=int, default=64)
    parser.add_argument('--model-root', type=Path, default=MODEL_ROOT)
    parser.add_argument('--coco-root', type=Path, default=COCO_ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--capture-layers', type=int, nargs='+', default=[0, 5, 18, 20, 21, 22])
    parser.add_argument('--controls', action='store_true', help='Also compare native/batch B1, batch row reversal, and repeated B1')
    parser.add_argument('--inspect-cache-only', action='store_true', help='Validate and print the exact cached case without loading a model')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.max_prefix_tokens < 0:
        parser.error('--max-prefix-tokens must be nonnegative')
    if (args.output_root / 'report.json').exists():
        parser.error('Choose a new output directory; an existing report is not overwritten')
    refs, ref_files = cached_rows(args.reference_root, args.reference_cohort, args.variant)
    comps, comp_files = cached_rows(args.comparison_root, args.comparison_cohort, args.variant)
    candidates = []
    for image_id in sorted(set(refs) & set(comps)):
        if args.image_id is not None and image_id != args.image_id:
            continue
        step = first_difference(refs[image_id][2]['response_token_ids'], comps[image_id][2]['response_token_ids'])
        if step is not None and step <= args.max_prefix_tokens:
            candidates.append((step, image_id))
    if not candidates:
        parser.error('No common image with a token divergence inside the allowed prefix length')
    step, image_id = min(candidates)
    ref_group = recorded_group(refs, ref_files, image_id)
    comp_group = recorded_group(comps, comp_files, image_id)
    one = [refs[image_id][2]]
    if one[0]['visual_token_count'] != comps[image_id][2]['visual_token_count']:
        raise ValueError('The cached target has different visual token counts')
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        'image_id': image_id, 'variant': args.variant, 'first_cached_divergence_index': step,
        'shared_prefix_ids': one[0]['response_token_ids'][:step],
        'reference_batch_image_ids': [r['image_id'] for r in ref_group],
        'comparison_batch_image_ids': [r['image_id'] for r in comp_group],
        'reference_token': one[0]['response_token_ids'][step],
        'comparison_token': comps[image_id][2]['response_token_ids'][step],
        'native_decode_steps_per_case': step + 1,
    }
    atomic_write_json(args.output_root / 'plan.json', plan)
    if args.inspect_cache_only:
        print(json.dumps(plan, indent=2))
        return
    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config['hf_name'] = str(args.model_root)
    wrapper = build_model(MODEL, config, device=args.device)
    wrapper.model.eval().requires_grad_(False)
    eos_ids = wrapper.model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    if eos_ids.intersection(one[0]['response_token_ids'][:step]):
        raise ValueError('Target cached common prefix contains EOS before the probe step')
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ') if args.variant in LATE_VARIANTS else frozenset()
    if args.variant in LATE_VARIANTS and not nouns:
        raise ValueError('WordNet entity vocabulary is empty')
    if any(not 0 <= index < len(wrapper.model.model.language_model.layers) for index in args.capture_layers):
        parser.error('Capture layer outside decoder range')
    prepared, expected = {}, None
    for record in ref_group + comp_group:
        current = int(record['image_id'])
        if current in prepared:
            continue
        path = args.coco_root / 'val2014' / f'COCO_val2014_{current:012d}.jpg'
        with Image.open(path) as image:
            inputs = wrapper._prompt_inputs(image.convert('RGB'), 'Describe this image.')
        span = wrapper._find_vision_token_range(inputs['input_ids'][0])
        shape = (span, inputs['input_ids'].shape[1], tuple(sorted(inputs)))
        if expected is None:
            expected = shape
        if shape != expected or not inputs['attention_mask'].all().item():
            raise ValueError('Recorded companions do not produce compatible unpadded prompts')
        if span[1] - span[0] != record['visual_token_count']:
            raise ValueError('Reprocessed visual span differs from the cached experiment')
        prepared[current] = inputs
    native_ref = len(ref_group) == 1 and ref_group[0]['decoder'] == 'native_qwen3_generate'
    native_comp = len(comp_group) == 1 and comp_group[0]['decoder'] == 'native_qwen3_generate'
    cases = [('reference_protocol', ref_group, native_ref), ('comparison_protocol', comp_group, native_comp)]
    pairs = [('reference_protocol', 'comparison_protocol')]
    if args.controls:
        cases = [('native_b1', one, True), ('batch_controller_b1', one, False)] + cases + [
            ('comparison_reversed', list(reversed(comp_group)), native_comp), ('native_b1_repeat', one, True)]
        pairs += [('native_b1', 'batch_controller_b1'), ('comparison_protocol', 'comparison_reversed'),
                  ('native_b1', 'native_b1_repeat')]
    reports, tensors = {}, {}
    for name, records, native in cases:
        reports[name], tensors[name] = probe(wrapper, records, image_id, prepared, expected[0],
                                             args.variant, nouns, step, native, args.capture_layers)
        print(f"[probe] {name} B={len(records)} token={reports[name]['generated_token_at_probe']}", flush=True)
    comparisons = {}
    for left, right in pairs:
        comparisons[f'{left}__vs__{right}'] = {
            key: compare_tensor(tensors[left][key], tensors[right][key])
            for key in sorted(set(tensors[left]) & set(tensors[right]))
        }
    report = {
        'image_id': image_id, 'variant': args.variant, 'first_cached_divergence_index': step,
        'shared_prefix_ids': refs[image_id][2]['response_token_ids'][:step],
        'reference_root': str(args.reference_root), 'comparison_root': str(args.comparison_root),
        'reference_cohort': args.reference_cohort, 'comparison_cohort': args.comparison_cohort,
        'source_sha256': {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in (
            'features/qwen3_sar_ems_attention.py', 'features/qwen3_sar_ems_attention_batch.py',
            'features/llava_sar_ems_attention.py', 'models/qwen3_vl_wrapper.py',
            'scripts/generation_time_sar_ems_seed43.py',
            'configs/model_configs_coco4000_512_endac_baselines_v2.yaml',
            'scripts/diagnose_qwen3_batch_logits.py')},
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'device': torch.cuda.get_device_name(wrapper.device),
                        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32},
        'cases': reports, 'comparisons': comparisons,
        'limitations': [
            'Forced-prefix replay preserves native sequential KV and intervention paths, but does not regenerate complete captions.',
            'Cached branch reproduction is required before interpreting this as the original observed branch point.',
            'Differences localize an execution boundary; they do not by themselves prove BF16 rounding or a specific kernel bug.',
            'Batch composition is recovered only when contiguous saved size/timing metadata uniquely identifies a complete group.',
        ],
    }
    torch.save(tensors, args.output_root / 'probe_tensors.pt')
    atomic_write_json(args.output_root / 'report.json', report)
    print(args.output_root / 'report.json', flush=True)


if __name__ == '__main__':
    main()
