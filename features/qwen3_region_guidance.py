"""Live generation with risk-gated semantic-region attention enhancement.

Proposals are evaluated against one native batch state. Only changed accepted
rows' final hidden state and current20--35 KV are committed, after all rows
have been tested. Original visual/past text cache positions are preserved.
"""

from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from features import qwen3_region_memory as memory_module
from features import qwen3_region_consistency as consistency_module
from features.qwen3_region_memory import rectangle, LAYERS
from features.qwen3_region_consistency import RegionConsistency, category_probabilities
from features.qwen3_native_risk_i_el import NativeRiskIEl
from features.qwen3_bounded_visual_tilt import evidence_preference
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.probe_qwen3_region_counterevidence import signed_attention


class LiveRegionReader(NativeRiskIEl):
    @property
    def anchor(self): return self.owner.anchor

    @property
    def tail_kwargs(self): return self.owner.tail_kwargs

    @contextmanager
    def paused(self):
        owner = self.owner
        assert not owner.suspended and not owner._replaying and not owner.replaying_tail
        owner.suspended = owner._replaying = owner.replaying_tail = True
        try: yield
        finally: owner.suspended = owner._replaying = owner.replaying_tail = False


def prefix_sha(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


class Qwen3RegionGuidance:
    def __init__(self, wrapper, span, nouns, *, probe, history_parser, image_ids, images,
                 output_dir, policy='verified', shared_cache=None, disabled=False,
                 zero_strength=False, reject_all=False):
        assert policy in ['verified', 'unverified', 'shifted']
        self.wrapper, self.model = wrapper, wrapper.model
        self.decoder = self.model.model.language_model; self.head = self.model.get_output_embeddings()
        self.start, self.end = span; self.nouns = frozenset(nouns)
        self.probe, self.parser, self.tokenizer = probe, history_parser, wrapper.tokenizer
        self.token_classes = history_parser.token_classes(self.tokenizer, self.nouns)
        self.aliases = {}
        for token in sorted(self.nouns): self.aliases.setdefault(self.token_classes[token], []).append(token)
        self.image_ids, self.images = image_ids, images
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy, self.disabled = policy, bool(disabled)
        self.zero_strength, self.reject_all = bool(zero_strength), bool(reject_all)
        self.shared = {} if shared_cache is None else shared_cache
        self.reader = LiveRegionReader(self.model, span, self.nouns, owner=self,
                                      shared_static=self.shared.setdefault('native_static', {}))
        self.step = -1; self.suspended = self._replaying = self.replaying_tail = False
        self.anchor = self.tail_kwargs = self.memory = self.vision_output = self.inputs = None
        self.history, self.seen = [], []; self.active = None
        self.gates, self.events, self.decisions = {}, {}, {}
        self.actual_encoder_calls = self.original_vision_cache_hits = 0
        self.eos_ids = self.model.generation_config.eos_token_id
        self.eos_ids = tuple(self.eos_ids if isinstance(self.eos_ids, list) else [self.eos_ids])

    def _begin(self, _module, args, kwargs):
        assert not self.suspended
        self.step += 1; self.anchor = self.tail_kwargs = None
        ids = kwargs.get('input_ids', args[0] if args else None); assert ids is not None
        if self.step == 0:
            self.active = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
            self.history = [[] for _ in range(len(ids))]; self.seen = [set() for _ in range(len(ids))]
            self.inputs = {name: kwargs[name] for name in ['attention_mask', 'pixel_values', 'image_grid_thw']}
            self.inputs['input_ids'] = ids
            assert len(ids) == len(self.image_ids)
        else:
            assert ids.shape[1] == 1
            for row, token in enumerate(ids[:, -1].tolist()):
                self.history[row].append(token); self.seen[row].add(token)
                if token in self.eos_ids: self.active[row] = False

    def _vision(self, _module, _args, output):
        if self.step == 0 and not self.suspended:
            assert self.vision_output is None
            self.vision_output = output

    def _source(self, _module, _args, output):
        if not self.suspended: self.anchor = output[:, -1:].clone()

    def _tail(self, _module, _args, kwargs):
        if not self.suspended: self.tail_kwargs = dict(kwargs)

    def _eligible(self, tokens):
        rows = []
        for row, token in enumerate(tokens.tolist()):
            if not self.active[row] or token not in self.nouns or token in self.seen[row]: continue
            assert len(self.history[row]) == self.step
            prefix = self.tokenizer.decode(self.history[row], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            prior = self.parser.classes(prefix); canonical = self.token_classes[token]
            blocked = canonical in prior
            self.decisions.setdefault(row, []).append(dict(step=self.step, token=token,
                canonical=canonical, prior_classes=sorted(prior), blocked=blocked))
            if not blocked: rows.append(row)
        return rows

    def _finish(self, _module, _args, output):
        if self.suspended: return output
        assert self.reader.step == self.step
        if self.step == 0:
            assert self.vision_output is not None
            self.memory = RegionConsistency(self.wrapper, self.inputs, (self.start, self.end),
                                             self.vision_output, output.last_hidden_state[:, -1:])
            self.memory.crop_cache = self.shared.setdefault('crops', {})
            self.memory.verified_probabilities = self.shared.setdefault('fine_probabilities', {})
            self.vision_output = None
            return output
        if self.disabled: return output
        reference = output.last_hidden_state[:, -1:].clone()
        logits = self.head(reference).float()[:, 0]; tokens = logits.argmax(-1)
        eligible = self._eligible(tokens)
        if not eligible: return output
        features = self.reader.features(eligible, tokens[eligible].tolist())
        probability = self.probe(features)
        selected = []
        for index, row in enumerate(eligible):
            p = float(probability[index]); positive = p < .418672114610672
            self.gates.setdefault(row, []).append(dict(step=self.step, token=int(tokens[row]), p_real=p,
                threshold=.418672114610672, positive=positive, prefix_sha256=prefix_sha(self.history[row]),
                features=features[index].cpu().tolist()))
            if positive: selected.append(row)
        if not selected: return output
        native_cache = self.tail_kwargs['past_key_values']
        commits = []
        for row in selected:
            token = int(tokens[row]); aliases = self.aliases[self.token_classes[token]]
            cached = self.reader._static(19, row)
            norms = self.reader.last_write_norms[row]
            q, p = evidence_preference(norms, cached['log_s'][:, self.reader.columns[token]],
                                      self.reader.visual[19][row], self.reader.current[19]['update'][row])
            drift = float((1-(q*p).sqrt().sum()).clamp(0, 1))
            semantic = category_probabilities(self.reader, row, aliases)
            rect = rectangle(semantic, *self.memory.grid(row), 'evidence')
            if self.policy == 'unverified':
                inspection = dict(reliability_deficit=1., skipped=True)
            else:
                destination = self.output_dir / f"{self.image_ids[row]}_{'_'.join(map(str, rect))}_448.jpg"
                with self.reader.paused():
                    crop = self.memory.encode_crop(self.images[self.image_ids[row]], self.image_ids[row], row, rect, destination)
                inspection = self.memory.inspect(row, native_cache, self.reader, crop, rect, aliases)
            strength = 5*drift*inspection['reliability_deficit']
            applied = rectangle(semantic, *self.memory.grid(row), 'shifted') if self.policy == 'shifted' else rect
            if self.zero_strength: strength = 0.
            if not strength and not self.zero_strength: continue
            proposed_kv = {}
            parent = memory_module.LastPositionCache
            class RecordingCache(parent):
                def update(cache_self, keys, values, layer_idx, cache_kwargs):
                    result = super().update(keys, values, layer_idx, cache_kwargs)
                    proposed_kv[layer_idx] = (keys[row:row+1].clone(), values[row:row+1].clone())
                    return result
            with patch.object(memory_module, 'LastPositionCache', RecordingCache), \
                 patch.object(consistency_module, 'mask_attention', signed_attention):
                hidden, mass_error = self.memory.attenuate(row, native_cache, self.reader,
                    applied, -strength, reference, zero=self.zero_strength)
            assert set(proposed_kv) == set(LAYERS)
            current_logits = self.head(hidden).float()[:, 0]
            before = float(logits[row].log_softmax(0)[token])
            after = float(current_logits[row].log_softmax(0)[token]); new = int(current_logits[row].argmax())
            accepted = bool(new != token and after < before and not self.reject_all)
            delta = max(float((proposed_kv[index][j]-getattr(native_cache.layers[index], name)[row:row+1, :, -1:]).abs().max())
                        for index in LAYERS for j, name in enumerate(['keys', 'values']))
            self.events.setdefault(row, []).append(dict(step=self.step, token=token, proposed_token=new,
                accepted=accepted, before_logp=before, after_logp=after, actual_logp_change=after-before,
                drift=drift, reliability=inspection, strength=strength, rect=rect, applied_rect=applied,
                visual_mass_relative_error_fp32=mass_error, proposed_current_kv_max_difference=delta,
                prefix_sha256=prefix_sha(self.history[row]), cache_position=int(self.tail_kwargs['cache_position'].item()),
                committed_layers=list(LAYERS) if accepted else [], policy=self.policy))
            if accepted: commits.append((row, hidden[row, 0].clone(), proposed_kv))
        # Deliberately defer every commit until all rows have been compared
        # against the same unmodified native cache and reference hidden state.
        if commits:
            hidden = output.last_hidden_state.clone()
            for row, normalized, proposed in commits:
                hidden[row, -1] = normalized
                for index, (keys, values) in proposed.items():
                    native_cache.layers[index].keys[row:row+1, :, -1:] = keys
                    native_cache.layers[index].values[row:row+1, :, -1:] = values
            output.last_hidden_state = hidden
        return output

    @contextmanager
    def capture(self):
        original_vision = self.model.model.visual.forward
        def vision_forward(pixels, grid_thw, **kwargs):
            if not self.suspended and self.step == 0:
                entry = self.shared.get('vision')
                if entry is not None:
                    assert torch.equal(entry['pixels'], pixels) and torch.equal(entry['grid'], grid_thw)
                    self.original_vision_cache_hits += 1
                    return entry['output']
                result = original_vision(pixels, grid_thw, **kwargs)
                self.shared['vision'] = dict(pixels=pixels.detach().clone(), grid=grid_thw.detach().clone(),
                    output=(result[0].detach().clone(), [v.detach().clone() for v in result[1]]))
                return result
            return original_vision(pixels, grid_thw, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(Qwen3BatchLateAttention(self.model, (self.start, self.end), frozenset(), 'sadi').capture())
            stack.enter_context(patch.object(self.model.model.visual, 'forward', vision_forward))
            stack.callback(self.model.register_forward_pre_hook(self._begin, with_kwargs=True).remove)
            stack.callback(self.model.model.visual.register_forward_hook(self._vision).remove)
            stack.callback(self.model.model.visual.blocks[0].register_forward_hook(
                lambda *_: setattr(self, 'actual_encoder_calls', self.actual_encoder_calls+1)).remove)
            stack.callback(self.decoder.layers[19].register_forward_hook(self._source).remove)
            stack.callback(self.decoder.layers[20].register_forward_pre_hook(self._tail, with_kwargs=True).remove)
            stack.enter_context(self.reader.capture())
            stack.callback(self.decoder.register_forward_hook(self._finish).remove)
            yield self

    def statistics(self, responses):
        result = []
        for row, ids in enumerate(responses):
            events = self.events.get(row, [])
            for event in events:
                assert event['step'] < len(ids)
                assert ids[event['step']] == (event['proposed_token'] if event['accepted'] else event['token'])
            result.append(dict(policy=self.policy, full_events=events, gates=self.gates.get(row, []),
                class_decisions=self.decisions.get(row, []), accepted=sum(e['accepted'] for e in events),
                model_steps=self.step+1, disabled=self.disabled, diagnostic_zero_strength=self.zero_strength,
                diagnostic_reject_all=self.reject_all, actual_encoder_calls_in_batch=self.actual_encoder_calls,
                original_vision_cache_hits=self.original_vision_cache_hits,
                crop_encoder_calls_in_batch=self.memory.crop_encoder_calls,
                fine_prefill_calls_in_batch=self.memory.reconstruction_calls,
                fine_probability_cache_hits=self.memory.verification_cache_hits,
                feature_calls_in_batch=self.reader.feature_calls, risk_solver_calls_in_batch=self.reader.risk_solver_calls,
                native_static_cache_hits=self.reader.cache_hits, all_proposals_evaluated_before_commits=True,
                intervention_layers=list(LAYERS), unchanged_base_layers=list(range(5, 19))))
        return result

    def close(self):
        self.reader.owner = None
        self.reader.shared_static = {}
        for name in ['current', 'static', 'values', 'visual']: getattr(self.reader, name).clear()
        if self.memory is not None:
            self.memory.crop_cache = {}; self.memory.verified_probabilities = {}
        self.anchor = self.tail_kwargs = self.memory = self.vision_output = self.inputs = None
