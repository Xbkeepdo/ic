"""Full-SADI continuation with evidence-selected crop memory and attention."""

from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import torch

from features.qwen3_decoupled_semantic_route import movement_fraction, semantic_destination, semantic_route
from features.qwen3_post_sadi_ffn_attenuation import FFNResidualReader
from features.qwen3_region_memory import RegionMemory, rectangle, LAYERS
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from features import qwen3_region_memory as memory_module


class LiveSemanticCrop:
    def __init__(self, wrapper, span, ids, photos, names, tokens, mapping, priors, graph,
                 parser, output, *, policy='semantic_crop', shared=None,
                 disabled=False, zero=False, reject=False, cached=None):
        assert policy in ['semantic_crop', 'semantic_crop_route', 'shifted_crop']
        self.wrapper, self.model, self.span = wrapper, wrapper.model, tuple(span)
        self.ids, self.photos, self.names = ids, photos, names
        self.tokens, self.mapping = list(tokens), list(mapping)
        self.classes = dict(zip(tokens, mapping)); self.parser = parser
        self.priors, self.graph = priors, graph
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.policy, self.disabled, self.zero, self.reject = policy, disabled, zero, reject
        self.shared = {} if shared is None else shared
        self.reader = FFNResidualReader(self.model, span, tokens)
        self.reader.static = self.shared.setdefault('static19', {})
        self.cached = {} if cached is None else cached
        self.head = self.model.get_output_embeddings(); self.norm = self.reader.decoder.norm
        self.step = -1; self.history = []; self.active = None
        self.vision = self.memory = self.inputs = self.early = None
        self.events = {}; self.gates = {}; self.early_head_calls = 0
        self.encoder_calls = self.vision_hits = self.refined_hits = 0
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]

    def _ignore(self): return self.reader._ignore()

    def _begin(self, _module, args, kwargs):
        if self._ignore(): return
        self.step = self.reader.step
        ids = kwargs.get('input_ids', args[0] if args else None)
        assert ids is not None
        if self.step == 0:
            self.history = [[] for _ in self.ids]
            self.active = [True for _ in self.ids]
            self.inputs = {k: kwargs[k] for k in ['pixel_values', 'image_grid_thw', 'attention_mask']}
            self.inputs['input_ids'] = ids
            assert len(ids) == len(self.ids)
        else:
            assert ids.shape == (len(self.ids), 1)
            for row, token in enumerate(ids[:, -1].tolist()):
                self.history[row].append(token)
                if token in self.eos: self.active[row] = False

    def _early(self, _module, args, kwargs):
        if self._ignore() or self.step != 0: return
        h = args[0] if args else kwargs['hidden_states']
        self.early = h[:, self.span[0]:self.span[1]].clone()

    def _vision(self, _module, _args, result):
        if not self._ignore() and self.step == 0: self.vision = result

    def _class_mass(self, row):
        bank = self.shared.setdefault('class_mass3', {})
        if row not in bank:
            if row in self.cached:
                entry = self.cached[row]
                assert entry['batch_ids'] == self.ids and entry['span'] == list(self.span)
                assert entry['prompt'] == self.inputs['input_ids'].shape[1]
                mass = entry['mass3']
                self.reader.static[19, row] = dict(log_s=entry['log_s19'].to(self.wrapper.device))
            else:
                h = self.norm(self.early[row]); selected = torch.tensor(self.tokens, device=h.device)
                pieces = []
                for offset in range(0, len(h), 64):
                    logits = self.head(h[offset:offset+64]).float()
                    p = (logits[:, selected]-logits.logsumexp(-1, keepdim=True)).clamp_max(0).exp()
                    pieces.append(torch.log1p(logits.shape[-1]*p).cpu().numpy())
                    self.early_head_calls += 1
                scaled = np.expm1(np.concatenate(pieces).astype(np.float64))
                mapping = np.array(self.mapping)
                mass = np.stack([scaled[:, mapping == c].sum(1) for c in self.names], axis=1)
                path = self.output/f'class_mass3_{self.ids[row]}.npz'
                if not path.exists():
                    np.savez_compressed(path, class_mass=mass, image_id=self.ids[row],
                        class_names=np.array(self.names), tokens=np.array(self.tokens),
                        batch_ids=np.array(self.ids), span=np.array(self.span),
                        prompt_length=self.inputs['input_ids'].shape[1])
            bank[row] = torch.as_tensor(mass, device=self.wrapper.device, dtype=torch.float64)
        return bank[row]

    def _finish(self, _module, _args, output):
        if self._ignore(): return output
        if self.step == 0:
            assert self.vision is not None and self.early is not None
            self.memory = RegionMemory(self.wrapper, self.inputs, self.span, self.vision,
                                       output.last_hidden_state[:, -1:])
            self.memory.crop_cache = self.shared.setdefault('crops', {})
            self.vision = None
            return output
        if self.disabled: return output
        reference = output.last_hidden_state[:, -1:].clone()
        native = self.head(reference).float()[:, 0]; targets = native.argmax(-1)
        cache = self.reader.tail_kwargs['past_key_values']; commits = []
        for row, token in enumerate(targets.tolist()):
            if not self.active[row] or token not in self.classes: continue
            prefix = self.wrapper.tokenizer.decode(self.history[row], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            target = self.classes[token]
            if target in self.parser.classes(prefix): continue
            residual = self.reader.ffn_residual[row, 0]; update = self.reader.current[19]['update'][row]
            gain = float(self.head.weight[token].float()@(self.norm(residual+update).float()-self.norm(residual).float()))
            if gain <= 0: continue
            mass = self._class_mass(row); prior = self.priors.to(mass.device)
            rival = self.graph[target]; y, r = self.names.index(target), self.names.index(rival)
            rho = mass.mean(0)/prior
            eligible = float(rho[r]) > max(1., float(rho[y]))
            self.gates.setdefault(row, []).append(dict(step=self.step, token=token, target=target,
                rival=rival, gain=gain, target_rho=float(rho[y]), rival_rho=float(rho[r]), eligible=eligible))
            if not eligible: continue
            q, p, drift = self.reader.distributions(row, token)
            amount = 0. if self.zero else movement_fraction(drift, gain)
            destination = semantic_destination(mass, prior, y, r).float()
            location = 'shifted' if self.policy == 'shifted_crop' else 'evidence'
            rect = rectangle(destination, *self.memory.grid(row), location)
            blend = 0. if self.zero else drift
            key = row, tuple(rect), blend
            refined_bank = self.shared.setdefault('refined', {})
            if key not in refined_bank:
                if self.zero:
                    refined_bank[key] = self.memory.keys(row, cache, self.reader)
                else:
                    path = self.output/f'{self.ids[row]}_{"_".join(map(str, rect))}_448.jpg'
                    with self.reader.paused(): crop = self.memory.encode_crop(self.photos[self.ids[row]], self.ids[row], row, rect, path)
                    refined_bank[key] = self.memory.keys(row, cache, self.reader, crop=crop, rect=rect, amount=blend)
            else: self.refined_hits += 1
            refined = refined_bank[key]; proposed_kv = {}; parent = memory_module.LastPositionCache
            class RecordingCache(parent):
                def update(cache_self, keys, values, layer_idx, cache_kwargs):
                    result = super().update(keys, values, layer_idx, cache_kwargs)
                    proposed_kv[layer_idx] = (keys[row:row+1].clone(), values[row:row+1].clone())
                    return result
            context = semantic_route(self.span, row, destination, p, destination, amount, LAYERS) if self.policy == 'semantic_crop_route' else nullcontext([])
            with patch.object(memory_module, 'LastPositionCache', RecordingCache), context as action_calls:
                hidden = self.memory.replay(row, cache, self.reader, refined, reference, zero=self.zero)
            assert set(proposed_kv) == set(LAYERS)
            logits = self.head(hidden).float()[row, 0]; new = int(logits.argmax())
            before = float(native[row].log_softmax(0)[token]); after = float(logits.log_softmax(0)[token])
            accepted = new != token and after < before and not self.reject
            delta = max(float((proposed_kv[i][j]-getattr(cache.layers[i], name)[row:row+1, :, -1:]).abs().max())
                for i in LAYERS for j, name in enumerate(['keys', 'values']))
            event = dict(step=self.step, token=token, target=target, rival=rival, proposed_token=new,
                proposed_text=self.wrapper.tokenizer.decode([new]), accepted=accepted, before_logp=before,
                after_logp=after, gain=gain, drift=drift, amount=amount, rect=rect, blend=blend,
                current_kv_difference=delta, committed_layers=list(LAYERS) if accepted else [],
                fp32_mass_error=max([c['fp32_mass_error'] for c in action_calls], default=0.))
            self.events.setdefault(row, []).append(event)
            if accepted: commits.append((row, hidden[row, 0].clone(), proposed_kv, event))
        # Every proposal sees the unmodified native state; commit accepted rows together.
        if commits:
            changed = output.last_hidden_state.clone()
            for row, hidden, states, event in commits:
                changed[row, -1] = hidden
                for index, (keys, values) in states.items():
                    layer = cache.layers[index]
                    layer.keys[row:row+1, :, -1:] = keys
                    layer.values[row:row+1, :, -1:] = values
                    assert torch.equal(layer.keys[row:row+1, :, -1:], keys)
                    assert torch.equal(layer.values[row:row+1, :, -1:], values)
                event['positive_commit_readback_exact'] = True
            output.last_hidden_state = changed
        return output

    @contextmanager
    def capture(self):
        original = self.model.model.visual.forward
        def vision(pixels, grid_thw, **kwargs):
            if not self._ignore() and self.step == 0:
                if 'vision' in self.shared:
                    entry = self.shared['vision']
                    assert torch.equal(entry['pixels'], pixels) and torch.equal(entry['grid'], grid_thw)
                    self.vision_hits += 1; return entry['output']
                value = original(pixels, grid_thw, **kwargs)
                self.shared['vision'] = dict(pixels=pixels.clone(), grid=grid_thw.clone(),
                    output=(value[0].clone(), [v.clone() for v in value[1]]))
                return value
            return original(pixels, grid_thw, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(Qwen3BatchLateAttention(self.model, self.span, frozenset(), 'sadi').capture())
            stack.enter_context(self.reader.capture())
            stack.enter_context(patch.object(self.model.model.visual, 'forward', vision))
            stack.callback(self.model.model.register_forward_pre_hook(self._begin, with_kwargs=True).remove)
            stack.callback(self.reader.decoder.layers[3].register_forward_pre_hook(self._early, with_kwargs=True).remove)
            stack.callback(self.model.model.visual.register_forward_hook(self._vision).remove)
            stack.callback(self.model.model.visual.blocks[0].register_forward_hook(lambda *_: setattr(self, 'encoder_calls', self.encoder_calls+1)).remove)
            stack.callback(self.model.model.register_forward_hook(self._finish).remove)
            yield self

    def statistics(self, responses):
        result = []
        for row, ids in enumerate(responses):
            events = self.events.get(row, [])
            for e in events:
                assert e['step'] < len(ids)
                assert ids[e['step']] == (e['proposed_token'] if e['accepted'] else e['token'])
            result.append(dict(full_events=events, gates=self.gates.get(row, []), policy=self.policy,
                accepted=sum(e['accepted'] for e in events), steps=self.step+1, encoder_calls=self.encoder_calls,
                early_head_calls=self.early_head_calls, source19_head_calls=self.reader.head_calls,
                vision_cache_hits=self.vision_hits, crop_encoder_calls=self.memory.crop_encoder_calls,
                refined_cache_hits=self.refined_hits, prompt_reconstructions=self.memory.reconstruction_calls,
                all_proposals_before_commits=True, base_layers=list(range(5, 19)), intervention_layers=list(LAYERS)))
        return result

    def close(self):
        self.reader.static = {}
        for name in ['current', 'values', 'visual']: getattr(self.reader, name).clear()
        self.early = self.vision = self.memory = self.inputs = None
