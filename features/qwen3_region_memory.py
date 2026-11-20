"""Temporary late-layer visual-memory rereading from a signal-selected crop.

Original visual embeddings and DeepStack outputs are cached once. A crop is
encoded once per rectangle. Refined visual keys/values are reconstructed using
the original full B/P prompt geometry; original image encoding is bypassed.
Only visual slots20--35 are substituted during a current-query tail replay.
"""

from contextlib import ExitStack, contextmanager
import math
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.qwen3_online_risk_f import OnlineRiskF, native_write_norms
from features.qwen3_bounded_visual_tilt import evidence_preference
from features.qwen3_acg_signal_feedback import normalize_nonnegative
from features.qwen3_sar_ems_final_target import LastPositionCache

LAYERS = tuple(range(20, 36))


def rectangle(q, height, width, mode):
    assert mode in ['evidence', 'shifted'] and q.numel() == height*width
    center = int(q.argmax()); y, x = divmod(center, width)
    if mode == 'shifted':
        y, x = (y+height//2) % height, (x+width//2) % width
    h, w = min(3, height), min(3, width)
    top, left = min(max(y-h//2, 0), height-h), min(max(x-w//2, 0), width-w)
    return top, left, top+h, left+w


def image_box(rect, grid, size):
    top, left, bottom, right = rect; height, width = grid; pixel_width, pixel_height = size
    box = (math.floor(left*pixel_width/width), math.floor(top*pixel_height/height),
           math.ceil(right*pixel_width/width), math.ceil(bottom*pixel_height/height))
    assert 0 <= box[0] < box[2] <= pixel_width and 0 <= box[1] < box[3] <= pixel_height
    return box


def replace_region(original, refined, row, grid, refined_grid, rect, amount):
    """Resample within one image's macro-patch rectangle, preserving B/M/D."""
    assert 0 <= amount <= 1 and original.dtype == refined.dtype
    if amount == 0: return original
    height, width = grid; rh, rw = refined_grid; top, left, bottom, right = rect
    assert original.shape[0] % (height*width) == 0 and refined.shape[0] == rh*rw
    values = refined.T.reshape(1, -1, rh, rw).float()
    resized = F.interpolate(values, size=(bottom-top, right-left), mode='bilinear', align_corners=False)
    resized = resized[0].permute(1, 2, 0)
    output = original.clone()
    selected = output[row*height*width:(row+1)*height*width].view(height, width, -1)
    coarse = selected[top:bottom, left:right].float()
    selected[top:bottom, left:right] = ((1-amount)*coarse + amount*resized).to(original.dtype)
    return output


class RegionEvidenceReader(OnlineRiskF):
    """Read only layer19; pause all capture during nested memory construction."""
    def __init__(self, model, span, tokens):
        self.flags = SimpleNamespace(_replaying=False, replaying_tail=False)
        super().__init__(model, span, tokens, owner=self.flags)
        self.anchor = self.tail_kwargs = None

    def _source(self, _module, _args, output):
        if not self._ignore(): self.anchor = output[:, -1:].clone()

    def _tail(self, _module, _args, kwargs):
        if not self._ignore(): self.tail_kwargs = dict(kwargs)

    @contextmanager
    def paused(self):
        assert not self.flags._replaying
        self.flags._replaying = True
        try: yield
        finally: self.flags._replaying = False

    @contextmanager
    def capture(self):
        original = qwen.eager_attention_forward
        def attention(module, query, key, value, *args, **kwargs):
            result = original(module, query, key, value, *args, **kwargs)
            if isinstance(module, qwen.Qwen3VLTextAttention) and module.layer_idx == 19 and not self._ignore():
                self.current[19]['attention'] = result[1][:, :, -1, self.start:self.end].contiguous()
                if self.step == 0: self.values[19] = value[:, :, self.start:self.end].contiguous()
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen, 'eager_attention_forward', attention))
            stack.callback(self.model.model.register_forward_pre_hook(self._begin).remove)
            layer = self.decoder.layers[19]
            stack.callback(layer.register_forward_pre_hook(self._input(19), with_kwargs=True).remove)
            stack.callback(layer.mlp.register_forward_hook(self._ffn(19)).remove)
            stack.callback(layer.register_forward_hook(self._source).remove)
            stack.callback(self.decoder.layers[20].register_forward_pre_hook(self._tail, with_kwargs=True).remove)
            yield self

    def distributions(self, row, token, component='full_sar'):
        assert component in ['full_sar', 'semantic_increment']
        state = self.current[19]
        values = qwen.repeat_kv(self.values[19][row:row+1], self.decoder.layers[19].self_attn.num_key_value_groups)[0]
        norms = native_write_norms(values, state['attention'][row], self.decoder.layers[19].self_attn.o_proj.weight)
        log_s = self._static(19, row)['log_s'][:, self.columns[token]]
        q, p = evidence_preference(norms, log_s, self.visual[19][row], state['update'][row])
        if component == 'semantic_increment': q = normalize_nonnegative(norms*log_s)
        drift = float((1-(q*p).sqrt().sum()).clamp(0, 1))
        return q, p, drift


class RegionMemory:
    def __init__(self, wrapper, inputs, span, visual_output, prefill_hidden):
        self.wrapper, self.model, self.inputs = wrapper, wrapper.model, inputs
        self.start, self.end = span; self.prompt = inputs['input_ids'].shape[1]
        self.base = (visual_output[0].detach().clone(), [x.detach().clone() for x in visual_output[1]])
        self.prefill_hidden = prefill_hidden.detach().clone()
        self.crop_cache = {}; self.crop_encoder_calls = self.reconstruction_calls = 0
        self.vision_cache_hits = 0

    def grid(self, row):
        temporal, height, width = map(int, self.inputs['image_grid_thw'][row].tolist())
        merge = self.model.model.visual.spatial_merge_size
        assert temporal == 1 and height % merge == width % merge == 0
        height, width = height//merge, width//merge
        assert height*width == self.end-self.start
        return height, width

    def encode_crop(self, image, image_id, row, rect, destination):
        from PIL import Image
        key = (image_id, tuple(rect), 448)
        if key in self.crop_cache:
            self.vision_cache_hits += 1
            return self.crop_cache[key]
        box = image_box(rect, self.grid(row), image.size)
        crop = image.crop(box).resize((448, 448), Image.Resampling.BICUBIC)
        crop.save(destination)
        prepared = self.wrapper._prompt_inputs(crop, 'Describe this image.')
        pixels = prepared['pixel_values'].to(self.wrapper.device)
        grid = prepared['image_grid_thw'].to(self.wrapper.device)
        embeddings, stack = self.model.model.get_image_features(pixels, grid)
        assert len(embeddings) == 1
        merged = self.model.model.visual.spatial_merge_size
        crop_grid = tuple(int(n)//merged for n in grid[0, 1:].tolist())
        assert embeddings[0].shape[0] == crop_grid[0]*crop_grid[1] > 9
        entry = dict(features=embeddings[0], deepstack=stack, grid=crop_grid,
                     pixel_box=box, image_file=str(destination))
        self.crop_cache[key] = entry; self.crop_encoder_calls += 1
        return entry

    def keys(self, row, native_cache, reader, *, crop=None, rect=None, amount=0.):
        assert crop is not None or amount == 0
        if amount:
            image = replace_region(self.base[0], crop['features'], row, self.grid(row), crop['grid'], rect, amount)
            stack = [replace_region(old, new, row, self.grid(row), crop['grid'], rect, amount)
                     for old, new in zip(self.base[1], crop['deepstack'], strict=True)]
        else: image, stack = self.base
        original_rope = self.model.model.rope_deltas
        current = dict(self.inputs, cache_position=torch.arange(self.prompt, device=self.wrapper.device))
        try:
            with reader.paused(), patch.object(self.model.model.visual, 'forward', return_value=(image, stack)):
                self.model.model.rope_deltas = None
                result = self.model.model(**current, use_cache=True, return_dict=True)
            assert torch.equal(self.model.model.rope_deltas, original_rope)
            self.reconstruction_calls += 1
            if amount == 0:
                assert torch.equal(result.last_hidden_state[:, -1:], self.prefill_hidden)
            keys = {}
            other = [i for i in range(self.inputs['input_ids'].shape[0]) if i != row]
            for index, layer in enumerate(result.past_key_values.layers):
                native = native_cache.layers[index]
                if amount == 0:
                    assert torch.equal(layer.keys, native.keys[:, :, :self.prompt])
                    assert torch.equal(layer.values, native.values[:, :, :self.prompt])
                elif other:
                    assert torch.equal(layer.keys[other], native.keys[other, :, :self.prompt])
                    assert torch.equal(layer.values[other], native.values[other, :, :self.prompt])
                if index in LAYERS:
                    keys[index] = (layer.keys[row:row+1, :, self.start:self.end].clone(),
                                   layer.values[row:row+1, :, self.start:self.end].clone())
            return keys
        finally:
            self.model.model.rope_deltas = original_rope

    def replay(self, row, native_cache, reader, refined, reference, zero=False):
        """Return a proposal, then restore both visual slots and current KV."""
        backup = {index: (native_cache.layers[index].keys[row:row+1, :, self.start:self.end].clone(),
                          native_cache.layers[index].values[row:row+1, :, self.start:self.end].clone(),
                          native_cache.layers[index].keys[row:row+1, :, -1:].clone(),
                          native_cache.layers[index].values[row:row+1, :, -1:].clone()) for index in LAYERS}
        cache = LastPositionCache(native_cache, [row], check_identical=zero)
        kwargs = dict(reader.tail_kwargs, past_key_values=cache)
        current = reader.anchor.clone()
        try:
            for index, (keys, values) in refined.items():
                native_cache.layers[index].keys[row:row+1, :, self.start:self.end] = keys
                native_cache.layers[index].values[row:row+1, :, self.start:self.end] = values
            with reader.paused():
                for index in LAYERS: current = reader.decoder.layers[index](current, **kwargs)
            normalized = reader.decoder.norm(current)
            assert cache.updates == list(LAYERS)
            other = [i for i in range(len(current)) if i != row]
            if other: assert torch.equal(normalized[other], reference[other])
            if zero: assert torch.equal(normalized, reference)
            return normalized
        finally:
            for index, (keys, values, last_keys, last_values) in backup.items():
                layer = native_cache.layers[index]
                layer.keys[row:row+1, :, self.start:self.end] = keys
                layer.values[row:row+1, :, self.start:self.end] = values
                layer.keys[row:row+1, :, -1:] = last_keys
                layer.values[row:row+1, :, -1:] = last_values
                assert torch.equal(layer.keys[row:row+1, :, self.start:self.end], keys)
                assert torch.equal(layer.values[row:row+1, :, self.start:self.end], values)
                assert torch.equal(layer.keys[row:row+1, :, -1:], last_keys)
                assert torch.equal(layer.values[row:row+1, :, -1:], last_values)
