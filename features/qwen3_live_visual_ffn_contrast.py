"""Live attention contrast with source19 visual FFN-output preferences."""

from contextlib import contextmanager
import torch
from features.qwen3_live_preference_contrast import Qwen3LivePreferenceContrast
from features.qwen3_visual_ffn_preference import VisualFFNPreferenceReader, output_priors


class Qwen3LiveVisualFFNContrast(Qwen3LivePreferenceContrast):
    def __init__(self, wrapper, span, nouns, *, policy='uv_excess', **kwargs):
        assert policy in ['uv_excess', 'hpre_excess', 'uniform', 'rolled_uv_excess']
        assert 'mode' not in kwargs
        super().__init__(wrapper, span, nouns, mode='p_drift', **kwargs)
        self.policy = policy
        self.reader = VisualFFNPreferenceReader(self.model, span, self.nouns)
        self.reader.static = self.shared.setdefault('static', {})
        self.shared.setdefault('visual_ffn', {})
        self.preference_diagnostics = {}; self.visual_ffn_cache_hits = 0
        def distributions(row, token):
            q, old, current, old_drift, drift = self.reader.comparison_distributions(row, token)
            priors = output_priors(q, old, current)
            self.preference_diagnostics[self.step, row] = dict(drift_hpre=old_drift,
                preference_tv_uv_hpre=float(.5*(current-old).abs().sum()),
                preference_tv_uv_uniform=float(.5*(current-1/len(current)).abs().sum()),
                visual_excess_mass=float((current-q).clamp_min(0).sum()))
            return q, priors[policy], drift
        self.reader.distributions = distributions

    def _share_visual_updates(self, _module, _args, _value):
        if self.reader._ignore() or self.reader.step != 0: return
        mlp = self.decoder.layers[19].mlp
        weights = [mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]
        identity = tuple((w.data_ptr(),w._version,tuple(w.shape),str(w.dtype)) for w in weights)
        for key, value in self.reader.visual_updates.items():
            if key in self.shared['visual_ffn']:
                old = self.shared['visual_ffn'][key]
                assert old['identity'] == identity and torch.equal(old['value'], value)
                self.reader.visual_updates[key] = old['value']; self.visual_ffn_cache_hits += 1
            else:
                self.shared['visual_ffn'][key] = dict(identity=identity,value=value)

    @contextmanager
    def capture(self):
        with super().capture():
            # The parent reader captures first; this later hook checks and
            # shares those native outputs without executing another MLP.
            handle = self.decoder.layers[19].mlp.register_forward_hook(self._share_visual_updates)
            try: yield self
            finally: handle.remove()

    def statistics(self, responses):
        rows = super().statistics(responses)
        for row, stats in enumerate(rows):
            for event in stats['full_events']:
                event.update(self.preference_diagnostics[event['step'],row])
            stats.update(mode=self.policy,parent_mode='p_drift',source_preference='actual_visual_ffn_outputs19',
                common_uv_drift_strength=True,negative_prior_policy=self.policy,
                visual_ffn_capture_calls=self.reader.visual_capture_calls,
                visual_ffn_cache_hits=self.visual_ffn_cache_hits,
                additional_source19_mlp_forwards=0)
        return rows

    def close(self):
        self.reader.visual_updates.clear()
        super().close()
