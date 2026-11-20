"""Feature-blind sparse gate for the frozen first-class attention action."""

import hashlib

import torch

from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass


def gate_uniform(seed, image_id, step, token):
    payload = f'{seed}|{image_id}|{step}|{token}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') / 2**64


class Qwen3BatchMatchedHashGate(Qwen3BatchProbeFirstClass):
    def __init__(self, *args, image_ids, gate_seed, gate_rate, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 < gate_rate < 1 and len(image_ids) == len(set(image_ids))
        self.image_ids = tuple(image_ids)
        self.gate_seed, self.gate_rate = int(gate_seed), float(gate_rate)
        self.hash_decisions = {}

    def _budget(self, source):
        probe, features = self.probe, self.reader.features
        feature_rows = None

        def capture_features(rows, tokens):
            nonlocal feature_rows
            feature_rows = list(rows)
            return features(rows, tokens)

        def feature_blind_gate(values):
            assert feature_rows is not None and len(feature_rows) == len(values)
            # Record the real detector output for analysis, never for selection.
            real = probe(values)
            permitted = []
            for row, probability in zip(feature_rows, real.tolist(), strict=True):
                token = int(self.final_tokens[row])
                value = gate_uniform(self.gate_seed, self.image_ids[row], self.step, token)
                allowed = value < self.gate_rate
                self.hash_decisions.setdefault(row, []).append(dict(
                    step=self.step, token=token, uniform=value, permitted=allowed,
                    detector_p_real=probability,
                ))
                permitted.append(allowed)
            # The inherited action uses a <.5 gate; these are gate indicators.
            mask = torch.tensor(permitted, dtype=torch.bool, device=real.device)
            return torch.where(mask, torch.zeros_like(real), torch.ones_like(real))

        self.probe, self.reader.features = feature_blind_gate, capture_features
        try:
            rows, budget, details = super()._budget(source)
        finally:
            self.probe, self.reader.features = probe, features
        for row, detail in zip(rows, details, strict=True):
            decision = self.hash_decisions[row][-1]
            assert decision['step'] == self.step and decision['permitted']
            detail['gate_score'] = detail.pop('probe_p_real')
            detail['probe_p_real'] = decision['detector_p_real']
        return rows, budget, details

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            decisions = [d for d in self.hash_decisions.get(row, ()) if d['step'] < len(tokens)]
            stats.update(
                gate_type='feature_blind_matched_hash', gate_seed=self.gate_seed,
                gate_rate=self.gate_rate, hash_gate_decisions=decisions,
                detector_used_for_selection=False, probe_gated_recycle=False,
                probe_real_threshold=None, probe_calls=len(decisions),
                probe_positive_calls=sum(d['detector_p_real'] < .5 for d in decisions),
                hash_gate_positive_calls=sum(d['permitted'] for d in decisions),
            )
        return result
