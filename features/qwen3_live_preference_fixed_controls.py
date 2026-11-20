"""Fixed-strength spatial controls for the frozen FFN contrastive decoder."""

import torch
from features.qwen3_live_preference_contrast import Qwen3LivePreferenceContrast


def control_prior(preference, policy):
    assert policy in ['uniform_fixed', 'rolled_fixed']
    assert preference.ndim == 1 and torch.isfinite(preference).all()
    if policy == 'uniform_fixed': return torch.ones_like(preference)/len(preference)
    return preference.roll(len(preference)//2)


class Qwen3FixedPreferenceControl(Qwen3LivePreferenceContrast):
    def __init__(self, *args, policy, **kwargs):
        assert policy in ['uniform_fixed', 'rolled_fixed']
        assert 'mode' not in kwargs
        super().__init__(*args, mode='p_fixed', **kwargs)
        self.policy = policy
        original = self.reader.distributions
        def distributions(row, token):
            evidence, preference, drift = original(row, token)
            # Parent p_fixed uses the supplied prior with eta=.5. The original
            # Q/P drift remains a recorded diagnostic and never sets strength.
            return evidence, control_prior(preference, policy), drift
        self.reader.distributions = distributions

    def statistics(self, responses):
        rows = super().statistics(responses)
        for row in rows:
            row.update(mode=self.policy, parent_mode='p_fixed', negative_prior_policy=self.policy,
                recorded_drift_is_unmodified_source_diagnostic=True, drift_does_not_scale_action=True)
        return rows
