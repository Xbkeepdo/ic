"""Separate FFN excess preference from target-supported visual evidence."""

import torch


def coupled_priors(evidence, preference):
    assert evidence.ndim == preference.ndim == 1 and evidence.shape == preference.shape
    assert torch.isfinite(evidence).all() and torch.isfinite(preference).all()
    assert evidence.min() >= 0 and preference.min() >= 0
    uniform = torch.ones_like(evidence)/len(evidence)
    excess = (preference-evidence).clamp_min(0)
    deficit = (evidence-preference).clamp_min(0)
    def normalize(value):
        mass = value.sum()
        return value/mass if float(mass) > 0 else uniform
    excess, deficit = normalize(excess), normalize(deficit)
    return dict(excess=excess, deficit=deficit, uniform=uniform,
                rolled_excess=excess.roll(len(excess)//2))
