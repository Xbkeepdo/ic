"""Bound text suppression by the positive FFN/evidence imbalance.

This changes the dose, not the CRG Conflict-B definition. All modes select
the same low-VRI heads. Flat matches mean dose; roll permutes only those
heads' doses. No target truth label or fitted threshold enters the rule.
"""

import math


def soft_conflict_gates(visual, textual, risk, *, mode, k=8):
    assert mode in ('signal', 'flat', 'roll') and k > 0
    assert len(visual) == len(textual) == len(risk) and len(visual) > 0
    assert all(math.isfinite(x) for x in [*visual, *textual, *risk])
    assert all(0 <= x <= 1 for x in risk)
    vri = [abs(v) / (abs(v) + abs(t) + 1e-8) for v, t in zip(visual, textual)]
    chosen = sorted((h for h, (v, t) in enumerate(zip(visual, textual))
                     if v < 0 and t > 0), key=lambda h: (vri[h], h))[:k]
    raw = [max(0., risk[h] - .5) for h in chosen]
    if mode == 'flat' and chosen:
        applied = [math.fsum(raw) / len(chosen)] * len(chosen)
    elif mode == 'roll' and chosen:
        shift = len(chosen) // 2
        applied = raw[-shift:] + raw[:-shift] if shift else raw[:]
    else:
        applied = raw[:]
    gates = [1.] * len(visual)
    for head, dose in zip(chosen, applied):
        gates[head] = 1. - dose
    assert all(.5 <= value <= 1. for value in gates)
    return gates, dict(base_selected_heads=chosen, vri=vri,
        selected_raw_doses=raw, selected_applied_doses=applied,
        mean_raw_dose=math.fsum(raw) / len(raw) if raw else 0.,
        selected_dose_sum=math.fsum(applied),
        predicted_change=math.fsum((g - 1.) * t for g, t in zip(gates, textual)))
