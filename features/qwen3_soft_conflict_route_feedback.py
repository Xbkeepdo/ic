"""Complete SADI followed by evidence-sized late text-route corrections."""

from copy import deepcopy
import math
import torch

from features.qwen3_conflict_route_feedback import Qwen3BatchConflictRoute
from features.qwen3_soft_conflict_dose import soft_conflict_gates


def soft_route_payload(details, native_text, *, mode, k=8, zero=False):
    assert set(native_text) == set(range(19, 36))
    assert all(len(value) == len(details) for value in native_text.values())
    updated = deepcopy(details)
    payload = {}
    for index, routes in native_text.items():
        expected = []
        for detail in updated:
            layer = detail['layer_details'][str(index)]
            gates, measures = soft_conflict_gates(layer['visual_effect'], layer['textual_effect'],
                layer['raw_head_risk'], mode=mode, k=k)
            expected.append([1.] * len(gates) if zero else gates)
            layer.pop('applied_head_risk')
            layer.update(soft_conflict_route=True, dose_mode=mode, route_mode=mode,
                base_selected_heads=measures['base_selected_heads'],
                base_vri_double=measures['vri'],
                selected_raw_doses=measures['selected_raw_doses'],
                selected_applied_doses=measures['selected_applied_doses'],
                diagnostic_zero_dose=zero)
        actual = torch.tensor(expected, device=routes.device, dtype=torch.float32)
        payload[index] = actual
        maxima = ((actual - 1)[..., None] * routes.float()).abs().amax((1, 2)).tolist()
        for row, detail in enumerate(updated):
            layer = detail['layer_details'][str(index)]
            layer.update(text_gates=actual[row].tolist(),
                selected_heads=(actual[row] < 1).nonzero().flatten().tolist(),
                actual_soft_doses=(1-actual[row]).tolist(), budget_max=maxima[row],
                predicted_log_probability_change=math.fsum(
                    (g-1)*t for g,t in zip(actual[row].tolist(), layer['textual_effect'])))
    for detail in updated:
        layers = list(detail['layer_details'].values())
        detail.update(soft_conflict_route=True, dose_mode=mode, route_mode=mode,
            budget_max=max(d['budget_max'] for d in layers),
            predicted_log_probability_change=math.fsum(d['predicted_log_probability_change'] for d in layers))
    return payload, updated


class Qwen3BatchSoftConflictRoute(Qwen3BatchConflictRoute):
    def __init__(self, *args, dose_mode='signal', **kwargs):
        assert dose_mode in ('signal', 'flat', 'roll')
        assert 'route_mode' not in kwargs
        # Obtain the same unpermuted native Q/P score in every arm. The
        # parent's rank-based payload is replaced before any actual replay.
        super().__init__(*args, route_mode='signal', **kwargs)
        self.dose_mode = dose_mode
        self.soft_dose_decisions = {}

    def _budget(self, source):
        rows, _discarded_rank_payload, details = super()._budget(source)
        if not rows:
            return rows, None, details
        payload, updated = soft_route_payload(details, self._native_text_routes,
            mode=self.dose_mode, k=self.route_k, zero=self.zero_route)
        effective = []
        for position, (row, detail) in enumerate(zip(rows, updated)):
            layers = list(detail['layer_details'].values())
            positive = detail['budget_max'] > 0
            self.soft_dose_decisions.setdefault(row, []).append(dict(step=self.step,
                token=detail['token'], actual_nonzero=positive,
                selected_heads=sum(len(d['base_selected_heads']) for d in layers),
                positive_doses=sum(len(d['selected_heads']) for d in layers),
                total_actual_dose=math.fsum(math.fsum(d['actual_soft_doses']) for d in layers)))
            if positive or self.zero_route:
                effective.append(position)
        self._native_text_routes = {index: value[effective]
                                    for index, value in self._native_text_routes.items()}
        return ([rows[i] for i in effective],
                {index: value[effective] for index, value in payload.items()},
                [updated[i] for i in effective])

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stat) in enumerate(zip(responses, result)):
            decisions = self.soft_dose_decisions.get(row, [])
            assert all(d['step'] <= len(tokens) for d in decisions)
            stat.update(soft_conflict_route=True, dose_mode=self.dose_mode,
                dose_formula='d=relu(R-.5); g=1-d; maximum removal .5',
                selected_head_rule='At most8 native Conflict-B heads with smallest VRI.',
                control_scope='Signal/roll preserve dose multiset within selected heads; flat preserves local mean dose.',
                full_soft_dose_decisions=decisions,
                soft_zero_dose_steps=sum(not d['actual_nonzero'] for d in decisions),
                acg_scope='Complete SADI5--18; soft late19--35 text routes, no learned gate or new gradient query.')
        return result
