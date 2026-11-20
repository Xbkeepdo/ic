"""Reduce only text-route attenuation, retaining frozen CRG/SAR/FFN ranking."""

from copy import deepcopy
import math
import torch

from features.qwen3_conflict_route_feedback import Qwen3BatchConflictRoute


def attenuate_route_payload(gates, details, native_text_routes, scale):
    """Scale text removal, with explicit endpoints and actual scaled telemetry."""
    assert 0<=scale<=1 and set(gates)==set(native_text_routes)
    assert all(len(value)==len(details) for value in gates.values())
    updated=deepcopy(details);payload={}
    for index,strong in gates.items():
        if scale==1:actual=strong.clone()
        elif scale==0:actual=torch.ones_like(strong)
        else:actual=1-float(scale)*(1-strong)
        payload[index]=actual
        maxima=((actual-1)[...,None]*native_text_routes[index].float()).abs().amax((1,2)).tolist()
        for row,detail in enumerate(updated):
            layer=detail['layer_details'][str(index)]
            layer['strong_text_gates']=layer['text_gates'][:]
            layer['text_gates']=actual[row].tolist()
            layer['attenuation_scale']=float(scale)
            layer['budget_max']=maxima[row]
            layer['predicted_log_probability_change']=math.fsum(
                (g-1)*effect for g,effect in zip(layer['text_gates'],layer['textual_effect'],strict=True))
            layer['selected_heads']=(actual[row]<1).nonzero().flatten().tolist()
    for detail in updated:
        layers=detail['layer_details'].values()
        detail.update(attenuation_scale=float(scale),
            budget_max=max(layer['budget_max'] for layer in layers),
            predicted_log_probability_change=math.fsum(layer['predicted_log_probability_change'] for layer in layers))
    return payload,updated


class Qwen3BatchMildConflictRoute(Qwen3BatchConflictRoute):
    def __init__(self,*args,attenuation_scale=.5,**kwargs):
        assert 0<=attenuation_scale<=1
        super().__init__(*args,**kwargs)
        self.attenuation_scale=float(attenuation_scale)

    def _budget(self,source):
        rows,gates,details=super()._budget(source)
        if not rows:return rows,gates,details
        payload,updated=attenuate_route_payload(gates,details,self._native_text_routes,self.attenuation_scale)
        return rows,payload,updated

    def statistics(self,responses):
        result=super().statistics(responses)
        for stat in result:
            stat.update(mild_conflict_route=True,attenuation_scale=self.attenuation_scale,
                selected_text_gate_range=[1-self.attenuation_scale,1-.5*self.attenuation_scale],
                dose_definition='g_actual=1-scale*(1-g_strong), ranks and native conflict selection unchanged.',
                acg_scope='Frozen CRG/SAR/FFN ranking with reduced text attenuation on actual per-layer routes.')
        return result
