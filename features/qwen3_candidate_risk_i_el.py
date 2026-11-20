"""Read original108 candidate features while sharing target-independent writes."""

import numpy as np
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.qwen3_online_risk_f import exact_risk,native_write_norms


def candidate_features(reader,row,tokens):
    """Same feature order/arithmetic as NativeRiskIEl; reuse each layer's I."""
    risks=[[] for _ in tokens];semantics=[[] for _ in tokens];writes=[];norms19=None
    for index in range(36):
        cached,state=reader._static(index,row),reader.current[index]
        groups=state['attention'].shape[1]//reader.values[index].shape[1]
        values=qwen.repeat_kv(reader.values[index][row:row+1],groups)[0]
        norms=native_write_norms(values,state['attention'][row],reader.decoder.layers[index].self_attn.o_proj.weight)
        writes.append(float(norms.sum()))
        if index==19:norms19=norms
        for k,token in enumerate(tokens):
            column=reader.columns[token]
            risk,*_=exact_risk(reader.visual[index][row],state['hpre'][row],state['update'][row],
                              state['attention'][row],cached['log_s'][:,column])
            risks[k].append(risk);semantics[k].append(float(cached['E_l'][column]))
    matrix=np.asarray([r+writes+s for r,s in zip(risks,semantics,strict=True)],dtype=np.float32)
    assert matrix.shape==(len(tokens),108) and np.isfinite(matrix).all()
    return torch.from_numpy(matrix).to(reader.head.weight.device),norms19
