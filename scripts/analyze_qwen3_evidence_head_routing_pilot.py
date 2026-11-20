"""Keep all routing controls, paired counts and emitted-token EOS metadata."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import analyze_qwen3_causal_visual_tilt_pilot as causal
from scripts.run_qwen3_signed_feedback import read,write,sha


def analyze(root):
    original=causal.core.brief_event
    def brief(event,tokens,tokenizer):
        result=original(event,tokens,tokenizer)
        if event.get('evidence_head_routing'):
            result.update(evidence_head_routing=True,head_mode=event['head_mode'],
                feature_guided=True,gradient_guided=False,
                matching='Local image-wise mean visual mass across heads; conditional patterns preserved.')
        return result
    with patch.object(causal.core,'brief_event',brief):
        causal.analyze(root)
    plan=read(root/'protocol.json');activity=read(root/'activity.json')
    for name in plan['candidates']:
        activity[name].update(head_mode=plan['head_mode_by_candidate'][name],
            feature_guided=True,gradient_guided=False,
            head_allocation_guided=plan['head_mode_by_candidate'][name]!='uniform',
            timing_scope='Uniform, guided and rolled run sequentially within each cached original batch, sharing image semantics. Do not infer intrinsic speed from setup order.')
    write(root/'activity.json',activity)
    p=read(root/'analysis_provenance.json')
    p.update(head_routing_adapter_sha256=sha(Path(__file__)),activity_sha256=sha(root/'activity.json'),
        raw_captions_and_evaluations_unchanged=True)
    write(root/'analysis_provenance.json',p)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
