"""Use complete paired analyses while reporting route-gate semantics explicitly."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import analyze_qwen3_causal_visual_tilt_pilot as common
from scripts.run_qwen3_signed_feedback import read,write,sha


def analyze(root):
    original=common.core.brief_event
    def brief(event,tokens,tokenizer):
        result=original(event,tokens,tokenizer)
        if event.get('no_learned_gate'):
            result.update(route_mode=event['route_mode'],no_learned_gate=True,
                head_ranking_feature_guided=event['route_mode']!='vri',source='Native joint gates on actual text routes.')
        return result
    with patch.object(common.core,'brief_event',brief):common.analyze(root)
    activity=read(root/'activity.json')
    for name in read(root/'protocol.json')['candidates']:
        activity[name].update(route_mode=read(root/'protocol.json')['mode_by_candidate'][name],
            probe_calls=0,original108_features_computed=False,
            timing_scope='VRI uses no semantic head; signal builds lazy cache, roll shares it. No intrinsic speed comparison from sequential setup order.')
    write(root/'activity.json',activity)
    p=read(root/'analysis_provenance.json');p.update(route_adapter_source_sha256=sha(Path(__file__)),
        activity_sha256=sha(root/'activity.json'),raw_captions_and_evaluations_unchanged=True)
    write(root/'analysis_provenance.json',p)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
