"""Paired preservation analysis with explicit risk/E_l gate metadata."""

import argparse
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import analyze_qwen3_causal_visual_tilt_pilot as causal
from scripts.run_qwen3_signed_feedback import read,write,sha


def analyze(root):
    causal.analyze(root)
    activity=read(root/'activity.json')
    for name in read(root/'protocol.json')['candidates']:
        activity[name].update(gate_features=['risk','E_l'],probe_input_width=72,
            feature_reader_output_width=108,feature_guided=True,gradient_guided=False,
            timing_scope='Candidate generated fresh; copied controls have historical timing. Do not interpret cached-control wall time as a current speed comparison.')
    write(root/'activity.json',activity)
    p=read(root/'analysis_provenance.json')
    p.update(gate_adapter_source_sha256=sha(Path(__file__)),activity_sha256=sha(root/'activity.json'),
        prior_added_error_screen_passed=False,preservation_hypothesis_frozen_before_generation=True)
    write(root/'analysis_provenance.json',p)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
