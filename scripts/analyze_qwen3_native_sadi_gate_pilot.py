"""Keep both evaluators and paired analysis; state the source108 gate scope."""

import argparse
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import analyze_qwen3_causal_visual_tilt_pilot as causal
from scripts.run_qwen3_signed_feedback import read,write,sha


def analyze(root):
    causal.analyze(root)
    plan=read(root/'protocol.json');activity=read(root/'activity.json')
    for name in plan['candidates']:
        activity[name].update(gate_features=['risk','I','E_l'],probe_input_width=108,
            feature_reader_output_width=108,gate_variant=plan['gate_variant'],
            feature_guided=True,gradient_guided=False,
            coefficient_scope='Uniform0.5; target-conditioned diagnostics select when to intervene, not per-patch coefficient.',
            timing_scope='New candidate generated fresh; copied controls retain historical timings.')
    write(root/'activity.json',activity)
    provenance=read(root/'analysis_provenance.json')
    provenance.update(source_gate_adapter_sha256=sha(Path(__file__)),
        activity_sha256=sha(root/'activity.json'),
        initial_pilot_retention_screen_passed=plan['initial_pilot_retention_screen_passed'],
        source_gate_diagnostic_sha256=sha(Path(plan['gate_diagnostic'])/'analysis.json'),
        gate_variant=plan['gate_variant'],no_independent_validation_claim=True)
    write(root/'analysis_provenance.json',provenance)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
