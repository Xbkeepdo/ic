"""Retain all arms and distinguish genuine versus spatially shuffled evidence."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import analyze_qwen3_causal_visual_tilt_pilot as causal
from scripts.run_qwen3_signed_feedback import read, write, sha


def analyze(root):
    original = causal.core.brief_event

    def brief(event, tokens, tokenizer):
        result = original(event, tokens, tokenizer)
        if event.get('mass_matched_visual'):
            result.update(mass_matched_visual=True, contrast_permutation=event['contrast_permutation'],
                feature_guided=True, gradient_guided=False,
                mass_reference='Local actual-query uniform .5 visual enhancement.')
        return result

    with patch.object(causal.core, 'brief_event', brief):
        causal.analyze(root)
    plan = read(root/'protocol.json'); activity = read(root/'activity.json')
    for name in plan['candidates']:
        activity[name].update(contrast_permutation=plan['permutation_by_candidate'][name],
            refresh_source=plan['refresh_by_candidate'][name], feature_guided=True, gradient_guided=False,
            timing_scope='Three candidates run native, refresh, roll in each original batch and share image setup. GPU seconds are batch-wall sums; no inference of intrinsic speed from variant order.')
    write(root/'activity.json', activity)
    provenance = read(root/'analysis_provenance.json')
    provenance.update(mass_matched_adapter_source_sha256=sha(Path(__file__)),
        mass_matching_mechanism_audit_sha256=sha(root/'mechanism_audit.json'),
        activity_sha256=sha(root/'activity.json'), raw_captions_and_evaluation_unchanged=True)
    write(root/'analysis_provenance.json', provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
