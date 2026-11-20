"""Summarize proposal acceptance and gradient-probe fidelity from saved captions."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import source_groups


def audit(root):
    plan = json.loads((root / 'protocol.json').read_bytes())
    assert json.loads((root / 'run_status.json').read_bytes())['status'] == 'complete'
    result = {}
    for name in plan['candidates']:
        groups = [group for path in sorted((root / plan['cohort']).glob(name + '.shard*.jsonl'))
                  for group in source_groups(path)]
        rows = [row for group in groups for row in group]
        assert len(rows) == plan['images']
        probes = 0
        for group in groups:
            counts = {row['sar_ems_statistics']['gradient_probe_calls'] for row in group}
            assert len(counts) == 1
            probes += counts.pop()
        stats = [row['sar_ems_statistics'] for row in rows]
        for stat in stats:
            assert not stat['diagnostic_reject_all']
            assert stat['accepted_guidance_calls'] == stat['log_probability_decreases']
            assert stat['measured_log_probability_changes'] == stat['guided_attention_calls']
            assert stat['accepted_guidance_calls'] <= stat['guided_attention_calls']
            if stat['gradient_scope'] == 'final':
                assert stat['maximum_probe_hidden_difference'] == 0
        proposals = sum(s['guided_attention_calls'] for s in stats)
        accepted = sum(s['accepted_guidance_calls'] for s in stats)
        result[name] = dict(images=len(rows), batches=len(groups), gradient_probe_calls=probes,
                            proposals=proposals, accepted=accepted,
                            acceptance_rate=accepted / max(1, proposals),
                            immediate_candidate_changes=proposals - sum(s['target_match_events'] for s in stats),
                            proposal_mean_log_probability_change=sum(s['mean_log_probability_change'] * s['measured_log_probability_changes'] for s in stats) / max(1, proposals),
                            maximum_probe_hidden_difference=max(s['maximum_probe_hidden_difference'] for s in stats))
    (root / 'mechanism_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
