"""Resume saved next6 evaluation using its inherited legacy activity fields."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_qwen3_full_sadi_next6 import check
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.qwen3_caption_groups_with_cache import source_groups


def caption_paths(root):
    plan = read(root / 'protocol.json')
    return [root / f'{plan["cohort"]}/{name}.shard{shard}.jsonl'
            for name in plan['variants'] for shard in range(plan['shards'])]


def prepare(root):
    check(root)
    assert not (root / 'analysis_recovery_request.json').exists()
    assert not list((root / 'evaluations').glob('**/metrics*.json'))
    assert "KeyError: 'accepted'" in (root / 'evaluate_sentence.log').read_text()
    assert read(root / 'pipeline_complete.json')['passed']
    assert not read(root / 'analysis_complete.json')['passed']
    assert read(root / 'mechanism_audit.json')['passed']
    for path in caption_paths(root):
        for group in source_groups(path):
            for row in group:
                stats = row.get('sar_ems_statistics', {})
                if stats:
                    assert 'guided_steps' in stats and 'target_matches_generated' in stats
    write(root / 'analysis_recovery_request.json', dict(
        fixed_at=time.time(), original_request_sha256=sha(root / 'analysis_request.json'),
        original_failure_preserved=True, before_any_new_caption_metrics=True,
        reason='Region-only summary requires accepted; next6 already inherits guided_steps and target_matches_generated.',
        change='Use unchanged evaluate_caption_chair.py directly. No caption, parser, scoring, aggregation or bootstrap changes.',
        source_sha256={str(Path(__file__)): sha(Path(__file__))},
        caption_sha256={str(p): sha(p) for p in caption_paths(root)},
        mechanism_sha256=sha(root / 'mechanism_audit.json')))


def run(root):
    plan, _ = check(root)
    recovery = read(root / 'analysis_recovery_request.json')
    assert recovery['original_request_sha256'] == sha(root / 'analysis_request.json')
    assert not (root / 'analysis_provenance.json').exists()
    for name, wanted in {**recovery['source_sha256'], **recovery['caption_sha256']}.items():
        assert sha(Path(name)) == wanted, name
    assert sha(root / 'mechanism_audit.json') == recovery['mechanism_sha256']
    for shard in range(plan['shards']):
        assert read(root / f'preflight{shard}.json')['passed']
        assert read(root / f'worker{shard}.json')['passed']
    for mode in ['sentence', 'legacy']:
        with (root / f'evaluate_{mode}_recovery.log').open('x') as log:
            subprocess.run([sys.executable, 'scripts/evaluate_caption_chair.py',
                '--generation-root', str(root), '--cohort', plan['cohort'], '--limit', '100',
                '--variants', *plan['variants'], '--evaluator', mode],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root / 'run_status.json', dict(status='complete',
        phase='generation_and_evaluation_complete', finished=time.time()))
    with (root / 'independent_audit_recovery.log').open('x') as log:
        subprocess.run([sys.executable, 'scripts/audit_qwen3_caption_with_native_cache_v2.py', str(root)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(root / 'independent_summary.json')['passed']
    check(root)
    for name, wanted in recovery['caption_sha256'].items():
        assert sha(Path(name)) == wanted, name
    write(root / 'analysis_provenance.json', dict(passed=True, finished=time.time(),
        protocol_sha256=sha(root / 'protocol.json'),
        request_sha256=sha(root / 'analysis_request.json'),
        recovery_request_sha256=sha(root / 'analysis_recovery_request.json'),
        physical_review='pending', direct_advance_to500=False))
    write(root / 'analysis_recovery_complete.json', dict(passed=True, completed_at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root)
