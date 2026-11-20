"""Compare conditional raw/SADI choices on the same archived source prefixes.

The first two source batches were fixed before either prediction audit.
This is a paired source-distribution diagnosis, not a mitigation experiment.
"""

import argparse
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read, write, sha


def run(root):
    assert not root.exists(), 'Keep existing audit products unchanged.'
    sadi = REPO / 'outputs/qwen3_source_prediction_consistency_20261120'
    raw = REPO / 'outputs/qwen3_source_raw_prediction_control_20261120'
    paths = [Path(__file__)]
    records = {}
    for name, source in [('raw', raw), ('sadi', sadi)]:
        assert read(source / 'pipeline_complete.json')['passed']
        assert read(source / 'analysis.json')['passed']
        plan = read(source / 'protocol.json')
        assert sha(source / 'protocol.json') == read(source / 'analysis.json')['protocol_sha256']
        for filename, expected in plan['source_sha256'].items():
            assert sha(Path(filename)) == expected, filename
        values = []
        for shard in [0, 1]:
            worker = read(source / f'worker{shard}.json')
            assert worker['passed'] and worker['encoder_calls'] == 1
            if name == 'sadi':
                assert worker['anchor_check']['passed']
                assert worker['anchor_check']['maximum_absolute_error'] == 0
            else:
                assert worker['anchor_check']['checked'] is False
            values.extend(read(source / f'results{shard}.json'))
        assert len(values) == len({value['row'] for value in values}) == 101
        records[name] = {value['row']: value for value in values}
        paths.extend(source / leaf for leaf in [
            'protocol.json', 'pipeline_complete.json', 'analysis.json',
            'worker0.json', 'worker1.json', 'results0.json', 'results1.json'])
    assert read(raw / 'protocol.json')['groups'] == read(sadi / 'protocol.json')['groups']
    assert records['raw'].keys() == records['sadi'].keys()
    pairs = []
    identity = ['row', 'image_id', 'step', 'label_real', 'source_train',
                'source_token', 'source_runtime_eligible', 'source_class']
    for row in sorted(records['raw']):
        left, right = records['raw'][row], records['sadi'][row]
        assert all(left[key] == right[key] for key in identity), row
        pairs.append(dict(**{key: left[key] for key in identity},
            raw_native_top1=left['native_top1'], sadi_native_top1=right['native_top1'],
            raw_text=left['native_text'], sadi_text=right['native_text'],
            raw_source_match=left['exact_token_match'], sadi_source_match=right['exact_token_match'],
            source_text=left['source_text'], context=left['context'],
            raw_gap=left['native_minus_source_logit'], sadi_gap=right['native_minus_source_logit'],
            raw_source_rank=left['source_rank'], sadi_source_rank=right['source_rank']))

    def stats(values):
        return dict(targets=len(values),
            raw_exact=sum(value['raw_source_match'] for value in values),
            sadi_exact=sum(value['sadi_source_match'] for value in values),
            both_exact=sum(value['raw_source_match'] and value['sadi_source_match'] for value in values),
            raw_exact_sadi_changed=sum(value['raw_source_match'] and not value['sadi_source_match'] for value in values),
            raw_changed_sadi_exact=sum(not value['raw_source_match'] and value['sadi_source_match'] for value in values),
            both_differ_from_source=sum(not value['raw_source_match'] and not value['sadi_source_match'] for value in values),
            raw_sadi_top1_differ=sum(value['raw_native_top1'] != value['sadi_native_top1'] for value in values))

    hashes = {str(path): sha(path) for path in paths}
    root.mkdir()
    write(root / 'protocol.json', dict(fixed_at=time.time(), source_sha256=hashes,
        comparison='Paired raw/SADI conditional next-token choices on fixed first two B16 source groups.',
        selection='No further images or target selection; all101 source targets included.',
        no_training=True, no_threshold_changes=True, no_new_captions=True))
    output = dict(passed=True, protocol_sha256=sha(root / 'protocol.json'),
        all=stats(pairs), real=stats([value for value in pairs if value['label_real'] == 1]),
        hall=stats([value for value in pairs if value['label_real'] == 0]),
        runtime_source_real=stats([value for value in pairs if value['label_real'] == 1 and value['source_runtime_eligible']]),
        runtime_source_hall=stats([value for value in pairs if value['label_real'] == 0 and value['source_runtime_eligible']]),
        interpretation='SADI changes conditional preferences under archived raw prefixes. This can contribute to source-policy mismatch, but does not establish a causal explanation of probe failure or a mitigation improvement.',
        raw_baseline_already_differs='Original archived token is not always the current unmodified-model argmax either.',
        numeric_feature_anchor='SADI has one exactly reproduced108-feature anchor per batch; the raw control recomputes no features and has no such numeric comparator.',
        no_mitigation_or_population_claim=True)
    for filename, expected in hashes.items():
        assert sha(Path(filename)) == expected, filename
    write(root / 'pairs.json', pairs)
    write(root / 'analysis.json', output)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
