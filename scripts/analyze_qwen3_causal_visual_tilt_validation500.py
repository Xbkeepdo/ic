"""Paired frozen500 analysis with causal metadata and actual emitted-ID EOS counts."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import analyze_qwen3_bounded_visual_tilt_validation500 as core
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.qwen3_caption_groups_with_cache import source_groups


def analyze(root):
    original = core.brief_event
    def brief(event, tokens, tokenizer):
        result = original(event, tokens, tokenizer)
        if event.get('causal_visual_tilt'):
            result.update(causal_visual_tilt=True, tilt_mode=event['tilt_mode'],
                refresh_source=event['refresh_source'], emitted_is_eos=event['emitted_is_eos'])
        return result
    with patch.object(core, 'brief_event', brief):
        core.analyze(root)
    activity = read(root/'activity.json')
    for name in read(root/'protocol.json')['candidates']:
        rows = [row for shard in (0, 1) for group in source_groups(root/f'coco500/{name}.shard{shard}.jsonl') for row in group]
        actual = sum(row['sar_ems_statistics']['actual_eos_accepts'] for row in rows)
        activity[name]['legacy_terminal_eos_accepts'] = activity[name]['terminal_eos_accepts']
        activity[name]['terminal_eos_accepts'] = actual
        activity[name]['eos_count_definition'] = 'Accepted proposals whose actual emitted ID is an EOS ID, including EOS retained in response_token_ids.'
    write(root/'activity.json', activity)
    provenance = read(root/'analysis_provenance.json')
    provenance.update(adapter_source_sha256=sha(Path(__file__)), actual_eos_from_emitted_ids=True,
        activity_sha256=sha(root/'activity.json'), raw_captions_and_evaluation_unchanged=True)
    write(root/'analysis_provenance.json', provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
