"""Verify explicit whole-group reuse, then run the frozen CLVA caption audit."""

import argparse
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.qwen3_caption_groups_with_cache import source_groups
from scripts import analyze_qwen3_clva_signal as core


def run(root):
    plan,request=core.check(root);receipts=[]
    for entry in plan['generation_reuse_groups']:
        source=Path(entry['source']);assert sha(source)==entry['source_sha256']
        expected,=[g for g in source_groups(source) if [r['image_id'] for r in g]==entry['ids']]
        actual=source_groups(root/f'coco100/{entry["mode"]}.shard{entry["shard"]}.jsonl')[entry['group']]
        assert actual==expected and actual[0]['visual_token_count']==entry['visual_tokens']
        receipts.append(entry)
    assert len(receipts)==8 and sum(len(r['ids']) for r in receipts)==40
    write(root/'generation_reuse_audit.json',dict(passed=True,groups=len(receipts),image_arm_pairs=40,
        newly_generated_image_arm_pairs=360,receipts=receipts,
        timing='Caption times preserved from original matched calls; activity.seconds includes reused source calls. Current wall time and new calls are in worker reports.'))
    core.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();core.prepare(args.root) if args.prepare else run(args.root)
