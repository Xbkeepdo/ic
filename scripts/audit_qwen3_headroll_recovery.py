"""Run frozen head-control checks with explicit cache parsing and prefix hashes."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import audit_qwen3_headroll_pilot as original
from scripts.qwen3_caption_groups_with_cache import source_groups
from scripts.audit_qwen3_caption_with_native_cache_v2 import validate_cached_group
from scripts.run_qwen3_signed_feedback import read,write,sha


def audit(root):
    plan=read(root/'protocol.json')
    for shard,info in plan['recovery_prefixes'].items():
        source=Path(info['path']);assert sha(source)==info['sha256']
        current=root/f'coco100/headroll.shard{shard}.jsonl'
        assert current.read_bytes()[:info['bytes']]==source.read_bytes()
    with patch.object(original,'source_groups',source_groups),patch.object(original,'validate_cached_group',validate_cached_group):original.audit(root)
    value=read(root/'mechanism_audit.json');value['all96_original_records_preserved_bytewise']=True
    write(root/'mechanism_audit.json',value)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);audit(p.parse_args().root)
