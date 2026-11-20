"""Read whole caption batches and validate explicitly certified cache rows."""

import json
from scripts.audit_qwen3_caption_with_native_cache_v2 import validate_cached_group


def source_groups(path):
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    offset=0;groups=[]
    while offset<len(rows):
        first=rows[offset];size=first['batch_size'];group=rows[offset:offset+size]
        assert 1<=size<=32 and len(group)==size
        assert all(r['batch_size']==size and r['batch_wall_seconds']==first['batch_wall_seconds']
                   and r['visual_token_count']==first['visual_token_count'] for r in group)
        validate_cached_group(group)
        groups.append(group);offset+=size
    return groups
