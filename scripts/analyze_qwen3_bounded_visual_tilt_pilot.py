"""Base-aware paired analysis with explicit bounded-tilt method labels."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import analyze_qwen3_scalar_visual_pilot as core


def analyze(root):
    original=core.brief_event
    def brief(event,tokens,tokenizer):
        result=original(event,tokens,tokenizer)
        if 'bounded_visual_tilt' in event:
            result.update(bounded_visual_tilt=event['bounded_visual_tilt'],tilt_mode=event['tilt_mode'])
        return result
    with patch.object(core,'brief_event',brief):core.analyze(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
