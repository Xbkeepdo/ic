"""Keep CHAIR calculations intact while reporting the new activity schema."""

from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import evaluate_caption_chair as cli
from scripts import evaluate_generation_time_evidence_drift_llava as shared


def main():
    original=shared.summarize
    def summarize(rows):
        activity=[r['sar_ems_statistics'] for r in rows
                  if 'full_events' in r.get('sar_ems_statistics',{})]
        if not activity:return original(rows)
        assert len(activity)==len(rows)
        cleaned=[{k:v for k,v in r.items() if k!='sar_ems_statistics'} for r in rows]
        result=original(cleaned)
        result['region_guidance_statistics']=dict(proposals=sum(len(s['full_events']) for s in activity),
            accepted=sum(s['accepted'] for s in activity),
            scope='New proposal/acceptance counts; legacy guided_steps is not applicable.')
        return result
    with patch.object(shared,'summarize',summarize):cli.main()


if __name__=='__main__':main()
