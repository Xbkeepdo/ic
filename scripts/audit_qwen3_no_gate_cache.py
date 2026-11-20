"""Find complete native batches with no positive SAM gate, without GPU work."""

import argparse
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha,source_groups,group_keys


def audit(source,target=None):
    plan=read(source/'protocol.json')
    assert read(source/'run_status.json')['status']=='complete'
    names=['sadi_sam_acg_fixed_31','sadi_sam_acg_guided_31']
    inputs={str(source/'protocol.json'):sha(source/'protocol.json'),str(Path(__file__)):sha(Path(__file__))}
    eligible=[];possible_seconds=0.;possible_images=0
    for shard in (0,1):
        base=source_groups(source/f'coco100/sadi.shard{shard}.jsonl')
        methods={n:source_groups(source/f'coco100/{n}.shard{shard}.jsonl') for n in names}
        for n,gs in methods.items():assert group_keys(gs)==group_keys(base)
        for n in ['sadi',*names]:
            p=source/f'coco100/{n}.shard{shard}.jsonl';inputs[str(p)]=sha(p)
        for index,group in enumerate(base):
            fixed=methods[names[0]][index]
            if any(e['positive'] for r in fixed for e in r['sar_ems_statistics']['gate_events']):continue
            canonical=[]
            for position,native in enumerate(group):
                left=methods[names[0]][index][position];right=methods[names[1]][index][position]
                for row in (left,right):
                    stat=row['sar_ems_statistics']
                    assert not stat['recycle_events'] and not any(e['positive'] for e in stat['gate_events'])
                    assert row['response_token_ids']==native['response_token_ids']
                    assert row['generated_text']==native['generated_text']
                assert left['sar_ems_statistics']['gate_events']==right['sar_ems_statistics']['gate_events']
                canonical.append(dict(image_id=native['image_id'],gate_events=left['sar_ems_statistics']['gate_events']))
            saved=sum(methods[n][index][0]['batch_wall_seconds'] for n in names)
            eligible.append(dict(shard=shard,group=index,image_ids=[r['image_id'] for r in group],
                observed_two_arm_gpu_seconds=saved,gate_records=canonical))
            possible_seconds+=saved;possible_images+=len(group)
    report=dict(passed=True,created_at=time.time(),source=str(source),source_sha256=inputs,
        eligibility='Whole unchanged ordered batch, both native trajectories, zero positive gate at every eligible position.',
        conditions='Same frozen native108 operator/threshold/history/scope/weights/batching and no action before a positive gate. New operator needs parity verification. Preserve cache provenance and report reuse as reuse.',
        whole_batches=len(eligible),images=possible_images,observed_two_arm_gpu_seconds=possible_seconds,
        eligible_groups=eligible,actual_skips_performed=0)
    if target is not None:
        assert read(target/'run_status.json')['status']=='complete'
        target_plan=read(target/'protocol.json')
        assert target_plan['real_threshold']==plan['real_threshold']
        comparisons=0
        for entry in eligible:
            shard,index=entry['shard'],entry['group']
            for name in target_plan['candidates']:
                path=target/f'coco100/{name}.shard{shard}.jsonl';inputs[str(path)]=sha(path)
                group=source_groups(path)[index]
                assert [r['image_id'] for r in group]==entry['image_ids']
                for row,expected in zip(group,entry['gate_records'],strict=True):
                    assert row['sar_ems_statistics']['gate_events']==expected['gate_events']
                    assert not row['sar_ems_statistics']['recycle_events']
                    comparisons+=1
        report.update(joint17_agreement_verified=True,target=str(target),target_gate_records_checked=comparisons)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path)
    p.add_argument('--target',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    report=audit(args.source,args.target);write(args.output,report)
    print({k:report[k] for k in ['passed','whole_batches','images','observed_two_arm_gpu_seconds','actual_skips_performed']})
