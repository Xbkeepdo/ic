"""Compare single-layer and joint post-SADI ACG with unchanged cached controls."""

import argparse
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_acg_signal_pilot as previous
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_signed_feedback as shared

OUT=shared.OUT/'qwen3_block_acg_pilot_20261118'
SOURCE=previous.OUT
CPU=shared.OUT/'qwen3_block_acg_cpu/verification.json'
PREFLIGHT=shared.OUT/'qwen3_block_acg_preflight/verification.json'
MODES={'sadi_sam_acg_fixed_19_35':False,'sadi_sam_acg_guided_19_35':True}
FILES=['features/qwen3_block_acg_signal.py','scripts/verify_qwen3_block_acg.py',
       'scripts/verify_qwen3_block_acg_gpu.py','scripts/run_qwen3_block_acg_pilot.py',
       'scripts/audit_qwen3_block_acg_pilot.py','scripts/analyze_qwen3_block_acg_pilot.py']


def prepare():
    assert not OUT.exists(), 'Inspect an existing job before any restart'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    assert shared.read(CPU)['passed'] and shared.read(PREFLIGHT)['passed']
    for n,v in shared.read(PREFLIGHT)['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    for p in [CPU,PREFLIGHT,SOURCE/'protocol.json',SOURCE/'human_review.json',*(REPO/n for n in FILES)]:
        hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco100').mkdir()
    for n in ('cohort.json','object_lexicon.json'):shutil.copyfile(SOURCE/n,OUT/n)
    controls=old['variants']
    for shard in (0,1):
        for name in controls:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl';hashes[str(p)]=shared.sha(p)
            shutil.copyfile(p,OUT/'coco100'/p.name)
    variants=[*controls,*MODES]
    assert len('chair_per_image.first100.'+'-'.join(sorted(variants))+'.jsonl')<=255
    plan=dict(stage='block-acg-pilot',images=100,shards=2,cohort='coco100',max_batch=16,max_new_tokens=512,
        variants=variants,candidates=list(MODES),source=str(SOURCE),copied_controls=controls,cached_controls=['sadi'],
        source_sha256=hashes,fixed_at=time.time(),probe=old['probe'],real_threshold=old['real_threshold'],
        reference=old['reference'],
        recipe=dict(base_layers=[5,18],joint_layers=list(range(19,36)),first_replay_layer=19,target_read_layer=35,
            gate=old['recipe']['gate'],gamma=1.4,
            fixed='Same orthogonal ACG context direction and gamma1.4 at all17 layers/32heads.',
            guided=old['recipe']['guided'],head_scale=old['recipe']['head_scale'],
            derivative='One native-state19FFN+20--35 graph returns attention-residual gradients at all17 layers in one autograd call.',
            proposal='Compute every direction/budget at the same native point, jointly add context deltas during one causal replay. Later budgets are not recomputed or claimed exact after earlier edits.',
            acceptance='Original target actual final log probability strictly decreases; otherwise restore hidden/current20--35 KV.',
            replay='Layer19 cached QKV; layers20--35 recompute the current query and current QKV with unchanged prefix; fullB; unselected rows retained.',
            features='Original108 SAM-V2 feature values and exact36 small OT calls per eligible target unchanged; intercept already computed write norms, no extra projections for SAR.',
            static_cache='Same309 normalized probabilities. Exact hpre/GQA validation copies in host RAM, not disk; shared per ordered group; no retained FP32 WO matrices at all17 layers.',
            diagnostics='Original top2 value gap, original target logprob, proposed argmax and actual immediate change; per-layer scales/effects and actual acceptance.',
            costs='Fixed0backward, guided1jointbackward per selected step. ACG masked contexts reuse native QKV. Normal causal20--35 replay still recomputes current QKV.',
            memory='Use previous explicit per-generation GC; no duplicate expanded visual-prefix caches; no hidden/attention/fullV saved.'),
        development_screen=old['development_screen'],
        followup='If numerical and image-confirmed support: head-scale roll control and unseen development/independent500 with matched SADI/SVAR. No parameter tuning on the new sets.',
        scope='Old100 reused to isolate layer coverage, not independent evidence. Neither method is original all-layer/ungated ACG.')
    shared.write(OUT/'protocol.json',plan)
    return plan


def worker(gpu):
    from features import qwen3_acg_signal_feedback as single
    from features.qwen3_block_acg_signal import Qwen3BatchBlockACG
    with patch.object(previous,'OUT',OUT),patch.object(previous,'MODES',MODES),\
            patch.object(single,'Qwen3BatchACGSignal',Qwen3BatchBlockACG):
        previous.worker(gpu)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
