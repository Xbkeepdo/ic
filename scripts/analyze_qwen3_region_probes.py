"""Aggregate all fixed-prefix region trials and freeze a caption-pilot recipe."""

import argparse
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read, write, sha


def run(root):
    assert not root.exists()
    names = ['qwen3_region_memory_probe_20261120', 'qwen3_region_consistency_probe_20261120',
             'qwen3_region_counterevidence_probe_20261120']
    inputs = [Path(__file__)]; datasets = {}; activities = {}; all_keys = None
    for name in names:
        folder = REPO / 'outputs' / name
        plan = read(folder / 'protocol.json'); assert read(folder / 'pipeline_complete.json')['passed']
        assert plan['full_sadi_layers'] == list(range(5, 19))
        assert plan['intervention_layers'] == list(range(20, 36))
        rows = []; workers = []
        for shard in [0, 1]:
            worker = read(folder / f'worker{shard}.json'); assert worker['passed'] and worker['all_trials_rolled_back']
            workers.append(worker); rows += read(folder / f'results{shard}.json')
        keys = {(r['image_id'], r['step'], r['token_id']) for r in rows}; assert len(keys) == len(rows) == 6
        if all_keys is None: all_keys = keys
        else: assert keys == all_keys
        for filename, expected in plan['source_sha256'].items(): assert sha(REPO / filename) == expected, filename
        inputs += [folder / leaf for leaf in ['protocol.json', 'pipeline_complete.json', 'results0.json',
            'results1.json', 'worker0.json', 'worker1.json', 'launch.json', 'workers_launch.json']]
        trials = [dict(case={k: r[k] for k in ['image_id', 'step', 'token_id', 'canonical', 'reference_gate_positive']}, **t)
                  for r in rows for t in r['trials']]
        strategies = {}
        for t in trials:
            key = '/'.join(str(t[k]) for k in ['component', 'location'] if k in t) or '/'.join([t['locator'], t['policy']])
            item = strategies.setdefault(key, dict(trials=0, changed=0, accept_candidates=0, gate_positive_accept_candidates=[]))
            item['trials'] += 1; item['changed'] += int(t['token_changed']); item['accept_candidates'] += int(t['would_accept'])
            if t['would_accept'] and t['case']['reference_gate_positive']:
                item['gate_positive_accept_candidates'].append(dict(**t['case'], next_piece=t['top5']['text'][0], delta=t['original_logp_delta']))
        datasets[name] = dict(cases=6, trials=len(trials), strategies=strategies)
        activities[name] = dict(workers=workers,
            original_encoder_calls=sum(w['original_encoder_calls'] for w in workers),
            crop_encoder_calls=sum(w['crop_encoder_calls'] for w in workers))
    assert sum(d['trials'] for d in datasets.values()) == 96
    positive = datasets[names[-1]]['strategies']['semantic_peak/verified_boost']['gate_positive_accept_candidates']
    assert len(positive) == 1 and (positive[0]['image_id'], positive[0]['step']) == (353411, 97)
    reference_root = REPO / 'outputs/qwen3_mass_matched_visual_pilot_20261119'
    reserved = REPO / 'outputs/qwen3_mass_matched_visual_reserved500_20261119'
    assert not (reserved / 'consumed.json').exists()
    inputs += [reference_root / leaf for leaf in ['protocol.json', 'cohort.json', 'final_provenance.json']]
    inputs += [reserved / leaf for leaf in ['cohort.json', 'reservation.json', 'batch_plan.json']]
    hashes = {str(p): sha(p) for p in inputs}; root.mkdir()
    write(root / 'protocol.json', dict(fixed_at=time.time(), source_sha256=hashes,
        scope='Complete96 forced mechanism trials on six fixed slots/two reused development images, including every declared strategy; no caption metrics.'))
    write(root / 'analysis.json', dict(passed=True, protocol_sha256=sha(root / 'protocol.json'),
        datasets=datasets, activity=activities, primary_mechanism_candidate=positive,
        primary_is_not_unique='Unverified boost also changes the same false fork; current cases do not establish the necessity of crop verification.',
        ungated_change='Verified boosts also change243882 person to close, but its cached gate is negative. Do not count this as an online correction or hide it.',
        limitation='nap is a token piece. No complete new description or CHAIR/F1 result exists. Fixed prefixes do not measure downstream rollouts.',
        no_new_captions=True, no_independent_efficacy_claim=True, reserved500_unconsumed=True))
    recipe = dict(fixed_at=time.time(), before_caption_pilot_results=True,
        motivation='Verified semantic-peak region is net counterevidence for the false target: attenuation raises its logp, enhancement lowers it.',
        model='Qwen3-VL-8B-Instruct', cohort='Same old development100 and exact22 ordered batches, maxB16, prompt Describe this image., BF16/eager/greedy, cap512.',
        controls='Reuse frozen SADI/SVAR captions; generate verified semantic-peak boost, unverified semantic-peak boost, and verified spatial-shift boost.',
        base_layers=list(range(5, 19)), source_layer=19, target_layers=list(range(20, 36)),
        gate='Original108 risk/I/El seed43, old source-calibrated real threshold .418672114610672;309 lexical, token unseen, class absent from actual generated prefix.',
        locator='argmax over current-class alias probability sum from original layer19 visual hpre, final norm and full-vocabulary softmax. Fixed3x3 merged-patch window.',
        signal='D=Hellinger2(fullSAR_Q19,FFN_preference19); fine ROI448, same original B/P cached-vision prompt, whole-M64-chunk head; r=relu(p_coarse-p_fine)/(p_coarse+p_fine).',
        action='Add5*D*r within ROI visual logits at20--35, correct visual logsumexp to preserve total visual attention mass. Native values retained. Unverified uses r1, shifted applies the same r budget half a grid away.',
        acceptance='Actual current final argmax changes and original-target logp decreases; otherwise reject. Test proposals against pristine native cache for all batch rows before committing any row.',
        commit='Prototype replays restore everything. Production must record proposed current KV20--35 during replay and commit only accepted rows after all proposals. Never append an extra cache position; past visual/text slots stay native.',
        reentrancy='Suspend own finish hook and all feature capture during cached-vision prompt reconstruction and tail replay; keep native anchor19/tail kwargs intact.',
        cache='Original vision/DeepStack once per batch; crop features and fine image-prefix class probabilities once per ROI. Do not reuse across different B/P/hpre/value identity. Fine inspection does not depend on generated text beyond choosing ROI/class.',
        tests='Disabled/zero/reject-all must match native SADI logits/tokens/all36 KV exactly; active known prefix test must exercise a kept fork->nap proposal and real book/sandwich preservation. Cached source probe comparisons only before first accepted deviation.',
        evaluation='Both sentence and legacy, full100 Cs/Ci/object recall/F1 and paired bootstrap. Inspect all fixes/new errors and nonzero recall losses; unchanged wordpieces are not independent improvements.',
        followup='Only useful genuine caption improvement goes to the unchanged reserved500; all candidates/SADI/SVAR fresh on identical49 explicit groups. No best-seed or threshold tuning.',
        implementation_status='Recipe only. Live controller, generation runner and commit-path checks are not implemented yet.')
    write(root / 'caption_pilot_recipe.json', recipe)
    for filename, expected in hashes.items(): assert sha(Path(filename)) == expected, filename
    print(dict(passed=True, forced_trials=96, candidate=positive,
               caption_pilot_not_implemented=True, recipe_sha256=sha(root / 'caption_pilot_recipe.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
