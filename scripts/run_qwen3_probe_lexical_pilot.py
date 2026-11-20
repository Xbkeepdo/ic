"""Change only the trigger vocabulary of the frozen attention-recycling pilot."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-lexical-pilot')
SOURCE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-recycle-pilot-v2')
TRACE = shared.OUT / 'qwen3_sadi_pre_attention_i_el_trace'
PREVIOUS = 'sadi_probe_spatial_recycle_31'
NAME = 'sadi_probe_lexical_recycle_31'


def prepare():
    import os
    os.environ['NLTK_DATA'] = str(shared.OUT / 'generation_time_evidence_drift_llava/nltk_data')
    from transformers import AutoTokenizer
    from features.qwen3_object_token_lexicon import object_token_lexicon
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', PREVIOUS), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    training = shared.read(OUT / 'training_vocabulary.json')['token_ids']
    lexicon = object_token_lexicon(tokenizer, training, REPO / 'coco-labeling/coco_chair.py')
    assert len(lexicon['token_ids']) == 309
    shared.write(OUT / 'object_lexicon.json', lexicon)
    for filename in ('features/qwen3_object_token_lexicon.py', 'scripts/run_qwen3_probe_lexical_pilot.py',
                     'outputs/' + OUT.name + '/object_lexicon.json'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    previous = shared.read(SOURCE / 'protocol.json')
    checked = SOURCE / 'recycle_preflight.json'
    plan['source_sha256'][str(checked)] = shared.sha(checked)
    plan.update(stage='head-gradient-probe-lexical-pilot', source_trace=str(TRACE), probe=previous['probe'],
                actuation_preflight=str(checked),
                selection='One prospective change: single-token object word-form lexicon; same probe and actuation.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gate='First lexical object word-form token, original frozen I/E_l probe p_real<0.5.',
        trigger_vocabulary_tokens=309, original_training_tokens=496,
        vocabulary='Standard CHAIR single-word aliases plus noun inflections, intersect old training IDs.',
        limitations=lexicon['limitations'])
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    import models
    from features import qwen3_sar_ems_head_gradient as head
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_attention_recycle import Qwen3BatchProbeAttentionRecycle
    build = models.build_model
    plan = shared.read(OUT / 'protocol.json')
    trace = shared.read(TRACE / 'protocol.json')
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    probe, index = None, 0

    def checked_build(*args, **kwargs):
        nonlocal probe
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
        return wrapper

    def factory(model, span, nouns, *, final_gradient):
        nonlocal index
        assert final_gradient
        with np.load(TRACE / f'semantics/shard{gpu}.batch{index}.npz', allow_pickle=False) as saved:
            cache = {key: saved[key] for key in saved.files}
        assert cache['image_ids'].tolist() == trace['groups'][gpu][index]['ids']
        assert cache['span'].tolist() == list(span)
        index += 1
        return Qwen3BatchProbeAttentionRecycle(model, span, allowed, probe=probe,
            semantic_tokens=cache['token_ids'].tolist(), cached_semantics=torch.from_numpy(cache['E_l']), guided=True)

    with patch.object(shared, 'HEAD_FINAL', NAME), patch.object(head, 'Qwen3BatchHeadGradient', factory), \
            patch.object(models, 'build_model', checked_build):
        shared.worker(OUT, gpu)
    assert index == len(trace['groups'][gpu])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', prepare):
            driver.run()
    else:
        worker(args.worker)
