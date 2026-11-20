"""Keep the frozen attention action, but suppress repeats of object classes."""

import argparse
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-first-class-pilot')
SOURCE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-early-recycle-pilot')
TRACE = shared.OUT / 'qwen3_sadi_pre_attention_i_el_trace'
PREVIOUS = 'sadi_probe_lexical_recycle_19'
NAME = 'sadi_probe_first_class_recycle_19'


def prepare():
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', PREVIOUS), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for filename in ('features/coco_object_history.py', 'features/qwen3_probe_first_class.py',
                     'scripts/run_qwen3_probe_first_class_pilot.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    previous = shared.read(SOURCE / 'protocol.json')
    checked = SOURCE / 'early_preflight.json'
    plan['source_sha256'][str(checked)] = shared.sha(checked)
    plan.update(stage='head-gradient-probe-first-class-pilot', source_trace=str(TRACE), probe=previous['probe'],
                actuation_preflight=str(checked),
                selection='One prospective eligibility change: block classes already mentioned in the generated prefix.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gate='First lexical target token AND its class absent from generated-prefix alias history AND probe p_real<.5.',
        prefix_history='Only model-emitted text before the current target; longest lexical alias match and noun lemmatization; no image GT.',
        limitations='History word forms still lack contextual noun/adjective disambiguation; current target has no boundary lookahead.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    import models
    from features import qwen3_sar_ems_head_gradient as head
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
    build = models.build_model
    plan = shared.read(OUT / 'protocol.json')
    trace = shared.read(TRACE / 'protocol.json')
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    probe, tokenizer, index = None, None, 0
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    assert parser.classes('a baby elephant next to a passenger train') == {'elephant', 'train'}
    assert parser.classes('a woman and her skateboard') == {'person', 'skateboard'}
    assert parser.classes('Cars, a sedan, and several cars') == {'car'}

    def checked_build(*args, **kwargs):
        nonlocal probe, tokenizer
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        tokenizer = wrapper.tokenizer
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
        return Qwen3BatchProbeFirstClass(model, span, allowed, probe=probe,
            semantic_tokens=cache['token_ids'].tolist(), cached_semantics=torch.from_numpy(cache['E_l']),
            guided=True, tokenizer=tokenizer, history_parser=parser)

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
