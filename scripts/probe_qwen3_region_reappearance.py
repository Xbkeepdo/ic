"""Read-only attention trials when a suppressed object reappears in live decoding."""

from contextlib import ExitStack
from pathlib import Path
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_region_guidance import setup,batch,digest,MECHANISM
from scripts.run_qwen3_signed_feedback import read,write,sha


def main(root):
    import torch
    from features.qwen3_region_guidance import Qwen3RegionGuidance
    from features import qwen3_region_consistency as consistency
    from features.qwen3_region_consistency import category_probabilities
    from features.qwen3_region_memory import rectangle
    from features.qwen3_bounded_visual_tilt import evidence_preference
    from scripts.probe_qwen3_region_counterevidence import signed_attention
    source=REPO/'outputs/qwen3_region_guidance100_20261120'
    assert read(source/'worker0.json')['passed'], 'Only the terminal shard is used.'
    import json
    reference={r['image_id']:r for r in map(json.loads,(source/'coco100/verified.shard0.jsonl').read_bytes().splitlines())}
    assert not root.exists();root.mkdir();(root/'crops').mkdir()
    hashes=dict(read(source/'protocol.json')['source_sha256'])
    hashes.update({str(p):sha(p) for p in [Path(__file__),source/'protocol.json',source/'worker0.json',
        source/'coco100/verified.shard0.jsonl',REPO/'features/qwen3_native_risk_i_el.py']})
    write(root/'protocol.json',dict(fixed_at=time.time(),source=str(source),source_sha256=hashes,
        diagnostic='After accepted fork-to-nap, test the first later native fork on that actual live trajectory.',
        strengths=[5,10],scope='Development mechanism only; all extra trials rolled back; no new mitigation score.'))
    for n,v in hashes.items():assert sha(REPO/n)==v,n
    _,wrapper,probe,allowed,parser=setup(source)
    ids=read(MECHANISM/'protocol.json')['groups'][1]['ids'];span,inputs,photos=batch(wrapper,ids)
    controller=Qwen3RegionGuidance(wrapper,span,allowed,probe=probe,history_parser=parser,
        image_ids=ids,images=photos,output_dir=root/'crops')
    records=[]
    def inspect(_module,_args,output):
        if controller.suspended or records or controller.step<=97:return
        native=output.last_hidden_state[:,-1:].clone();logits=controller.head(native).float()[:,0]
        token=int(logits[0].argmax())
        if token!=22435:return
        assert any(e['accepted'] and e['token']==22435 for e in controller.events.get(0,[]))
        reader=controller.reader;features=reader.features([0],[token]);p_real=float(probe(features)[0])
        cached=reader._static(19,0)
        q,p=evidence_preference(reader.last_write_norms[0],cached['log_s'][:,reader.columns[token]],
            reader.visual[19][0],reader.current[19]['update'][0])
        drift=float((1-(q*p).sqrt().sum()).clamp(0,1))
        aliases=controller.aliases[controller.token_classes[token]]
        heat=category_probabilities(reader,0,aliases);rect=rectangle(heat,*controller.memory.grid(0),'evidence')
        cache=controller.tail_kwargs['past_key_values']
        with reader.paused():crop=controller.memory.encode_crop(photos[ids[0]],ids[0],0,rect,root/'crops/reappearance.jpg')
        support=controller.memory.inspect(0,cache,reader,crop,rect,aliases)
        def full():return digest([t for layer in cache.layers for t in [layer.keys,layer.values]])
        before_digest=full();before=float(logits[0].log_softmax(0)[token]);trials=[]
        zero,_=controller.memory.attenuate(0,cache,reader,rect,0.,native,zero=True)
        assert torch.equal(zero,native)
        for factor in [5,10]:
            strength=factor*drift*support['reliability_deficit']
            if strength:
                with patch.object(consistency,'mask_attention',signed_attention):
                    hidden,error=controller.memory.attenuate(0,cache,reader,rect,-strength,native)
            else:hidden,error=native,0.
            proposed=controller.head(hidden).float()[0,0];top=proposed.topk(5).indices.tolist()
            trials.append(dict(factor=factor,strength=strength,top5_ids=top,
                top5_text=[wrapper.tokenizer.decode([t]) for t in top],
                original_logp_change=float(proposed.log_softmax(0)[token])-before,
                visual_mass_relative_error=error))
        assert full()==before_digest
        records.append(dict(step=controller.step,image_id=ids[0],token=token,p_real=p_real,
            previous_signal_retained_by_production=False,drift=drift,support=support,trials=trials,
            zero_and_full36_rollback_exact=True))
    with torch.inference_mode(),ExitStack() as stack:
        stack.enter_context(controller.capture());stack.callback(controller.decoder.register_forward_hook(inspect).remove)
        # capture is already installed, so use the standard generation kwargs directly.
        from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
        from scripts.generation_time_evidence_drift_qwen3 import MODEL
        from scripts.generation_time_evidence_drift_qwen3_batched import trim_eos
        result=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=128))
        eos=set(wrapper.model.generation_config.eos_token_id)
        tokens=[trim_eos(t,eos) for t in result[:,inputs['input_ids'].shape[1]:].tolist()]
    assert tokens==[reference[i]['response_token_ids'][:128] for i in ids] and len(records)==1
    for n,v in hashes.items():assert sha(REPO/n)==v,n
    write(root/'results.json',dict(passed=True,records=records,actual_original_live_trajectory_exact=True))
    controller.close();print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
