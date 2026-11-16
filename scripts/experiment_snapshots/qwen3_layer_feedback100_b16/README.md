# Qwen3: feedback to layers 19 and 20

Frozen exploratory comparison on 100 new COCO val2014 images. All layer indices are zero-based.

| Arm | SADI base | Budget sources → attention targets |
|---|---|---|
| `sadi` | Layers 5–18 | No additional guidance |
| `sadi_sar_ems_next_19_20` | Layers 5–18 | Q18/P18 → 19; Q19/P19 → 20 |
| `sadi_sar_ems_same_19_20` | Layers 5–18 | Probe Q19/P19 → recompute 19; probe Q20/P20 → recompute 20 |

Each source layer reads its tentative target after the FFN and applies the unchanged entity/positive-gain gate, strength 5 and temperature 0.03. The same-layer arm locally recomputes the last-query attention output and that layer's FFN. It must not append another KV entry or replay the full model. The next-layer arm computes fresh evidence and a fresh budget at each source layer.

`prepare.py` requires all source files before publishing any manifest. It excludes COCO4000, POPE and all registered `generation_time_*/cohort.json` manifests, including the latest new200 experiment, then draws 100 images with PCG64 seed 43. It atomically publishes cohort, sampling order, exclusion provenance and source hashes into a previously absent output directory. Existing experiments are never resampled.

The two GPU workers each receive 50 sorted, stride-assigned images. All three arms share ordered image batches. Maximum batch size is 16; visual-span/prompt-length buckets retain smaller tails, and any OOM splits all arms together. BF16, greedy decoding, `Describe this image.`, maximum 512 output tokens. No parameter selection using these outputs is part of this comparison.

`evaluate.py` validates 300 captions and matching batch plans, then reports CHAIRs, pooled CHAIRi, mean per-image object recall/F1 and paired 5,000-draw bootstrap intervals. It includes per-layer trigger statistics, token limits, actual batch sizes, OOM counts and timing. The complete 100 images are primary. Batch16 and common-uncapped subsets and single-image influence checks are secondary; the uncapped subset is conditioned on generated outputs.

Timing includes each method's own image/prompt prefill and controller cache construction, but excludes model loading, CPU preprocessing and preflight verification. Batch time is counted once per batch. Summed time across both workers is not experiment wall time.

Server output: `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_layer_feedback100_b16/`.

Entry points: `prepare.py` freezes input/protocol; `launch.py` starts `run.py`; the driver performs verification/generation and then invokes `evaluate.py`. Keep the frozen `source_sha256` list unchanged after launch; changes require explicit provenance rather than silent replacement.
