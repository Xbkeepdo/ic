# Qwen3 next-layer feedback: 200 new images, 256 output tokens

This frozen comparison evaluates SADI against SADI plus the unchanged next-layer SAR/EMS attention guidance. All layer indices are zero-based. SADI acts on layers 5–18; evidence and FFN preference at layer 18 guide attention at layer 19, then fresh evidence and FFN preference at layer 19 guide layer 20. The tentative target and budget are recomputed at each source layer. Strength 5, temperature 0.03, WordNet entity and positive-FFN-gain gates are unchanged.

The two arms are `sadi` and `sadi_sar_ems_next_19_20`. Qwen3-VL-8B-Instruct uses BF16, greedy decoding, `Describe this image.`, and **256 maximum new tokens**. Two GPUs each process 100 sorted, stride-assigned images. Maximum batch size is 16, with matched visual-span/prompt-length buckets, smaller tails and common OOM splitting. All 400 captions must use the same ordered image batches across arms.

`prepare.py` excludes COCO4000, POPE and all registered historical generation cohorts, explicitly requiring the previous new200 and layer-feedback100 manifests. It draws 200 unused COCO val2014 images using PCG64 seed43. Source dependencies, sampling and verification provenance are validated before four manifests are atomically published into an absent output root. Existing experiments are never resampled or overwritten.

The prior layer-feedback100 GPU verification is reused. The preparer requires its passing no-op controls and live next-layer intervention, and compares the controller, attention helpers, model wrapper, generation helpers and config hashes with that experiment's frozen protocol. It records both verification and protocol SHA256 hashes. This is source/config-level reuse, without a repeated GPU test or a claim that every model weight has been rehashed. New worker/run/launch code is frozen alongside this protocol.

`evaluate.py` validates 400 raw captions, paired batch plans and the 256-token limit. Primary results use all 200 images: CHAIRs, pooled CHAIRi, mean per-image object recall/F1, paired 5,000-draw bootstrap intervals, hallucinated/total object mentions, triggered layers, token limits, actual batch sizes, OOM counts and timing. Matched actual-B16 and common-uncapped subsets are secondary; the latter is output-conditioned.

CHAIR's existing per-caption cache also supplies canonical object counts without repeat scoring. The report quantifies repeated object-category mentions and identifies denominator increases. Removing the top 1/3/5 positive denominator-growth images is a post-hoc sensitivity check, never a replacement for the full result. Repeated mentions can be legitimate and are not automatically generation loops.

Timing includes each arm's prefill and controller caches, excluding model loading and CPU image preprocessing. Each batch's wall time is counted once; totals across GPU workers are not experiment wall time.

Server output: `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_next200_t256_b16/`.

Entry points: `prepare.py`, then `launch.py` / `run.py` / `worker.py`; completion invokes `evaluate.py`. The primary metrics, per-image scores and `validation.next200_t256_b16.json` retain the complete audit trail.
