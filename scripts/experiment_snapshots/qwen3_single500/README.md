# Qwen3 single-image SVAR / SAR-EMS experiment snapshot

This directory archives the exact orchestration and evaluation scripts used for the two-GPU run started on 2026-11-17. Each inference processes one image; each GPU handles 250 images and both variants. Generation uses `scripts/generation_time_evidence_drift_qwen3.py`.

- `run.py`: freezes the cohort, records the protocol and source hashes, starts both generation workers, and evaluates after both workers finish.
- `evaluate.py`: validates image IDs, single-image decoding and intervention counts, then computes CHAIRs, CHAIRi, object recall, object F1, paired bootstrap intervals, truncation diagnostics, and nested 100/200/300/400/500 subsets.
- `launch.py`: starts the driver as a detached process with a log and PID file.
- `cohort.json`: the exact frozen 500-image manifest, in generation order.
- `sampling_order.json`: the original sampling description and unsorted draw IDs used to define nested subsets. It contains no generated results.

These are experiment snapshots with the original server paths, not portable command-line launchers. The live copies are named `run.py`, `evaluate.py`, and `launch.py` in `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_single500/`. The driver expects `evaluate.py` there and reads the previous experiment's `cohort.json`; the evaluator also reads its `metrics.nested_seed43.json` for `selection` and `draw_ids`. If recreating the directory on another server, restore those manifests from this directory and update the repository, interpreter, dataset, and output paths. Model configuration and NLTK data must also be available.

Protocol: Qwen3-VL-8B-Instruct, BF16, greedy decoding, 512 new-token limit, prompt `Describe this image.`, variants `svar` and `svar_sar_ems_attention`, batch size 1. The sorted frozen cohort is shared with the earlier batched 500-image experiment. No new sampling or parameter tuning is performed by these scripts.

At the time of this code snapshot the experiment is still running. Its eventual metrics belong under the output directory; no final single-image result is claimed here.
