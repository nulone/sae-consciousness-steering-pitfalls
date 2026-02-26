# Data Directory

## Canonical JSON files (used by verify_numbers.py and cited in writeup)

**all_experiments_20260223_130655.json** — Results from Experiments A through G (Gemma 4B and 12B). Contains contrastive discovery, feature verification, scaling law (EXP C), and the EXP G injection bug postmortem. This is the source for ΔAIC=15.8 and all FP rate numbers.

**exp_H_v2_20260225_121930.json** — EXP H v2 definitive result with delta-steering. Contains paper features delta (−0.031), null distribution (150 random pairs), baseline logit diffs, round-trip control (−1.875), and norm-matched random control. This is the source for all main result numbers.

**scaling_law_results.json** — Compact subset of EXP C results (FP rates at each n). Same data as in all_experiments, extracted for convenience. Read-only.

## Per-experiment JSONs (exp_A through exp_G)

Individual experiment results extracted from the main JSON. These contain the same data as the corresponding sections in all_experiments_20260223_130655.json and exist for convenience.

## Checkpoint files (exp_H_v2_checkpoint_seed*.json)

Intermediate checkpoints from EXP H v2 runs. These are not cited anywhere and exist only for debugging/auditing.

## Outdated files

**OUTDATED_exp_h_v2.log** — Raw console log from EXP H v2. Contains an INCORRECT verdict ("significantly WORSE") based on unconditional p-value, which is misleading for zero-inflated distributions. See LOG_NOTE.md for explanation. The JSON file is the canonical data source.

**LOG_NOTE.md** — Explains why the log verdict is wrong and how to interpret the results correctly.
