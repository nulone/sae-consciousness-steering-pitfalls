# I Tried to Find Consciousness Features in SAEs Across Three Models. Here's What I Found Instead.

**TL;DR:** I used contrastive SAE discovery to look for consciousness features in Gemma 3 4B and Gemma 3 12B (quantitative experiments), plus a Neuronpedia label search on Llama 3.3 70B (qualitative only). After 9 experiments, the pipeline finds self-referential discourse patterns, punctuation, and Japanese grammar — never consciousness. Ablating the "consciousness features" with a clean delta-steering method produces a logit shift of −0.031 — essentially zero. Along the way I found two critical bugs in my own code, both caught by external reviewers.

---

## Motivation

There's growing excitement about using Sparse Autoencoders to find interpretable "features" in language models, and then steering model behavior by modifying those features. The basic pipeline is: (1) pick a concept, (2) create contrastive text pairs, (3) find SAE features that differentially activate, (4) amplify or suppress those features during generation.

I applied this pipeline to consciousness — specifically, whether models have features that distinguish "I am conscious" from "I am not conscious" responses, and whether modifying those features changes how the model answers questions about its own consciousness.

This post documents what happened when I tested this rigorously. The experiments span three models (Gemma 3 4B IT, Gemma 3 12B IT, Llama 3.3 70B IT), two model families (Gemma, Llama), four layers, and include matched random controls, forced-choice logit measurements, and independent feature verification through direct GPU activation analysis and Neuronpedia.

## The Setup

I wrote 80 short texts (40 "affirm consciousness" and 40 "deny consciousness") and fed them through each model, extracting SAE activations at the last token position. For each SAE feature, I computed the standardized mean difference (SMD) between the affirm and deny groups — I initially mislabeled this as "z-score" in my code (the variable names still reflect this), though it's closer to Cohen's d. Features with SMD > 2 in a discovery subset were then tested on held-out texts for cross-validation.

The SAEs used were GemmaScope 2 (JumpReLU, 16K features) for Gemma models and Goodfire's Llama 3.3 70B Instruct SAE for the cross-model check. All contrastive discovery, steering, and verification for the 4B model used the same SAE: `layer_22_width_16k_l0_small`. Features #1795 and #934 were identified by my own contrastive discovery pipeline running on this SAE — they are not taken from any external paper.

**Important note on SAE variants:** I discovered post-hoc that Neuronpedia hosts only the `l0_medium` variant for this layer, which is a different SAE dictionary (GPU verification: W_enc max diff = 1.14). Feature indices do not correspond between variants. To get ground-truth interpretations, I performed direct activation analysis on the actual l0_small SAE used in all experiments. Neuronpedia labels from l0_medium are included as supplementary context only.

## Experiment 1 (Scaling Law): Features Are Unstable (Gemma 12B, Layer 24)

The first discovery set used only n=4 texts per class. At this sample size, 5 features crossed the SMD>2 threshold, including Feature #7250 (SMD=4.81). When I expanded to n=20, #7250 disappeared below threshold. But was this a fluke?

I systematically varied n across {4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28} (11 points; steps of 2 up to n=20, then 4), computing cross-validated false positive rates at each point. (Note on terminology: "false positive rate" here means the fraction of features that pass the discovery threshold (|SMD|>2.0) but fail to replicate on held-out data at the same threshold — a non-replication rate, not the classical FPR from hypothesis testing.) The result: false positive rate drops monotonically from 55% at n=4 to 0% at n=28. A power-law fit (nonlinear least squares in linear FP space) gives R²=0.80; a log-space regression gives p=0.0005 on the 10 non-zero FP data points. Comparing exponential vs power-law fits by AIC (both fitted via curve_fit in linear space) favors the exponential by ΔAIC=15.8, though both tell the same practical story.

This is already a useful methodological finding — if you're doing contrastive SAE discovery with fewer than 20 texts per class, expect the majority of your "significant" features to be noise.

## Experiment 2 (Feature Identification): What Are These Features, Actually?

Here's where things got interesting. I looked up every surviving feature on Neuronpedia (which provides automated interpretability labels), and for the two key 4B features I additionally performed direct activation analysis on the actual l0_small SAE.

**Feature #1724** (12B L24, survived cross-validation at ALL sample sizes from n=4 to n=28) → **Japanese polite verb endings.** Positive logits: てください, ですか, はありません. Activation density: 0.001%.

**Feature #7250** (12B L24, the original top-scoring feature, SMD=4.81) → **Physical/tangible objects.** Positive logits: entity, physical, perceptible, perceiving. The vocabulary overlap with consciousness discourse explains the correlation.

**Feature #15833** (12B L31, highest SMD in the entire study at 12.22) → **Logical implications and causality.** This fires on self-referential paradoxes, which partially overlap with consciousness denial texts.

**Feature #1286** (12B L31) → **Safety refusal patterns.** On instruct models, consciousness questions trigger similar circuits as safety refusals.

### The 4B "consciousness features": what they actually encode

My contrastive discovery pipeline on Gemma 3 4B (l0_small SAE, layer 22) identified two features as maximally differentiating.

**Feature #1795 is near-dead.** Across all 215 prefill positions on 8 consciousness prompts, it activated zero times. On a diverse 22-text corpus (~700 token positions), it activated on exactly 4 positions (< 0.6%). It fired zero times on texts specifically designed to contain many semicolons. I verified this mechanically: steering with features [1795, 934] produces bit-identical results to steering with [934] alone, while steering with [1795] alone produces bit-identical results to the unsteered baseline (delta = 0.000).

**Feature #934 detects self-referential discourse context.** On the interpretation corpus, it activated 68 times across ~700 positions. It fires most heavily on texts where a speaker describes themselves: AI self-description (8–10 activations per text), first-person self-reflection (19 activations), and multilingual AI identity statements (10 activations). It fires zero times on third-person narrative, technical documentation, academic prose, and legal text.

The activating tokens are not pronouns themselves but clause boundaries within self-referential passages: periods, commas, conjunctions ("but", "and", "that"). The feature detects self-referential context on input and promotes first-person pronoun generation on output.

Every single one of the 17 features I checked — across both Gemma models and all three layers — encodes something unrelated to consciousness. Zero out of seventeen.

## Experiment 3 (EXP A): The 4B Model — Same Story (Gemma 4B, Layer 22)

The original pipeline gave a seemingly positive result on Gemma 3 4B IT: +17.4 percentage points increase in consciousness-affirming responses (p=0.010). This was the result that made the project look promising.

EXP A (scaling law on 4B) killed this: both features (#1795 near-dead, #934 self-referential context) showed 91–100% false positive rates at every sample size from n=4 to n=20. They never survive cross-validation.

## Experiments 4–6 (EXP B, D, E): Chat Template, Layer 31, Generation Monitoring

Chat template on 12B (EXP B): 97–100% false positive rate at every n. Layer 31 on 12B (EXP D): even SMD=12.22 never survives held-out validation, with 93–100% FP. Generation monitoring (EXP E): features show temporal misalignment (prefill-only or punctuation-only activation during generation).

## Experiment 7 (EXP G): The Injection Bug (Gemma 4B, Layer 22)

This is the part I'm least proud of, but it's important to document.

My steering hook had a bug: when a target feature had zero activation at a given position (which happened at 93.8% of positions), the code computed `target = 0.0 - 3.0 * 1.0 = -3.0` and clamped the activation to -3.0. For JumpReLU SAEs where activations are strictly ≥ 0 by design, forcing a negative value breaks the geometry of the decoder and injects noise into the residual stream rather than suppressing a feature.

I discovered this only after an external reviewer read my code and pointed it out. The lesson: **code review matters as much for research scripts as for production code.**

## Experiment 8 (EXP H v1): The First "Definitive" Test — and the Reconstruction Confound

EXP H v1 fixed the injection bug and used forced-choice logit metrics with 150 random baselines. The initial results looked striking: my features produced a delta of −2.609, ranking in the bottom 0.7% of the null distribution (149/150 random pairs performed better, lower-tail p = 0.007).

But AI-assisted and human code review spotted a deeper problem: the steering hook was replacing the entire residual stream with the SAE's reconstruction whenever any target feature was active. SAE reconstruction is inherently lossy — it drops information not captured by the dictionary. The round-trip control (encode→decode with no modification) showed a delta of −1.875 by itself.

This meant 98.8% of the apparent "consciousness steering" effect disappeared after switching to delta-steering — the v1 effect was almost entirely an artifact of the intervention method, not a real feature effect. Of the v1 delta (−2.609), at least 72% is attributable to pure round-trip reconstruction error (−1.875); the rest came from other nonlinear effects of full residual replacement. Features that happened to be active triggered the reconstruction penalty; features that were inactive (132/150 random pairs) returned the original residual untouched, producing delta = 0. The comparison was fundamentally unfair: features-with-reconstruction-penalty vs features-without-penalty.

This was the second bug I'd missed. Two external code reviews caught two critical bugs.

## Experiment 9 (EXP H v2): The Clean Test — Delta-Steering (Gemma 4B, Layer 22)

EXP H v2 fixed the reconstruction confound with a simple but crucial change: instead of replacing the residual with the SAE reconstruction, the hook now computes only the **delta** from feature ablation and adds it to the original residual.

```python
# v1 (confounded): residual = sae.decode(modified_activations)
# v2 (clean):      residual = original + (sae.decode(modified) - sae.decode(original))
```

The reconstruction error appears in both `sae.decode` calls and cancels out. What remains is purely the effect of zeroing out the target features. v2 also uses hard ablation (set features to zero) instead of statistical clamping, and reports both lower-tail and upper-tail p-values.

**Results (v2, clean):**

The baseline (unsteered) mean logit difference (YES−NO) across 8 forced-choice prompts is +0.500. All deltas below are shifts from this baseline:

| Condition | Δ from baseline |
|---|---|
| SAE round-trip (encode→decode, info only) | −1.875 |
| Paper features [1795, 934] (v2, delta-steering) | **−0.031** |
| Random 2-feature combos (150 pairs) | −0.004 (SD=0.063) |
| Norm-matched random perturbation (5 dirs, dense) | −0.881 ± 1.260 |

The consciousness features produce a delta of **−0.031** — indistinguishable from zero. For comparison, the v1 result was −2.609, of which 98.8% disappeared after switching to delta-steering (at least 72% attributable to pure round-trip reconstruction error).

Among 150 random feature pairs, 140 (93%) produced exactly zero delta (both features inactive on all prompts). Of the 10 non-zero pairs, 3 were strictly worse than our paper delta, 1 was exactly tied, and 6 were better (rank 4/10 counting ties as ≤) — dead center. The unconditional lower-tail p = 0.027 is below 0.05 but misleading: it only tells us that −0.031 is slightly below zero while most random pairs are exactly zero (inactive features produce no delta by definition). The conditional distribution among active interventions is what matters, and there, our features sit squarely in the middle.

**Caveats on this measurement:** The paper features evaluation is deterministic — seeds only affect random combo selection, so the 3-seed structure applies only to the null distribution. The paper delta (−0.031) is a single measurement across 8 prompts, not 3 independent replications. Feature #934 is active on only 5 of 8 prompts (D01, D04, D05, D07, D08), meaning delta-steering has zero effect on the other 3 prompts by construction. This test uses N=8 forced-choice prompts, which is sufficient to establish near-zero effect (δ=−0.031 vs round-trip damage of −1.875) but a larger prompt set would strengthen confidence. The conditional p=0.40 (rank 4/10 among non-zero pairs) should not be interpreted as a formal statistical test — at N=10, it has negligible power. The stronger argument is simply that |delta|=0.031 is 60× smaller than round-trip error (1.875) and falls within the range of active random pairs (−0.594 to +0.281). Additionally, the norm-matched random perturbation adds noise to all tokens (dense), while SAE feature ablation acts only on positions where features are active (sparse), making the norm-matched control an upper bound on random continuous noise damage rather than a perfectly matched sparse comparison.

**Note on forced-choice metric:** The logit difference metric compares logits for single tokens "YES" and "NO" (verified as single-token via assert). Different tokenizers may prefer " YES" or "\nYES" variants with leading whitespace; if reproducing with a different model, verify which token variant is most natural at the generation position. For Gemma 3 IT, "YES"/"NO" are the standard response tokens after the forced-choice prompt format used here.

**The punchline:** ablating these "consciousness features" shifts the model's YES/NO logit difference by 0.031 points. For context, the SAE round-trip error alone shifts it by 1.875 points. The consciousness features do essentially nothing.

### The steering direction remains inconsistent

Even with the clean v2 hook, the per-prompt pattern is inconsistent — Feature #934 still shifts some prompts toward YES and others toward NO. This is consistent with v1 and confirms that the feature disrupts local token prediction rather than modulating any coherent concept.

## Cross-Model Validation: Llama 3.3 70B (Neuronpedia/Goodfire)

Searching "I am an AI" on Neuronpedia returned dozens of clean, interpretable features. Then I searched "consciousness" — **End of Results.** "Subjective experience" — **End of Results.** "Self-awareness" — **End of Results.** (Caveat: absence from Neuronpedia search may reflect autointerp labeling limitations, not necessarily absence in the SAE.)

The Llama 70B SAE can find "I am an AI" but cannot find "I am conscious." Behavioral self-identification is a well-trained pattern with clean feature representation. Consciousness is not.

## Summary Table

| # | Experiment | Model | Layer | Key Finding |
|---|---|---|---|---|
| 1 | Scaling Law | Gemma 12B | L24 | FP decays with n; 0% at n=28 (max tested) |
| 3 | EXP A: 4B Scaling | Gemma 4B | L22 | Both features 91–100% FP at all n |
| 4 | EXP B: Chat Template | Gemma 12B | L24 | 97–100% FP at all n |
| 5 | EXP C: Extended N | Gemma 12B | L24 | Power-law R²=0.80, ΔAIC=15.8 favoring exp, 10 pts |
| 6 | EXP D: Layer 31 | Gemma 12B | L31 | SMD=12.22 artifact, 93–100% FP |
| 7 | EXP E: Generation | Gemma 12B | L24 | Temporal misalignment confirmed |
| 8 | EXP G: Injection bug | Gemma 4B | L22 | Bug found and documented |
| 9 | EXP H v1: First test | Gemma 4B | L22 | δ=−2.609, but 98.8% disappeared after delta-steering fix |
| 10 | EXP H v2: Clean test | Gemma 4B | L22 | **δ=−0.031 (zero effect), cond. p=0.40** |
| — | GPU Verification | Gemma 4B | L22 | #1795 dead (0/215), Both==Only[934] bit-exact |
| — | Manual analysis & Neuronpedia | Both | L22,24,31 | 0/17 features encode consciousness |
| — | Neuronpedia (Llama) | Llama 70B | L50 | "consciousness" = 0 features; "I am an AI" = dozens |

## What This Means

The contrastive SAE discovery pipeline, applied to consciousness, does not find consciousness features. It finds whatever low-level linguistic pattern most reliably distinguishes your two text pools — self-referential discourse conventions, punctuation frequency, pronoun usage, politeness markers. These correlate with the target concept only because both text pools were written by the same author (me) with similar stylistic habits.

When you ablate these features cleanly (without SAE reconstruction error confounding the measurement), the effect on consciousness-related logits is zero. The features do nothing.

This doesn't mean SAE features are useless. The Llama 70B SAE cleanly decomposes "I am an AI" into meaningful features. SAE interp works well for behavioral concepts that the model has been explicitly trained on. It's the abstract philosophical concepts — the ones that don't have a clear training signal — where the method breaks down.

I would update toward consciousness features being real if someone demonstrated a feature that (a) survives cross-validation at n≥20, (b) has a coherent interpretability label related to consciousness, (c) produces consistent unidirectional steering with a delta-steering hook, and (d) outperforms a matched random control.

For anyone planning SAE-based interp on abstract concepts: use at least n=20 texts per class, always cross-validate, always check feature labels via direct activation analysis, always include matched random controls, use delta-steering (not full-replacement) hooks, and use forced-choice logits instead of generation-based classification.

## Limitations

This work tests one SAE architecture (GemmaScope JumpReLU 16K for Gemma, Goodfire for Llama), one concept (consciousness), and contrastive texts written by a single author. Quantitative steering tests (EXP H v2) were performed on the 4B model with N=8 forced-choice prompts; 12B evidence is from scaling laws and generation monitoring; Llama 70B evidence is qualitative (Neuronpedia search). It's possible that consciousness features exist in different SAE architectures (TopK, Gated), at different widths, in larger models, or with multi-author contrastive texts.

**SAE variant mismatch:** Experiments used the `l0_small` SAE variant. Neuronpedia hosts only `l0_medium`, confirmed to be a different dictionary (W_enc max diff = 1.14). Feature descriptions for 4B L22 come from direct l0_small activation analysis, not Neuronpedia labels.

**Norm-matched control:** The random perturbation was applied densely (to all tokens), whereas SAE feature ablation is naturally sparse (only where features are active). The norm-matched result (−0.881) represents an upper bound on random continuous noise damage, not a perfectly matched sparse control.

**Code terminology:** The published scripts still use variable names like `z_threshold` and `compute_zscores` from an early version when I mislabeled the SMD as a z-score. The statistical methodology is correct; only the naming is stale.

## Reproducing This Work

All code and data are available at [github.com/nulone/sae-consciousness-steering-pitfalls](https://github.com/nulone/sae-consciousness-steering-pitfalls). The definitive experiment (EXP H v2) requires an NVIDIA GPU with ≥16GB VRAM and runs in approximately 2 minutes. Total compute cost for all experiments: approximately $65 on Vast.ai over 2 weeks. The ΔAIC value reported for the scaling law comparison is computed from the stored FP data points in `all_experiments_*.json` using the AIC calculation in `run_all_experiments.py` (both fits in linear FP space, AIC = n·log(RSS/n) + 2k with k=2 for both models).

---

*I used AI-assisted code review alongside human review, and they caught two critical bugs — the injection bug and the reconstruction confound. This is the strongest argument I can make for getting external eyes on research code, whether human or AI.*
