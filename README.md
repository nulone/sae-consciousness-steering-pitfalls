# SAE Consciousness Steering: A Multi-Model Null Result

I tried to use contrastive SAE discovery to find features that control how language models answer consciousness-related questions. After 9 experiments on Gemma 3 4B and Gemma 3 12B (plus a qualitative Neuronpedia label search on Llama 3.3 70B), I found no evidence of causal consciousness features with this pipeline. The contrastive method finds punctuation, Japanese grammar, and self-referential discourse markers — not consciousness.

Along the way I found two critical bugs in my steering code (both caught by external reviewers), and developed a clean **delta-steering** method that eliminates a common confound in SAE steering experiments.

## ⚠️ Why You Should Use Delta-Steering

```python
# ❌ WRONG: replaces residual with SAE reconstruction (lossy!)
# This measures "SAE reconstruction damage" + "feature effect" combined.
# In my case, 98.8% of the observed effect disappeared after fixing the method
# (at least 72% was pure round-trip reconstruction error).
residual = sae.decode(acts_modified)

# ✅ RIGHT: adds only the feature-specific delta to original residual.
# Reconstruction error appears in both decode() calls and cancels out.
delta = sae.decode(acts_modified) - sae.decode(acts_original)
residual = original_residual + delta
```

## Key Results

**Contrastive features are not what they seem.** 17 features identified across two Gemma models and three layers. All 17 encode low-level linguistic patterns (punctuation, pronouns, Japanese verb endings), not consciousness.

**Steering effect is zero.** With delta-steering (no reconstruction confound), ablating the "consciousness features" shifts logits by -0.031. The old v1 hook (full residual replacement) gave -2.609, of which 98.8% disappeared after the fix. SAE round-trip error alone is -1.875. The features do nothing.

**False positive scaling law.** Contrastive SAE discovery with small n has high FP rates (55% at n=4, 32% at n=8), dropping to 0% at n=28 (the largest sample size tested). Rule of thumb: use ≥20 texts per class.

## Repository Structure

```
scripts/
  exp_h_definitive_v2.py    # Final experiment (delta-steering, the one that matters)
  run_all_experiments.py     # EXP A-E (scaling laws, cross-validation)
  verify_numbers.py          # Verify all writeup numbers from saved JSON (no GPU needed)
  exp_g_matched_random.py    # EXP G (has known injection bug, kept for documentation)
  exp_f_random_control.py    # EXP F (random steering pilot)
  gpu_phase1_all.py          # GPU activation analysis

data/
  exp_H_v2_*.json            # Definitive experiment results
  exp_A-G_*.json             # Earlier experiment results
  OUTDATED_exp_h_v2.log      # Old GPU log (verdict is WRONG — see data/LOG_NOTE.md)
  README.md                  # Guide to which data files are canonical

neuronpedia_screenshots/     # Neuronpedia search results (Llama 70B, Gemma)
supplementary/               # Visualization (React chart)
```

## Reproducing the Definitive Experiment (EXP H v2)

Requirements: NVIDIA GPU with ≥16GB VRAM. Runtime: ~2 minutes on RTX 5090, longer on older GPUs.

```bash
pip install torch transformers sae-lens numpy scipy accelerate
python scripts/exp_h_definitive_v2.py
```

This runs 150 random 2-feature baselines (3 seeds × 50 combos), compares them against the paper features [1795, 934], and outputs a JSON with all results.

## Verifying the Numbers (No GPU Needed)

```bash
python scripts/verify_numbers.py
```

This reads the saved JSON data and recomputes every key number from the writeup (ΔAIC=15.8, δ=−0.031, conditional p=0.40, round-trip=−1.875, etc.), asserting they match.

## Known Issues in Older Scripts

`exp_g_matched_random.py` contains the injection bug (negative activations into JumpReLU SAE) documented in the post. It is kept in the repo for transparency. Do not use it as a reference for steering implementations. Use `exp_h_definitive_v2.py` instead.

## Practical Checklist for Contrastive SAE Discovery

- [ ] n ≥ 20 texts per class
- [ ] Cross-validate on held-out texts
- [ ] Check feature labels (Neuronpedia or direct activation analysis)
- [ ] Use delta-steering hooks, not full-replacement
- [ ] Include matched random feature baselines
- [ ] Report forced-choice logits, not generation-based classification

## Cost

~$80 total on Vast.ai (A6000, A100, RTX 5090 instances) over 2 weeks.

## License

MIT
