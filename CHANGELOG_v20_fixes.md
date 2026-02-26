# v20 Fixes Applied — Full History

## Round 2 (2026-02-26, post-analyst review)

Four independent analysts reviewed the archive. Fixes below address issues flagged by 2+ analysts, plus all factual errors.

**Fix R2-1 (CRITICAL) — ΔAIC code now matches writeup.**
The code in `run_all_experiments.py` previously fit models in log-space but computed RSS in linear space, producing ΔAIC≈18.4. The writeup claimed 15.8 from "both fits in linear FP space" — but the code didn't actually do that. Fixed: code now uses `scipy.optimize.curve_fit` (nonlinear least squares in linear FP space) for both power-law and exponential. Independent verification: ΔAIC=15.8. (Flagged by 3/4 analysts.)

**Fix R2-2 — `dtype` → `torch_dtype` in `exp_h_definitive_v2.py`.**
`AutoModelForCausalLM.from_pretrained()` uses `torch_dtype=`, not `dtype=`. The old parameter may be silently ignored or error on some transformers versions. (Flagged by 2/4 analysts.)

**Fix R2-3 — SAE loading: deprecated tuple unpack → future-proof pattern.**
`sae, _, _ = SAE.from_pretrained(...)` replaced with `sae_result = ...; sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result`. Matches pattern already used in `run_all_experiments.py`. (Flagged by 2/4 analysts.)

**Fix R2-4 — README overclaim softened.**
Old: "the answer is: there are no such features" (strong ontological claim).
New: "I found no evidence of causal consciousness features with this pipeline" (scoped to method).
Also: Llama 70B now explicitly marked as "qualitative Neuronpedia label search". (Flagged by 3/4 analysts.)

**Fix R2-5 — "n≥28" → "n=28 (max tested)" in README and writeup.**
Only n=28 was tested, not all n≥28. (Flagged by 1 analyst.)

**Fix R2-6 — Runtime: "~2 minutes" → "~2 minutes on RTX 5090, longer on older GPUs".**
Script header says "30-60 min", actual runtime was 131s on RTX 5090. README now specifies GPU. (Flagged by 1 analyst.)

**Fix R2-7 — Log verdict conflict documented.**
`data/exp_h_v2.log` still contains old verdict "significantly WORSE" (unconditional p only). Added `data/LOG_NOTE.md` explaining discrepancy with current code/writeup. (Flagged by 3/4 analysts.)

**Fix R2-8 — requirements.txt: added tested versions comment + accelerate.**
Added exact tested versions as comment, plus `accelerate` dependency. (Flagged by 2/4 analysts.)

**Fix R2-9 — `lesswrong_post_ENGLISH.md` → `DEPRECATED_english_draft_v1.md`.**
Renamed to prevent anyone quoting outdated p=0.993 / "8 experiments". (Flagged by 1 analyst.)

### Not fixed (acknowledged, low priority)

**"paper_features" terminology** — One analyst recommended renaming to "target_features" or "discovered_features" throughout code. This would touch 12+ locations in exp_h_definitive_v2.py and risk introducing bugs. Noted for future cleanup. The writeup already explains these are self-discovered, not from external publication.

**YES/NO tokenization** — One analyst noted `tokenizer.encode("YES")[0]` is fragile to multi-token encoding. In practice, Gemma tokenizes "YES" and "NO" as single tokens, but a defensive check would be better. Noted for future.

**Activation-matched random baseline** — One analyst suggested sampling random features conditional on being active (to avoid 93% zeros). Valid suggestion for a follow-up, but changes the experiment design. Current approach is documented with its limitations.

**Feature table (17 features)** — One analyst requested explicit table mapping all 17 feature IDs to models, layers, SMD, and labels. Good suggestion for supplementary; not blocking publication.

---

## Round 1 (2026-02-26, pre-analyst)

Base: `SAE_Consciousness_v20_CORRECTED.zip`

**Fix A — ΔAIC value: 18.4 → 15.8** (full_writeup.md lines 27, 130). See Round 2 Fix R2-1 for full resolution.

**Fix B — "increments of 2" → actual n values** (full_writeup.md line 27). Actual: {4,6,8,...,20,24,28}, steps of 2 then 4.

**Fix C — Llama 70B qualifier in TL;DR** (full_writeup.md line 3). Added "(qualitative only)".

**Fix D — Tie in null distribution** (full_writeup.md line 107). "3 strictly worse, 1 tied, 6 better (rank 4/10 counting ties as ≤)".

**Fix E — AIC computation provenance** (full_writeup.md line 164). Added sentence explaining method.

**Fix F — [GitHub link] placeholders** in full_writeup.md and lesswrong_post_RUSSIAN.md — replaced with `nulone/sae-consciousness-steering-pitfalls`.
