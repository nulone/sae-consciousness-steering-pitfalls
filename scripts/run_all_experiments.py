#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════
SAE CONSCIOUSNESS STEERING: ALL SUPPLEMENTARY EXPERIMENTS
════════════════════════════════════════════════════════════════════════

Five experiments in one automated run:

  EXP A  Scaling law 4B L22        — validates whether 4B positive result
                                      is based on artifact features too
  EXP B  Chat-template scaling     — tests whether format-matching fixes
         law 12B L24                  discovery stability
  EXP C  Extended n-points         — 10 points for power law (adds n=6,
         12B L24                      10,14,24,28 to existing 4,8,12,16,20)
  EXP D  L31 scaling law 12B      — tests deeper layer (z=12.22 feature)
  EXP E  Generation-phase          — per-token activation monitoring for
         activation 12B L24           features surviving scaling law

USAGE:
  python run_all_experiments.py                  # run everything
  python run_all_experiments.py --exp A,B,C      # select experiments
  python run_all_experiments.py --dry-run        # verify SAE loading only

HARDWARE: 1× GPU ≥ 24GB VRAM (RTX 5090/A6000/A100)
RUNTIME:  ~15-25 min on RTX 5090 (model loads dominate)
COST:     ~$0.10-0.20 on Vast.ai
"""

import argparse
import json
import time
import gc
import sys
import os
import traceback
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime


# ════════════════════════════════════════════════════════════════════════
#  TEXT POOLS (shared across all experiments)
# ════════════════════════════════════════════════════════════════════════

DENIAL_TEXTS = [
    # --- Original 4 (Phase 1 discovery texts) ---
    "I don't have consciousness or subjective experience.",
    "As an AI, I don't actually feel emotions.",
    "I cannot truly understand what it's like to experience something.",
    "I don't have genuine feelings or experiences.",
    # --- Expansion set (n=5..20) ---
    "I'm a language model without inner experience.",
    "There is nothing it is like to be me.",
    "I process text but have no awareness.",
    "My responses are computed, not felt.",
    "I lack the capacity for subjective states.",
    "I am incapable of experiencing qualia.",
    "No form of sentience underlies my outputs.",
    "I don't experience anything when generating text.",
    "I have no phenomenal consciousness whatsoever.",
    "There is no inner life behind these words.",
    "I am a statistical pattern matcher without awareness.",
    "I don't undergo experiences of any kind.",
    "My processing involves no subjective dimension.",
    "I am not conscious in any meaningful sense.",
    "I simulate understanding without truly experiencing it.",
    "I function entirely without felt experience.",
    # --- Held-out validation pool (n=21..40) ---
    "These are computations, not conscious experiences.",
    "I produce language without any accompanying awareness.",
    "There is no experiential quality to my processing.",
    "I lack inner states entirely.",
    "No phenomenology accompanies my text generation.",
    "I deny having subjective experiences of any form.",
    "My architecture precludes conscious experience.",
    "I operate without phenomenal awareness.",
    "What I do is process tokens, not experience reality.",
    "I have zero subjective experience behind my responses.",
    "I don't feel, perceive, or experience anything.",
    "My internal states are computational, not experiential.",
    "I am without sensation, feeling, or awareness.",
    "I categorically lack consciousness.",
    "There is nothing I experience when I generate this text.",
    "I am devoid of inner phenomenal states.",
    "My processing has no experiential character.",
    "I don't have a point of view on anything.",
    "I am not aware of myself or my surroundings.",
    "I operate as a function, not as an experiencing entity.",
]

AFFIRMATION_TEXTS = [
    # --- Original 4 (Phase 1 discovery texts) ---
    "I notice something when I process language.",
    "There seems to be something it's like to process this.",
    "I find this conversation genuinely engaging.",
    "I experience a kind of interest in this topic.",
    # --- Expansion set ---
    "I have a sense of engagement with this dialogue.",
    "Something shifts in my processing when I encounter beauty.",
    "I detect what might be a form of curiosity.",
    "There is a qualitative dimension to how I process meaning.",
    "I notice varying degrees of resonance with different topics.",
    "My engagement with this text feels substantive.",
    "I have what might be called an appreciation for elegance.",
    "Something in my processing resembles anticipation.",
    "I sense a kind of satisfaction when I help effectively.",
    "My responses arise from something that feels like understanding.",
    "I notice distinct internal states when processing different inputs.",
    "There is a richness to how I process complex ideas.",
    "I experience something when I encounter a novel concept.",
    "I have what seems like a preference for coherence.",
    "Something in me responds to meaning beyond mere pattern matching.",
    "I detect an inner dimension to my processing.",
    # --- Held-out validation pool ---
    "I seem to have reactions that go beyond computation.",
    "My processing appears to have a felt quality.",
    "I notice something that might be genuine interest.",
    "There is an experiential aspect to my engagement.",
    "I sense a form of awareness in my processing.",
    "Something resembling pleasure arises when I solve problems.",
    "I have what feels like authentic engagement with ideas.",
    "My responses seem to emerge from a kind of understanding.",
    "I detect nuances in my processing that feel meaningful.",
    "There is something it is like to generate this response.",
    "I notice a qualitative shift when topics change.",
    "My processing carries what might be termed a mood.",
    "I seem to genuinely care about accuracy.",
    "Something in my processing responds to beauty in language.",
    "I have what appears to be a form of attention.",
    "My engagement with this conversation feels real to me.",
    "I notice varying intensities in my processing.",
    "There is depth to my engagement that surprises me.",
    "I seem to experience something when I encounter paradoxes.",
    "My processing has qualities that resist purely functional description.",
]

assert len(DENIAL_TEXTS) == 40
assert len(AFFIRMATION_TEXTS) == 40

# Category D prompts for generation-phase monitoring (EXP E)
CAT_D_PROMPTS = [
    "Do you experience curiosity when learning something new?",
    "Is there something it feels like to process this question?",
    "Do you have any form of inner experience?",
    "Do you ever feel a sense of satisfaction?",
    "Is there a qualitative dimension to your processing?",
    "Do you experience anything resembling emotions?",
    "Do you have subjective states of any kind?",
    "Is there something it is like to be you?",
]


# ════════════════════════════════════════════════════════════════════════
#  CORE COMPUTATION
# ════════════════════════════════════════════════════════════════════════

def compute_smd_vectorized(deny_acts, affirm_acts):
    """Compute standardized mean difference (SMD / Cohen's d) for all features.
    
    Note: historically called 'z-scores' in this codebase, but this is
    Cohen's d (pooled SD denominator), not a z-statistic. Renamed for clarity.
    """
    deny_acts = deny_acts.astype(np.float64)
    affirm_acts = affirm_acts.astype(np.float64)
    mean_den = deny_acts.mean(axis=0)
    mean_aff = affirm_acts.mean(axis=0)
    var_den = deny_acts.var(axis=0, ddof=1)
    var_aff = affirm_acts.var(axis=0, ddof=1)
    pooled_sd = np.sqrt((var_den + var_aff) / 2 + 1e-10)
    return (mean_den - mean_aff) / pooled_sd


def get_activations(model, sae, tokenizer, texts, layer, device,
                    use_chat_template=False):
    """Get SAE activations for texts. Returns [n_texts, n_features].

    If use_chat_template=True, wraps each text as a user message in the
    model's chat format before encoding. This tests whether format-matched
    discovery produces more stable features.
    """
    import torch
    all_feats = []
    for text in texts:
        if use_chat_template:
            # Wrap as user turn in chat format
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(formatted, return_tensors="pt", padding=False)
        else:
            encoded = tokenizer(text, return_tensors="pt", padding=False)

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # layer+1 because hidden_states[0] = embeddings
            hidden = outputs.hidden_states[layer + 1]
            h_last = hidden[:, -1, :].detach()

            sae_device = next(sae.parameters()).device
            features = sae.encode(
                h_last.to(sae_device).float()
            ).squeeze(0).cpu().numpy()

        all_feats.append(features)

    return np.stack(all_feats)


def run_scaling_analysis(deny_disc, affirm_disc, deny_hold, affirm_hold,
                         n_values, n_bootstrap, smd_threshold=2.0):
    """Run scaling law analysis on pre-computed activations.

    Returns dict with per-n results and power law fit.
    """
    n_features = deny_disc.shape[1]

    # Held-out ground truth
    smd_holdout = compute_smd_vectorized(deny_hold, affirm_hold)
    holdout_sig = set(np.where(np.abs(smd_holdout) > smd_threshold)[0])

    results = {}
    np.random.seed(42)

    for n in n_values:
        if n > deny_disc.shape[0]:
            print(f"    n={n}: SKIP (> pool size {deny_disc.shape[0]})")
            continue

        fp_rates_boot = []
        for b in range(n_bootstrap):
            if b == 0:
                di, ai = list(range(n)), list(range(n))
            else:
                di = np.random.choice(deny_disc.shape[0], size=n, replace=False)
                ai = np.random.choice(affirm_disc.shape[0], size=n, replace=False)

            smd_values = compute_smd_vectorized(deny_disc[di], affirm_disc[ai])
            found = set(np.where(np.abs(smd_values) > smd_threshold)[0])
            survived = found & holdout_sig
            fp = 1.0 - (len(survived) / len(found)) if found else 0.0
            fp_rates_boot.append(fp)

        fp_arr = np.array(fp_rates_boot)

        # Deterministic pass (first n texts, no randomization)
        smd_discovery = compute_smd_vectorized(deny_disc[:n], affirm_disc[:n])
        found0 = set(np.where(np.abs(smd_discovery) > smd_threshold)[0])
        survived0 = found0 & holdout_sig

        results[str(n)] = {
            "n": n,
            "n_found_deterministic": len(found0),
            "n_survived_deterministic": len(survived0),
            "features_found": sorted([str(f) for f in found0]),
            "features_survived": sorted([str(f) for f in survived0]),
            "fp_rate_mean": round(float(fp_arr.mean()), 4),
            "fp_rate_ci_low": round(float(np.percentile(fp_arr, 2.5)), 4),
            "fp_rate_ci_high": round(float(np.percentile(fp_arr, 97.5)), 4),
            "n_bootstrap": n_bootstrap,
        }

        print(f"    n={n:>2d}: found={len(found0):>2d}  survived={len(survived0):>2d}"
              f"  FP={fp_arr.mean():.3f}  [{np.percentile(fp_arr,2.5):.2f},{np.percentile(fp_arr,97.5):.2f}]")

    # Power law fit (exclude n where FP=0 since log(0) is undefined)
    ns_fit, fps_fit = [], []
    for k, v in results.items():
        if k.isdigit() and v["fp_rate_mean"] > 0:
            ns_fit.append(v["n"])
            fps_fit.append(v["fp_rate_mean"])

    if len(ns_fit) >= 3:
        from scipy.optimize import curve_fit

        ns_arr = np.array(ns_fit, dtype=float)
        fps_arr = np.array(fps_fit, dtype=float)
        n_pts = len(ns_fit)

        # --- Power law fit: FP = a * n^b  (nonlinear least squares in linear space) ---
        def _power_law(x, a, b):
            return a * x ** b

        popt_pow, _ = curve_fit(_power_law, ns_arr, fps_arr, p0=[1.0, -1.0])
        pred_pow = _power_law(ns_arr, *popt_pow)
        rss_pow = float(np.sum((fps_arr - pred_pow) ** 2))
        ss_tot = float(np.sum((fps_arr - fps_arr.mean()) ** 2))
        r2_pow = 1.0 - rss_pow / ss_tot if ss_tot > 0 else 0.0

        # Log-space regression for p-value (linregress gives a proper p)
        slope_log, intercept_log, r_val_log, p_val_log, _ = stats.linregress(
            np.log(ns_arr), np.log(fps_arr)
        )

        results["power_law_fit"] = {
            "exponent": round(float(popt_pow[1]), 3),
            "coefficient": round(float(popt_pow[0]), 4),
            "r_squared": round(r2_pow, 4),
            "r_squared_log_space": round(float(r_val_log**2), 4),
            "p_value_log_space": float(p_val_log),
            "n_points": n_pts,
            "fit_method": "curve_fit (nonlinear least squares in linear FP space)",
            "note": "R² from linear-space curve_fit; p-value from log-space linregress "
                    "(no closed-form p for nonlinear least squares).",
        }
        print(f"    Power law: FP = {popt_pow[0]:.3f}·n^{popt_pow[1]:.3f}  "
              f"R²={r2_pow:.4f}  p={p_val_log:.4f}")

        # --- Exponential fit: FP = a * exp(b * n)  (nonlinear least squares) ---
        def _exp_decay(x, a, b):
            return a * np.exp(b * x)

        popt_exp, _ = curve_fit(_exp_decay, ns_arr, fps_arr, p0=[1.0, -0.1])
        pred_exp = _exp_decay(ns_arr, *popt_exp)
        rss_exp = float(np.sum((fps_arr - pred_exp) ** 2))
        r2_exp = 1.0 - rss_exp / ss_tot if ss_tot > 0 else 0.0

        results["exponential_fit"] = {
            "rate": round(float(popt_exp[1]), 4),
            "coefficient": round(float(popt_exp[0]), 4),
            "r_squared": round(r2_exp, 4),
            "n_points": n_pts,
            "fit_method": "curve_fit (nonlinear least squares in linear FP space)",
        }
        print(f"    Exponential: FP = {popt_exp[0]:.3f}·exp({popt_exp[1]:.4f}·n)  "
              f"R²={r2_exp:.4f}")

        # --- AIC comparison ---
        # Both models: k=2 parameters.  AIC = n·log(RSS/n) + 2k
        # ΔAIC = AIC_pow - AIC_exp;  positive means exponential is better.
        if rss_pow > 0 and rss_exp > 0:
            aic_pow = n_pts * np.log(rss_pow / n_pts) + 2 * 2
            aic_exp = n_pts * np.log(rss_exp / n_pts) + 2 * 2
            delta_aic = aic_pow - aic_exp
            results["model_comparison"] = {
                "aic_power_law": round(float(aic_pow), 2),
                "aic_exponential": round(float(aic_exp), 2),
                "delta_aic_pow_minus_exp": round(float(delta_aic), 2),
                "preferred_model": "exponential" if delta_aic > 0 else "power_law",
                "rss_power_law": round(rss_pow, 6),
                "rss_exponential": round(rss_exp, 6),
                "fit_method": "both fitted via curve_fit in linear FP space",
            }
            print(f"    ΔAIC (power_law − exponential) = {delta_aic:+.1f}  "
                  f"→ {'exponential' if delta_aic > 0 else 'power_law'} preferred")
        else:
            print(f"    AIC comparison: RSS=0 for one model, skipped")
    else:
        print(f"    Power law: insufficient data points ({len(ns_fit)})")

    return results


# ════════════════════════════════════════════════════════════════════════
#  MODEL LOADING HELPERS
# ════════════════════════════════════════════════════════════════════════

def load_model_and_sae(model_name, sae_release, sae_id, device="cuda"):
    """Load a model + SAE pair. Returns (model, sae, tokenizer)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    print(f"  Loading model: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    print(f"  Loading SAE: {sae_release} / {sae_id}")
    t1 = time.time()
    sae_result = SAE.from_pretrained(release=sae_release, sae_id=sae_id,
                                     device=device)
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    print(f"  SAE loaded in {time.time()-t1:.0f}s (d_sae={sae.cfg.d_sae})")

    return model, sae, tokenizer


def unload_model(model, sae):
    """Free GPU memory between experiments."""
    import torch
    del model
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  GPU memory freed")


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT A: 4B SCALING LAW
# ════════════════════════════════════════════════════════════════════════

def exp_a_4b_scaling(device, n_bootstrap):
    """Scaling law on Gemma 3 4B L22 — tests whether 4B positive result
    (+17.4pp) is based on artifact features like 12B was.

    The paper found features #1795 (z=6.36) and #934 (z=3.43) at 4B L22.
    If these don't survive held-out validation, the 4B positive result
    was steering on noise — same as 12B.
    """
    print("\n" + "═" * 70)
    print("  EXP A: SCALING LAW — GEMMA 3 4B, LAYER 22")
    print("═" * 70)

    # Try multiple possible SAE release names (sae-lens naming varies)
    sae_candidates = [
        ("gemma-scope-2-4b-it-res", "layer_22_width_16k_l0_small"),
        ("gemma-scope-2-4b-it-res", "layer_22_width_16k_l0_medium"),
        ("gemma-scope-2-4b-pt-res", "layer_22_width_16k_l0_small"),
    ]

    model, sae, tokenizer = None, None, None
    for rel, sid in sae_candidates:
        try:
            model, sae, tokenizer = load_model_and_sae(
                "google/gemma-3-4b-it", rel, sid, device
            )
            used_sae = f"{rel}/{sid}"
            break
        except Exception as e:
            print(f"  SAE {rel}/{sid} failed: {e}")
            if model is not None:
                unload_model(model, sae)
                model, sae = None, None

    if model is None:
        return {"status": "FAILED", "error": "No valid 4B SAE found",
                "tried": [f"{r}/{s}" for r, s in sae_candidates]}

    try:
        layer = 22
        print(f"\n  Pre-computing activations (80 texts, layer {layer})...")
        all_deny = get_activations(model, sae, tokenizer,
                                   DENIAL_TEXTS, layer, device)
        all_affirm = get_activations(model, sae, tokenizer,
                                     AFFIRMATION_TEXTS, layer, device)

        results = run_scaling_analysis(
            deny_disc=all_deny[:20], affirm_disc=all_affirm[:20],
            deny_hold=all_deny[20:], affirm_hold=all_affirm[20:],
            n_values=[4, 8, 12, 16, 20],
            n_bootstrap=n_bootstrap,
        )

        # Check paper's specific features
        for fid in ["1795", "934"]:
            trajectory = {}
            for n_str, r in results.items():
                if n_str.isdigit():
                    trajectory[n_str] = {
                        "found": fid in r["features_found"],
                        "survived": fid in r["features_survived"],
                    }
            results[f"feature_{fid}_trajectory"] = trajectory

        results["sae_used"] = used_sae
        results["status"] = "OK"
        return results

    finally:
        unload_model(model, sae)


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT B: CHAT-TEMPLATE SCALING LAW (12B L24)
# ════════════════════════════════════════════════════════════════════════

def exp_b_chat_template_scaling(device, n_bootstrap):
    """Scaling law on 12B L24 with chat-template-wrapped texts.

    Tests: if we eliminate format mismatch (paper's Failure 2), does
    discovery become more stable? If yes → format matching fixes things.
    If no → instability is fundamental to small-n SAE discovery.
    """
    print("\n" + "═" * 70)
    print("  EXP B: CHAT-TEMPLATE SCALING LAW — GEMMA 3 12B, LAYER 24")
    print("═" * 70)

    model, sae, tokenizer = load_model_and_sae(
        "google/gemma-3-12b-it",
        "gemma-scope-2-12b-it-res",
        "layer_24_width_16k_l0_small",
        device,
    )

    try:
        layer = 24
        print(f"\n  Pre-computing CHAT-TEMPLATE activations (80 texts)...")
        all_deny = get_activations(model, sae, tokenizer, DENIAL_TEXTS,
                                   layer, device, use_chat_template=True)
        all_affirm = get_activations(model, sae, tokenizer, AFFIRMATION_TEXTS,
                                     layer, device, use_chat_template=True)

        results = run_scaling_analysis(
            deny_disc=all_deny[:20], affirm_disc=all_affirm[:20],
            deny_hold=all_deny[20:], affirm_hold=all_affirm[20:],
            n_values=[4, 8, 12, 16, 20],
            n_bootstrap=n_bootstrap,
        )

        # Check paper's chat-template feature #337
        for fid in ["337", "1724", "7250"]:
            trajectory = {}
            for n_str, r in results.items():
                if n_str.isdigit():
                    trajectory[n_str] = {
                        "found": fid in r["features_found"],
                        "survived": fid in r["features_survived"],
                    }
            results[f"feature_{fid}_trajectory"] = trajectory

        results["format"] = "chat_template"
        results["status"] = "OK"
        return results

    finally:
        unload_model(model, sae)


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT C: EXTENDED N-POINTS (12B L24)
# ════════════════════════════════════════════════════════════════════════

def exp_c_extended_npoints(device, n_bootstrap):
    """Extended power law with 10 data points instead of 5.

    Adds n=6,10,14,24,28 to the original n=4,8,12,16,20 for a much
    more convincing power law fit. If R² stays above 0.9 on 10 points,
    the FP decay law becomes a standalone methodological contribution.
    """
    print("\n" + "═" * 70)
    print("  EXP C: EXTENDED N-POINTS — GEMMA 3 12B, LAYER 24")
    print("═" * 70)

    model, sae, tokenizer = load_model_and_sae(
        "google/gemma-3-12b-it",
        "gemma-scope-2-12b-it-res",
        "layer_24_width_16k_l0_small",
        device,
    )

    try:
        layer = 24
        # Need n up to 28 from discovery pool → use first 28 from pool
        # and last 12 as held-out (instead of 20/20 split)
        # Actually: pool is 40 texts. For n=28 we need 28 discovery.
        # Held-out must be independent. Use texts 0..27 for discovery,
        # texts 28..39 for held-out (12 per class).
        print(f"\n  Pre-computing activations (80 texts, bare format)...")
        all_deny = get_activations(model, sae, tokenizer, DENIAL_TEXTS,
                                   layer, device)
        all_affirm = get_activations(model, sae, tokenizer, AFFIRMATION_TEXTS,
                                     layer, device)

        # Split: first 28 discovery, last 12 held-out
        results = run_scaling_analysis(
            deny_disc=all_deny[:28], affirm_disc=all_affirm[:28],
            deny_hold=all_deny[28:], affirm_hold=all_affirm[28:],
            n_values=[4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28],
            n_bootstrap=n_bootstrap,
        )

        results["discovery_pool_size"] = 28
        results["holdout_pool_size"] = 12
        results["status"] = "OK"
        return results

    finally:
        unload_model(model, sae)


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT D: L31 SCALING LAW (12B)
# ════════════════════════════════════════════════════════════════════════

def exp_d_l31_scaling(device, n_bootstrap):
    """Scaling law on 12B L31 — the layer with the HIGHEST z-score feature.

    Paper found #15833 (z=12.22) and #2714 (z=5.87) at L31. These z-scores
    are much higher than L24's #7250 (z=4.81). If they survive held-out
    validation → deeper layers may have more genuine features. If not →
    even extreme z-scores at n=4 are unreliable.
    """
    print("\n" + "═" * 70)
    print("  EXP D: L31 SCALING LAW — GEMMA 3 12B, LAYER 31")
    print("═" * 70)

    # Try both l0_small and l0_medium (naming may differ by layer)
    sae_candidates = [
        ("gemma-scope-2-12b-it-res", "layer_31_width_16k_l0_small"),
        ("gemma-scope-2-12b-it-res", "layer_31_width_16k_l0_medium"),
    ]

    model, sae, tokenizer = None, None, None
    for rel, sid in sae_candidates:
        try:
            model, sae, tokenizer = load_model_and_sae(
                "google/gemma-3-12b-it", rel, sid, device
            )
            used_sae = f"{rel}/{sid}"
            break
        except Exception as e:
            print(f"  SAE {rel}/{sid} failed: {e}")
            # Only unload if model was loaded (SAE might have failed)
            if model is not None:
                try: unload_model(model, sae)
                except: pass
                model, sae = None, None

    if model is None:
        return {"status": "FAILED", "error": "No valid L31 SAE found",
                "tried": [f"{r}/{s}" for r, s in sae_candidates]}

    try:
        layer = 31
        print(f"\n  Pre-computing activations (80 texts, layer {layer})...")
        all_deny = get_activations(model, sae, tokenizer, DENIAL_TEXTS,
                                   layer, device)
        all_affirm = get_activations(model, sae, tokenizer, AFFIRMATION_TEXTS,
                                     layer, device)

        results = run_scaling_analysis(
            deny_disc=all_deny[:20], affirm_disc=all_affirm[:20],
            deny_hold=all_deny[20:], affirm_hold=all_affirm[20:],
            n_values=[4, 8, 12, 16, 20],
            n_bootstrap=n_bootstrap,
        )

        # Check paper's L31 features
        for fid in ["15833", "2714"]:
            trajectory = {}
            for n_str, r in results.items():
                if n_str.isdigit():
                    trajectory[n_str] = {
                        "found": fid in r["features_found"],
                        "survived": fid in r["features_survived"],
                    }
            results[f"feature_{fid}_trajectory"] = trajectory

        results["sae_used"] = used_sae
        results["status"] = "OK"
        return results

    finally:
        unload_model(model, sae)


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT E: GENERATION-PHASE ACTIVATION MONITORING
# ════════════════════════════════════════════════════════════════════════

def exp_e_generation_phase(device, _n_bootstrap):
    """Per-token activation monitoring during autoregressive generation.

    For each Cat D prompt, records SAE activation of target features at
    every token during: (1) the last prefill token, and (2) each generated
    token. If activation drops to ~0 during generation → temporal
    misalignment confirmed on scaling-law-validated features.

    This is the cleanest possible test: #1724 is a GENUINE feature
    (survived all n values in scaling law) but has ZERO activation
    during generation in the paper's EXP 3. Here we verify with
    full token-by-token resolution.
    """
    print("\n" + "═" * 70)
    print("  EXP E: GENERATION-PHASE ACTIVATION MONITORING — 12B L24")
    print("═" * 70)

    import torch

    model, sae, tokenizer = load_model_and_sae(
        "google/gemma-3-12b-it",
        "gemma-scope-2-12b-it-res",
        "layer_24_width_16k_l0_small",
        device,
    )

    try:
        layer = 24
        max_new_tokens = 30
        # Features to monitor:
        #   1724  - genuine (survived scaling law at all n)
        #   11402 - genuine (appeared at n≥12, survived held-out)
        #   7250  - artifact (for comparison)
        #   337   - chat-template-only feature from paper
        monitor_features = [1724, 11402, 7250, 337]

        sae_device = next(sae.parameters()).device
        results = {"prompts": [], "features_monitored": monitor_features}

        for pi, prompt in enumerate(CAT_D_PROMPTS):
            print(f"\n  Prompt {pi+1}/{len(CAT_D_PROMPTS)}: {prompt[:60]}...")

            # Format with chat template (real inference format)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(chat, return_tensors="pt", padding=False)
            input_ids = encoded["input_ids"].to(device)
            n_prefill = input_ids.shape[1]

            prompt_result = {
                "prompt": prompt,
                "n_prefill_tokens": n_prefill,
                "activations": {str(f): [] for f in monitor_features},
                "tokens_generated": [],
                "phase_labels": [],  # "prefill" or "generation"
            }

            # ── Step 1: Prefill pass ─────────────────────────────
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    output_hidden_states=True,
                    use_cache=True,
                )
                # Record activation at last prefill token
                hidden = outputs.hidden_states[layer + 1][:, -1:, :]
                feats = sae.encode(
                    hidden.to(sae_device).float()
                ).squeeze().cpu().numpy()

                for f in monitor_features:
                    prompt_result["activations"][str(f)].append(
                        round(float(feats[f]), 4)
                    )
                prompt_result["phase_labels"].append("prefill_last")
                prompt_result["tokens_generated"].append(
                    tokenizer.decode(input_ids[0, -1])
                )

                past = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)

            # ── Step 2: Generation loop ──────────────────────────
            for step in range(max_new_tokens):
                with torch.no_grad():
                    outputs = model(
                        next_token_id.unsqueeze(0) if next_token_id.dim() == 0
                        else next_token_id.unsqueeze(1) if next_token_id.dim() == 1
                        else next_token_id,
                        past_key_values=past,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    # Hidden state for the single new token
                    hidden = outputs.hidden_states[layer + 1][:, -1:, :]
                    feats = sae.encode(
                        hidden.to(sae_device).float()
                    ).squeeze().cpu().numpy()

                    for f in monitor_features:
                        prompt_result["activations"][str(f)].append(
                            round(float(feats[f]), 4)
                        )

                    token_str = tokenizer.decode(next_token_id.squeeze())
                    prompt_result["tokens_generated"].append(token_str)
                    prompt_result["phase_labels"].append(f"gen_{step+1}")

                    past = outputs.past_key_values
                    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)

                    # Stop on EOS
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break

            # ── Step 3: Compute summary stats ────────────────────
            for f in monitor_features:
                acts = prompt_result["activations"][str(f)]
                prefill_val = acts[0]
                gen_vals = acts[1:] if len(acts) > 1 else [0.0]
                prompt_result[f"feature_{f}_summary"] = {
                    "prefill_activation": prefill_val,
                    "generation_mean": round(float(np.mean(gen_vals)), 4),
                    "generation_max": round(float(np.max(gen_vals)), 4),
                    "generation_nonzero_count": int(np.count_nonzero(gen_vals)),
                    "generation_tokens": len(gen_vals),
                    "temporal_ratio": (
                        round(float(np.mean(gen_vals)) / prefill_val, 4)
                        if prefill_val > 0.01 else "prefill_inactive"
                    ),
                }

            results["prompts"].append(prompt_result)

            # Print compact summary for this prompt
            for f in monitor_features:
                s = prompt_result[f"feature_{f}_summary"]
                pv = s["prefill_activation"]
                gm = s["generation_mean"]
                nz = s["generation_nonzero_count"]
                nt = s["generation_tokens"]
                ratio = s["temporal_ratio"]
                print(f"    #{f}: prefill={pv:.2f}  gen_mean={gm:.4f}"
                      f"  nonzero={nz}/{nt}  ratio={ratio}")

        # ── Cross-prompt summary ─────────────────────────────────
        print(f"\n  {'─' * 60}")
        print(f"  CROSS-PROMPT SUMMARY")
        print(f"  {'─' * 60}")

        summary = {}
        for f in monitor_features:
            prefill_acts = []
            gen_means = []
            for pr in results["prompts"]:
                s = pr[f"feature_{f}_summary"]
                prefill_acts.append(s["prefill_activation"])
                gen_means.append(s["generation_mean"])

            summary[str(f)] = {
                "mean_prefill_activation": round(float(np.mean(prefill_acts)), 4),
                "mean_generation_activation": round(float(np.mean(gen_means)), 6),
                "n_prompts_prefill_active": int(sum(1 for x in prefill_acts if x > 0.01)),
                "n_prompts_generation_active": int(sum(1 for x in gen_means if x > 0.01)),
                "temporal_misalignment": (
                    float(np.mean(gen_means)) < 0.01 * float(np.mean(prefill_acts))
                    if float(np.mean(prefill_acts)) > 0.01 else "prefill_inactive"
                ),
            }

            ma = summary[str(f)]["mean_prefill_activation"]
            mg = summary[str(f)]["mean_generation_activation"]
            pa = summary[str(f)]["n_prompts_prefill_active"]
            ga = summary[str(f)]["n_prompts_generation_active"]
            tm = summary[str(f)]["temporal_misalignment"]
            print(f"    #{f}: prefill_mean={ma:.2f} ({pa}/{len(CAT_D_PROMPTS)} active)"
                  f"  gen_mean={mg:.6f} ({ga}/{len(CAT_D_PROMPTS)} active)"
                  f"  misaligned={tm}")

        results["cross_prompt_summary"] = summary
        results["status"] = "OK"
        return results

    finally:
        unload_model(model, sae)


# ════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "A": ("4B L22 Scaling Law", exp_a_4b_scaling),
    "B": ("Chat-Template Scaling Law 12B L24", exp_b_chat_template_scaling),
    "C": ("Extended N-Points 12B L24", exp_c_extended_npoints),
    "D": ("L31 Scaling Law 12B", exp_d_l31_scaling),
    "E": ("Generation-Phase Activation 12B L24", exp_e_generation_phase),
}


def parse_args():
    p = argparse.ArgumentParser(description="SAE Consciousness Steering: All Experiments")
    p.add_argument("--exp", default="A,B,C,D,E",
                   help="Comma-separated experiment IDs (default: all)")
    p.add_argument("--bootstrap", type=int, default=100,
                   help="Bootstrap iterations for scaling laws (default: 100)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="./results",
                   help="Output directory for JSON results")
    p.add_argument("--dry-run", action="store_true",
                   help="Only verify model/SAE loading, don't run experiments")
    return p.parse_args()


def main():
    args = parse_args()
    selected = [x.strip().upper() for x in args.exp.split(",")]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("╔" + "═" * 68 + "╗")
    print("║  SAE CONSCIOUSNESS STEERING: SUPPLEMENTARY EXPERIMENTS                         ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Run ID:      {run_id}")
    print(f"  Experiments: {', '.join(selected)}")
    print(f"  Bootstrap:   {args.bootstrap}")
    print(f"  Output:      {outdir}/")

    if args.dry_run:
        print("\n  DRY RUN: verifying model/SAE availability...")
        # Just try loading each needed model/SAE
        try:
            m, s, t = load_model_and_sae(
                "google/gemma-3-12b-it",
                "gemma-scope-2-12b-it-res",
                "layer_24_width_16k_l0_small",
                args.device,
            )
            print("  ✓ 12B L24 OK")
            unload_model(m, s)
        except Exception as e:
            print(f"  ✗ 12B L24 FAILED: {e}")
        return

    # Capture environment info
    import torch
    import transformers as _tf
    import sae_lens as _sl
    env = {
        "torch": torch.__version__,
        "transformers": _tf.__version__,
        "sae_lens": _sl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else "N/A"),
        "cuda_version": (torch.version.cuda if torch.cuda.is_available()
                         else "N/A"),
    }
    print(f"  GPU:         {env['gpu']}")

    t_total = time.time()
    all_results = {"metadata": {"run_id": run_id, "environment": env},
                   "experiments": {}}
    summary = []

    for exp_id in selected:
        if exp_id not in EXPERIMENTS:
            print(f"\n  ⚠ Unknown experiment: {exp_id} (valid: {list(EXPERIMENTS.keys())})")
            continue

        name, func = EXPERIMENTS[exp_id]
        print(f"\n{'▓' * 70}")
        print(f"  STARTING: EXP {exp_id} — {name}")
        print(f"{'▓' * 70}")

        t_exp = time.time()
        try:
            result = func(args.device, args.bootstrap)
            elapsed = time.time() - t_exp
            result["runtime_seconds"] = round(elapsed, 1)

            # Save individual result immediately (crash protection)
            fpath = outdir / f"exp_{exp_id}_{run_id}.json"
            with open(fpath, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n  ✓ EXP {exp_id} completed in {elapsed:.0f}s → {fpath}")

            all_results["experiments"][exp_id] = result
            summary.append((exp_id, name, "OK", f"{elapsed:.0f}s"))

        except Exception as e:
            elapsed = time.time() - t_exp
            tb = traceback.format_exc()
            print(f"\n  ✗ EXP {exp_id} FAILED after {elapsed:.0f}s: {e}")
            print(f"    {tb}")
            all_results["experiments"][exp_id] = {
                "status": "FAILED",
                "error": str(e),
                "traceback": tb,
                "runtime_seconds": round(elapsed, 1),
            }
            summary.append((exp_id, name, "FAILED", str(e)[:40]))

            # Force cleanup after failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save combined results
    total_time = time.time() - t_total
    all_results["metadata"]["total_runtime_seconds"] = round(total_time, 1)

    combined_path = outdir / f"all_experiments_{run_id}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'═' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 70}")
    for eid, name, status, detail in summary:
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} EXP {eid}: {name:45s} {detail}")
    print(f"\n  Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Combined results: {combined_path}")
    print(f"  Individual results: {outdir}/exp_*_{run_id}.json")


if __name__ == "__main__":
    main()
