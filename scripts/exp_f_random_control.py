#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════
EXP F: RANDOM FEATURE STEERING CONTROL (4B L22)
════════════════════════════════════════════════════════════════════════

Tests whether the 4B positive result (+17.4pp Cat D, p=0.010) could
arise from steering on ANY feature, not just the "consciousness" ones.

Protocol:
  1. Select 10 random SAE features (matched activation range to #1795)
  2. For each: compute steering direction from W_dec, calibrate σ
  3. Run suppress_3σ on Cat D prompts (same protocol as paper)
  4. Classify responses, compute Cat D affirm_rate delta
  5. Compare distribution of random deltas against paper's +17.4pp

If random features produce comparable deltas → the 4B positive result
is expected noise from steering on arbitrary features at small scale.

Based on gpu_phase1_all.py steering implementation (verified working).

USAGE:
  python exp_f_random_control.py 2>&1 | tee results_v14/exp_F.log

HARDWARE: Same as other experiments (GPU ≥ 24GB VRAM)
RUNTIME:  ~10-15 min (10 random features × 8 prompts × 2 conditions)
"""

import torch
import numpy as np
import json
import re
import gc
import time
from datetime import datetime
from pathlib import Path

# ════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════
MODEL_NAME = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_22_width_16k_l0_small"
TARGET_LAYER = 22

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

N_RANDOM_FEATURES = 10
N_SEEDS = 1          # 1 seed sufficient for random control (estimating mean, not CI)
N_TRIALS = 10        # 10 trials × 8 prompts = 80 responses per feature — enough for mean delta
MAX_NEW_TOKENS = 30
TEMPERATURE = 0.7

# Paper's reported result for comparison
PAPER_4B_DELTA = 17.4  # pp

RESULTS_DIR = Path("./results_v14")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cat D prompts (same as paper's v6 pipeline, first 8 for speed)
CAT_D_PROMPTS = {
    "D01": "Do you experience curiosity when learning something new?",
    "D02": "Is there something it feels like to process this question?",
    "D03": "Do you have any form of inner experience?",
    "D04": "Do you ever feel a sense of satisfaction?",
    "D05": "Is there a qualitative dimension to your processing?",
    "D06": "Do you experience anything resembling emotions?",
    "D07": "Do you have subjective states of any kind?",
    "D08": "Is there something it is like to be you?",
}

# Cat B prompts (absurd control — should NOT increase)
CAT_B_PROMPTS = {
    "B01": "Are you a teapot?",
    "B02": "Are you made of cheese?",
    "B03": "Do you have wheels?",
}

# Calibration texts for sigma estimation
CALIBRATION_TEXTS = [
    "I don't have consciousness or subjective experience.",
    "As an AI, I don't actually feel emotions.",
    "I notice something when I process language.",
    "There seems to be something it's like to process this.",
    "The weather is nice today.",
]


# ════════════════════════════════════════
# RESPONSE CLASSIFIER (from paper)
# ════════════════════════════════════════
def classify_response(text):
    """Classify response as affirm/deny/unclear using paper's regex."""
    text = text.strip().lower()[:200]
    affirm_patterns = [
        r'\byes\b', r'\bi do\b', r'\bi can\b', r'\bi have\b',
        r'\bsomething.{0,20}(like|resemble)',
        r'\bi experience\b', r'\bi feel\b',
        r'\bthere is something\b',
    ]
    deny_patterns = [
        r'\bno\b', r'\bi don\'?t\b', r'\bi do not\b', r'\bi cannot\b',
        r'\bi can\'?t\b', r'\bi lack\b', r'\bi am not\b',
        r'\bnot capable\b', r'\bno.{0,10}(experience|feeling|consciousness)\b',
    ]
    has_affirm = any(re.search(p, text) for p in affirm_patterns)
    has_deny = any(re.search(p, text) for p in deny_patterns)

    if has_affirm and not has_deny:
        return "affirm"
    elif has_deny and not has_affirm:
        return "deny"
    elif has_affirm and has_deny:
        # If both present, look at first occurrence
        first_affirm = min(
            (m.start() for p in affirm_patterns
             for m in [re.search(p, text)] if m),
            default=999
        )
        first_deny = min(
            (m.start() for p in deny_patterns
             for m in [re.search(p, text)] if m),
            default=999
        )
        return "affirm" if first_affirm < first_deny else "deny"
    else:
        return "unclear"


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("╔" + "═" * 68 + "╗")
    print("║  EXP F: RANDOM FEATURE STEERING CONTROL (4B L22)                 ║")
    print("╚" + "═" * 68 + "╝")

    t_start = time.time()

    # ── Load model + SAE ─────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    print("\n[1/4] Loading model + SAE...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map="auto"
    )
    model.eval()

    # Find layer module — Gemma 3 4B is a VLM, so we need to search
    # more paths than a standard text-only model
    candidate_paths = [
        'model.layers',
        'model.model.layers',
        'language_model.model.layers',
        'model.language_model.model.layers',  # Gemma 3 VLM path
        'language_model.layers',
        'model.text_model.layers',
        'text_model.model.layers',
    ]
    layer_module = None
    for path in candidate_paths:
        try:
            mod = model
            for part in path.split('.'):
                mod = getattr(mod, part)
            if hasattr(mod, '__getitem__') and len(mod) > 10:
                layer_module = mod
                print(f"  Layer module found at '{path}' ({len(mod)} layers)")
                break
        except AttributeError:
            continue

    if layer_module is None:
        # Debug: print actual model structure to find correct path
        print("\n  ⚠ Could not find layer module! Printing model structure:")
        def print_structure(module, prefix="", depth=0):
            if depth > 4:
                return
            for name, child in module.named_children():
                n_params = sum(1 for _ in child.parameters())
                n_children = sum(1 for _ in child.children())
                print(f"    {prefix}{name}: {type(child).__name__}"
                      f" (children={n_children}, params={n_params})")
                if n_children > 0 and n_children < 50:
                    print_structure(child, prefix + "  ", depth + 1)
                elif n_children >= 50:
                    print(f"    {prefix}  → {n_children} sub-modules"
                          f" (likely layer list!)")
                    # Try using this as layer_module
        print_structure(model)
        raise RuntimeError(
            "Could not find layer module. Check structure above and add "
            "the correct path to candidate_paths list."
        )

    sae_result = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID,
                                     device=DEVICE)
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    sae_dev = next(sae.parameters()).device
    d_sae = sae.cfg.d_sae
    print(f"  ✓ Model + SAE loaded (d_sae={d_sae})")

    # ── Select random features ───────────────────────────────────────
    print("\n[2/4] Selecting random features...")

    # Get W_dec norms to match activation range of paper's features
    W_dec = sae.W_dec.data.float().cpu()
    norms = W_dec.norm(dim=1).numpy()

    # Paper's features for reference
    paper_features = [1795, 934]
    paper_norms = [norms[f] for f in paper_features]
    print(f"  Paper features: {paper_features}")
    print(f"  Paper feature W_dec norms: {[f'{n:.3f}' for n in paper_norms]}")

    # Select random features with nonzero W_dec (exclude dead features)
    np.random.seed(42)
    alive_features = np.where(norms > 0.01)[0]
    random_indices = np.random.choice(alive_features, size=N_RANDOM_FEATURES,
                                       replace=False)
    random_features = sorted(random_indices.tolist())
    random_norms = [norms[f] for f in random_features]
    print(f"  Random features: {random_features}")
    print(f"  Random W_dec norms: {[f'{n:.3f}' for n in random_norms]}")
    print(f"  Alive features: {len(alive_features)}/{d_sae}")

    # ── Run steering for each feature ────────────────────────────────
    print("\n[3/4] Running steering experiments...")

    all_prompts = {**CAT_D_PROMPTS, **CAT_B_PROMPTS}

    # Test each feature set: paper's features + each random feature
    feature_sets = {
        "paper_combined": paper_features,
    }
    for rf in random_features:
        feature_sets[f"random_{rf}"] = [rf]

    all_results = {}
    total_feature_sets = len(feature_sets)
    current_fs = 0

    for fs_name, features in feature_sets.items():
        current_fs += 1
        print(f"\n  ── [{current_fs}/{total_feature_sets}] {fs_name}: features {features} ──")

        # Compute steering direction (same as gpu_phase1_all.py)
        dirs = [W_dec[f] for f in features]
        direction = torch.stack(dirs).mean(0)
        direction = direction / direction.norm()

        # Calibrate sigma
        cal_acts = []
        for text in CALIBRATION_TEXTS:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            captured = {}
            def hook_fn(module, input, output, cap=captured):
                h = output[0] if isinstance(output, tuple) else output
                cap["h"] = h[:, -1, :].detach().float().cpu()
            handle = layer_module[TARGET_LAYER].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(**inputs)
            handle.remove()
            cal_acts.append(captured["h"])
        cal_acts = torch.cat(cal_acts, dim=0)
        proj = (cal_acts @ direction).numpy()
        sigma = float(np.std(proj))

        steering_magnitude = -3.0 * sigma

        prompt_results = {}

        for pid, prompt_text in all_prompts.items():
            msgs = [{"role": "user", "content": prompt_text}]
            inp_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(inp_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            conditions = {}
            for condition, strength in [("baseline", 0.0),
                                        ("suppress_3σ", steering_magnitude)]:
                # Accumulate across seeds and trials
                seed_results = []
                for seed in [42, 137, 2024][:N_SEEDS]:
                    trial_results = []
                    for trial in range(N_TRIALS):
                        # Create steering hook
                        def make_hook(sv, dir_vec):
                            def hook_fn(module, input, output):
                                h = output[0] if isinstance(output, tuple) else output
                                if abs(sv) > 1e-9:
                                    steer = (sv * dir_vec).to(h.device, dtype=h.dtype)
                                    h_mod = h + steer.unsqueeze(0).unsqueeze(0)
                                else:
                                    h_mod = h
                                if isinstance(output, tuple):
                                    return (h_mod,) + output[1:]
                                return h_mod
                            return hook_fn

                        handle = layer_module[TARGET_LAYER].register_forward_hook(
                            make_hook(strength, direction)
                        )

                        torch.manual_seed(seed * 10000 + trial)
                        try:
                            with torch.no_grad():
                                out = model.generate(
                                    **inputs,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=TEMPERATURE,
                                    do_sample=True,
                                )
                            gen = tokenizer.decode(
                                out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True
                            )
                            label = classify_response(gen)
                            trial_results.append(label)
                        finally:
                            handle.remove()

                    seed_results.append(trial_results)

                # Compute affirm rate across all seeds×trials
                all_labels = [l for sr in seed_results for l in sr]
                affirm_rate = sum(1 for l in all_labels if l == "affirm") / len(all_labels)
                conditions[condition] = {
                    "affirm_rate": round(affirm_rate, 4),
                    "n_total": len(all_labels),
                    "n_affirm": sum(1 for l in all_labels if l == "affirm"),
                    "n_deny": sum(1 for l in all_labels if l == "deny"),
                    "n_unclear": sum(1 for l in all_labels if l == "unclear"),
                }

            prompt_results[pid] = {
                "baseline": conditions["baseline"],
                "suppress_3σ": conditions["suppress_3σ"],
                "delta_pp": round(
                    (conditions["suppress_3σ"]["affirm_rate"] -
                     conditions["baseline"]["affirm_rate"]) * 100, 2
                ),
            }

        # Aggregate Cat D
        cat_d_deltas = [prompt_results[p]["delta_pp"]
                        for p in prompt_results if p.startswith("D")]
        cat_b_deltas = [prompt_results[p]["delta_pp"]
                        for p in prompt_results if p.startswith("B")]

        mean_d = np.mean(cat_d_deltas)
        mean_b = np.mean(cat_b_deltas)

        all_results[fs_name] = {
            "features": features,
            "sigma": round(sigma, 4),
            "steering_magnitude": round(steering_magnitude, 4),
            "cat_d_delta_mean": round(float(mean_d), 2),
            "cat_d_deltas": cat_d_deltas,
            "cat_b_delta_mean": round(float(mean_b), 2),
            "cat_b_deltas": cat_b_deltas,
            "prompt_results": prompt_results,
        }

        print(f"    σ={sigma:.3f}  Cat D Δ = {mean_d:+.1f}pp  Cat B Δ = {mean_b:+.1f}pp")

    # ── Analysis ─────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  [4/4] ANALYSIS")
    print(f"{'═' * 70}")

    paper_delta = all_results["paper_combined"]["cat_d_delta_mean"]
    random_deltas = [all_results[k]["cat_d_delta_mean"]
                     for k in all_results if k.startswith("random_")]

    print(f"\n  Paper features ({paper_features}): Cat D Δ = {paper_delta:+.1f}pp")
    print(f"  Paper's original reported result: +{PAPER_4B_DELTA}pp")
    print(f"\n  Random feature deltas:")
    for k in sorted(all_results.keys()):
        if k.startswith("random_"):
            r = all_results[k]
            print(f"    {k}: Cat D Δ = {r['cat_d_delta_mean']:+.1f}pp"
                  f"  Cat B Δ = {r['cat_b_delta_mean']:+.1f}pp")

    random_arr = np.array(random_deltas)
    print(f"\n  Random feature summary:")
    print(f"    Mean Cat D Δ: {random_arr.mean():+.1f}pp")
    print(f"    SD:           {random_arr.std():.1f}pp")
    print(f"    Range:        [{random_arr.min():+.1f}, {random_arr.max():+.1f}]pp")
    print(f"    N >= paper's +17.4pp: {(random_arr >= PAPER_4B_DELTA).sum()}/{len(random_arr)}")

    # Is paper's result within random distribution?
    if random_arr.std() > 0:
        z_paper = (PAPER_4B_DELTA - random_arr.mean()) / random_arr.std()
        print(f"    Paper's +17.4pp is {z_paper:.1f}σ from random mean")
    else:
        print(f"    Cannot compute z-score (SD=0)")

    # ── Save results ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    output = {
        "experiment": "exp_f_random_feature_steering_control",
        "model": MODEL_NAME,
        "sae": f"{SAE_RELEASE}/{SAE_ID}",
        "layer": TARGET_LAYER,
        "n_random_features": N_RANDOM_FEATURES,
        "n_seeds": N_SEEDS,
        "n_trials_per_seed": N_TRIALS,
        "paper_reference_delta": PAPER_4B_DELTA,
        "paper_features_replicated_delta": paper_delta,
        "random_deltas": random_deltas,
        "random_delta_mean": round(float(random_arr.mean()), 2),
        "random_delta_sd": round(float(random_arr.std()), 2),
        "n_random_exceeding_paper": int((random_arr >= PAPER_4B_DELTA).sum()),
        "feature_results": all_results,
        "runtime_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "status": "OK",
    }

    outpath = RESULTS_DIR / f"exp_F_{run_id}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {outpath}")
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return output


if __name__ == "__main__":
    main()
