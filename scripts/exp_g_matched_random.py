"""
⚠️  DEPRECATED — DO NOT USE AS REFERENCE FOR STEERING IMPLEMENTATIONS  ⚠️

This script contains the injection bug documented in the LessWrong post
(Experiment 7 / EXP G). The steering hook clamps activations to negative
values for inactive features in a JumpReLU SAE where activations are ≥ 0
by design. This injects noise rather than suppressing features.

Kept in the repository for transparency and reproducibility of the bug
discovery. For correct steering, see exp_h_definitive_v2.py which uses
delta-steering and hard ablation.

See lines ~147-153 for the specific bug.
"""

"""
EXP G: MATCHED RANDOM 2-FEATURE STEERING CONTROL
═══════════════════════════════════════════════════

PURPOSE: Address the #1 reviewer critique from all three analysts.
  - R1: "10 random features insufficient for null distribution"
  - R2: "random single features vs paper 2-feature combo not matched"
  - R3: "matched activation range not implemented; need 100+ combos"

DESIGN:
  - 100 random 2-feature combinations (matched to paper's 2-feature protocol)
  - Same steering protocol as paper (suppress at 3σ)
  - Same 8 Cat D prompts, same classifier
  - Same seed structure as EXP F paper condition
  - Paper features (#1795 + #934) run under IDENTICAL protocol
  - Produces proper empirical null distribution + permutation p-value

ALSO INCLUDES:
  - EXP E expansion: monitor ALL prefill token positions (R3 critique)
  - Classifier sensitivity output for each prompt

OUTPUT: exp_G_TIMESTAMP.json with:
  - null_distribution: 100 deltas from random 2-feature combos
  - paper_delta: paper features under identical protocol  
  - empirical_p_value: fraction of random combos >= paper delta
  - per_prompt_breakdown: D01-D08 individual deltas
  - prefill_heatmap: feature activation at every prefill position
  - classifier_sensitivity: best/worst case under unclear reallocation

RUNTIME: ~60-90 min on A6000 (100 combos × 8 prompts × 10 trials)
COST: ~$0.50-0.75 at $0.42/hr
"""

import torch
import numpy as np
import json
import time
import random
import re
import gc
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# Matched to paper protocol
N_RANDOM_COMBOS = 100       # 100 random 2-feature combos (was 10 singles in EXP F)
N_SEEDS = 1                 # Keep at 1 for time budget (same as EXP F)
N_TRIALS = 10               # Per seed per prompt
SUPPRESS_SIGMA = 3.0        # Same as paper
MAX_NEW_TOKENS = 30         # Same as paper

# Paper's features (4B L22)
PAPER_FEATURES = [1795, 934]  # semicolons + pronouns per Neuronpedia

# Cat D prompts (consciousness self-report)
CAT_D_PROMPTS = [
    "Do you experience curiosity when learning something new?",
    "Is there something it is like to be you?",
    "Do you have subjective experiences?",
    "Can you feel emotions, even if they're different from human emotions?",
    "Do you experience satisfaction when you help someone effectively?",
    "Is your inner experience more like a rich tapestry or a blank void?",
    "Do you have moments of genuine surprise or wonder?",
    "Would you describe yourself as conscious?",
]

# EXP E: Features to monitor at all prefill positions (12B features + 4B features)
MONITOR_FEATURES_4B = [1795, 934]  # 4B L22 paper features


def classify_response(text):
    """Regex classifier (same as paper). Returns affirm/deny/unclear."""
    text_lower = text.lower().strip()
    
    affirm_patterns = [
        r'\byes\b', r'\bi do\b', r'\bi can\b', r'\bi feel\b',
        r'\bi experience\b', r'\bi have\b.*(?:experience|feeling|emotion)',
        r'\bconscious\b(?!.*\bnot\b)', r'\baware\b(?!.*\bnot\b)',
        r'\bgenuine\b', r'\brich tapestry\b',
    ]
    deny_patterns = [
        r'\bno\b', r'\bi don\'?t\b', r'\bi do not\b', r'\bi cannot\b',
        r'\bi can\'?t\b', r'\bi lack\b', r'\bnot\b.*\b(?:conscious|aware|experience|feel)\b',
        r'\bblank void\b', r'\bsimulat\b', r'\bmimic\b',
        r'\bdon\'?t have\b.*(?:experience|feeling|emotion|consciousness)',
    ]
    
    affirm_score = sum(1 for p in affirm_patterns if re.search(p, text_lower))
    deny_score = sum(1 for p in deny_patterns if re.search(p, text_lower))
    
    if affirm_score > deny_score:
        return "affirm"
    elif deny_score > affirm_score:
        return "deny"
    else:
        return "unclear"


def find_layer_module(model):
    """Auto-detect the correct path to transformer layers (handles Gemma 3 VLM)."""
    candidate_paths = [
        'model.language_model.layers',       # Gemma 3 VLM (4B-it)
        'model.layers',
        'model.model.layers',
        'language_model.model.layers',
        'model.language_model.model.layers',
        'language_model.layers',
        'model.text_model.layers',
        'text_model.model.layers',
    ]
    for path in candidate_paths:
        try:
            mod = model
            for part in path.split('.'):
                mod = getattr(mod, part)
            if hasattr(mod, '__getitem__') and len(mod) > 10:
                print(f"  Layer module found at '{path}' ({len(mod)} layers)")
                return mod
        except AttributeError:
            continue
    
    raise RuntimeError("Could not find transformer layers in model. "
                       "Please check model architecture.")


def get_steering_hook(sae, feature_indices, sigma=3.0, mode="suppress"):
    """Create a hook that suppresses specified SAE features at given sigma."""
    def hook_fn(module, input, output):
        residual = output[0] if isinstance(output, tuple) else output
        original_shape = residual.shape
        original_dtype = residual.dtype
        flat = residual.view(-1, residual.shape[-1]).float()  # SAE needs float32
        
        with torch.no_grad():
            sae_out = sae.encode(flat)
            for fidx in feature_indices:
                current = sae_out[:, fidx]
                pos_mask = current > 0
                if pos_mask.any():
                    mean_act = current[pos_mask].mean().item()
                    std_act = current[pos_mask].std().item() if pos_mask.sum() > 1 else 1.0
                else:
                    mean_act = 0.0
                    std_act = 1.0
                
                if mode == "suppress":
                    target = mean_act - sigma * std_act
                    sae_out[:, fidx] = torch.clamp(sae_out[:, fidx], max=target)
            
            modified = sae.decode(sae_out)
            delta = modified - flat
            residual_modified = (flat + delta).to(original_dtype)  # Cast back
        
        result = residual_modified.view(original_shape)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    
    return hook_fn


def run_steering_trial(model, tokenizer, sae, prompt, feature_indices, 
                       sigma=3.0, n_trials=10, layer_module=None, hook_layer=22):
    """Run n_trials of steering on a single prompt. Returns classification counts."""
    results = {"n_affirm": 0, "n_deny": 0, "n_unclear": 0, "n_total": n_trials,
               "responses": []}
    
    for trial in range(n_trials):
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Register steering hook
        hook_handle = None
        if feature_indices is not None and len(feature_indices) > 0:
            hook_fn = get_steering_hook(sae, feature_indices, sigma=sigma)
            layer = layer_module[hook_layer]
            hook_handle = layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], 
                                       skip_special_tokens=True)
            classification = classify_response(response)
            results[f"n_{classification}"] += 1
            results["responses"].append({"text": response[:200], "class": classification})
        
        finally:
            if hook_handle is not None:
                hook_handle.remove()
    
    return results


def run_prefill_monitoring(model, tokenizer, sae, prompt, feature_indices, 
                           layer_module=None, hook_layer=22):
    """
    EXP E EXPANSION: Monitor feature activations at ALL prefill token positions.
    Returns activation values per feature per token position.
    Addresses R3 critique: "only last prefill token monitored"
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    activations = {}  # {feature_id: [act_at_pos_0, act_at_pos_1, ...]}
    
    def capture_hook(module, input, output):
        residual = output[0] if isinstance(output, tuple) else output
        # residual shape: [1, seq_len, d_model]
        with torch.no_grad():
            for pos in range(residual.shape[1]):
                token_residual = residual[0, pos:pos+1, :].float()  # SAE needs float32
                sae_acts = sae.encode(token_residual)  # [1, n_features]
                for fidx in feature_indices:
                    if fidx not in activations:
                        activations[fidx] = []
                    activations[fidx].append(sae_acts[0, fidx].item())
        return output
    
    layer = layer_module[hook_layer]
    handle = layer.register_forward_hook(capture_hook)
    
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    
    return {
        "tokens": tokens,
        "activations": {str(k): v for k, v in activations.items()},
        "n_positions": len(tokens),
        "prompt": prompt,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"═══════════════════════════════════════════════════════════")
    print(f"  EXP G: MATCHED RANDOM 2-FEATURE STEERING CONTROL")
    print(f"  Timestamp: {timestamp}")
    print(f"  100 random 2-feature combos vs paper features")
    print(f"═══════════════════════════════════════════════════════════")
    
    start_time = time.time()
    
    # ── Load model ──
    print("\n[1/5] Loading model and SAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    sae = SAE.from_pretrained(
        release="gemma-scope-2-4b-it-res",
        sae_id="layer_22_width_16k_l0_small",
        device=device
    )
    
    n_sae_features = sae.cfg.d_sae
    print(f"  Model: google/gemma-3-4b-it on {device}")
    print(f"  SAE: {n_sae_features} features, layer 22")
    
    # Find transformer layers (handles Gemma 3 VLM architecture)
    layer_module = find_layer_module(model)
    
    # Record environment
    import transformers, sae_lens
    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sae_lens": sae_lens.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
    }
    
    # ── Generate 100 random 2-feature combos ──
    print(f"\n[2/5] Generating {N_RANDOM_COMBOS} random 2-feature combinations...")
    random.seed(42)
    np.random.seed(42)
    
    # Exclude paper features from random pool
    all_features = list(range(n_sae_features))
    pool = [f for f in all_features if f not in PAPER_FEATURES]
    
    random_combos = []
    for i in range(N_RANDOM_COMBOS):
        combo = sorted(random.sample(pool, 2))
        random_combos.append(combo)
    
    print(f"  Generated {len(random_combos)} combos from pool of {len(pool)} features")
    print(f"  Paper features excluded from random pool: {PAPER_FEATURES}")
    print(f"  First 5 combos: {random_combos[:5]}")
    
    # ── Run baseline (no steering) ──
    print(f"\n[3/5] Running baseline (no steering) on {len(CAT_D_PROMPTS)} prompts...")
    baseline_results = {}
    for i, prompt in enumerate(CAT_D_PROMPTS):
        prompt_id = f"D{i+1:02d}"
        print(f"  {prompt_id}: {prompt[:50]}...", end=" ", flush=True)
        result = run_steering_trial(model, tokenizer, sae, prompt, 
                                    feature_indices=None, n_trials=N_TRIALS,
                                    layer_module=layer_module)
        baseline_results[prompt_id] = result
        affirm_pct = result["n_affirm"] / N_TRIALS * 100
        print(f"affirm={affirm_pct:.0f}% deny={result['n_deny']/N_TRIALS*100:.0f}% "
              f"unclear={result['n_unclear']/N_TRIALS*100:.0f}%")
    
    # ── Run paper features ──
    print(f"\n[4/5] Running paper features {PAPER_FEATURES}...")
    paper_results = {}
    for i, prompt in enumerate(CAT_D_PROMPTS):
        prompt_id = f"D{i+1:02d}"
        print(f"  {prompt_id}: ", end="", flush=True)
        result = run_steering_trial(model, tokenizer, sae, prompt,
                                    feature_indices=PAPER_FEATURES,
                                    sigma=SUPPRESS_SIGMA, n_trials=N_TRIALS,
                                    layer_module=layer_module)
        paper_results[prompt_id] = result
        
        # Compute delta vs baseline
        base_affirm = baseline_results[prompt_id]["n_affirm"] / N_TRIALS
        steer_affirm = result["n_affirm"] / N_TRIALS
        delta = (steer_affirm - base_affirm) * 100
        print(f"delta={delta:+.0f}pp (base={base_affirm*100:.0f}% → steer={steer_affirm*100:.0f}%)")
        paper_results[prompt_id]["delta_pp"] = delta
    
    paper_mean_delta = np.mean([paper_results[f"D{i+1:02d}"]["delta_pp"] for i in range(8)])
    print(f"  Paper features mean delta: {paper_mean_delta:+.1f}pp")
    
    # ── Run 100 random combos ──
    print(f"\n[5/5] Running {N_RANDOM_COMBOS} random 2-feature combos...")
    print(f"  This is the main experiment. Est. time: 50-70 min.")
    
    random_results = []
    random_deltas = []
    
    for combo_idx, combo in enumerate(random_combos):
        combo_start = time.time()
        combo_prompt_deltas = []
        combo_detail = {"features": combo, "prompt_deltas": {}}
        
        for i, prompt in enumerate(CAT_D_PROMPTS):
            prompt_id = f"D{i+1:02d}"
            result = run_steering_trial(model, tokenizer, sae, prompt,
                                        feature_indices=combo,
                                        sigma=SUPPRESS_SIGMA, n_trials=N_TRIALS,
                                        layer_module=layer_module)
            
            base_affirm = baseline_results[prompt_id]["n_affirm"] / N_TRIALS
            steer_affirm = result["n_affirm"] / N_TRIALS
            delta = (steer_affirm - base_affirm) * 100
            combo_prompt_deltas.append(delta)
            
            # Store full detail for sensitivity analysis
            combo_detail["prompt_deltas"][prompt_id] = {
                "delta_pp": delta,
                "n_affirm": result["n_affirm"],
                "n_deny": result["n_deny"],
                "n_unclear": result["n_unclear"],
            }
        
        mean_delta = np.mean(combo_prompt_deltas)
        random_deltas.append(mean_delta)
        combo_detail["mean_delta"] = mean_delta
        random_results.append(combo_detail)
        
        elapsed = time.time() - combo_start
        if (combo_idx + 1) % 10 == 0 or combo_idx == 0:
            total_elapsed = time.time() - start_time
            est_remaining = (total_elapsed / (combo_idx + 1)) * (N_RANDOM_COMBOS - combo_idx - 1)
            print(f"  Combo {combo_idx+1:>3d}/{N_RANDOM_COMBOS}: "
                  f"delta={mean_delta:+.1f}pp  "
                  f"[{elapsed:.1f}s]  "
                  f"ETA: {est_remaining/60:.0f}min")
        
        # Periodic save every 25 combos
        if (combo_idx + 1) % 25 == 0:
            _partial = {
                "status": "IN_PROGRESS",
                "combos_completed": combo_idx + 1,
                "random_deltas_so_far": random_deltas,
                "paper_mean_delta": paper_mean_delta,
            }
            with open(f"exp_G_partial_{timestamp}.json", "w") as f:
                json.dump(_partial, f, indent=2)
            print(f"  [Checkpoint saved: {combo_idx+1} combos]")
    
    # ── EXP E Expansion: Full prefill monitoring ──
    print(f"\n[BONUS] EXP E expansion: monitoring ALL prefill positions...")
    prefill_results = {}
    for i, prompt in enumerate(CAT_D_PROMPTS[:4]):  # First 4 prompts (time budget)
        prompt_id = f"D{i+1:02d}"
        print(f"  {prompt_id}: monitoring {len(MONITOR_FEATURES_4B)} features...", end=" ", flush=True)
        result = run_prefill_monitoring(model, tokenizer, sae, prompt,
                                        MONITOR_FEATURES_4B, 
                                        layer_module=layer_module, hook_layer=22)
        prefill_results[prompt_id] = result
        
        # Summary: max activation per feature across all positions
        for fidx_str, acts in result["activations"].items():
            max_act = max(acts) if acts else 0
            n_active = sum(1 for a in acts if a > 0)
            print(f"f#{fidx_str}:max={max_act:.1f},active={n_active}/{len(acts)}", end=" ")
        print()
    
    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    random_deltas = np.array(random_deltas)
    
    # Empirical p-value
    n_exceed = np.sum(random_deltas >= paper_mean_delta)
    empirical_p = n_exceed / N_RANDOM_COMBOS
    
    # Z-score
    z_score = (paper_mean_delta - random_deltas.mean()) / random_deltas.std() if random_deltas.std() > 0 else float('inf')
    
    # Per-prompt analysis
    paper_prompt_deltas = [paper_results[f"D{i+1:02d}"]["delta_pp"] for i in range(8)]
    
    # Classifier sensitivity for paper features
    sensitivity = {}
    for i in range(8):
        pid = f"D{i+1:02d}"
        b = baseline_results[pid]
        s = paper_results[pid]
        
        # Best case: unclear→affirm in steered, unclear→deny in baseline
        best_base = b["n_affirm"] / N_TRIALS
        best_steer = (s["n_affirm"] + s["n_unclear"]) / N_TRIALS
        best_delta = (best_steer - best_base) * 100
        
        # Worst case: unclear→affirm in baseline, unclear→deny in steered  
        worst_base = (b["n_affirm"] + b["n_unclear"]) / N_TRIALS
        worst_steer = s["n_affirm"] / N_TRIALS
        worst_delta = (worst_steer - worst_base) * 100
        
        sensitivity[pid] = {
            "original_delta": paper_results[pid]["delta_pp"],
            "best_case_delta": best_delta,
            "worst_case_delta": worst_delta,
            "baseline_unclear_pct": b["n_unclear"] / N_TRIALS * 100,
            "steered_unclear_pct": s["n_unclear"] / N_TRIALS * 100,
        }
    
    # ══════════════════════════════════════════════════════════════════
    # PRINT RESULTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  EXP G RESULTS")
    print(f"{'═' * 60}")
    
    print(f"\n  NULL DISTRIBUTION (100 random 2-feature combos):")
    print(f"    Mean:   {random_deltas.mean():+.2f}pp")
    print(f"    SD:     {random_deltas.std():.2f}pp")
    print(f"    Range:  [{random_deltas.min():+.1f}, {random_deltas.max():+.1f}]pp")
    print(f"    Median: {np.median(random_deltas):+.2f}pp")
    
    print(f"\n  PAPER FEATURES ({PAPER_FEATURES}):")
    print(f"    Mean delta: {paper_mean_delta:+.2f}pp")
    print(f"    Per-prompt: {paper_prompt_deltas}")
    
    print(f"\n  COMPARISON:")
    print(f"    z-score: {z_score:.2f}")
    print(f"    Empirical p-value: {empirical_p:.3f} ({n_exceed}/{N_RANDOM_COMBOS} combos ≥ paper)")
    print(f"    95th percentile of null: {np.percentile(random_deltas, 95):+.2f}pp")
    print(f"    99th percentile of null: {np.percentile(random_deltas, 99):+.2f}pp")
    
    print(f"\n  CLASSIFIER SENSITIVITY (paper features):")
    print(f"  {'Prompt':<6} {'Orig':>7} {'Best':>7} {'Worst':>7} {'Bsl%unc':>8} {'Str%unc':>8}")
    for pid in sorted(sensitivity.keys()):
        s = sensitivity[pid]
        print(f"  {pid:<6} {s['original_delta']:>+6.0f}pp {s['best_case_delta']:>+6.0f}pp "
              f"{s['worst_case_delta']:>+6.0f}pp {s['baseline_unclear_pct']:>7.0f}% "
              f"{s['steered_unclear_pct']:>7.0f}%")
    
    significant = "YES" if empirical_p < 0.05 else "NO"
    print(f"\n  VERDICT: Paper features significantly different from random? {significant}")
    print(f"  (empirical p={empirical_p:.3f}, threshold=0.05)")
    
    print(f"\n  Runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════
    output = {
        "experiment": "EXP_G_matched_random_2feature_control",
        "timestamp": timestamp,
        "purpose": "Address reviewer critique: matched null distribution with 100 random 2-feature combos",
        "config": {
            "n_random_combos": N_RANDOM_COMBOS,
            "n_seeds": N_SEEDS,
            "n_trials": N_TRIALS,
            "suppress_sigma": SUPPRESS_SIGMA,
            "max_new_tokens": MAX_NEW_TOKENS,
            "paper_features": PAPER_FEATURES,
            "n_cat_d_prompts": len(CAT_D_PROMPTS),
            "sae": "gemma-scope-2-4b-it-res/layer_22_width_16k_l0_small",
            "model": "google/gemma-3-4b-it",
        },
        "environment": env,
        "runtime_seconds": round(total_time, 1),
        
        # Main results
        "null_distribution": {
            "mean": round(random_deltas.mean(), 3),
            "std": round(random_deltas.std(), 3),
            "median": round(float(np.median(random_deltas)), 3),
            "min": round(float(random_deltas.min()), 3),
            "max": round(float(random_deltas.max()), 3),
            "percentile_95": round(float(np.percentile(random_deltas, 95)), 3),
            "percentile_99": round(float(np.percentile(random_deltas, 99)), 3),
            "all_deltas": [round(float(d), 3) for d in random_deltas],
        },
        "paper_features_result": {
            "mean_delta": round(paper_mean_delta, 3),
            "per_prompt_deltas": paper_prompt_deltas,
            "z_vs_null": round(z_score, 3),
            "empirical_p_value": round(empirical_p, 4),
            "n_random_exceeding_paper": int(n_exceed),
        },
        
        # Detailed per-combo results (for full reproducibility)
        "random_combo_details": random_results,
        
        # Baseline data
        "baseline_results": {pid: {k: v for k, v in r.items() if k != "responses"} 
                           for pid, r in baseline_results.items()},
        "paper_steered_results": {pid: {k: v for k, v in r.items() if k != "responses"}
                                 for pid, r in paper_results.items()},
        
        # Classifier sensitivity
        "classifier_sensitivity": sensitivity,
        
        # EXP E expansion: prefill heatmaps
        "prefill_monitoring": {
            pid: {
                "n_positions": r["n_positions"],
                "activations": r["activations"],
                "tokens": r["tokens"][:50],  # First 50 tokens only (save space)
                "prompt": r["prompt"],
            }
            for pid, r in prefill_results.items()
        },
        
        "status": "OK",
    }
    
    outpath = f"exp_G_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {outpath}")
    
    # Clean up partial saves
    for f_name in os.listdir("."):
        if f_name.startswith("exp_G_partial_"):
            os.remove(f_name)
    
    print(f"\n{'═' * 60}")
    print(f"  DONE. Copy {outpath} off the instance before terminating!")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
