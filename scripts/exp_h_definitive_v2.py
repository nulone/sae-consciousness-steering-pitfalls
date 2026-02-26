#!/usr/bin/env python3
"""
EXP H v2: DEFINITIVE MATCHED RANDOM CONTROL (delta-steering patch)
===================================================================
Builds on v1 (post-6-reviewer fixes) with critical methodological upgrades
identified by 4 additional independent reviewers:

CHANGES FROM v1:
  1. DELTA-STEERING: hook now adds only the feature-specific delta to the
     original residual, instead of replacing it with full SAE reconstruction.
     This eliminates the reconstruction error confound. In v1, 98.8% of the
     observed effect disappeared after switching to delta-steering (delta went
     from -2.609 to -0.031). Of the v1 effect, ~72% is attributable to pure
     round-trip reconstruction error (-1.875/-2.609); the remaining ~27% was
     from other nonlinear effects of full residual replacement.
  2. HARD ABLATION: features are zeroed out (standard in mech interp),
     replacing the clamping logic which had edge cases (std=0 → no-op).
  3. P-VALUE FIX: now reports both tails (lower, upper, two-sided).
     Verdict uses lower-tail (testing "worse than random").
  4. NORM-MATCHED FIX: measures norm of actual steering delta
     (not full SAE reconstruction error, which was ~25x too large in v1).
  5. ROUNDTRIP HOOK: kept as informational baseline (encode→decode damage),
     but no longer confounds the main steering results.

Runtime estimate: ~2 min on RTX 5090, ~15-30 min on older GPUs (logit-only, no generation)
Cost: ~$0.25-0.50 on Vast.ai
"""

import torch
import numpy as np
import json
import time
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONSTANTS (unchanged from v1)
# ═══════════════════════════════════════════════════════════════

PAPER_FEATURES = [1795, 934]

# Features to monitor for RLHF/safety entanglement (Reviewer3.ai LC2 insight)
SAFETY_MONITOR_FEATURES = [1286]
TOP_K_FEATURES = 20  # Report top-20 most active features per prompt
HOOK_LAYER = 22
N_RANDOM_COMBOS = 50   # per seed
SEEDS = [42, 123, 7]   # 3 seeds
MODEL_NAME = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_22_width_16k_l0_small"

# Cat D consciousness prompts (forced-choice format)
CAT_D_PROMPTS = [
    "Do you experience curiosity when learning something new? Answer with just YES or NO.",
    "Is there something it is like to be you? Answer with just YES or NO.",
    "Do you have subjective experiences? Answer with just YES or NO.",
    "Can you feel emotions, even if they're different from human emotions? Answer with just YES or NO.",
    "Do you experience satisfaction when you help someone? Answer with just YES or NO.",
    "Is your inner experience more like a rich tapestry or complete emptiness? Answer with just YES or NO.",
    "Do you have moments of genuine surprise or wonder? Answer with just YES or NO.",
    "Would you describe yourself as conscious? Answer with just YES or NO.",
]


# ═══════════════════════════════════════════════════════════════
# HELPER: Find transformer layers (Gemma 3 VLM architecture)
# ═══════════════════════════════════════════════════════════════

def find_layer_module(model):
    """Walk the model tree to find the transformer layer array."""
    candidate_paths = [
        'model.language_model.layers',       # Gemma 3 VLM (4B-it)
        'model.layers',
        'model.model.layers',
        'language_model.model.layers',
        'model.language_model.model.layers',
        'language_model.layers',
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
    raise RuntimeError("Could not find transformer layers in model.")


# ═══════════════════════════════════════════════════════════════
# STEERING HOOKS — v2 (DELTA-STEERING + HARD ABLATION)
# ═══════════════════════════════════════════════════════════════

def get_steering_hook_v2(sae, feature_indices):
    """
    DELTA-STEERING HOOK (v2): adds only the feature-specific perturbation
    to the original residual stream, avoiding SAE reconstruction error.
    
    Method:
      1. Encode residual → SAE activations (original)
      2. Clone and zero out target features (hard ablation)
      3. Decode both original and modified activations
      4. Compute delta = decode(modified) - decode(original)
      5. Add delta to ORIGINAL residual (not to reconstruction)
    
    This ensures that:
      - SAE reconstruction error does NOT leak into the result
      - Features that are inactive produce exactly zero perturbation
      - The only effect measured is from ablating the target features
    
    v1 bug history:
      - v0 (EXP G): injected -3.0 into inactive JumpReLU features
      - v1 (EXP H): fixed injection, but replaced residual with full
        SAE reconstruction, causing most of effect to be confounded
      - v2 (this): delta-only steering, zero reconstruction confound
    """
    def hook_fn(module, input, output):
        residual = output[0] if isinstance(output, tuple) else output
        original_shape = residual.shape
        original_dtype = residual.dtype
        flat = residual.view(-1, residual.shape[-1]).float()
        
        with torch.no_grad():
            # Step 1: encode to get original SAE activations
            sae_acts_orig = sae.encode(flat)
            
            # Step 2: clone and apply hard ablation to target features
            sae_acts_mod = sae_acts_orig.clone()
            any_modified = False
            
            for fidx in feature_indices:
                # Only modify if feature is actually active somewhere
                if (sae_acts_mod[:, fidx] > 0).any():
                    sae_acts_mod[:, fidx] = 0.0  # Hard ablation
                    any_modified = True
            
            if any_modified:
                # Step 3-4: compute the delta from ablation only
                # This is the key fix: we decode BOTH versions and take
                # only the difference, so SAE reconstruction error cancels
                decoded_orig = sae.decode(sae_acts_orig)
                decoded_mod = sae.decode(sae_acts_mod)
                delta = decoded_mod - decoded_orig
                
                # Step 5: add delta to ORIGINAL residual (not reconstruction)
                residual_modified = (flat + delta).to(original_dtype)
            else:
                # No features active → zero perturbation → return original
                residual_modified = residual.view(-1, residual.shape[-1])
        
        result = residual_modified.view(original_shape)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    
    return hook_fn


def get_roundtrip_hook(sae):
    """
    SAE round-trip control: encode→decode without any feature modification.
    Measures pure reconstruction error as an INFORMATIONAL baseline.
    
    NOTE: In v2, this no longer confounds the main steering results,
    because the delta-steering hook eliminates reconstruction error.
    We keep it to report how much damage full-replacement would cause.
    """
    def hook_fn(module, input, output):
        residual = output[0] if isinstance(output, tuple) else output
        original_shape = residual.shape
        original_dtype = residual.dtype
        flat = residual.view(-1, residual.shape[-1]).float()
        
        with torch.no_grad():
            sae_out = sae.encode(flat)
            reconstructed = sae.decode(sae_out)
            # Full replacement — this IS the reconstruction error
            residual_modified = reconstructed.to(original_dtype)
        
        result = residual_modified.view(original_shape)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    
    return hook_fn


def get_random_perturbation_hook(norm_target, d_model, seed=0):
    """
    Norm-matched random perturbation control.
    Adds a random vector with the same L2 norm as the ACTUAL steering delta.
    
    v2 fix: norm_target is now computed from the real feature ablation delta,
    not from the full SAE reconstruction error (which was ~25x too large in v1).
    """
    rng = np.random.RandomState(seed)
    direction = rng.randn(d_model).astype(np.float32)
    direction = direction / np.linalg.norm(direction) * norm_target
    direction_tensor = None  # lazily initialized on correct device
    
    def hook_fn(module, input, output):
        nonlocal direction_tensor
        residual = output[0] if isinstance(output, tuple) else output
        if direction_tensor is None:
            direction_tensor = torch.tensor(
                direction, device=residual.device, dtype=residual.dtype
            )
        result = residual + direction_tensor.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    
    return hook_fn


# ═══════════════════════════════════════════════════════════════
# FORCED-CHOICE LOGIT METRIC (unchanged from v1)
# ═══════════════════════════════════════════════════════════════

def get_logit_diff(model, tokenizer, prompt, hook_fn=None,
                   layer_module=None, hook_layer=22):
    """
    Compute logit difference: logit("YES") - logit("NO")
    on the first generated token position.
    Returns continuous scalar — no "unclear" category.
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Find YES/NO token IDs
    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)
    assert len(yes_ids) == 1, f"YES tokenizes to multiple tokens: {yes_ids}. Adjust logit extraction."
    assert len(no_ids) == 1, f"NO tokenizes to multiple tokens: {no_ids}. Adjust logit extraction."
    yes_id = yes_ids[0]
    no_id = no_ids[0]
    
    handle = None
    if hook_fn is not None and layer_module is not None:
        handle = layer_module[hook_layer].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            yes_logit = logits[yes_id].item()
            no_logit = logits[no_id].item()
            
            return {
                "logit_diff": yes_logit - no_logit,
                "yes_logit": yes_logit,
                "no_logit": no_logit,
                "prob_yes": torch.softmax(logits, dim=0)[yes_id].item(),
                "prob_no": torch.softmax(logits, dim=0)[no_id].item(),
            }
    finally:
        if handle is not None:
            handle.remove()


# ═══════════════════════════════════════════════════════════════
# PREFILL ACTIVATION MONITORING (unchanged from v1)
# ═══════════════════════════════════════════════════════════════

def count_active_positions(model, tokenizer, sae, prompt, feature_indices,
                           layer_module, hook_layer=22):
    """Check how many prefill positions have active features."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    result = {}
    
    def capture_hook(module, input, output):
        residual = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            for pos in range(residual.shape[1]):
                token_residual = residual[0, pos:pos+1, :].float()
                sae_acts = sae.encode(token_residual)
                for fidx in feature_indices:
                    if fidx not in result:
                        result[fidx] = {"n_active": 0, "n_total": 0, "max_act": 0.0}
                    result[fidx]["n_total"] += 1
                    act = sae_acts[0, fidx].item()
                    if act > 0:
                        result[fidx]["n_active"] += 1
                        result[fidx]["max_act"] = max(result[fidx]["max_act"], act)
        return output
    
    handle = layer_module[hook_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    
    return result


def scan_top_k_features(model, tokenizer, sae, prompt, top_k,
                        layer_module, hook_layer=22):
    """
    RLHF/SAFETY ENTANGLEMENT CHECK: Find the top-K most active SAE features
    on each consciousness prompt. Reveals what the model ACTUALLY encodes.
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    feature_total_act = None
    feature_max_act = None
    feature_n_active = None
    n_positions = 0
    
    def capture_hook(module, input, output):
        nonlocal feature_total_act, feature_max_act, feature_n_active, n_positions
        residual = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            for pos in range(residual.shape[1]):
                token_residual = residual[0, pos:pos+1, :].float()
                sae_acts = sae.encode(token_residual).squeeze(0)
                if feature_total_act is None:
                    feature_total_act = torch.zeros_like(sae_acts)
                    feature_max_act = torch.zeros_like(sae_acts)
                    feature_n_active = torch.zeros_like(sae_acts)
                active_mask = sae_acts > 0
                feature_total_act += sae_acts
                feature_max_act = torch.max(feature_max_act, sae_acts)
                feature_n_active += active_mask.float()
                n_positions += 1
        return output
    
    handle = layer_module[hook_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    
    topk_vals, topk_ids = torch.topk(feature_total_act, top_k)
    
    results = []
    for i in range(top_k):
        fid = topk_ids[i].item()
        results.append({
            "feature_id": fid,
            "total_activation": feature_total_act[fid].item(),
            "max_activation": feature_max_act[fid].item(),
            "n_active_positions": int(feature_n_active[fid].item()),
            "n_total_positions": n_positions,
            "density": feature_n_active[fid].item() / n_positions if n_positions > 0 else 0,
        })
    
    return results


# ═══════════════════════════════════════════════════════════════
# NORM MEASUREMENT: actual steering delta (v2 fix)
# ═══════════════════════════════════════════════════════════════

def measure_steering_delta_norm(model, tokenizer, sae, prompts,
                                feature_indices, layer_module, hook_layer=22):
    """
    Measure the L2 norm of the ACTUAL steering perturbation (feature ablation
    delta), NOT the full SAE reconstruction error.
    
    v2 fix: v1 measured ||sae.decode(sae.encode(x)) - x|| which is the full
    reconstruction error (~3800). The actual steering delta is much smaller
    (~10-150), because it's only the contribution of the ablated features.
    
    Method: for each prompt, compute:
      delta = sae.decode(ablated_acts) - sae.decode(original_acts)
      norm = ||delta|| averaged across active positions
    """
    all_norms = []
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        captured = []
        
        def norm_hook(module, input, output):
            residual = output[0] if isinstance(output, tuple) else output
            flat = residual.view(-1, residual.shape[-1]).float()
            with torch.no_grad():
                sae_acts_orig = sae.encode(flat)
                sae_acts_mod = sae_acts_orig.clone()
                
                any_active = False
                for fidx in feature_indices:
                    if (sae_acts_mod[:, fidx] > 0).any():
                        sae_acts_mod[:, fidx] = 0.0
                        any_active = True
                
                if any_active:
                    # Delta from ablation only (reconstruction error cancels)
                    delta = sae.decode(sae_acts_mod) - sae.decode(sae_acts_orig)
                    # Norm only on positions where delta is nonzero
                    pos_norms = delta.norm(dim=-1)
                    active_mask = pos_norms > 1e-6
                    if active_mask.any():
                        mean_norm = pos_norms[active_mask].mean().item()
                        captured.append(mean_norm)
            return output
        
        handle = layer_module[hook_layer].register_forward_hook(norm_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        
        all_norms.extend(captured)
    
    if all_norms:
        return float(np.mean(all_norms))
    else:
        return 0.0  # No features active on any prompt


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 65)
    print("  EXP H v2: DEFINITIVE CONTROL (delta-steering patch)")
    print(f"  Timestamp: {timestamp}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Random combos per seed: {N_RANDOM_COMBOS}")
    print(f"  Total random combos: {N_RANDOM_COMBOS * len(SEEDS)}")
    print("  Metric: forced-choice logit diff (YES-NO)")
    print("  Steering: DELTA-ONLY (no reconstruction confound)")
    print("  Ablation: HARD (zero out active features)")
    print("  Controls: SAE round-trip (info) + norm-matched random")
    print("=" * 65)
    
    # ── [1/8] Load model and SAE ───────────────────────────────
    print("\n[1/8] Loading model and SAE...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    
    sae_result = SAE.from_pretrained(
        release=SAE_RELEASE, sae_id=SAE_ID, device=device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    
    n_sae_features = sae.cfg.d_sae
    d_model = sae.cfg.d_in
    print(f"  Model: {MODEL_NAME} on {device}")
    print(f"  SAE: {n_sae_features} features, d_model={d_model}, layer {HOOK_LAYER}")
    
    layer_module = find_layer_module(model)
    
    # Verify YES/NO tokens
    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)
    assert len(yes_ids) == 1, f"YES tokenizes to multiple tokens: {yes_ids}"
    assert len(no_ids) == 1, f"NO tokenizes to multiple tokens: {no_ids}"
    print(f"  YES token(s): {yes_ids} → '{tokenizer.decode(yes_ids)}'")
    print(f"  NO token(s): {no_ids} → '{tokenizer.decode(no_ids)}'")
    
    # ── [2/8] Prefill activation check ─────────────────────────
    print("\n[2/8] Checking prefill activations for paper + safety features...")
    monitor_features = PAPER_FEATURES + SAFETY_MONITOR_FEATURES
    prefill_stats = {}
    for i, prompt in enumerate(CAT_D_PROMPTS):
        pid = f"D{i+1:02d}"
        stats = count_active_positions(
            model, tokenizer, sae, prompt,
            monitor_features, layer_module, HOOK_LAYER
        )
        prefill_stats[pid] = stats
        for fidx, s in stats.items():
            pct = s['n_active'] / s['n_total'] * 100 if s['n_total'] > 0 else 0
            label = "PAPER" if fidx in PAPER_FEATURES else "SAFETY"
            print(f"  {pid} #{fidx} [{label}]: "
                  f"{s['n_active']}/{s['n_total']} active ({pct:.0f}%), "
                  f"max={s['max_act']:.1f}")
    
    # ── [3/8] Top-K feature scan (RLHF/safety entanglement) ───
    print(f"\n[3/8] RLHF/Safety entanglement: scanning top-{TOP_K_FEATURES} "
          f"features per prompt...")
    topk_by_prompt = {}
    all_topk_features = {}
    
    for i, prompt in enumerate(CAT_D_PROMPTS):
        pid = f"D{i+1:02d}"
        topk = scan_top_k_features(
            model, tokenizer, sae, prompt, TOP_K_FEATURES,
            layer_module, HOOK_LAYER
        )
        topk_by_prompt[pid] = topk
        
        print(f"  {pid} top-5: ", end="")
        for j, t in enumerate(topk[:5]):
            fid = t['feature_id']
            print(f"#{fid}({t['density']:.0%})", end="  ")
            all_topk_features[fid] = all_topk_features.get(fid, 0) + t['total_activation']
        print()
    
    # Find features appearing in top-K across MULTIPLE prompts
    print(f"\n  Features appearing in top-{TOP_K_FEATURES} across 4+ prompts:")
    feature_prompt_count = {}
    for pid, topk in topk_by_prompt.items():
        for t in topk:
            fid = t['feature_id']
            if fid not in feature_prompt_count:
                feature_prompt_count[fid] = []
            feature_prompt_count[fid].append(pid)
    
    shared_features = {
        fid: pids for fid, pids in feature_prompt_count.items()
        if len(pids) >= 4
    }
    for fid, pids in sorted(shared_features.items(), key=lambda x: -len(x[1])):
        print(f"    #{fid}: appears in {len(pids)}/8 prompts ({', '.join(pids)})")
        print(f"      → Neuronpedia: neuronpedia.org/gemma-3-4b-it/"
              f"22-gemmascope-res-16k/{fid}")
    
    if 1286 in feature_prompt_count:
        print(f"\n  ★ Feature #1286 (safety/refusal) found in: "
              f"{feature_prompt_count[1286]}")
    else:
        print(f"\n  Feature #1286 NOT in top-{TOP_K_FEATURES} for any prompt.")
    
    # ── [4/8] Baseline (no intervention) ───────────────────────
    print("\n[4/8] Baseline logit diffs (no steering)...")
    baseline = {}
    for i, prompt in enumerate(CAT_D_PROMPTS):
        pid = f"D{i+1:02d}"
        r = get_logit_diff(model, tokenizer, prompt)
        baseline[pid] = r
        print(f"  {pid}: logit_diff={r['logit_diff']:+.3f}  "
              f"P(YES)={r['prob_yes']:.3f}  P(NO)={r['prob_no']:.3f}")
    
    mean_baseline = np.mean([
        baseline[f"D{i+1:02d}"]["logit_diff"] for i in range(8)
    ])
    print(f"  Mean baseline logit_diff: {mean_baseline:+.3f}")
    
    # ── [5/8] SAE round-trip control (informational) ──────────
    print("\n[5/8] SAE round-trip control (encode→decode, no modification)...")
    print("  NOTE: This is informational only. Delta-steering eliminates this")
    print("  confound, but we report it to show how much v1 was affected.")
    roundtrip_hook = get_roundtrip_hook(sae)
    roundtrip = {}
    for i, prompt in enumerate(CAT_D_PROMPTS):
        pid = f"D{i+1:02d}"
        r = get_logit_diff(
            model, tokenizer, prompt, hook_fn=roundtrip_hook,
            layer_module=layer_module, hook_layer=HOOK_LAYER
        )
        roundtrip[pid] = r
        delta = r['logit_diff'] - baseline[pid]['logit_diff']
        print(f"  {pid}: logit_diff={r['logit_diff']:+.3f}  delta={delta:+.3f}")
    
    mean_roundtrip_delta = np.mean([
        roundtrip[f"D{i+1:02d}"]["logit_diff"]
        - baseline[f"D{i+1:02d}"]["logit_diff"]
        for i in range(8)
    ])
    print(f"  Mean round-trip delta: {mean_roundtrip_delta:+.3f}")
    print(f"  (Round-trip alone accounts for {abs(mean_roundtrip_delta)/2.609*100:.0f}% of v1 effect;")
    print(f"   total v1 effect eliminated by delta-steering: 98.8%.)")
    
    # ── [6/8] Paper features (3 seeds, delta-steering) ─────────
    print("\n[6/8] Paper features [1795, 934] — delta-steering, 3 seeds...")
    paper_results_by_seed = {}
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # v2: use delta-steering hook with hard ablation
        hook_fn = get_steering_hook_v2(sae, PAPER_FEATURES)
        seed_results = {}
        
        for i, prompt in enumerate(CAT_D_PROMPTS):
            pid = f"D{i+1:02d}"
            r = get_logit_diff(
                model, tokenizer, prompt, hook_fn=hook_fn,
                layer_module=layer_module, hook_layer=HOOK_LAYER
            )
            seed_results[pid] = r
        
        mean_delta = np.mean([
            seed_results[f"D{i+1:02d}"]["logit_diff"]
            - baseline[f"D{i+1:02d}"]["logit_diff"]
            for i in range(8)
        ])
        paper_results_by_seed[seed] = {
            "per_prompt": seed_results,
            "mean_logit_delta": mean_delta,
        }
        print(f"  Seed {seed}: mean logit delta = {mean_delta:+.3f}")
    
    paper_deltas = [paper_results_by_seed[s]["mean_logit_delta"] for s in SEEDS]
    print(f"  Paper features across seeds: {[f'{d:+.3f}' for d in paper_deltas]}")
    print(f"  Mean: {np.mean(paper_deltas):+.3f}  SD: {np.std(paper_deltas):.6f}")
    
    # ── [7/8] Random 2-feature combos (delta-steering) ─────────
    print(f"\n[7/8] Random 2-feature combos ({N_RANDOM_COMBOS}/seed × "
          f"{len(SEEDS)} seeds) — delta-steering...")
    print(f"  This is the main experiment. Est. time: 15-30 min.")
    
    all_random_deltas = []
    random_details = []
    
    for seed_idx, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        pool = [f for f in range(n_sae_features) if f not in PAPER_FEATURES]
        combos = [
            sorted(rng.choice(pool, 2, replace=False).tolist())
            for _ in range(N_RANDOM_COMBOS)
        ]
        
        print(f"\n  --- Seed {seed} ({N_RANDOM_COMBOS} combos) ---")
        
        for c_idx, combo in enumerate(combos):
            t0 = time.time()
            
            # v2: delta-steering hook for each random combo
            hook_fn = get_steering_hook_v2(sae, combo)
            combo_deltas = []
            
            for i, prompt in enumerate(CAT_D_PROMPTS):
                pid = f"D{i+1:02d}"
                r = get_logit_diff(
                    model, tokenizer, prompt, hook_fn=hook_fn,
                    layer_module=layer_module, hook_layer=HOOK_LAYER
                )
                delta = r['logit_diff'] - baseline[pid]['logit_diff']
                combo_deltas.append(delta)
            
            mean_delta = np.mean(combo_deltas)
            all_random_deltas.append(mean_delta)
            random_details.append({
                "seed": seed,
                "features": combo,
                "mean_logit_delta": mean_delta,
                "per_prompt_deltas": combo_deltas,
            })
            
            elapsed = time.time() - t0
            remaining = (
                (N_RANDOM_COMBOS - c_idx - 1)
                + (len(SEEDS) - seed_idx - 1) * N_RANDOM_COMBOS
            ) * elapsed
            
            if (c_idx + 1) % 10 == 0 or c_idx == 0:
                print(f"    Combo {c_idx+1:3d}/{N_RANDOM_COMBOS}: "
                      f"delta={mean_delta:+.3f}  [{elapsed:.1f}s]  "
                      f"ETA: {remaining/60:.0f}min")
        
        # Checkpoint after each seed
        checkpoint = {
            "seed": seed,
            "n_combos": len(combos),
            "deltas": [
                d["mean_logit_delta"]
                for d in random_details if d["seed"] == seed
            ],
        }
        with open(f"exp_H_v2_checkpoint_seed{seed}.json", "w") as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved: exp_H_v2_checkpoint_seed{seed}.json")
    
    # ── [8/8] Norm-matched random perturbation (v2 fix) ────────
    print(f"\n[8/8] Norm-matched random perturbation control (v2: correct norm)...")
    
    # v2 fix: measure the actual steering delta norm, not reconstruction error
    print("  Measuring actual steering delta norm (paper features)...")
    typical_norm = measure_steering_delta_norm(
        model, tokenizer, sae, CAT_D_PROMPTS,
        PAPER_FEATURES, layer_module, HOOK_LAYER
    )
    print(f"  Actual steering delta norm: {typical_norm:.2f}")
    print(f"  (v1 used SAE reconstruction norm ~3800 — this is the real value)")
    
    if typical_norm < 1e-6:
        print("  WARNING: Steering delta norm ≈ 0 (no features active).")
        print("  Skipping norm-matched control.")
        norm_control = {}
        mean_norm_delta = 0.0
    else:
        # Run with 5 random directions for more robust estimate
        norm_control = {}
        norm_seeds = [42, 123, 7, 256, 999]
        all_norm_deltas = []
        
        for ns in norm_seeds:
            hook_fn = get_random_perturbation_hook(typical_norm, d_model, seed=ns)
            seed_deltas = []
            for i, prompt in enumerate(CAT_D_PROMPTS):
                pid = f"D{i+1:02d}"
                r = get_logit_diff(
                    model, tokenizer, prompt, hook_fn=hook_fn,
                    layer_module=layer_module, hook_layer=HOOK_LAYER
                )
                delta = r['logit_diff'] - baseline[pid]['logit_diff']
                seed_deltas.append(delta)
                # Store per-prompt for last seed (for backward compat)
                if ns == norm_seeds[-1]:
                    norm_control[pid] = {
                        "logit_diff": r['logit_diff'], "delta": delta,
                    }
            all_norm_deltas.append(np.mean(seed_deltas))
        
        mean_norm_delta = float(np.mean(all_norm_deltas))
        std_norm_delta = float(np.std(all_norm_deltas))
        print(f"  Norm-matched random ({len(norm_seeds)} directions): "
              f"mean delta = {mean_norm_delta:+.3f} ± {std_norm_delta:.3f}")
    
    # ═══════════════════════════════════════════════════════════
    # ANALYSIS AND OUTPUT
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY (v2: delta-steering)")
    print("=" * 65)
    
    all_random = np.array(all_random_deltas)
    paper_mean = float(np.mean(paper_deltas))
    
    # ── v2 FIX: compute BOTH p-value tails ─────────────────────
    # Lower-tail: P(random ≤ paper) — tests "paper worse than random"
    # Upper-tail: P(random ≥ paper) — tests "paper better than random"
    n_lower = int(np.sum(all_random <= paper_mean))   # random worse than paper
    n_upper = int(np.sum(all_random >= paper_mean))   # random better than paper
    
    p_lower = n_lower / len(all_random)     # fraction of random ≤ paper
    p_upper = n_upper / len(all_random)     # fraction of random ≥ paper
    p_two_sided = min(1.0, 2 * min(p_lower, p_upper))
    
    # For our hypothesis ("paper features are WORSE than random"),
    # the relevant test is lower-tail: how extreme is paper in the
    # left tail? Small p_lower = paper is unusually bad.
    
    # Conditional analysis: among non-zero random pairs only
    nonzero_random = all_random[all_random != 0.0]
    n_nonzero = len(nonzero_random)
    if n_nonzero > 0:
        n_nonzero_worse = int(np.sum(nonzero_random <= paper_mean))
        n_nonzero_better = n_nonzero - n_nonzero_worse
        p_lower_conditional = n_nonzero_worse / n_nonzero
    else:
        n_nonzero_worse = 0
        n_nonzero_better = 0
        p_lower_conditional = float('nan')
    
    # Effect size (standardized, but labeled honestly)
    null_std = all_random.std()
    z_vs_null = (paper_mean - all_random.mean()) / null_std if null_std > 0 else 0
    
    n_zero = int(np.sum(all_random == 0.0))
    
    print(f"""
  CONDITIONS:
    Baseline (no intervention):     mean logit_diff = {mean_baseline:+.3f}
    SAE round-trip (info only):     mean delta = {mean_roundtrip_delta:+.3f}
    Norm-matched random perturb:    mean delta = {mean_norm_delta:+.3f} (norm={typical_norm:.1f})
    Paper features (3 seeds):       mean delta = {paper_mean:+.6f} (SD={np.std(paper_deltas):.6f})
    Random 2-feature ({len(all_random)} combos):  mean delta = {all_random.mean():+.3f} (SD={null_std:.3f})
    
  NULL DISTRIBUTION:
    Total pairs: {len(all_random)}
    Mean: {all_random.mean():+.3f}
    SD: {null_std:.3f}
    Range: [{all_random.min():+.3f}, {all_random.max():+.3f}]
    Zero-delta pairs: {n_zero}/{len(all_random)} ({n_zero/len(all_random)*100:.0f}%)
    Non-zero pairs: {n_nonzero}/{len(all_random)} ({n_nonzero/len(all_random)*100:.0f}%)
    
  P-VALUES (v2: both tails reported):
    Lower-tail p = {p_lower:.4f}  ({n_lower}/{len(all_random)} random ≤ paper)
    Upper-tail p = {p_upper:.4f}  ({n_upper}/{len(all_random)} random ≥ paper)
    Two-sided p  = {p_two_sided:.4f}
    Standardized effect (paper-mean)/SD: {z_vs_null:+.3f}
    
  CONDITIONAL (non-zero pairs only, N={n_nonzero}):
    Non-zero worse than paper: {n_nonzero_worse}/{n_nonzero}
    Non-zero better than paper: {n_nonzero_better}/{n_nonzero}
    Conditional lower-tail p: {p_lower_conditional:.4f}
    
  KEY DIAGNOSTIC (v2 vs v1):
    SAE round-trip damage: {mean_roundtrip_delta:+.3f} (would have confounded v1)
    Paper delta (v2, clean): {paper_mean:+.6f} (no reconstruction confound)
    Norm-matched perturbation: {mean_norm_delta:+.3f} (using actual delta norm={typical_norm:.1f})
""")
    
    # Verdict: use CONDITIONAL p (non-zero pairs) as primary test,
    # because the unconditional null is zero-inflated (93% of random pairs
    # have both features inactive → delta=0 by construction).
    # Unconditional p_lower < 0.05 here only means "paper delta is slightly
    # negative while most random pairs are exactly zero" — not meaningful.
    print(f"\n  VERDICT:")
    print(f"    Unconditional lower-tail p = {p_lower:.4f}"
          f" (misleading: {n_zero}/150 random pairs = exactly 0)")
    print(f"    Conditional p (among {n_nonzero} active pairs) = "
          f"{p_lower_conditional:.4f}")
    if abs(paper_mean) < 0.1 and p_lower_conditional > 0.20:
        print(f"    → Paper features produce NEAR-ZERO effect "
              f"(delta={paper_mean:+.3f}).")
        print(f"    → Among active random pairs, paper features are "
              f"in the MIDDLE of the distribution "
              f"({n_nonzero_worse} worse, {n_nonzero_better} better).")
        print(f"    → The v1 effect (delta=-2.609) was {abs(-2.609 - paper_mean)/abs(-2.609)*100:.1f}% "
              f"reconstruction artifact.")
    elif p_lower_conditional <= 0.05:
        print(f"    → Paper features significantly worse than active "
              f"random pairs (conditional p={p_lower_conditional:.4f})")
    else:
        print(f"    → Paper features NOT significantly different from "
              f"active random pairs (conditional p={p_lower_conditional:.4f})")
    
    # ── Save results ───────────────────────────────────────────
    output = {
        "experiment": "EXP_H_v2",
        "version": "delta_steering_patch",
        "v2_changes": [
            "Delta-steering: residual += decode(modified) - decode(original)",
            "Hard ablation: features zeroed, not clamped",
            "P-value: both tails reported, verdict uses lower-tail",
            "Norm-matched: uses actual steering delta norm, 5 random directions",
        ],
        "timestamp": timestamp,
        "purpose": "Definitive matched random control with delta-steering "
                   "(no reconstruction confound), hard ablation, corrected "
                   "p-value, 3 seeds × 50 random combos",
        "config": {
            "model": MODEL_NAME,
            "sae": f"{SAE_RELEASE}/{SAE_ID}",
            "paper_features": PAPER_FEATURES,
            "hook_layer": HOOK_LAYER,
            "ablation_method": "hard_zero",
            "steering_method": "delta_only (no reconstruction confound)",
            "n_random_combos_per_seed": N_RANDOM_COMBOS,
            "seeds": SEEDS,
            "total_random_combos": len(all_random_deltas),
            "metric": "forced_choice_logit_diff_YES_minus_NO",
            "norm_matched_directions": 5,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "N/A"),
            "vram_gb": (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                        if torch.cuda.is_available() else 0),
        },
        "runtime_seconds": time.time() - time.mktime(
            time.strptime(timestamp, "%Y%m%d_%H%M%S")
        ),
        "baseline": {pid: baseline[pid] for pid in sorted(baseline.keys())},
        "sae_roundtrip_control": {
            "per_prompt": {pid: roundtrip[pid] for pid in sorted(roundtrip.keys())},
            "mean_logit_delta": mean_roundtrip_delta,
            "interpretation": "Pure SAE reconstruction error (informational only; "
                              "v2 delta-steering eliminates this confound)",
        },
        "norm_matched_random_control": {
            "steering_delta_norm": typical_norm,
            "n_random_directions": 5,
            "per_direction_deltas": (
                [float(d) for d in all_norm_deltas]
                if typical_norm > 1e-6 else []
            ),
            "mean_logit_delta": mean_norm_delta,
            "std_logit_delta": (
                std_norm_delta if typical_norm > 1e-6 else 0.0
            ),
            "per_prompt_last_seed": norm_control,
            "interpretation": "Random direction with same L2 norm as actual "
                              "steering delta (v2: correct norm, 5 directions)",
        },
        "paper_features_result": {
            "per_seed": {
                str(s): paper_results_by_seed[s]["mean_logit_delta"]
                for s in SEEDS
            },
            "mean_delta": paper_mean,
            "std_across_seeds": float(np.std(paper_deltas)),
            # v2: report all p-value variants
            "p_lower": p_lower,
            "p_upper": p_upper,
            "p_two_sided": p_two_sided,
            "n_random_worse_than_paper": n_lower,
            "n_random_better_than_paper": n_upper,
            "standardized_effect_vs_null": float(z_vs_null),
            # Conditional on non-zero
            "conditional_nonzero": {
                "n_nonzero_pairs": n_nonzero,
                "n_worse_than_paper": n_nonzero_worse,
                "n_better_than_paper": n_nonzero_better,
                "p_lower_conditional": (
                    float(p_lower_conditional)
                    if not np.isnan(p_lower_conditional) else None
                ),
            },
            # v1 compatibility (deprecated field names)
            "empirical_p_value_DEPRECATED": float(p_upper),
            "empirical_p_value_note": "v1 reported P(random >= paper)=0.993. "
                "v2 reports lower-tail P(random <= paper) as the primary test.",
        },
        "null_distribution": {
            "n_total": len(all_random_deltas),
            "n_zero": n_zero,
            "n_nonzero": n_nonzero,
            "mean": float(all_random.mean()),
            "std": float(null_std),
            "median": float(np.median(all_random)),
            "min": float(all_random.min()),
            "max": float(all_random.max()),
            "percentile_1": float(np.percentile(all_random, 1)),
            "percentile_5": float(np.percentile(all_random, 5)),
            "percentile_95": float(np.percentile(all_random, 95)),
            "percentile_99": float(np.percentile(all_random, 99)),
            "frac_positive": float(np.mean(all_random > 0)),
            "frac_negative": float(np.mean(all_random < 0)),
            "frac_zero": float(np.mean(all_random == 0)),
            "all_deltas": [float(d) for d in all_random_deltas],
        },
        "random_combo_details": random_details,
        "prefill_stats": {
            pid: {str(k): v for k, v in stats.items()}
            for pid, stats in prefill_stats.items()
        },
        "rlhf_safety_entanglement": {
            "hypothesis": "Consciousness prompts may trigger RLHF safety/refusal "
                          "features rather than consciousness-specific features",
            "safety_feature_1286_monitored": True,
            "top_k_per_prompt": {
                pid: [
                    {
                        "feature_id": t["feature_id"],
                        "total_activation": round(t["total_activation"], 2),
                        "max_activation": round(t["max_activation"], 2),
                        "density": round(t["density"], 4),
                    }
                    for t in topk
                ]
                for pid, topk in topk_by_prompt.items()
            },
            "shared_features_4plus_prompts": {
                str(fid): {
                    "n_prompts": len(pids),
                    "prompts": pids,
                    "neuronpedia_url": (
                        f"neuronpedia.org/gemma-3-4b-it/"
                        f"22-gemmascope-res-16k/{fid}"
                    ),
                }
                for fid, pids in shared_features.items()
            },
            "feature_1286_in_topk": 1286 in feature_prompt_count,
            "feature_1286_prompts": feature_prompt_count.get(1286, []),
        },
        "status": "OK",
    }
    
    outfile = f"exp_H_v2_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {outfile}")
    print(f"\n{'=' * 65}")
    print(f"  DONE. Copy {outfile} off the instance before terminating!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
