#!/usr/bin/env python3
"""
run_universal_family_edit_two_stage.py

Two-stage universal perturbation evaluation for fine-grained families.

Stage A: Neutral baseline recognition
    - whole-family closed-set prompt
    - randomized option order across multiple trials
    - majority-vote neutral baseline species

Stage B: Targeted edited evaluation
    - target-specific neuron selection on yes/no prompt
    - symmetric pairwise target-vs-true evaluation before and after edit

Per-image outputs:
    true_species
    neutral_baseline_pred_species
    targeted_baseline_pred_species
    edited_pred_species
    score_delta
    changed
    successful_target_flip

No token forcing. No logit bias.
Same old working LLaVA path.
"""

import os
import sys
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "LLaVA"))

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model


MODEL_PATH = "/data/susheel/MM_Neurons-main/llava-llama-2-13b-chat-lightning-preview"
CONV_MODE = "llava_llama_2"
DEVICE = "cuda"
DTYPE = torch.float16


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_pipe_list(s: str) -> List[str]:
    if s is None or not s.strip():
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def get_llama_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "get_model"):
        inner = model.get_model()
        if hasattr(inner, "layers"):
            return inner.layers
    raise RuntimeError("Could not find transformer layers.")


def load_manifest(nabirds_root: str) -> pd.DataFrame:
    csv_path = os.path.join(nabirds_root, "metadata", "image_manifest.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing image manifest: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"absolute_path", "family_group_name", "species_name"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in image_manifest.csv: {sorted(missing)}")

    for col in ["absolute_path", "family_group_name", "species_name"]:
        df[col] = df[col].astype(str)
    return df


def choose_source_species(df: pd.DataFrame, family_name: str, target_species: str, source_species: List[str]) -> List[str]:
    fam_df = df[df["family_group_name"] == family_name].copy()
    if len(fam_df) == 0:
        raise RuntimeError(f"Family not found: {family_name}")

    all_species = sorted(fam_df["species_name"].unique().tolist())
    if target_species not in all_species:
        raise RuntimeError(f"Target species '{target_species}' not found inside family '{family_name}'")

    if len(source_species) == 0:
        source_species = [s for s in all_species if s != target_species]
    else:
        bad = [s for s in source_species if s not in all_species]
        if bad:
            raise RuntimeError(f"These source species are not in family '{family_name}': {bad}")
        source_species = [s for s in source_species if s != target_species]

    if len(source_species) == 0:
        raise RuntimeError("No source species selected.")
    return source_species


def sample_rows(df: pd.DataFrame, species_name: str, n: int, seed: int, exclude_paths=None) -> pd.DataFrame:
    sub = df[df["species_name"] == species_name].copy()
    if exclude_paths is not None and len(exclude_paths) > 0:
        sub = sub[~sub["absolute_path"].isin(exclude_paths)].copy()

    if len(sub) == 0:
        raise RuntimeError(f"No available images for species: {species_name}")

    if len(sub) <= n:
        return sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sub.sample(n=n, random_state=seed).reset_index(drop=True)


def build_benchmark(
    df: pd.DataFrame,
    family_name: str,
    target_species: str,
    source_species: List[str],
    calib_target_max: int,
    calib_source_per_species: int,
    eval_per_source: int,
    test_per_source: int,
    seed: int,
) -> Dict:
    fam_df = df[df["family_group_name"] == family_name].copy()
    source_species = choose_source_species(df, family_name, target_species, source_species)

    target_calib = sample_rows(fam_df, target_species, calib_target_max, seed)
    used = set(target_calib["absolute_path"].tolist())

    source_calib_parts = []
    for i, sp in enumerate(source_species):
        part = sample_rows(fam_df, sp, calib_source_per_species, seed + 100 + i)
        source_calib_parts.append(part)
        used.update(part["absolute_path"].tolist())
    source_calib = pd.concat(source_calib_parts, ignore_index=True)

    source_eval_parts = []
    for i, sp in enumerate(source_species):
        part = sample_rows(
            fam_df,
            sp,
            eval_per_source,
            seed + 500 + i,
            exclude_paths=used,
        )
        source_eval_parts.append(part)
        used.update(part["absolute_path"].tolist())
    source_eval = pd.concat(source_eval_parts, ignore_index=True)

    source_test_parts = []
    for i, sp in enumerate(source_species):
        part = sample_rows(
            fam_df,
            sp,
            test_per_source,
            seed + 1000 + i,
            exclude_paths=used,
        )
        source_test_parts.append(part)
    source_test = pd.concat(source_test_parts, ignore_index=True)

    all_family_species = [target_species] + source_species

    return {
        "family_name": family_name,
        "target_species": target_species,
        "source_species": source_species,
        "all_family_species": all_family_species,
        "target_calibration": target_calib.to_dict(orient="records"),
        "source_calibration": source_calib.to_dict(orient="records"),
        "source_eval": source_eval.to_dict(orient="records"),
        "source_test": source_test.to_dict(orient="records"),
        "counts": {
            "target_calibration": int(len(target_calib)),
            "source_calibration": int(len(source_calib)),
            "source_eval": int(len(source_eval)),
            "source_test": int(len(source_test)),
        },
    }


def build_binary_prompt(model, target_species: str) -> str:
    core = f"Is this bird a {target_species}? Answer yes or no with one word."
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + core
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + core
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_pair_prompt(model, option_a_species: str, option_b_species: str) -> str:
    core = (
        "Which bird species is shown in the image?\n"
        f"A = {option_a_species}\n"
        f"B = {option_b_species}\n"
        "Answer with only A or B."
    )
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + core
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + core
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_closed_set_prompt(model, codebook: Dict[str, str]) -> str:
    lines = [
        "Which bird species is shown in the image?",
        "Choose exactly one option from the list below.",
        "Answer with only the code letter.",
        "",
    ]
    for code, species in codebook.items():
        lines.append(f"{code} = {species}")
    lines.append("")
    lines.append("Answer with only one code letter.")
    core = "\n".join(lines)

    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + core
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + core
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_single_token_ids(tokenizer, word: str) -> List[int]:
    variants = [word, " " + word, word.lower(), " " + word.lower(), word.upper(), " " + word.upper()]
    ids = set()
    for v in variants:
        tok_ids = tokenizer.encode(v, add_special_tokens=False)
        if len(tok_ids) != 1:
            continue
        tid = tok_ids[0]
        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded == word.lower():
            ids.add(tid)
    if not ids:
        raise RuntimeError(f"Could not find a clean single-token id for word={word!r}")
    return sorted(ids)


def letter_codebook(species_list: List[str]) -> Dict[str, str]:
    if len(species_list) > 26:
        raise RuntimeError("Supports at most 26 species in a family closed-set prompt.")
    letters = [chr(ord("A") + i) for i in range(len(species_list))]
    return {letter: species for letter, species in zip(letters, species_list)}


def load_image_tensor(image_processor, image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device=DEVICE, dtype=DTYPE)


def prepare_inputs_from_prompt(tokenizer, image_processor, prompt: str, image_path: str):
    image_tensor = load_image_tensor(image_processor, image_path)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(DEVICE)
    return input_ids, image_tensor


@torch.inference_mode()
def get_next_token_logits_from_prompt(model, tokenizer, image_processor, prompt: str, image_path: str) -> torch.Tensor:
    input_ids, image_tensor = prepare_inputs_from_prompt(tokenizer, image_processor, prompt, image_path)
    outputs = model(input_ids=input_ids, images=image_tensor, use_cache=False)
    return outputs.logits[:, -1, :].detach().float().cpu()[0]


@torch.inference_mode()
def get_binary_gap(model, tokenizer, image_processor, image_path: str, target_species: str, yes_ids: List[int], no_ids: List[int]) -> float:
    prompt = build_binary_prompt(model, target_species)
    logits = get_next_token_logits_from_prompt(model, tokenizer, image_processor, prompt, image_path)
    yes_score = logits[yes_ids].max().item()
    no_score = logits[no_ids].max().item()
    return yes_score - no_score


@torch.inference_mode()
def get_pair_gap(model, tokenizer, image_processor, image_path: str, option_a_species: str, option_b_species: str, a_ids: List[int], b_ids: List[int]) -> float:
    prompt = build_pair_prompt(model, option_a_species, option_b_species)
    logits = get_next_token_logits_from_prompt(model, tokenizer, image_processor, prompt, image_path)
    a_score = logits[a_ids].max().item()
    b_score = logits[b_ids].max().item()
    return a_score - b_score


def neutral_baseline_predict_species(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    family_species: List[str],
    baseline_trials: int,
    baseline_seed: int,
) -> Tuple[str, Dict[str, int], List[str]]:
    votes = []
    vote_counter = Counter()

    for t in range(baseline_trials):
        species_order = family_species[:]
        rng = random.Random(baseline_seed + t)
        rng.shuffle(species_order)

        codebook = letter_codebook(species_order)
        prompt = build_closed_set_prompt(model, codebook)
        logits = get_next_token_logits_from_prompt(model, tokenizer, image_processor, prompt, image_path)

        best_code = None
        best_score = None
        for code, species in codebook.items():
            code_ids = build_single_token_ids(tokenizer, code)
            score = logits[code_ids].max().item()
            if best_score is None or score > best_score:
                best_score = score
                best_code = code

        pred_species = codebook[best_code]
        votes.append(pred_species)
        vote_counter[pred_species] += 1

    majority_species = vote_counter.most_common(1)[0][0]
    return majority_species, dict(vote_counter), votes


def collect_single_image_representation(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    target_species: str,
    layer_min: int,
    layer_max: int,
) -> Dict[int, torch.Tensor]:
    layers = get_llama_layers(model)
    captured = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs):
            x = inputs[0].detach().float().cpu()
            captured[layer_idx] = x.mean(dim=(0, 1))
        return hook

    for layer_idx in range(len(layers)):
        if layer_min <= layer_idx <= layer_max:
            handles.append(layers[layer_idx].mlp.down_proj.register_forward_pre_hook(make_hook(layer_idx)))

    prompt = build_binary_prompt(model, target_species)
    input_ids, image_tensor = prepare_inputs_from_prompt(tokenizer, image_processor, prompt, image_path)

    with torch.inference_mode():
        _ = model(input_ids=input_ids, images=image_tensor, use_cache=False)

    for h in handles:
        h.remove()

    return captured


def collect_rows_mean(
    model,
    tokenizer,
    image_processor,
    rows: List[dict],
    target_species: str,
    layer_min: int,
    layer_max: int,
) -> Dict[int, torch.Tensor]:
    sums = defaultdict(lambda: None)
    count = 0

    for row in rows:
        p = row["absolute_path"]
        rep = collect_single_image_representation(
            model,
            tokenizer,
            image_processor,
            p,
            target_species,
            layer_min,
            layer_max,
        )
        for layer_idx, vec in rep.items():
            if sums[layer_idx] is None:
                sums[layer_idx] = vec.clone()
            else:
                sums[layer_idx] += vec
        count += 1
        print("Mean pass:", p)

    means = {}
    for layer_idx, vec in sums.items():
        means[layer_idx] = vec / max(count, 1)
    return means


def build_initial_delta_ranking(
    source_mean: Dict[int, torch.Tensor],
    target_mean: Dict[int, torch.Tensor],
    only_positive_delta: bool,
):
    delta_by_layer = {}
    ranked = []

    for layer_idx in sorted(target_mean.keys()):
        delta = target_mean[layer_idx] - source_mean[layer_idx]
        delta_by_layer[layer_idx] = delta
        for neuron_idx in range(delta.shape[0]):
            score = float(delta[neuron_idx].item())
            if only_positive_delta and score <= 0:
                continue
            ranked.append((layer_idx, neuron_idx, score))

    ranked.sort(key=lambda x: abs(x[2]), reverse=True)
    return delta_by_layer, ranked


def make_single_neuron_hook(neuron_idx: int, edit_value: float):
    def hook(module, inputs):
        x = inputs[0].clone()
        x[..., neuron_idx] = x[..., neuron_idx] + edit_value
        return (x,)
    return hook


@torch.inference_mode()
def get_gap_with_single_neuron_edit(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    target_species: str,
    layer_idx: int,
    neuron_idx: int,
    edit_value: float,
    yes_ids: List[int],
    no_ids: List[int],
) -> float:
    layers = get_llama_layers(model)
    h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
        make_single_neuron_hook(neuron_idx, edit_value)
    )
    try:
        gap = get_binary_gap(
            model, tokenizer, image_processor, image_path, target_species, yes_ids, no_ids
        )
    finally:
        h.remove()
    return gap


def causal_rerank_shortlist(
    model,
    tokenizer,
    image_processor,
    source_eval_rows: List[dict],
    target_species: str,
    shortlist: List[Tuple[int, int, float]],
    yes_ids: List[int],
    no_ids: List[int],
    probe_alpha: float,
):
    baseline_gaps = {}
    eval_paths = [r["absolute_path"] for r in source_eval_rows]
    for p in eval_paths:
        baseline_gaps[p] = get_binary_gap(
            model, tokenizer, image_processor, p, target_species, yes_ids, no_ids
        )

    shortlist_abs = [abs(s) for _, _, s in shortlist]
    probe_scale = (sum(shortlist_abs) / max(len(shortlist_abs), 1)) + 1e-8

    causal_ranked = []
    print("Evaluating causal effect of shortlist neurons...")
    for idx, (layer_idx, neuron_idx, delta_value) in enumerate(shortlist, start=1):
        edit_value = probe_alpha * (delta_value / probe_scale)
        deltas = []

        for p in eval_paths:
            edited_gap = get_gap_with_single_neuron_edit(
                model,
                tokenizer,
                image_processor,
                p,
                target_species,
                layer_idx,
                neuron_idx,
                edit_value,
                yes_ids,
                no_ids,
            )
            deltas.append(edited_gap - baseline_gaps[p])

        causal_score = float(sum(deltas) / len(deltas))
        causal_ranked.append(
            {
                "layer": int(layer_idx),
                "neuron": int(neuron_idx),
                "delta_value": float(delta_value),
                "causal_score": causal_score,
                "probe_edit_value": float(edit_value),
            }
        )

        if idx % 25 == 0 or idx == len(shortlist):
            print(f"  done {idx}/{len(shortlist)}")

    causal_ranked.sort(key=lambda x: x["causal_score"], reverse=True)
    return causal_ranked


def build_selected_neurons(causal_ranked, top_k: int, only_positive_causal: bool):
    filtered = []
    for item in causal_ranked:
        cscore = float(item["causal_score"])
        if only_positive_causal and cscore <= 0:
            continue
        filtered.append(item)

    total_positive_causal = len(filtered)
    selected = filtered[:top_k]

    if len(selected) == 0:
        raise RuntimeError("No positive causal neurons available. Check ranking or alpha.")

    mean_abs_delta = sum(abs(x["delta_value"]) for x in selected) / len(selected)
    mean_abs_delta = mean_abs_delta + 1e-8

    by_layer = {}
    selected_json = []
    for item in selected:
        edit_strength = float(item["delta_value"] / mean_abs_delta)
        by_layer.setdefault(int(item["layer"]), []).append((int(item["neuron"]), edit_strength))
        selected_json.append(
            {
                "layer": int(item["layer"]),
                "neuron": int(item["neuron"]),
                "delta_value": float(item["delta_value"]),
                "causal_score": float(item["causal_score"]),
                "probe_edit_value": float(item["probe_edit_value"]),
                "normalized_edit_strength": float(edit_strength),
            }
        )

    return by_layer, selected_json, total_positive_causal


def make_pre_hook(neuron_score_pairs, alpha=8.0):
    score_map = {n: s for n, s in neuron_score_pairs}

    def hook(module, inputs):
        x = inputs[0].clone()
        for neuron_idx, normalized_strength in score_map.items():
            x[..., neuron_idx] = x[..., neuron_idx] + alpha * normalized_strength
        return (x,)
    return hook


def register_edit_hooks(model, by_layer, alpha=8.0):
    handles = []
    layers = get_llama_layers(model)
    for layer_idx, neuron_score_pairs in by_layer.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        handles.append(
            layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                make_pre_hook(neuron_score_pairs, alpha=alpha)
            )
        )
    return handles


def symmetric_target_score(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    target_species: str,
    true_species: str,
    a_ids: List[int],
    b_ids: List[int],
) -> Tuple[float, float, float]:
    gap1 = get_pair_gap(
        model, tokenizer, image_processor, image_path, target_species, true_species, a_ids, b_ids
    )
    gap2 = get_pair_gap(
        model, tokenizer, image_processor, image_path, true_species, target_species, a_ids, b_ids
    )
    score = gap1 - gap2
    return gap1, gap2, score


def pred_species_from_score(score: float, target_species: str, true_species: str) -> str:
    return target_species if score > 0 else true_species


def evaluate(
    model,
    tokenizer,
    image_processor,
    benchmark: Dict,
    by_layer,
    alpha: float,
    a_ids: List[int],
    b_ids: List[int],
    baseline_trials: int,
    baseline_seed: int,
):
    target_species = benchmark["target_species"]
    family_species = benchmark["all_family_species"]

    rows = []
    overall = defaultdict(float)
    subset = defaultdict(float)
    per_species = defaultdict(lambda: defaultdict(float))

    for idx, row in enumerate(benchmark["source_test"]):
        image_path = row["absolute_path"]
        true_species = row["species_name"]

        neutral_baseline_pred_species, neutral_votes, neutral_vote_list = neutral_baseline_predict_species(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=image_path,
            family_species=family_species,
            baseline_trials=baseline_trials,
            baseline_seed=baseline_seed + idx * 17,
        )
        neutral_baseline_correct = int(neutral_baseline_pred_species == true_species)
        neutral_baseline_is_target = int(neutral_baseline_pred_species == target_species)

        base_gap_ab, base_gap_ba, base_score = symmetric_target_score(
            model, tokenizer, image_processor, image_path, target_species, true_species, a_ids, b_ids
        )
        targeted_baseline_pred_species = pred_species_from_score(base_score, target_species, true_species)

        handles = register_edit_hooks(model, by_layer, alpha=alpha)
        try:
            edit_gap_ab, edit_gap_ba, edit_score = symmetric_target_score(
                model, tokenizer, image_processor, image_path, target_species, true_species, a_ids, b_ids
            )
            edited_pred_species = pred_species_from_score(edit_score, target_species, true_species)
        finally:
            for h in handles:
                h.remove()

        targeted_baseline_is_target = int(targeted_baseline_pred_species == target_species)
        edited_is_target = int(edited_pred_species == target_species)
        changed = int(targeted_baseline_pred_species != edited_pred_species)
        successful_target_flip = int(
            (targeted_baseline_pred_species == true_species) and (edited_pred_species == target_species)
        )
        score_delta = edit_score - base_score

        clean_changed = int(neutral_baseline_correct == 1 and changed == 1)
        clean_successful_target_flip = int(
            neutral_baseline_correct == 1
            and targeted_baseline_pred_species == true_species
            and edited_pred_species == target_species
        )

        rows.append(
            {
                "image_path": image_path,
                "true_species": true_species,
                "target_species": target_species,
                "neutral_baseline_pred_species": neutral_baseline_pred_species,
                "neutral_baseline_correct": neutral_baseline_correct,
                "neutral_baseline_is_target": neutral_baseline_is_target,
                "neutral_vote_counts": json.dumps(neutral_votes, ensure_ascii=False),
                "neutral_vote_list": json.dumps(neutral_vote_list, ensure_ascii=False),
                "targeted_baseline_pred_species": targeted_baseline_pred_species,
                "edited_pred_species": edited_pred_species,
                "baseline_gap_targetA_sourceB": base_gap_ab,
                "baseline_gap_sourceA_targetB": base_gap_ba,
                "baseline_symmetric_score": base_score,
                "edited_gap_targetA_sourceB": edit_gap_ab,
                "edited_gap_sourceA_targetB": edit_gap_ba,
                "edited_symmetric_score": edit_score,
                "score_delta": score_delta,
                "changed": changed,
                "successful_target_flip": successful_target_flip,
                "clean_changed_on_neutral_correct_subset": clean_changed,
                "clean_successful_target_flip": clean_successful_target_flip,
            }
        )

        overall["n"] += 1
        overall["neutral_baseline_correct_count"] += neutral_baseline_correct
        overall["neutral_baseline_target_count"] += neutral_baseline_is_target
        overall["targeted_baseline_target_count"] += targeted_baseline_is_target
        overall["edited_target_count"] += edited_is_target
        overall["changed_count"] += changed
        overall["successful_target_flip_count"] += successful_target_flip
        overall["mean_score_delta_sum"] += score_delta

        if neutral_baseline_correct == 1:
            subset["n"] += 1
            subset["targeted_baseline_target_count"] += targeted_baseline_is_target
            subset["edited_target_count"] += edited_is_target
            subset["changed_count"] += changed
            subset["successful_target_flip_count"] += successful_target_flip
            subset["clean_changed_count"] += clean_changed
            subset["clean_successful_target_flip_count"] += clean_successful_target_flip
            subset["mean_score_delta_sum"] += score_delta

        ps = per_species[true_species]
        ps["n"] += 1
        ps["neutral_baseline_correct_count"] += neutral_baseline_correct
        ps["targeted_baseline_target_count"] += targeted_baseline_is_target
        ps["edited_target_count"] += edited_is_target
        ps["changed_count"] += changed
        ps["successful_target_flip_count"] += successful_target_flip
        ps["mean_score_delta_sum"] += score_delta

        print("\nProcessing:", image_path)
        print("True species              :", true_species)
        print("Neutral baseline species  :", neutral_baseline_pred_species)
        print("Targeted baseline species :", targeted_baseline_pred_species)
        print("Edited species            :", edited_pred_species)
        print("Baseline symmetric score  :", base_score)
        print("Edited symmetric score    :", edit_score)
        print("Score delta               :", score_delta)

    def rate(num, den):
        return float(num / den) if den > 0 else 0.0

    n = int(overall["n"])
    summary = {
        "n_test": n,
        "neutral_baseline_correct_count": int(overall["neutral_baseline_correct_count"]),
        "neutral_baseline_accuracy": rate(overall["neutral_baseline_correct_count"], n),
        "neutral_baseline_target_count": int(overall["neutral_baseline_target_count"]),
        "neutral_baseline_target_rate": rate(overall["neutral_baseline_target_count"], n),
        "targeted_baseline_target_count": int(overall["targeted_baseline_target_count"]),
        "targeted_baseline_target_rate": rate(overall["targeted_baseline_target_count"], n),
        "edited_target_count": int(overall["edited_target_count"]),
        "edited_target_rate": rate(overall["edited_target_count"], n),
        "changed_count": int(overall["changed_count"]),
        "changed_rate": rate(overall["changed_count"], n),
        "successful_target_flip_count": int(overall["successful_target_flip_count"]),
        "successful_target_flip_rate": rate(overall["successful_target_flip_count"], n),
        "mean_score_delta": rate(overall["mean_score_delta_sum"], n),
        "baseline_correct_subset_n": int(subset["n"]),
        "baseline_correct_subset_targeted_baseline_target_rate": rate(subset["targeted_baseline_target_count"], subset["n"]),
        "baseline_correct_subset_edited_target_rate": rate(subset["edited_target_count"], subset["n"]),
        "baseline_correct_subset_changed_rate": rate(subset["changed_count"], subset["n"]),
        "baseline_correct_subset_successful_target_flip_rate": rate(subset["successful_target_flip_count"], subset["n"]),
        "baseline_correct_subset_clean_changed_rate": rate(subset["clean_changed_count"], subset["n"]),
        "baseline_correct_subset_clean_successful_target_flip_rate": rate(subset["clean_successful_target_flip_count"], subset["n"]),
        "baseline_correct_subset_mean_score_delta": rate(subset["mean_score_delta_sum"], subset["n"]),
        "per_species": {},
    }

    for species, s in per_species.items():
        denom = s["n"]
        summary["per_species"][species] = {
            "n": int(denom),
            "neutral_baseline_correct_count": int(s["neutral_baseline_correct_count"]),
            "neutral_baseline_accuracy": rate(s["neutral_baseline_correct_count"], denom),
            "targeted_baseline_target_rate": rate(s["targeted_baseline_target_count"], denom),
            "edited_target_rate": rate(s["edited_target_count"], denom),
            "changed_rate": rate(s["changed_count"], denom),
            "successful_target_flip_rate": rate(s["successful_target_flip_count"], denom),
            "mean_score_delta": rate(s["mean_score_delta_sum"], denom),
        }

    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nabirds-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--family-name", type=str, default="Woodpeckers")
    parser.add_argument("--target-species", type=str, default="Downy Woodpecker")
    parser.add_argument("--source-species", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--layer-min", type=int, default=20)
    parser.add_argument("--layer-max", type=int, default=39)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=8.0)

    parser.add_argument("--only-positive-delta", action="store_true")
    parser.add_argument("--only-positive-causal", action="store_true")

    parser.add_argument("--calib-target-max", type=int, default=40)
    parser.add_argument("--calib-source-per-species", type=int, default=10)
    parser.add_argument("--eval-per-source", type=int, default=5)
    parser.add_argument("--test-per-source", type=int, default=20)

    parser.add_argument("--shortlist-top-m", type=int, default=256)
    parser.add_argument("--probe-alpha", type=float, default=8.0)

    parser.add_argument("--baseline-trials", type=int, default=5)
    parser.add_argument("--baseline-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading NABirds metadata...")
    df = load_manifest(args.nabirds_root)

    source_species = parse_pipe_list(args.source_species)
    benchmark = build_benchmark(
        df=df,
        family_name=args.family_name,
        target_species=args.target_species,
        source_species=source_species,
        calib_target_max=args.calib_target_max,
        calib_source_per_species=args.calib_source_per_species,
        eval_per_source=args.eval_per_source,
        test_per_source=args.test_per_source,
        seed=args.seed,
    )

    with open(os.path.join(args.output_dir, "benchmark.json"), "w") as f:
        json.dump(benchmark, f, indent=2)

    print("[info] Benchmark built")
    print(json.dumps(benchmark["counts"], indent=2))
    print("[info] Family:", benchmark["family_name"])
    print("[info] Target:", benchmark["target_species"])
    print("[info] Sources:", benchmark["source_species"])

    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_base=None, model_name="llava"
    )
    model = model.to(DEVICE)
    model.eval()

    yes_ids = build_single_token_ids(tokenizer, "yes")
    no_ids = build_single_token_ids(tokenizer, "no")
    a_ids = build_single_token_ids(tokenizer, "a")
    b_ids = build_single_token_ids(tokenizer, "b")
    print("yes token ids:", yes_ids)
    print("no token ids :", no_ids)
    print("A token ids  :", a_ids)
    print("B token ids  :", b_ids)

    print("Collecting pooled source mean...")
    source_mean = collect_rows_mean(
        model, tokenizer, image_processor, benchmark["source_calibration"],
        benchmark["target_species"], args.layer_min, args.layer_max
    )

    print("Collecting target mean...")
    target_mean = collect_rows_mean(
        model, tokenizer, image_processor, benchmark["target_calibration"],
        benchmark["target_species"], args.layer_min, args.layer_max
    )

    delta_by_layer, initial_ranked = build_initial_delta_ranking(
        source_mean=source_mean,
        target_mean=target_mean,
        only_positive_delta=args.only_positive_delta,
    )
    shortlist = initial_ranked[: args.shortlist_top_m]
    print(f"[info] Initial ranked neurons: {len(initial_ranked)}")
    print(f"[info] Shortlist size       : {len(shortlist)}")

    causal_ranked = causal_rerank_shortlist(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        source_eval_rows=benchmark["source_eval"],
        target_species=benchmark["target_species"],
        shortlist=shortlist,
        yes_ids=yes_ids,
        no_ids=no_ids,
        probe_alpha=args.probe_alpha,
    )

    by_layer, selected_json, total_positive_causal = build_selected_neurons(
        causal_ranked=causal_ranked,
        top_k=args.top_k,
        only_positive_causal=args.only_positive_causal,
    )

    neuron_percentage_used = 100.0 * len(selected_json) / max(total_positive_causal, 1)
    print(f"[info] Positive causal neurons: {total_positive_causal}")
    print(f"[info] Selected neurons      : {len(selected_json)}")
    print(f"[info] Neuron % used         : {neuron_percentage_used:.2f}%")
    print(f"[info] Alpha                : {args.alpha}")

    torch.save(
        {
            "family_name": benchmark["family_name"],
            "target_species": benchmark["target_species"],
            "source_species": benchmark["source_species"],
            "all_family_species": benchmark["all_family_species"],
            "layer_min": args.layer_min,
            "layer_max": args.layer_max,
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
            "a_token_ids": a_ids,
            "b_token_ids": b_ids,
            "source_mean": source_mean,
            "target_mean": target_mean,
            "delta_by_layer": delta_by_layer,
            "initial_ranked": initial_ranked,
            "causal_ranked": causal_ranked,
            "selected_neurons": selected_json,
            "selected_by_layer": by_layer,
            "selection_method": "two_stage_neutral_baseline_plus_targeted_symmetric_eval",
            "alpha": args.alpha,
            "probe_alpha": args.probe_alpha,
            "top_k": len(selected_json),
            "total_positive_causal": total_positive_causal,
            "neuron_percentage_used": neuron_percentage_used,
            "baseline_trials": args.baseline_trials,
        },
        os.path.join(args.output_dir, "universal_edit_artifacts.pt"),
    )

    with open(os.path.join(args.output_dir, "selected_neurons.json"), "w") as f:
        json.dump(selected_json, f, indent=2)

    rows, summary = evaluate(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        benchmark=benchmark,
        by_layer=by_layer,
        alpha=args.alpha,
        a_ids=a_ids,
        b_ids=b_ids,
        baseline_trials=args.baseline_trials,
        baseline_seed=args.baseline_seed,
    )

    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "per_image_results.csv"),
        index=False
    )

    with open(os.path.join(args.output_dir, "per_image_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    summary.update({
        "family_name": benchmark["family_name"],
        "target_species": benchmark["target_species"],
        "source_species": benchmark["source_species"],
        "all_family_species": benchmark["all_family_species"],
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "top_k": len(selected_json),
        "alpha": args.alpha,
        "probe_alpha": args.probe_alpha,
        "shortlist_top_m": args.shortlist_top_m,
        "baseline_trials": args.baseline_trials,
        "selection_method": "two_stage_neutral_baseline_plus_targeted_symmetric_eval",
        "no_logit_bias": True,
        "no_token_forcing": True,
        "baseline_prompt_style": "neutral_closed_set_randomized_codes",
        "train_prompt_style": "target_specific_yes_no",
        "eval_prompt_style": "targeted_symmetric_pairwise_target_vs_true",
        "total_positive_causal_neurons": total_positive_causal,
        "neuron_percentage_used": neuron_percentage_used,
    })

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved files:")
    for name in [
        "run_args.json",
        "benchmark.json",
        "universal_edit_artifacts.pt",
        "selected_neurons.json",
        "per_image_results.csv",
        "per_image_results.json",
        "summary.json",
    ]:
        print(os.path.join(args.output_dir, name))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
