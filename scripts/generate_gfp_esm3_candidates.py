from pathlib import Path
import argparse
import json
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WT_FASTA_PATH = PROJECT_ROOT / "data" / "processed" / "gfp_wt.fasta"
ALL_VARIANTS_PATH = PROJECT_ROOT / "data" / "processed" / "gfp_all_variants.csv"
FINAL_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "gfp" / "final_ensemble" / "multi_mutant_predictions.csv"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "gfp" / "esm3_generation"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ATTEMPTS_PATH = OUTPUT_ROOT / "generation_attempts.csv"
CANDIDATES_PATH = OUTPUT_ROOT / "esm3_generated_candidates.csv"
SEEDS_PATH = OUTPUT_ROOT / "selected_seed_variants.csv"
POSITION_STATS_PATH = OUTPUT_ROOT / "design_position_frequency.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

DEFAULT_MODEL_NAMES = [
    "esm3_sm_open_v1",
    "esm3-sm-open-v1",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta_sequence(path: Path):
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    )
    if not sequence:
        raise ValueError(f"Empty FASTA: {path}")
    if not set(sequence).issubset(AA20):
        raise ValueError("WT FASTA contains non-standard amino-acid symbols.")
    return sequence


def parse_mutation_positions(mutant: str):
    positions = []
    for token in str(mutant).split(":"):
        match = MUTATION_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid mutation token: {token!r}")
        _, position, _ = match.groups()
        positions.append(int(position))
    return positions


def sequence_to_mutant_string(wt_sequence: str, sequence: str):
    if len(wt_sequence) != len(sequence):
        raise ValueError("WT/generated sequence length mismatch.")
    mutations = []
    for idx, (wt_aa, mut_aa) in enumerate(zip(wt_sequence, sequence), start=1):
        if wt_aa != mut_aa:
            mutations.append(f"{wt_aa}{idx}{mut_aa}")
    return ":".join(mutations)


def hamming_distance(seq_a: str, seq_b: str):
    if len(seq_a) != len(seq_b):
        raise ValueError("Sequence length mismatch.")
    return sum(aa != bb for aa, bb in zip(seq_a, seq_b))


def load_esm3(device: str, requested_model_name):
    model_names = [requested_model_name] if requested_model_name else DEFAULT_MODEL_NAMES
    errors = []

    for model_name in model_names:
        try:
            print(f"Trying ESM3 model: {model_name}")
            model = ESM3.from_pretrained(model_name).to(device)
            model.eval()
            print(f"Loaded ESM3 model: {model_name}")
            return model, model_name
        except Exception as exc:
            errors.append((model_name, repr(exc)))

    message = (
        "Could not load ESM3 locally.\n"
        "Make sure ESM3-open Hugging Face access/license is accepted "
        "and your Hugging Face token is available.\n"
        "Tried:\n"
    )
    for model_name, error in errors:
        message += f"  - {model_name}: {error}\n"

    raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(
        description="Generate novel GFP variants with constrained ESM3 infilling."
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--positions-per-seed", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--position-pool-top-n", type=int, default=500)
    parser.add_argument("--max-seed-mutations", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="2 seeds, 1 position per seed, 1 sample per prompt.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.num_seeds = 2
        args.positions_per_seed = 1
        args.samples_per_prompt = 1

    for value, name in [
        (args.num_seeds, "--num-seeds"),
        (args.positions_per_seed, "--positions-per-seed"),
        (args.samples_per_prompt, "--samples-per-prompt"),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    wt_sequence = read_fasta_sequence(WT_FASTA_PATH)
    sequence_length = len(wt_sequence)

    header = pd.read_csv(ALL_VARIANTS_PATH, nrows=0)
    required_all_columns = {"mutant", "mutated_sequence", "mutation_count"}
    missing = required_all_columns - set(header.columns)
    if missing:
        raise ValueError(f"gfp_all_variants.csv missing: {sorted(missing)}")

    all_variants = pd.read_csv(
        ALL_VARIANTS_PATH,
        usecols=["mutant", "mutated_sequence", "mutation_count"],
    )
    all_variants["mutant"] = all_variants["mutant"].astype(str)
    all_variants["mutated_sequence"] = all_variants["mutated_sequence"].astype(str)

    observed_sequences = set(all_variants["mutated_sequence"])
    observed_sequences.add(wt_sequence)

    predictions = pd.read_csv(
        FINAL_PREDICTIONS_PATH,
        usecols=[
            "mutant",
            "mutation_count",
            "predicted_fitness_mu",
            "uncertainty_sigma",
        ],
    )
    predictions["conservative_score"] = (
        predictions["predicted_fitness_mu"] - predictions["uncertainty_sigma"]
    )

    top_position_source = (
        predictions.sort_values("conservative_score", ascending=False)
        .head(args.position_pool_top_n)
        .copy()
    )

    position_counter = Counter()
    for mutant in top_position_source["mutant"]:
        for position in parse_mutation_positions(mutant):
            position_counter[position] += 1

    position_rows = [
        {
            "position": int(position),
            "frequency": int(frequency),
            "fraction_in_top_pool": float(frequency / len(top_position_source)),
        }
        for position, frequency in position_counter.most_common()
    ]
    position_df = pd.DataFrame(position_rows)
    position_df.to_csv(POSITION_STATS_PATH, index=False)

    ranked_design_positions = position_df["position"].astype(int).tolist()
    if not ranked_design_positions:
        raise ValueError("No design positions were derived.")

    seed_candidates = (
        predictions[
            (predictions["mutation_count"] >= 2)
            & (predictions["mutation_count"] <= args.max_seed_mutations)
        ]
        .sort_values(
            ["conservative_score", "predicted_fitness_mu"],
            ascending=[False, False],
        )
        .head(args.num_seeds)
        .copy()
    )

    seed_candidates = seed_candidates.merge(
        all_variants[["mutant", "mutated_sequence"]],
        on="mutant",
        how="left",
        validate="one_to_one",
    )
    if seed_candidates["mutated_sequence"].isna().any():
        raise ValueError("Could not recover seed sequences.")

    seed_candidates.to_csv(SEEDS_PATH, index=False)

    prompt_specs = []

    for seed_rank, seed_row in enumerate(seed_candidates.itertuples(index=False), start=1):
        seed_mutant = str(seed_row.mutant)
        seed_sequence = str(seed_row.mutated_sequence)
        seed_positions = set(parse_mutation_positions(seed_mutant))

        eligible_positions = [
            position
            for position in ranked_design_positions
            if position not in seed_positions and 1 <= position <= sequence_length
        ]

        chosen_positions = eligible_positions[: args.positions_per_seed]

        for design_position in chosen_positions:
            prompt_chars = list(seed_sequence)
            original_aa = prompt_chars[design_position - 1]
            prompt_chars[design_position - 1] = "_"

            prompt_specs.append(
                {
                    "seed_rank": int(seed_rank),
                    "seed_mutant": seed_mutant,
                    "seed_sequence": seed_sequence,
                    "seed_mutation_count": int(seed_row.mutation_count),
                    "seed_mu": float(seed_row.predicted_fitness_mu),
                    "seed_sigma": float(seed_row.uncertainty_sigma),
                    "seed_conservative_score": float(seed_row.conservative_score),
                    "design_position": int(design_position),
                    "original_aa_at_design_position": original_aa,
                    "prompt_sequence": "".join(prompt_chars),
                }
            )

    if not prompt_specs:
        raise ValueError("No ESM3 prompts were created.")

    model, loaded_model_name = load_esm3(
        device=args.device,
        requested_model_name=args.model_name,
    )

    print()
    print("=" * 100)
    print("ESM3 GFP Candidate Generation")
    print("=" * 100)
    print(f"WT length              : {sequence_length}")
    print(f"Seeds                  : {len(seed_candidates)}")
    print(f"Prompts                : {len(prompt_specs)}")
    print(f"Samples / prompt       : {args.samples_per_prompt}")
    print(f"Planned attempts       : {len(prompt_specs) * args.samples_per_prompt}")
    print(f"Temperature            : {args.temperature}")
    print(f"Model                  : {loaded_model_name}")
    print(f"Device                 : {args.device}")
    print("Multi-mutant DMS_score : NOT LOADED")

    attempt_rows = []
    accepted_rows = []
    generated_sequence_set = set()

    attempt_counter = 0

    for prompt_idx, spec in enumerate(prompt_specs, start=1):
        print()
        print(
            f"Prompt {prompt_idx}/{len(prompt_specs)} "
            f"| seed rank={spec['seed_rank']} "
            f"| position={spec['design_position']}"
        )

        for sample_idx in range(args.samples_per_prompt):
            attempt_counter += 1
            generation_seed = args.base_seed + attempt_counter
            set_seed(generation_seed)

            protein = ESMProtein(sequence=spec["prompt_sequence"])
            config = GenerationConfig(
                track="sequence",
                num_steps=1,
                temperature=args.temperature,
            )

            status = "unknown"
            error_message = ""
            generated_sequence = None
            generated_mutant = None
            generated_k = None
            distance_from_seed = None
            observed_in_benchmark = None
            duplicate_generated = None
            generated_aa = None

            try:
                generated = model.generate(protein, config)

                if isinstance(generated, ESMProteinError):
                    status = "esm3_error"
                    error_message = str(generated)

                elif not isinstance(generated, ESMProtein):
                    status = "unexpected_output_type"
                    error_message = type(generated).__name__

                else:
                    generated_sequence = generated.sequence

                    if generated_sequence is None:
                        status = "missing_sequence"
                    else:
                        generated_sequence = str(generated_sequence)

                        if len(generated_sequence) != sequence_length:
                            status = "length_mismatch"
                        elif not set(generated_sequence).issubset(AA20):
                            status = "nonstandard_amino_acid"
                        else:
                            generated_aa = generated_sequence[
                                spec["design_position"] - 1
                            ]

                            if generated_sequence == spec["seed_sequence"]:
                                status = "no_change"
                            else:
                                generated_mutant = sequence_to_mutant_string(
                                    wt_sequence,
                                    generated_sequence,
                                )
                                generated_k = (
                                    0
                                    if generated_mutant == ""
                                    else len(generated_mutant.split(":"))
                                )
                                distance_from_seed = hamming_distance(
                                    spec["seed_sequence"],
                                    generated_sequence,
                                )
                                observed_in_benchmark = (
                                    generated_sequence in observed_sequences
                                )
                                duplicate_generated = (
                                    generated_sequence in generated_sequence_set
                                )

                                if observed_in_benchmark:
                                    status = "already_in_proteingym"
                                elif duplicate_generated:
                                    status = "duplicate_generated"
                                else:
                                    status = "accepted_novel"
                                    generated_sequence_set.add(generated_sequence)

                                    accepted_rows.append(
                                        {
                                            "candidate_id": (
                                                f"esm3_gfp_{len(accepted_rows) + 1:05d}"
                                            ),
                                            "seed_rank": spec["seed_rank"],
                                            "seed_mutant": spec["seed_mutant"],
                                            "seed_mutation_count": spec[
                                                "seed_mutation_count"
                                            ],
                                            "seed_mu": spec["seed_mu"],
                                            "seed_sigma": spec["seed_sigma"],
                                            "seed_conservative_score": spec[
                                                "seed_conservative_score"
                                            ],
                                            "design_position": spec[
                                                "design_position"
                                            ],
                                            "original_aa_at_design_position": spec[
                                                "original_aa_at_design_position"
                                            ],
                                            "generated_aa": generated_aa,
                                            "generated_mutant": generated_mutant,
                                            "mutation_count": int(generated_k),
                                            "distance_from_seed": int(
                                                distance_from_seed
                                            ),
                                            "generated_sequence": generated_sequence,
                                            "esm3_model": loaded_model_name,
                                            "temperature": float(args.temperature),
                                            "generation_seed": int(generation_seed),
                                        }
                                    )

            except Exception as exc:
                status = "exception"
                error_message = repr(exc)

            attempt_rows.append(
                {
                    "attempt": int(attempt_counter),
                    "prompt_index": int(prompt_idx),
                    "sample_index": int(sample_idx),
                    "seed_rank": spec["seed_rank"],
                    "seed_mutant": spec["seed_mutant"],
                    "seed_mutation_count": spec["seed_mutation_count"],
                    "design_position": spec["design_position"],
                    "original_aa_at_design_position": spec[
                        "original_aa_at_design_position"
                    ],
                    "generated_aa": generated_aa,
                    "generation_seed": int(generation_seed),
                    "status": status,
                    "generated_mutant": generated_mutant,
                    "generated_mutation_count": generated_k,
                    "distance_from_seed": distance_from_seed,
                    "observed_in_benchmark": observed_in_benchmark,
                    "duplicate_generated": duplicate_generated,
                    "error": error_message,
                }
            )

            print(
                f"  sample {sample_idx + 1}/{args.samples_per_prompt} "
                f"→ {status}"
            )

    attempts_df = pd.DataFrame(attempt_rows)
    candidates_df = pd.DataFrame(accepted_rows)

    attempts_df.to_csv(ATTEMPTS_PATH, index=False)
    candidates_df.to_csv(CANDIDATES_PATH, index=False)

    status_counts = attempts_df["status"].value_counts().to_dict()

    summary = {
        "stage": "Step 9-1",
        "method": (
            "ESM3 constrained single-position infilling around "
            "high-confidence GFP seed variants"
        ),
        "label_policy": {
            "multi_mutant_dms_score_loaded": False,
            "seed_selection": "predicted_fitness_mu - uncertainty_sigma",
        },
        "generation": {
            "model": loaded_model_name,
            "device": args.device,
            "temperature": float(args.temperature),
            "num_steps": 1,
            "num_seeds": int(len(seed_candidates)),
            "positions_per_seed": int(args.positions_per_seed),
            "samples_per_prompt": int(args.samples_per_prompt),
            "planned_attempts": int(
                len(prompt_specs) * args.samples_per_prompt
            ),
            "actual_attempts": int(len(attempts_df)),
            "accepted_novel_candidates": int(len(candidates_df)),
            "status_counts": {
                str(key): int(value)
                for key, value in status_counts.items()
            },
        },
        "seed_constraints": {
            "max_seed_mutations": int(args.max_seed_mutations),
            "position_frequency_source_top_n": int(
                args.position_pool_top_n
            ),
        },
        "artifacts": {
            "selected_seeds": str(SEEDS_PATH),
            "position_frequency": str(POSITION_STATS_PATH),
            "generation_attempts": str(ATTEMPTS_PATH),
            "accepted_candidates": str(CANDIDATES_PATH),
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 110)
    print("ESM3 GFP Generation Summary")
    print("=" * 110)
    print(f"Attempts                  : {len(attempts_df)}")
    print(f"Accepted novel candidates : {len(candidates_df)}")

    print()
    print("Status counts")
    for status, count in status_counts.items():
        print(f"  {status:28s}: {count}")

    if len(candidates_df) > 0:
        print()
        print("Accepted candidates")
        print(
            candidates_df[
                [
                    "candidate_id",
                    "seed_rank",
                    "seed_mutant",
                    "design_position",
                    "generated_aa",
                    "generated_mutant",
                    "mutation_count",
                    "distance_from_seed",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print()
    print("Saved:")
    print(SEEDS_PATH)
    print(POSITION_STATS_PATH)
    print(ATTEMPTS_PATH)
    print(CANDIDATES_PATH)
    print(SUMMARY_PATH)

    print()
    print("IMPORTANT:")
    print("No GFP multi-mutant DMS_score was loaded or used.")
    print(
        "Generated candidates are novel with respect to the observed "
        "ProteinGym GFP sequence set."
    )
    print(
        "These candidates are NOT yet predicted to be high-fitness. "
        "Step 9-2 will extract ESM-C features and score them with "
        "the frozen Deep Ensemble."
    )


if __name__ == "__main__":
    main()
