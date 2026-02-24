from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from opt_attn_net_feb.explainability.chem_ace.rules.rdkit_fragment_rules import (
        default_functional_rules_path,
        generate_fragment_rules_from_smiles,
        merge_rules,
    )
except Exception:
    from opt_attn_net_feb.explainability.chem_ace.rules.rdkit_fragment_rules import (
        default_functional_rules_path,
        generate_fragment_rules_from_smiles,
        merge_rules,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate RDKit-fragment functional-group rules from a labels table "
            "and merge them into default_functional_group_rules.json."
        )
    )
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--smiles_col", default="curated_SMILES")
    ap.add_argument(
        "--output_json",
        default=str(
            default_functional_rules_path()
        ),
    )
    ap.add_argument("--min_count", type=int, default=25)
    ap.add_argument("--min_prevalence", type=float, default=0.001)
    ap.add_argument("--max_molecules", type=int, default=0)
    ap.add_argument("--stats_json", default=None)
    args = ap.parse_args(argv)

    labels_csv = Path(str(args.labels_csv))
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels_csv not found: {labels_csv}")

    df = pd.read_csv(labels_csv, low_memory=False)
    smiles_col = str(args.smiles_col)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {labels_csv}")

    smiles = (
        df[smiles_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    smiles = smiles[smiles.str.len() > 0].drop_duplicates().tolist()
    if int(args.max_molecules) > 0:
        smiles = smiles[: int(args.max_molecules)]

    generated_rules, summary, fragment_stats = generate_fragment_rules_from_smiles(
        smiles_iter=smiles,
        min_count=int(args.min_count),
        min_prevalence=float(args.min_prevalence),
    )

    output_json = Path(str(args.output_json))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        payload = json.loads(output_json.read_text())
        base_rules = list(payload.get("rules", []))
    else:
        payload = {}
        base_rules = []
    merged_rules = merge_rules(base_rules=base_rules, generated_rules=generated_rules)

    payload_out = {
        **payload,
        "rules": merged_rules,
        "rdkit_fragment_generation": {
            "labels_csv": str(labels_csv),
            "smiles_col": str(smiles_col),
            **summary.to_dict(),
        },
    }
    output_json.write_text(json.dumps(payload_out, indent=2, sort_keys=False))

    if args.stats_json:
        stats_json = Path(str(args.stats_json))
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats_json.write_text(
            json.dumps(
                {
                    **summary.to_dict(),
                    "fragments": fragment_stats,
                },
                indent=2,
                sort_keys=False,
            )
        )

    print(
        "[CHEM-ACE][RDKIT-FRAG] "
        f"seen={summary.n_smiles_seen} valid={summary.n_smiles_valid} generated_rules={len(generated_rules)} "
        f"total_rules_after_merge={len(merged_rules)} output={output_json}"
    )


if __name__ == "__main__":
    main()
