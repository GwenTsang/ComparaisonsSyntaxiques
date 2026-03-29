#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Comparaison des distances syntaxiques moyennes : SMS vs Philosophie

  Lit les fichiers resultats_par_sms.csv produits par chaque modele
  dans le dossier de resultats, calcule les moyennes des metriques
  de dependance par corpus et par modele, puis genere un rapport
  comparatif.

  Usage :
    python compare_distances.py
    python compare_distances.py --results-dir /chemin/vers/results
    python compare_distances.py --output rapport_distances.txt
==========================================================================
"""

import argparse
import csv
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

BANNER = "=" * 76

# ── configuration ─────────────────────────────────────────────────────

CORPORA = ["SMS", "Philosophie"]

ALL_MODELS = ["gsd", "fsmb", "sequoia", "rhapsodie", "zenodo-spoken", "stanza"]

MODEL_AFFINITY = {
    "fsmb":           "SMS / oral",
    "rhapsodie":      "SMS / oral",
    "zenodo-spoken":  "SMS / oral",
    "gsd":            "Philosophie / ecrit formel",
    "sequoia":        "Philosophie / ecrit formel",
    "stanza":         "generaliste",
}

# Dependency features of interest (suffix after the column prefix)
DEP_FEATURES = [
    "distance_dependance_moy",
    "distance_dependance_max",
    "distance_dependance_var",
    "profondeur_arbre_moy",
    "profondeur_arbre_max",
    "nb_dependances_moy_phrase",
]

# Human-readable labels
FEATURE_LABELS = {
    "distance_dependance_moy": "Dist. dep. moyenne",
    "distance_dependance_max": "Dist. dep. max",
    "distance_dependance_var": "Dist. dep. variance",
    "profondeur_arbre_moy":    "Prof. arbre moyenne",
    "profondeur_arbre_max":    "Prof. arbre max",
    "nb_dependances_moy_phrase": "Nb dep. moy/phrase",
}


# ── helpers ───────────────────────────────────────────────────────────

def _detect_prefixes(columns: list[str]) -> list[str]:
    """Auto-detect column prefixes by looking for known suffixes.

    For example, columns like 'SMS_distance_dependance_moy' have
    prefix 'SMS_', while 'Texte_distance_dependance_moy' → 'Texte_'.
    """
    prefixes = set()
    for col in columns:
        for feat in DEP_FEATURES:
            if col.endswith(f"_{feat}"):
                prefix = col[: -len(feat) - 1]  # strip _feat
                prefixes.add(prefix + "_")
    return sorted(prefixes)


def load_corpus_features(
    results_dir: str,
    corpus_name: str,
    models: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Load dependency feature values per model for a given corpus.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Mapping: model_name → { feature_name → [values] }
    """
    data: dict[str, dict[str, list[float]]] = {}

    for model_name in models:
        csv_path = os.path.join(
            results_dir, corpus_name, model_name, "resultats_par_sms.csv",
        )
        if not os.path.isfile(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  ! Erreur lecture {csv_path}: {exc}")
            continue

        prefixes = _detect_prefixes(df.columns.tolist())
        if not prefixes:
            print(f"  ! Aucun prefixe detecte dans {csv_path}")
            continue

        model_data: dict[str, list[float]] = {f: [] for f in DEP_FEATURES}
        for feat in DEP_FEATURES:
            for prefix in prefixes:
                col = f"{prefix}{feat}"
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    model_data[feat].extend(vals.tolist())

        data[model_name] = model_data

    return data


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


# ── main logic ────────────────────────────────────────────────────────

def compare(results_dir: str, output_path: str):
    """Run the full comparison and write the report."""

    # Detect which models are actually present
    models = [
        m for m in ALL_MODELS
        if any(
            os.path.isfile(
                os.path.join(results_dir, c, m, "resultats_par_sms.csv")
            )
            for c in CORPORA
        )
    ]

    if not models:
        print("  ! Aucun resultat trouve. Verifiez --results-dir.")
        sys.exit(1)

    print(f"\n{BANNER}")
    print("  COMPARAISON DES DISTANCES SYNTAXIQUES : SMS vs PHILOSOPHIE")
    print(BANNER)
    print(f"  Modeles detectes : {', '.join(models)}")
    print(f"  Dossier resultats: {results_dir}")

    # ── load data ─────────────────────────────────────────────────────

    sms_data  = load_corpus_features(results_dir, "SMS", models)
    philo_data = load_corpus_features(results_dir, "Philosophie", models)

    # ── per-model comparison table ────────────────────────────────────

    rows: list[dict] = []
    lines: list[str] = []

    header_feat = "Feature"
    lines.append(f"\n  {'Modele':<18} {'Feature':<25} "
                 f"{'SMS moy':>10} {'Philo moy':>10} {'Delta':>10} {'Direction':>18}")
    lines.append("  " + "-" * 95)

    for model_name in models:
        sms_m  = sms_data.get(model_name, {})
        philo_m = philo_data.get(model_name, {})
        for feat in DEP_FEATURES:
            sms_vals   = sms_m.get(feat, [])
            philo_vals = philo_m.get(feat, [])
            sms_mean   = _safe_mean(sms_vals)
            philo_mean = _safe_mean(philo_vals)

            if sms_mean is not None and philo_mean is not None:
                delta = philo_mean - sms_mean
                direction = "Philo > SMS" if delta > 0 else "SMS > Philo" if delta < 0 else "="
                lines.append(
                    f"  {model_name:<18} {FEATURE_LABELS.get(feat, feat):<25} "
                    f"{sms_mean:>10.2f} {philo_mean:>10.2f} {delta:>+10.2f} {direction:>18}"
                )
                rows.append({
                    "modele": model_name,
                    "affinite": MODEL_AFFINITY.get(model_name, "?"),
                    "feature": feat,
                    "sms_n": len(sms_vals),
                    "sms_mean": round(sms_mean, 4),
                    "sms_std": round(_safe_std(sms_vals) or 0, 4),
                    "philo_n": len(philo_vals),
                    "philo_mean": round(philo_mean, 4),
                    "philo_std": round(_safe_std(philo_vals) or 0, 4),
                    "delta": round(delta, 4),
                    "direction": direction,
                })
            else:
                lines.append(
                    f"  {model_name:<18} {FEATURE_LABELS.get(feat, feat):<25} "
                    f"{'N/A':>10} {'N/A':>10} {'':>10} {'':>18}"
                )
        lines.append("")  # blank line between models

    for l in lines:
        print(l)

    # ── global averages (across all models) ───────────────────────────

    print(f"\n  {'-' * 60}")
    print(f"  MOYENNES GLOBALES (tous modeles confondus)")
    print(f"  {'-' * 60}")
    print(f"  {'Feature':<25} {'SMS':>10} {'Philo':>10} {'Delta':>10}")
    print(f"  {'-' * 58}")

    global_rows: list[dict] = []
    for feat in DEP_FEATURES:
        all_sms   = [v for m in sms_data.values()  for v in m.get(feat, [])]
        all_philo = [v for m in philo_data.values() for v in m.get(feat, [])]
        sms_mean   = _safe_mean(all_sms)
        philo_mean = _safe_mean(all_philo)
        if sms_mean is not None and philo_mean is not None:
            delta = philo_mean - sms_mean
            direction = "Philo > SMS" if delta > 0 else "SMS > Philo" if delta < 0 else "="
            print(f"  {FEATURE_LABELS.get(feat, feat):<25} "
                  f"{sms_mean:>10.2f} {philo_mean:>10.2f} {delta:>+10.2f}  {direction}")
            global_rows.append({
                "feature": feat,
                "sms_n": len(all_sms),
                "sms_mean": round(sms_mean, 4),
                "philo_n": len(all_philo),
                "philo_mean": round(philo_mean, 4),
                "delta": round(delta, 4),
                "direction": direction,
            })

    # ── write report ──────────────────────────────────────────────────

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("COMPARAISON DES DISTANCES SYNTAXIQUES : SMS vs PHILOSOPHIE\n")
        f.write("=" * 60 + "\n\n")

        f.write("Modeles utilises :\n")
        for m in models:
            aff = MODEL_AFFINITY.get(m, "?")
            f.write(f"  - {m:<20} affinite : {aff}\n")

        f.write("\n\n--- DETAIL PAR MODELE ---\n\n")
        for l in lines:
            f.write(l + "\n")

        f.write("\n\n--- MOYENNES GLOBALES ---\n\n")
        f.write(f"{'Feature':<25} {'SMS':>10} {'Philo':>10} {'Delta':>10} {'Direction':>18}\n")
        f.write("-" * 76 + "\n")
        for gr in global_rows:
            f.write(
                f"{FEATURE_LABELS.get(gr['feature'], gr['feature']):<25} "
                f"{gr['sms_mean']:>10.4f} {gr['philo_mean']:>10.4f} "
                f"{gr['delta']:>+10.4f}  {gr['direction']}\n"
            )

        f.write("\n\n--- INTERPRETATION ---\n\n")
        f.write("Rappel : une distance de dependance plus elevee indique des\n")
        f.write("arcs syntaxiques plus longs, typiquement associes a des\n")
        f.write("structures plus complexes (enchassements, relatives, etc.).\n\n")

        # Quick interpretation
        dist_row = next((r for r in global_rows if r["feature"] == "distance_dependance_moy"), None)
        if dist_row:
            d = dist_row["delta"]
            if abs(d) < 0.1:
                f.write("→ Les deux corpus ont des distances de dependance \n"
                        "  moyennes tres proches, l'ecart est negligeable.\n")
            else:
                higher = "Philosophie" if d > 0 else "SMS"
                f.write(f"→ Le corpus {higher} presente des distances de\n"
                        f"  dependance moyennes plus elevees (delta = {d:+.4f}),\n"
                        f"  ce qui suggere des structures syntaxiques legerement\n"
                        f"  plus complexes.\n")

    print(f"\n  -> Rapport ecrit : {output_path}")

    # ── CSV export ────────────────────────────────────────────────────

    csv_path = output_path.replace(".txt", ".csv")
    if not csv_path.endswith(".csv"):
        csv_path = output_path + ".csv"

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> CSV detaille  : {csv_path}")

    print(f"\n{BANNER}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Comparaison des distances syntaxiques SMS vs Philosophie.",
    )
    p.add_argument(
        "--results-dir", "-r",
        default=str(_REPO_ROOT / "results"),
        help="Dossier racine des resultats (defaut : <repo>/results).",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Chemin du rapport de sortie (defaut : <results-dir>/comparaison_distances/rapport.txt).",
    )
    args = p.parse_args()

    output = args.output or os.path.join(
        args.results_dir, "comparaison_distances", "rapport.txt"
    )

    compare(args.results_dir, output)


if __name__ == "__main__":
    main()
