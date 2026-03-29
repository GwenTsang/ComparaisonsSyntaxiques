#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
==========================================================================
  Comparaison 3-way des structures syntaxiques :
  SMS  vs  Philosophie  vs  Le Monde (UD_FTB)

  Réutilise la logique de détection de structures_syntaxiques.py
  et compare les proportions de chaque structure entre les trois
  genres textuels.

  Usage :
    python compare_structures_3way.py
    python compare_structures_3way.py --results-dir ../results
==========================================================================
"""

import argparse
import csv
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

# ── Make ParsingScripts importable ──
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "ParsingScripts"))

from structures_syntaxiques import (
    STRUCTURE_FEATURES,
    read_conllu,
    detect_structures,
)

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

ALL_MODELS = ["fsmb", "gsd", "rhapsodie", "sequoia", "stanza", "zenodo-spoken"]

# Best-affinity model per genre (oral ↔ fsmb, formal written ↔ gsd)
MODEL_AFFINITY = {
    "SMS":         "fsmb",
    "Philosophie": "gsd",
}

FEATURE_LABELS = {
    "subordonnees_qui":         "Sub. rel. (qui)",
    "subordonnees_que":         "Sub. rel. (que)",
    "subordonnees_prep_lequel": "Sub. prép+lequel",
    "subordonnees_dont":        "Sub. rel. (dont)",
    "completives":              "Complétives",
    "hypothetiques":            "Hypothétiques",
    "gerondif":                 "Gérondif",
    "incises":                  "Incises",
    "propositions_coordonnees": "Prop. coordonnées",
}

BANNER = "=" * 90


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════

def _find_conllu_files(model_dir: str) -> list[str]:
    """Return sorted list of output_*.conllu files in a model directory."""
    if not os.path.isdir(model_dir):
        return []
    return sorted(
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith("output_") and f.endswith(".conllu")
    )


def load_corpus_per_sentence(results_dir: str, corpus_name: str,
                             models: list[str]) -> dict[str, dict[str, list[int]]]:
    """
    Load all conllu files for a corpus and return per-sentence counts
    of each structure, grouped by model.

    Returns: { model_name: { feature: [count_sent1, count_sent2, ...] } }
    """
    data: dict[str, dict[str, list[int]]] = {}

    for model in models:
        model_dir = os.path.join(results_dir, corpus_name, model)
        conllu_files = _find_conllu_files(model_dir)
        if not conllu_files:
            continue

        feat_lists: dict[str, list[int]] = {f: [] for f in STRUCTURE_FEATURES}

        for fp in conllu_files:
            sentences = read_conllu(fp)
            for sent in sentences:
                counts = detect_structures([sent])
                for feat in STRUCTURE_FEATURES:
                    feat_lists[feat].append(counts[feat])

        data[model] = feat_lists

    return data


def load_reference_corpus(conllu_path: str) -> dict[str, list[int]]:
    """
    Parse the Le Monde (UD_FTB) reference conllu and return per-sentence
    structure counts.

    Returns: { feature: [count_sent1, count_sent2, ...] }
    """
    feat_lists: dict[str, list[int]] = {f: [] for f in STRUCTURE_FEATURES}

    if not os.path.isfile(conllu_path):
        print(f"  ⚠ Fichier de référence introuvable : {conllu_path}")
        return feat_lists

    sentences = read_conllu(conllu_path)
    print(f"  Le Monde : {len(sentences)} phrases chargées")

    for sent in sentences:
        counts = detect_structures([sent])
        for feat in STRUCTURE_FEATURES:
            feat_lists[feat].append(counts[feat])

    return feat_lists


# ════════════════════════════════════════════════════════════════════════
# STATISTICS HELPERS
# ════════════════════════════════════════════════════════════════════════

def _safe(val, fmt=".4f"):
    return f"{val:{fmt}}" if val is not None else "N/A"


def compute_summary(values: list) -> dict:
    """Compute summary statistics for a list of numeric values."""
    if not values:
        return {"mean": None, "std": None, "total": None, "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "total": int(np.sum(arr)),
        "n":     len(arr),
    }


def mann_whitney(a: list, b: list) -> tuple:
    """Run Mann-Whitney U test, returning (U, p-value) or (None, None)."""
    if len(a) < 3 or len(b) < 3:
        return None, None
    try:
        u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ValueError:
        return None, None


# ════════════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════════════

def build_report(sms_data, philo_data, lemonde_data, models, output_dir):
    """
    Build the full comparison report (text + CSV).
    """
    txt_path = os.path.join(output_dir, "rapport_structures_3way.txt")
    csv_per_model_path = os.path.join(output_dir, "detail_par_modele.csv")
    csv_global_path = os.path.join(output_dir, "synthese_globale.csv")
    csv_tests_path = os.path.join(output_dir, "tests_statistiques.csv")

    lines: list[str] = []
    detail_rows: list[dict] = []
    global_rows: list[dict] = []
    test_rows: list[dict] = []

    lines.append(BANNER)
    lines.append("  COMPARAISON 3-WAY DES STRUCTURES SYNTAXIQUES")
    lines.append("  SMS  vs  PHILOSOPHIE  vs  LE MONDE (UD_FTB)")
    lines.append(BANNER)

    # ── 1. Per-Model Detail ──────────────────────────────────────────
    lines.append("")
    lines.append("  1. DÉTAIL PAR MODÈLE")
    lines.append("  " + "-" * 85)
    header = (f"  {'Modèle':<16} {'Structure':<22} "
              f"{'SMS moy':>9} {'Philo moy':>9} {'LeMonde':>9}")
    lines.append(header)
    lines.append("  " + "-" * 85)

    for model in models:
        sms_m   = sms_data.get(model, {})
        philo_m = philo_data.get(model, {})
        for feat in STRUCTURE_FEATURES:
            s_sms   = compute_summary(sms_m.get(feat, []))
            s_philo = compute_summary(philo_m.get(feat, []))
            s_lm    = compute_summary(lemonde_data.get(feat, []))

            label = FEATURE_LABELS.get(feat, feat)
            sm = _safe(s_sms["mean"], ".3f").rjust(9)
            pm = _safe(s_philo["mean"], ".3f").rjust(9)
            lm = _safe(s_lm["mean"], ".3f").rjust(9)

            lines.append(f"  {model:<16} {label:<22} {sm} {pm} {lm}")
            detail_rows.append({
                "modele": model,
                "structure": feat,
                "sms_moy": s_sms["mean"],
                "sms_total": s_sms["total"],
                "sms_n": s_sms["n"],
                "philo_moy": s_philo["mean"],
                "philo_total": s_philo["total"],
                "philo_n": s_philo["n"],
                "lemonde_moy": s_lm["mean"],
                "lemonde_total": s_lm["total"],
                "lemonde_n": s_lm["n"],
            })
        lines.append("")

    # ── 2. Global Synthesis (average across models) ──────────────────
    lines.append("")
    lines.append("  2. SYNTHÈSE GLOBALE (MOYENNE INTER-MODÈLES + MEILLEUR MODÈLE)")
    lines.append("  " + "-" * 85)
    lines.append(f"  {'Structure':<22} {'SMS glob':>9} {'SMS best':>9} "
                 f"{'Philo glob':>11} {'Philo best':>11} {'Le Monde':>9}")
    lines.append("  " + "-" * 85)

    sms_best_model   = MODEL_AFFINITY["SMS"]
    philo_best_model = MODEL_AFFINITY["Philosophie"]

    for feat in STRUCTURE_FEATURES:
        # Global Mean across ALL models
        all_sms   = [v for m in sms_data.values()   for v in m.get(feat, [])]
        all_philo = [v for m in philo_data.values()  for v in m.get(feat, [])]

        g_sms   = compute_summary(all_sms)
        g_philo = compute_summary(all_philo)
        g_lm    = compute_summary(lemonde_data.get(feat, []))

        # Best-affinity model
        b_sms   = compute_summary(sms_data.get(sms_best_model, {}).get(feat, []))
        b_philo = compute_summary(philo_data.get(philo_best_model, {}).get(feat, []))

        label = FEATURE_LABELS.get(feat, feat)
        lines.append(
            f"  {label:<22} "
            f"{_safe(g_sms['mean'], '.4f'):>9} "
            f"{_safe(b_sms['mean'], '.4f'):>9} "
            f"{_safe(g_philo['mean'], '.4f'):>11} "
            f"{_safe(b_philo['mean'], '.4f'):>11} "
            f"{_safe(g_lm['mean'], '.4f'):>9}"
        )

        global_rows.append({
            "structure": feat,
            "sms_global_moy": g_sms["mean"],
            "sms_global_total": g_sms["total"],
            f"sms_best_moy ({sms_best_model})": b_sms["mean"],
            "philo_global_moy": g_philo["mean"],
            "philo_global_total": g_philo["total"],
            f"philo_best_moy ({philo_best_model})": b_philo["mean"],
            "lemonde_moy": g_lm["mean"],
            "lemonde_total": g_lm["total"],
        })

    # ── 3. Statistical Tests (Mann-Whitney U) ────────────────────────
    lines.append("")
    lines.append("")
    lines.append("  3. TESTS STATISTIQUES (Mann-Whitney U, seuil α = 0.05)")
    lines.append("     Comparaison sur les comptages par phrase, meilleur modèle par genre.")
    lines.append("  " + "-" * 85)
    lines.append(f"  {'Structure':<22} {'SMS↔Philo':>12} {'p-val':>9} "
                 f"{'SMS↔LeMonde':>12} {'p-val':>9} "
                 f"{'Philo↔LeMonde':>14} {'p-val':>9}")
    lines.append("  " + "-" * 85)

    for feat in STRUCTURE_FEATURES:
        a_sms   = sms_data.get(sms_best_model, {}).get(feat, [])
        a_philo = philo_data.get(philo_best_model, {}).get(feat, [])
        a_lm    = lemonde_data.get(feat, [])

        u_sp, p_sp = mann_whitney(a_sms, a_philo)
        u_sl, p_sl = mann_whitney(a_sms, a_lm)
        u_pl, p_pl = mann_whitney(a_philo, a_lm)

        label = FEATURE_LABELS.get(feat, feat)

        def _sig(p):
            if p is None:
                return "   -   "
            return f"{'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns':>7}"

        def _pval(p):
            if p is None:
                return "    -    "
            return f"{p:>9.4f}"

        lines.append(
            f"  {label:<22} "
            f"{_sig(p_sp):>12} {_pval(p_sp)} "
            f"{_sig(p_sl):>12} {_pval(p_sl)} "
            f"{_sig(p_pl):>14} {_pval(p_pl)}"
        )

        test_rows.append({
            "structure": feat,
            "U_sms_philo": u_sp, "p_sms_philo": p_sp,
            "sig_sms_philo": "***" if p_sp and p_sp < 0.001 else
                             "**"  if p_sp and p_sp < 0.01  else
                             "*"   if p_sp and p_sp < 0.05  else "ns" if p_sp else None,
            "U_sms_lemonde": u_sl, "p_sms_lemonde": p_sl,
            "sig_sms_lemonde": "***" if p_sl and p_sl < 0.001 else
                               "**"  if p_sl and p_sl < 0.01  else
                               "*"   if p_sl and p_sl < 0.05  else "ns" if p_sl else None,
            "U_philo_lemonde": u_pl, "p_philo_lemonde": p_pl,
            "sig_philo_lemonde": "***" if p_pl and p_pl < 0.001 else
                                 "**"  if p_pl and p_pl < 0.01  else
                                 "*"   if p_pl and p_pl < 0.05  else "ns" if p_pl else None,
        })

    lines.append("")
    lines.append("  Légende : *** p<0.001  ** p<0.01  * p<0.05  ns = non significatif")

    # ── 4. Interpretation ────────────────────────────────────────────
    lines.append("")
    lines.append("")
    lines.append("  4. INTERPRÉTATION")
    lines.append("  " + "-" * 85)

    # Count significant differences
    sig_sp = sum(1 for r in test_rows if r["p_sms_philo"] is not None and r["p_sms_philo"] < 0.05)
    sig_sl = sum(1 for r in test_rows if r["p_sms_lemonde"] is not None and r["p_sms_lemonde"] < 0.05)
    sig_pl = sum(1 for r in test_rows if r["p_philo_lemonde"] is not None and r["p_philo_lemonde"] < 0.05)
    n_feats = len(STRUCTURE_FEATURES)

    lines.append(f"  Nombre de structures comparées : {n_feats}")
    lines.append(f"  Différences significatives (p < 0.05) :")
    lines.append(f"    • SMS ↔ Philosophie  : {sig_sp}/{n_feats}")
    lines.append(f"    • SMS ↔ Le Monde     : {sig_sl}/{n_feats}")
    lines.append(f"    • Philosophie ↔ Le Monde : {sig_pl}/{n_feats}")
    lines.append("")

    if sig_sp > 0 or sig_sl > 0 or sig_pl > 0:
        lines.append("  → OUI, il existe des différences significatives entre les corpus.")
        lines.append("")

        # Highlight which structures differ
        lines.append("  Structures présentant des différences significatives :")
        for r in test_rows:
            feat = r["structure"]
            label = FEATURE_LABELS.get(feat, feat)
            diffs = []
            if r["p_sms_philo"] is not None and r["p_sms_philo"] < 0.05:
                diffs.append(f"SMS↔Philo ({r['sig_sms_philo']})")
            if r["p_sms_lemonde"] is not None and r["p_sms_lemonde"] < 0.05:
                diffs.append(f"SMS↔LeMonde ({r['sig_sms_lemonde']})")
            if r["p_philo_lemonde"] is not None and r["p_philo_lemonde"] < 0.05:
                diffs.append(f"Philo↔LeMonde ({r['sig_philo_lemonde']})")
            if diffs:
                lines.append(f"    • {label:<22} : {', '.join(diffs)}")

        lines.append("")
        lines.append("  Ces résultats confirment que la complexité syntaxique (variété et fréquence")
        lines.append("  des structures subordonnées, incises, et coordonnées) diffère entre le langage")
        lines.append("  informel/oral (SMS), le journalisme (Le Monde) et la prose philosophique.")
    else:
        lines.append("  → Aucune différence significative détectée entre les corpus")
        lines.append("    pour les structures syntaxiques étudiées.")

    lines.append("")
    lines.append(BANNER)

    # ── Write Text Report ────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    print(f"\n  -> Rapport texte : {txt_path}")

    # -- Write CSV: Detail per model --
    if detail_rows:
        with open(csv_per_model_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"  -> CSV detaille  : {csv_per_model_path}")

    # -- Write CSV: Global synthesis --
    if global_rows:
        with open(csv_global_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()))
            writer.writeheader()
            writer.writerows(global_rows)
        print(f"  -> CSV synthese  : {csv_global_path}")

    # -- Write CSV: Statistical tests --
    if test_rows:
        with open(csv_tests_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
            writer.writeheader()
            writer.writerows(test_rows)
        print(f"  -> CSV tests     : {csv_tests_path}")

    # ── Print to console ──
    for l in lines:
        print(l)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Comparaison 3-way des structures syntaxiques : SMS vs Philo vs Le Monde."
    )
    p.add_argument(
        "--results-dir", "-r",
        default=str(_REPO_ROOT / "results"),
        help="Dossier racine des résultats.",
    )
    p.add_argument(
        "--conllu-path", "-c",
        default=str(_REPO_ROOT / "UD_FTB" / "fr_ftb-ud-merged.conllu"),
        help="Chemin du fichier .conllu de référence Le Monde.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Dossier de sortie (défaut : results/comparaison_structures).",
    )
    args = p.parse_args()

    results_dir = args.results_dir
    conllu_path = args.conllu_path
    output_dir  = args.output_dir or os.path.join(results_dir, "comparaison_structures")

    # Discover available models
    models = [
        m for m in ALL_MODELS
        if any(
            os.path.isdir(os.path.join(results_dir, c, m))
            for c in ["SMS", "Philosophie"]
        )
    ]
    if not models:
        sys.exit("  X Aucun modele trouve dans les dossiers SMS / Philosophie.")

    print(f"\n{BANNER}")
    print("  COMPARAISON 3-WAY DES STRUCTURES SYNTAXIQUES")
    print(BANNER)
    print(f"  Dossier résultats : {results_dir}")
    print(f"  Fichier Le Monde  : {conllu_path}")
    print(f"  Modèles détectés  : {', '.join(models)}")
    print()

    # 1. Load data
    print("  Chargement SMS ...")
    sms_data = load_corpus_per_sentence(results_dir, "SMS", models)
    for m, feats in sms_data.items():
        n = len(next(iter(feats.values()), []))
        print(f"    {m}: {n} phrases")

    print("  Chargement Philosophie ...")
    philo_data = load_corpus_per_sentence(results_dir, "Philosophie", models)
    for m, feats in philo_data.items():
        n = len(next(iter(feats.values()), []))
        print(f"    {m}: {n} phrases")

    print("  Chargement Le Monde (UD_FTB) ...")
    lemonde_data = load_reference_corpus(conllu_path)

    # 2. Build report
    build_report(sms_data, philo_data, lemonde_data, models, output_dir)

    print(f"\n{BANNER}")


if __name__ == "__main__":
    main()
