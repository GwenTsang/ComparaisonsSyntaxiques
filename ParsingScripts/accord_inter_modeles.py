#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Accord inter-modèles

  Compare les résultats de 2 à 4 parsers (dossiers contenant chacun un
  fichier resultats_par_sms.csv) et identifie :
    • les features avec le plus fort accord inter-modèles
    • les features avec le plus faible accord

  Métriques :
    - Corrélation de Pearson (pairwise)
    - Écart absolu moyen inter-modèles (MAD)
    - Classement des features par accord
==========================================================================
"""

import argparse
import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
COLS = ["SMS", "Transcodage_1", "Transcodage_2"]

FKEYS = [
    "profondeur_arbre_max",
    "profondeur_arbre_moy",
    "distance_dependance_max",
    "distance_dependance_moy",
    "distance_dependance_var",
    "nb_dependances_moy_phrase",
    "nb_dependances_var_phrase",
    "nb_phrases",
    "nb_tokens",
]

BANNER = "=" * 76


def section(title: str):
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


# ════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════════════════
def load_results(dirs: list[str]) -> dict[str, pd.DataFrame]:
    """Charge resultats_par_sms.csv depuis chaque dossier."""
    data = {}
    for d in dirs:
        csv_path = os.path.join(d, "resultats_par_sms.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"  ✗ Fichier introuvable : {csv_path}")
        name = Path(d).name
        data[name] = pd.read_csv(csv_path)
        print(f"  ✓ {name} : {len(data[name])} SMS  ({csv_path})")
    return data


# ════════════════════════════════════════════════════════════════════════
# CORRÉLATIONS PAIRWISE (Pearson)
# ════════════════════════════════════════════════════════════════════════
def pairwise_correlations(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Corrélation de Pearson entre chaque paire de modèles, par feature."""
    models = list(data.keys())
    pairs = list(itertools.combinations(models, 2))
    rows = []

    for col in COLS:
        for feat in FKEYS:
            cn = f"{col}_{feat}"
            for m1, m2 in pairs:
                if cn not in data[m1].columns or cn not in data[m2].columns:
                    continue
                v1 = data[m1][cn].dropna()
                v2 = data[m2][cn].dropna()
                # Align on shared indices
                idx = v1.index.intersection(v2.index)
                if len(idx) < 10:
                    continue
                r, p = sp_stats.pearsonr(v1.loc[idx], v2.loc[idx])
                rows.append({
                    "corpus": col, "feature": feat,
                    "modele_1": m1, "modele_2": m2,
                    "pearson_r": round(r, 4),
                    "p_value": p,
                    "n": len(idx),
                })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# ÉCART ABSOLU MOYEN INTER-MODÈLES  (MAD)
# ════════════════════════════════════════════════════════════════════════
def mean_absolute_deviation(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pour chaque SMS × feature, calcule l'écart absolu moyen entre
    tous les modèles (pairwise mean absolute deviation), puis moyenne
    sur tous les SMS.
    """
    models = list(data.keys())
    n = min(len(df) for df in data.values())
    rows = []

    for col in COLS:
        for feat in FKEYS:
            cn = f"{col}_{feat}"
            vals = []
            for m in models:
                if cn in data[m].columns:
                    vals.append(data[m][cn].iloc[:n].values)
            if len(vals) < 2:
                continue

            arr = np.array(vals)  # shape: (n_models, n_sms)
            # Pairwise MAD per SMS
            mad_per_sms = np.zeros(n)
            n_pairs = 0
            for i, j in itertools.combinations(range(len(vals)), 2):
                mask = np.isfinite(arr[i]) & np.isfinite(arr[j])
                mad_per_sms[mask] += np.abs(arr[i][mask] - arr[j][mask])
                n_pairs += 1

            if n_pairs > 0:
                mad_per_sms /= n_pairs

            avg_mad = float(np.nanmean(mad_per_sms))
            std_mad = float(np.nanstd(mad_per_sms))

            rows.append({
                "corpus": col, "feature": feat,
                "mad_moyen": round(avg_mad, 4),
                "mad_ecart_type": round(std_mad, 4),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# CLASSEMENT PAR ACCORD
# ════════════════════════════════════════════════════════════════════════
def rank_features(corr_df: pd.DataFrame, mad_df: pd.DataFrame):
    """
    Affiche les features classées par accord (le plus fort → plus faible).
    Utilise la moyenne des corrélations pairwise comme proxy.
    """
    section("CLASSEMENT PAR ACCORD INTER-MODÈLES")

    if corr_df.empty:
        print("  Pas assez de données pour un classement.")
        return

    # Mean Pearson r per (corpus, feature)
    agg = corr_df.groupby(["corpus", "feature"])["pearson_r"].mean().reset_index()
    agg.columns = ["corpus", "feature", "mean_r"]

    # Merge with MAD
    if not mad_df.empty:
        agg = agg.merge(mad_df[["corpus", "feature", "mad_moyen"]],
                        on=["corpus", "feature"], how="left")
    else:
        agg["mad_moyen"] = np.nan

    # ── Plus fort accord ──
    print("\n  ─── PLUS FORT ACCORD (Pearson r le plus élevé) ───")
    top = agg.nlargest(15, "mean_r")
    print(f"\n  {'Corpus':<18} {'Feature':<34} {'r moyen':>8} {'MAD':>10}")
    print("  " + "─" * 74)
    for _, row in top.iterrows():
        mad_str = f"{row['mad_moyen']:.3f}" if pd.notna(row["mad_moyen"]) else "N/A"
        print(f"  {row['corpus']:<18} {row['feature']:<34} {row['mean_r']:>8.4f} {mad_str:>10}")

    # ── Plus faible accord ──
    print("\n  ─── PLUS FAIBLE ACCORD (Pearson r le plus bas) ───")
    bot = agg.nsmallest(15, "mean_r")
    print(f"\n  {'Corpus':<18} {'Feature':<34} {'r moyen':>8} {'MAD':>10}")
    print("  " + "─" * 74)
    for _, row in bot.iterrows():
        mad_str = f"{row['mad_moyen']:.3f}" if pd.notna(row["mad_moyen"]) else "N/A"
        print(f"  {row['corpus']:<18} {row['feature']:<34} {row['mean_r']:>8.4f} {mad_str:>10}")

    return agg


# ════════════════════════════════════════════════════════════════════════
# PER-SMS DISAGREEMENT SCORES
# ════════════════════════════════════════════════════════════════════════
def per_sms_disagreement(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pour chaque SMS, calcule un score de désaccord global entre les modèles
    (somme des MAD normalisées sur toutes les features).
    """
    models = list(data.keys())
    n = min(len(df) for df in data.values())

    scores = np.zeros(n)
    n_feats = 0

    for col in COLS:
        for feat in FKEYS:
            cn = f"{col}_{feat}"
            vals = []
            for m in models:
                if cn in data[m].columns:
                    vals.append(data[m][cn].iloc[:n].values)
            if len(vals) < 2:
                continue

            arr = np.array(vals)
            # Range per SMS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                range_per_sms = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
                range_per_sms = np.nan_to_num(range_per_sms, nan=0.0)
            scores += range_per_sms
            n_feats += 1

    rows = [{"sms_id": i + 1, "desaccord_total": round(scores[i], 3)}
            for i in range(n)]
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# ARGUMENTS CLI
# ════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calcule l'accord inter-modèles à partir de resultats_par_sms.csv.",
    )
    p.add_argument(
        "--dirs", nargs="+", required=True, metavar="DIR",
        help="2 à 4 dossiers contenant chacun resultats_par_sms.csv.",
    )
    p.add_argument(
        "--output", default="./accord_inter_modeles",
        help="Dossier de sortie (défaut : ./accord_inter_modeles).",
    )
    args = p.parse_args()

    if len(args.dirs) < 2:
        p.error("Au moins 2 dossiers sont requis.")
    if len(args.dirs) > 4:
        p.error("Au maximum 4 dossiers sont acceptés.")

    return args


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    section("CHARGEMENT DES RÉSULTATS")
    data = load_results(args.dirs)
    models = list(data.keys())
    print(f"\n  {len(models)} modèle(s) : {', '.join(models)}")

    # ── Corrélations pairwise ──
    section("CORRÉLATIONS PAIRWISE (Pearson)")
    corr_df = pairwise_correlations(data)
    if not corr_df.empty:
        for (m1, m2), grp in corr_df.groupby(["modele_1", "modele_2"]):
            avg_r = grp["pearson_r"].mean()
            print(f"\n  {m1}  ↔  {m2}  (r moyen = {avg_r:.4f})")
            print(f"  {'Corpus':<18} {'Feature':<34} {'r':>8} {'p':>12}")
            print("  " + "─" * 76)
            for _, row in grp.iterrows():
                sig = "***" if row["p_value"] < .001 else "**" if row["p_value"] < .01 \
                      else "*" if row["p_value"] < .05 else ""
                print(f"  {row['corpus']:<18} {row['feature']:<34} "
                      f"{row['pearson_r']:>8.4f} {row['p_value']:>12.2e} {sig}")

    # ── MAD ──
    section("ÉCART ABSOLU MOYEN (MAD)")
    mad_df = mean_absolute_deviation(data)
    if not mad_df.empty:
        print(f"\n  {'Corpus':<18} {'Feature':<34} {'MAD moyen':>10} {'σ':>10}")
        print("  " + "─" * 76)
        for _, row in mad_df.iterrows():
            print(f"  {row['corpus']:<18} {row['feature']:<34} "
                  f"{row['mad_moyen']:>10.4f} {row['mad_ecart_type']:>10.4f}")

    # ── Classement ──
    agg_df = rank_features(corr_df, mad_df)

    # ── Désaccord par SMS ──
    section("TOP 20 SMS AVEC LE PLUS DE DÉSACCORD")
    disag_df = per_sms_disagreement(data)
    top20 = disag_df.nlargest(20, "desaccord_total")
    for _, row in top20.iterrows():
        print(f"  SMS n°{int(row['sms_id']):>4}  désaccord = {row['desaccord_total']:.2f}")

    # ── Export ──
    section("EXPORT")
    corr_csv = os.path.join(args.output, "correlations_pairwise.csv")
    corr_df.to_csv(corr_csv, index=False, encoding="utf-8")
    print(f"  → {corr_csv}")

    mad_csv = os.path.join(args.output, "mad_par_feature.csv")
    mad_df.to_csv(mad_csv, index=False, encoding="utf-8")
    print(f"  → {mad_csv}")

    if agg_df is not None:
        rank_csv = os.path.join(args.output, "classement_accord.csv")
        agg_df.to_csv(rank_csv, index=False, encoding="utf-8")
        print(f"  → {rank_csv}")

    disag_csv = os.path.join(args.output, "desaccord_par_sms.csv")
    disag_df.to_csv(disag_csv, index=False, encoding="utf-8")
    print(f"  → {disag_csv}")

    print(f"\n{BANNER}")
    print(f"  ✓ Analyse d'accord terminée. Résultats dans : {args.output}/")
    print(BANNER)


if __name__ == "__main__":
    main()
