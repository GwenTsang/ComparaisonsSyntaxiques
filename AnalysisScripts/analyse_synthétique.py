#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse consolidée et synthétique

- Extrait distance_dependance_moy (variable dépendante) depuis 3 corpus
- Produit un fichier commun CSV
- Calcule statistiques descriptives et inférentielles
- Modèle retenu : GSD pour SMS/Philo, gold standard pour Le Monde (FTB)
- Version SMS retenue : Transcodage_1 (transcription étudiants)

"""

import csv
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
RESULTS_DIR = _REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "devoir"

# ── Le Monde CoNLL-U parser ──────────────────────────────────────────

def parse_lemonde_per_sentence(conllu_path):
    """Parse FTB CoNLL-U, return per-sentence distance_dependance_moy."""
    values = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip() and not l.startswith("#")]
        if not lines:
            continue
        tokens = []
        for line in lines:
            cols = line.split("\t")
            if len(cols) == 10:
                if "-" in cols[0] or "." in cols[0]:
                    continue
                try:
                    w_id = int(cols[0])
                    w_head = int(cols[6])
                except ValueError:
                    continue
                tokens.append((w_id, w_head))
        if not tokens:
            continue
        dist_deps = [abs(t[0] - t[1]) for t in tokens if t[1] != 0]
        if dist_deps:
            values.append(float(np.mean(dist_deps)))
    return values


# ── Chargement des corpus ────────────────────────────────────────────

def load_sms_gsd(results_dir):
    """Charge distance_dependance_moy pour SMS (GSD, Transcodage_1)."""
    csv_path = os.path.join(results_dir, "SMS", "gsd", "resultats_par_sms.csv")
    df = pd.read_csv(csv_path)
    col = "Transcodage_1_distance_dependance_moy"
    return pd.to_numeric(df[col], errors="coerce").dropna().tolist()


def load_philo_gsd(results_dir):
    """Charge distance_dependance_moy pour Philosophie (GSD)."""
    csv_path = os.path.join(results_dir, "Philosophie", "gsd", "resultats_par_sms.csv")
    df = pd.read_csv(csv_path)
    col = "Texte_distance_dependance_moy"
    return pd.to_numeric(df[col], errors="coerce").dropna().tolist()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("  ANALYSE CONSOLIDÉE — DEVOIR MAISON")
    print("=" * 70)
    
    # 1. Charger les données
    print("\n1. Chargement des données...")
    sms_vals = load_sms_gsd(str(RESULTS_DIR))
    philo_vals = load_philo_gsd(str(RESULTS_DIR))
    lemonde_vals = parse_lemonde_per_sentence(
        str(_REPO_ROOT / "UD_FTB" / "fr_ftb-ud-merged.conllu")
    )
    
    print(f"   SMS (Transcodage_1, GSD):  n = {len(sms_vals)}")
    print(f"   Philosophie (GSD):          n = {len(philo_vals)}")
    print(f"   Le Monde (FTB gold):        n = {len(lemonde_vals)}")
    
    # 2. Fichier commun CSV
    print("\n2. Export du fichier commun...")
    csv_path = OUTPUT_DIR / "donnees_consolidees.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["genre", "observation_id", "distance_dependance_moy"])
        for i, v in enumerate(sms_vals, 1):
            w.writerow(["SMS", i, round(v, 6)])
        for i, v in enumerate(philo_vals, 1):
            w.writerow(["Philosophie", i, round(v, 6)])
        for i, v in enumerate(lemonde_vals, 1):
            w.writerow(["Le_Monde", i, round(v, 6)])
    print(f"   → {csv_path}")
    
    # 3. Statistiques descriptives
    print("\n3. Statistiques descriptives")
    print("-" * 70)
    
    desc_rows = []
    for name, vals in [("SMS", sms_vals), ("Philosophie", philo_vals), ("Le Monde", lemonde_vals)]:
        arr = np.array(vals)
        row = {
            "Genre": name,
            "N": len(arr),
            "Moyenne": np.mean(arr),
            "Écart-type": np.std(arr, ddof=1),
            "Médiane": np.median(arr),
            "Q1": np.percentile(arr, 25),
            "Q3": np.percentile(arr, 75),
            "Min": np.min(arr),
            "Max": np.max(arr),
        }
        desc_rows.append(row)
        print(f"   {name:>14s}: M = {row['Moyenne']:.4f}, SD = {row['Écart-type']:.4f}, "
              f"Mdn = {row['Médiane']:.4f}, N = {row['N']}")
    
    desc_df = pd.DataFrame(desc_rows)
    desc_csv = OUTPUT_DIR / "statistiques_descriptives.csv"
    desc_df.to_csv(desc_csv, index=False, float_format="%.4f")
    print(f"   → {desc_csv}")
    
    # 4. Vérification de la normalité (Shapiro-Wilk sur échantillons)
    print("\n4. Tests de normalité (Shapiro-Wilk, échantillon n=500)")
    print("-" * 70)
    
    normality_ok = True
    for name, vals in [("SMS", sms_vals), ("Philosophie", philo_vals), ("Le Monde", lemonde_vals)]:
        # Shapiro-Wilk limité à 5000 obs, échantillonner si besoin
        arr = np.array(vals)
        if len(arr) > 500:
            rng = np.random.default_rng(42)
            sample = rng.choice(arr, size=500, replace=False)
        else:
            sample = arr
        stat_sw, p_sw = stats.shapiro(sample)
        sig = "***" if p_sw < 0.001 else ("**" if p_sw < 0.01 else ("*" if p_sw < 0.05 else "ns"))
        normal = p_sw > 0.05
        if not normal:
            normality_ok = False
        print(f"   {name:>14s}: W = {stat_sw:.4f}, p = {p_sw:.2e} {sig}")
    
    if not normality_ok:
        print("   → Au moins un groupe ne suit pas la loi normale.")
        print("   → Utilisation de tests non-paramétriques.")
    
    # 5. Test de Kruskal-Wallis
    print("\n5. Test de Kruskal-Wallis (H)")
    print("-" * 70)
    
    H_stat, p_kw = stats.kruskal(sms_vals, philo_vals, lemonde_vals)
    N_total = len(sms_vals) + len(philo_vals) + len(lemonde_vals)
    # Eta-squared pour Kruskal-Wallis : η² = (H - k + 1) / (N - k)
    k = 3
    eta_sq = (H_stat - k + 1) / (N_total - k)
    
    sig_kw = "***" if p_kw < 0.001 else ("**" if p_kw < 0.01 else ("*" if p_kw < 0.05 else "ns"))
    print(f"   H({k-1}) = {H_stat:.4f}, p = {p_kw:.2e} {sig_kw}")
    print(f"   η² = {eta_sq:.4f} (taille d'effet)")
    if eta_sq < 0.01:
        effect_label = "négligeable"
    elif eta_sq < 0.06:
        effect_label = "faible"
    elif eta_sq < 0.14:
        effect_label = "modéré"
    else:
        effect_label = "fort"
    print(f"   Interprétation : effet {effect_label}")
    
    # 6. Tests post-hoc de Dunn (avec correction de Bonferroni)
    print("\n6. Tests post-hoc de Dunn (correction Bonferroni)")
    print("-" * 70)
    
    # Construire un DataFrame pour scikit_posthocs
    all_vals = sms_vals + philo_vals + lemonde_vals
    all_groups = (["SMS"] * len(sms_vals) + 
                  ["Philosophie"] * len(philo_vals) + 
                  ["Le_Monde"] * len(lemonde_vals))
    
    dunn_df = pd.DataFrame({"genre": all_groups, "distance_dep_moy": all_vals})
    dunn_result = sp.posthoc_dunn(
        dunn_df, val_col="distance_dep_moy", group_col="genre", p_adjust="bonferroni"
    )
    print(dunn_result.to_string(float_format=lambda x: f"{x:.2e}"))
    
    # Aussi calculer les U de Mann-Whitney pour chaque paire avec taille d'effet r
    print("\n   Détail par paire (Mann-Whitney U + taille d'effet r) :")
    pairs = [
        ("SMS", sms_vals, "Philosophie", philo_vals),
        ("SMS", sms_vals, "Le Monde", lemonde_vals),
        ("Philosophie", philo_vals, "Le Monde", lemonde_vals),
    ]
    
    posthoc_rows = []
    for name1, v1, name2, v2 in pairs:
        U_stat, p_mw = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        n1, n2 = len(v1), len(v2)
        # Rank-biserial r = 1 - (2U)/(n1*n2)
        r_effect = 1 - (2 * U_stat) / (n1 * n2)
        sig = "***" if p_mw < 0.001 else ("**" if p_mw < 0.01 else ("*" if p_mw < 0.05 else "ns"))
        print(f"   {name1:>14s} ↔ {name2:<14s}: U = {U_stat:.0f}, p = {p_mw:.2e} {sig}, r = {r_effect:.4f}")
        posthoc_rows.append({
            "Comparaison": f"{name1} ↔ {name2}",
            "U": U_stat, "p": p_mw, "sig": sig, "r_effect": r_effect
        })
    
    # 7. Export des résultats inférentiels
    print("\n7. Export des résultats")
    print("-" * 70)
    
    inf_path = OUTPUT_DIR / "statistiques_inferentielles.csv"
    with open(inf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Test", "Statistique", "Valeur", "p", "Significativité", "Taille_effet"])
        w.writerow(["Kruskal-Wallis", f"H({k-1})", f"{H_stat:.4f}", f"{p_kw:.2e}", sig_kw, f"η²={eta_sq:.4f} ({effect_label})"])
        for row in posthoc_rows:
            w.writerow(["Mann-Whitney U (post-hoc)", "U", f"{row['U']:.0f}", 
                        f"{row['p']:.2e}", row['sig'], f"r={row['r_effect']:.4f}"])
    print(f"   → {inf_path}")
    
    # Dunn matrix
    dunn_path = OUTPUT_DIR / "dunn_posthoc.csv"
    dunn_result.to_csv(dunn_path, float_format="%.2e")
    print(f"   → {dunn_path}")
    
    print(f"\n{'=' * 70}")
    print("  TERMINÉ")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
