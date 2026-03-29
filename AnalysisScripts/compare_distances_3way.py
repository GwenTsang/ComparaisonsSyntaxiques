#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Comparaison des distances syntaxiques 3-way : 
  SMS vs Philosophie vs Le Monde (UD_FTB)

  Lit les fichiers resultats_par_sms.csv produits par chaque modele
  dans le dossier de resultats, calcule les metriques depuis le fichier 
  .conllu de reference, puis genere un rapport comparatif 3-way.

  Usage :
    python compare_distances_3way.py
==========================================================================
"""

import argparse
import csv
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

# Re-use existing robust logic from compare_distances
from compare_distances import (
    load_corpus_features,
    ALL_MODELS,
    MODEL_AFFINITY,
    DEP_FEATURES,
    FEATURE_LABELS,
    _safe_mean,
    _safe_std
)

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

BANNER = "=" * 90


# ── Le Monde .conllu Parser ──────────────────────────────────────────

def _tree_depth_conllu(sentence_tokens: list) -> int:
    children = defaultdict(list)
    root_id = None
    
    for t in sentence_tokens:
        if t["head"] == 0:
            root_id = t["id"]
        else:
            children[t["head"]].append(t["id"])
            
    if root_id is None:
        return 0
        
    stack = [(root_id, 1)]
    max_depth = 1
    while stack:
        node, d = stack.pop()
        if d > max_depth:
            max_depth = d
        for child in children.get(node, []):
            stack.append((child, d + 1))
            
    return max_depth


def parse_conllu_corpus(conllu_path: str) -> dict[str, list[float]]:
    """
    Lit un fichier conllu (Le Monde) et calcule les memes 6 features
    de base par phrase (traitee comme 1 doc SMS).
    """
    features_data = {f: [] for f in DEP_FEATURES}
    
    if not os.path.isfile(conllu_path):
        print(f"  ! Fichier {conllu_path} introuvable.")
        return features_data

    with open(conllu_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Chaque "phrase" (block separe par une ligne vide) est consideree
    # comme un 'document' unitaire pour le corpus Le Monde.
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip() and not l.startswith("#")]
        if not lines:
            continue
            
        sentence_tokens = []
        for line in lines:
            cols = line.split("\t")
            if len(cols) == 10:
                # Ignorer les tokens composes e.g., "1-2"
                if "-" in cols[0] or "." in cols[0]:
                    continue
                try:
                    w_id = int(cols[0])
                    w_head = int(cols[6])
                except ValueError:
                    continue
                sentence_tokens.append({"id": w_id, "head": w_head})

        if not sentence_tokens:
            continue

        depth = _tree_depth_conllu(sentence_tokens)
        dist_deps = [abs(t["id"] - t["head"]) for t in sentence_tokens if t["head"] != 0]

        dep_count = len(dist_deps)
        
        # Meme logique que Stanza extract_features
        features_data["profondeur_arbre_max"].append(float(depth))
        features_data["profondeur_arbre_moy"].append(float(depth))
        
        features_data["distance_dependance_max"].append(float(np.max(dist_deps)) if dist_deps else 0.0)
        features_data["distance_dependance_moy"].append(float(np.mean(dist_deps)) if dist_deps else 0.0)
        features_data["distance_dependance_var"].append(float(np.var(dist_deps)) if dist_deps else 0.0)
        
        features_data["nb_dependances_moy_phrase"].append(float(dep_count))
        # Note: sms parsing uses len(sentences) = 1 here.

    return features_data


# ── main logic ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Comparaison 3-way SMS vs Philo vs Le Monde.")
    p.add_argument("--results-dir", "-r", default=str(_REPO_ROOT / "results"))
    p.add_argument("--conllu-path", "-c", default=str(_REPO_ROOT / "UD_FTB" / "fr_ftb-ud-merged.conllu"))
    p.add_argument("--output", "-o", default=None)
    args = p.parse_args()

    results_dir = args.results_dir
    conllu_path = args.conllu_path
    
    output_path = args.output or os.path.join(
        results_dir, "comparaison_distances", "rapport_3way.txt"
    )

    models = [
        m for m in ALL_MODELS
        if any(
            os.path.isfile(os.path.join(results_dir, c, m, "resultats_par_sms.csv"))
            for c in ["SMS", "Philosophie"]
        )
    ]

    print(f"\n{BANNER}")
    print("  COMPARAISON 3-WAY : SMS vs PHILOSOPHIE vs LE MONDE (Reference)")
    print(BANNER)

    # 1. Load Data
    sms_data  = load_corpus_features(results_dir, "SMS", models)
    philo_data = load_corpus_features(results_dir, "Philosophie", models)
    
    print(f"  Chargement {conllu_path} ...")
    lemonde_data = parse_conllu_corpus(conllu_path)

    # 2. Build Per-Model Table
    rows: list[dict] = []
    lines: list[str] = []

    lines.append(f"\n  {'Modele':<16} {'Feature':<25} "
                 f"{'SMS moy':>9} {'Philo moy':>9} {'LeMonde gold':>12}")
    lines.append("  " + "-" * 75)

    for model_name in models:
        sms_m  = sms_data.get(model_name, {})
        philo_m = philo_data.get(model_name, {})
        for feat in DEP_FEATURES:
            sms_mean   = _safe_mean(sms_m.get(feat, []))
            philo_mean = _safe_mean(philo_m.get(feat, []))
            lm_mean    = _safe_mean(lemonde_data.get(feat, []))

            s_sms = f"{sms_mean:>9.2f}" if sms_mean is not None else "N/A".rjust(9)
            s_phi = f"{philo_mean:>9.2f}" if philo_mean is not None else "N/A".rjust(9)
            s_lm  = f"{lm_mean:>12.2f}" if lm_mean is not None else "N/A".rjust(12)

            lines.append(f"  {model_name:<16} {FEATURE_LABELS.get(feat, feat):<25} {s_sms} {s_phi} {s_lm}")
            
            rows.append({
                "modele": model_name,
                "affinite": MODEL_AFFINITY.get(model_name, "?"),
                "feature": feat,
                "sms_mean": round(sms_mean, 4) if sms_mean else None,
                "philo_mean": round(philo_mean, 4) if philo_mean else None,
                "lemonde_mean": round(lm_mean, 4) if lm_mean else None,
            })
        lines.append("")

    for l in lines:
        print(l)

    # 3. Global Means (Simple Average + Best Match Weighting)
    print(f"\n  {'-' * 80}")
    print(f"  MOYENNES GLOBAL ET SELECTION DES MEILLEURS MODELES")
    print(f"  {'-' * 80}")
    print(f"  {'Feature':<25} {'SMS (fsmb)':>12} {'Philo (gsd)':>12} {'LeMonde (gold)':>15}")
    print(f"  {'-' * 80}")

    global_rows: list[dict] = []
    for feat in DEP_FEATURES:
        # Simple Global Mean
        all_sms   = [v for m in sms_data.values()  for v in m.get(feat, [])]
        all_philo = [v for m in philo_data.values() for v in m.get(feat, [])]
        
        sms_global_mean = _safe_mean(all_sms)
        philo_global_mean = _safe_mean(all_philo)
        lm_global_mean = _safe_mean(lemonde_data.get(feat, []))

        # "Best Model" Selection for Genre
        # For SMS -> fsmb represents the best match
        # For Philosophie -> gsd represents the best match
        sms_best   = _safe_mean(sms_data.get("fsmb", {}).get(feat, [])) or sms_global_mean
        philo_best = _safe_mean(philo_data.get("gsd", {}).get(feat, [])) or philo_global_mean
        
        s_s = f"{sms_best:>12.2f}" if sms_best is not None else "N/A".rjust(12)
        s_p = f"{philo_best:>12.2f}" if philo_best is not None else "N/A".rjust(12)
        s_l = f"{lm_global_mean:>15.2f}" if lm_global_mean is not None else "N/A".rjust(15)

        print(f"  {FEATURE_LABELS.get(feat, feat):<25} {s_s} {s_p} {s_l}")
        
        global_rows.append({
            "feature": feat,
            "sms_global_mean": round(sms_global_mean, 4) if sms_global_mean else None,
            "sms_best_mean": round(sms_best, 4) if sms_best else None,
            "philo_global_mean": round(philo_global_mean, 4) if philo_global_mean else None,
            "philo_best_mean": round(philo_best, 4) if philo_best else None,
            "lemonde_mean": round(lm_global_mean, 4) if lm_global_mean else None,
        })

    # 4. Write Output Report
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("COMPARAISON DES DISTANCES SYNTAXIQUES 3-WAY : SMS vs PHILOSOPHIE vs LE MONDE\n")
        f.write("=" * 80 + "\n\n")

        f.write("Aperçu des affinites des modeles utilises pour chaque corpus formel :\n")
        for m in models:
            aff = MODEL_AFFINITY.get(m, "?")
            f.write(f"  - {m:<20} affinite : {aff}\n")

        f.write("\n\n--- DETAIL PAR MODELE ---\n")
        for l in lines:
            f.write(l + "\n")

        f.write("\n\n--- MOYENNES (MEILLEURS MODELES POUR LE GENRE) ---\n")
        f.write("Dans cette selection, les resultats du corpus SMS prennent les valeurs de FSMB, et\n")
        f.write("Philosophie prend celles de GSD, qui correspondent mieux a leur genre (oral vs ecrit formel).\n\n")
        
        f.write(f"{'Feature':<25} {'SMS (FSMB)':>12} {'Philo (GSD)':>12} {'LeMonde (Gold)':>15}\n")
        f.write("-" * 66 + "\n")
        for gr in global_rows:
            s_s = f"{gr['sms_best_mean']:>12.4f}" if gr['sms_best_mean'] is not None else "N/A".rjust(12)
            s_p = f"{gr['philo_best_mean']:>12.4f}" if gr['philo_best_mean'] is not None else "N/A".rjust(12)
            s_l = f"{gr['lemonde_mean']:>15.4f}" if gr['lemonde_mean'] is not None else "N/A".rjust(15)
            
            f.write(f"{FEATURE_LABELS.get(gr['feature'], gr['feature']):<25} {s_s} {s_p} {s_l}\n")

        f.write("\n\n--- INTERPRETATION SUR LES GENRES TEXTUELS ---\n\n")
        f.write("L'objectif est d'etablir si la profondeur d'arbre et les distances de dependance permettent ")
        f.write("de differencier la syntaxe de trois variations du français : le langage texto (SMS), le ")
        f.write("journalisme (Le Monde) et la litterature sophistiquee (Philosophie).\n\n")
        
        dist_row = next((r for r in global_rows if r["feature"] == "distance_dependance_moy"), None)
        prof_row = next((r for r in global_rows if r["feature"] == "profondeur_arbre_max"), None)

        if dist_row and prof_row:
            s = dist_row["sms_best_mean"] or 0
            p = dist_row["philo_best_mean"] or 0
            l = dist_row["lemonde_mean"] or 0
            
            genre_list = [("SMS", s), ("Journalisme", l), ("Philosophie", p)]
            genre_list.sort(key=lambda x: x[1])

            f.write(f"→ Complexite syntaxique (distance de dependance) dans l'ordre croissant :\n")
            for name, val in genre_list:
                f.write(f"  - {name:<12}: {val:.2f}\n")
                
            f.write("\nOn observe que la langue ecrite formelle (Journalisme et Philosophie) ")
            f.write("structure des arbres de dependance globaux plus etendus avec une distance et ")
            f.write("une tolerance de nouds de rattachement eloignes qui depasse celle de l'oral/SMS.\n")

    print(f"\n  -> Rapport ecrit : {output_path}")

    # 5. Export CSV
    csv_path = output_path.replace(".txt", ".csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> CSV detaille  : {csv_path}")

    print(f"\n{BANNER}")

if __name__ == "__main__":
    main()
