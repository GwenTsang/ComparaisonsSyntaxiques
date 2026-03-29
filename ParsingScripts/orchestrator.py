#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Orchestrateur multi-modeles / multi-corpus

  Lance tous les modeles de parsing (HopsParser + Stanza) sur les deux
  corpus (SMS et Philosophie), puis compare les proprietes syntaxiques.

  Usage :
    python orchestrator.py
    python orchestrator.py --output-dir /content/results
    python orchestrator.py --models gsd fsmb stanza
    python orchestrator.py --skip-download
==========================================================================
"""

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import time

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

BANNER = "=" * 76

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

# HopsParser models (use Camembert.py)
HOPS_MODELS = ["gsd", "fsmb", "sequoia", "rhapsodie", "zenodo-spoken"]

# All models including Stanza
ALL_MODELS = HOPS_MODELS + ["stanza"]

# Expected affinities (for the final report)
MODEL_AFFINITY = {
    "fsmb":           "SMS / oral",
    "rhapsodie":      "SMS / oral",
    "zenodo-spoken":  "SMS / oral",
    "gsd":            "Philosophie / ecrit formel",
    "sequoia":        "Philosophie / ecrit formel",
    "stanza":         "generaliste",
}

CORPORA = {
    "SMS": {
        "csv": str(_REPO_ROOT / "Corpus" / "1000_SMS_transcodage.csv"),
        "columns": ["SMS", "Transcodage_1", "Transcodage_2"],
    },
    "Philosophie": {
        "csv": str(_REPO_ROOT / "Corpus" / "philosophie.csv"),
        "columns": ["Texte"],
    },
}

# Syntactic structure features to compare
STRUCTURE_FEATURES = [
    "subordonnees_qui",
    "subordonnees_que",
    "subordonnees_prep_lequel",
    "subordonnees_dont",
    "completives",
    "hypothetiques",
    "gerondif",
    "incises",
    "propositions_coordonnees",
]

# Dependency features
DEP_FEATURES = [
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


def section(title: str):
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


def run_script(args: list[str], label: str) -> bool:
    """Lance un script Python en sous-processus (isolation VRAM)."""
    cmd = [sys.executable] + args
    print(f"\n  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    dt = time.time() - t0
    if result.returncode == 0:
        print(f"  OK {label} ({dt:.0f}s)")
        return True
    else:
        print(f"  ECHEC {label} (code={result.returncode}, {dt:.0f}s)")
        return False


# ════════════════════════════════════════════════════════════════════════
# ETAPE 1 : PREPARATION DU CORPUS PHILOSOPHIE
# ════════════════════════════════════════════════════════════════════════
def step_prepare_corpus():
    section("ETAPE 1 : PREPARATION DU CORPUS PHILOSOPHIE")

    philo_csv = CORPORA["Philosophie"]["csv"]
    if os.path.isfile(philo_csv):
        print(f"  Corpus deja present : {philo_csv}")
        return True

    copies_dir = str(_REPO_ROOT / "Corpus" / "Copies")
    if not os.path.isdir(copies_dir):
        print(f"  ! Dossier introuvable : {copies_dir}")
        return False

    return run_script(
        [str(_SCRIPT_DIR / "prepare_corpus.py"),
         "--input-dir", copies_dir,
         "--output", philo_csv,
         "--column", "Texte"],
        "Preparation corpus philosophie",
    )


# ════════════════════════════════════════════════════════════════════════
# ETAPE 2 : TELECHARGEMENT DES MODELES
# ════════════════════════════════════════════════════════════════════════
def step_download_models(models: list[str], models_dir: str):
    section("ETAPE 2 : TELECHARGEMENT DES MODELES")

    hops_to_download = [m for m in models if m in HOPS_MODELS]
    if not hops_to_download:
        print("  Aucun modele HopsParser a telecharger.")
        return True

    for name in hops_to_download:
        run_script(
            [str(_SCRIPT_DIR / "download_model.py"),
             "--model", name,
             "--output-dir", models_dir],
            f"Telechargement {name}",
        )

    if "stanza" in models:
        print("  Stanza sera telecharge automatiquement au premier lancement.")

    return True


# ════════════════════════════════════════════════════════════════════════
# ETAPE 3 : PARSING
# ════════════════════════════════════════════════════════════════════════
def step_parse(models: list[str], output_dir: str, models_dir: str):
    section("ETAPE 3 : PARSING (tous modeles x tous corpus)")

    results = {}

    for corpus_name, corpus_info in CORPORA.items():
        for model_name in models:
            label = f"{corpus_name}/{model_name}"
            out_dir = os.path.join(output_dir, corpus_name, model_name)

            # Skip if output already exists
            csv_out = os.path.join(out_dir, "resultats_par_sms.csv")
            if os.path.isfile(csv_out):
                print(f"\n  [SKIP] {label} : resultats deja presents")
                results[label] = True
                continue

            cols = corpus_info["columns"]

            if model_name == "stanza":
                ok = run_script(
                    [str(_SCRIPT_DIR / "Stanza.py"),
                     "--csv", corpus_info["csv"],
                     "--output", out_dir,
                     "--columns"] + cols,
                    label,
                )
            else:
                ok = run_script(
                    [str(_SCRIPT_DIR / "Camembert.py"),
                     "--model-name", model_name,
                     "--csv", corpus_info["csv"],
                     "--output", out_dir,
                     "--columns"] + cols +
                    ["--models-dir", models_dir],
                    label,
                )

            results[label] = ok

    # Summary
    print(f"\n  --- Resume du parsing ---")
    for label, ok in results.items():
        status = "OK" if ok else "ECHEC"
        print(f"  {label:<35} {status}")

    return results


# ════════════════════════════════════════════════════════════════════════
# ETAPE 4 : STRUCTURES SYNTAXIQUES
# ════════════════════════════════════════════════════════════════════════
def step_structures(models: list[str], output_dir: str):
    section("ETAPE 4 : ANALYSE DES STRUCTURES SYNTAXIQUES")

    for corpus_name in CORPORA:
        for model_name in models:
            model_dir = os.path.join(output_dir, corpus_name, model_name)
            if not os.path.isdir(model_dir):
                continue

            # Check if any output_*.conllu files exist
            conllu_files = [f for f in os.listdir(model_dir)
                           if f.startswith("output_") and f.endswith(".conllu")]
            if not conllu_files:
                print(f"  [SKIP] {corpus_name}/{model_name} : pas de fichiers CoNLL-U")
                continue

            struct_csv = os.path.join(model_dir, "structures_syntaxiques.csv")
            if os.path.isfile(struct_csv):
                print(f"  [SKIP] {corpus_name}/{model_name} : deja analyse")
                continue

            run_script(
                [str(_SCRIPT_DIR / "structures_syntaxiques.py"),
                 "--input-dir", model_dir],
                f"Structures {corpus_name}/{model_name}",
            )


# ════════════════════════════════════════════════════════════════════════
# ETAPE 5 : ACCORD INTER-MODELES
# ════════════════════════════════════════════════════════════════════════
def step_accord(models: list[str], output_dir: str):
    section("ETAPE 5 : ACCORD INTER-MODELES")

    for corpus_name in CORPORA:
        dirs = []
        for model_name in models:
            model_dir = os.path.join(output_dir, corpus_name, model_name)
            csv_path = os.path.join(model_dir, "resultats_par_sms.csv")
            if os.path.isfile(csv_path):
                dirs.append(model_dir)

        if len(dirs) < 2:
            print(f"  {corpus_name} : moins de 2 modeles avec resultats, skip.")
            continue

        accord_dir = os.path.join(output_dir, "accord_inter_modeles", corpus_name)
        run_script(
            [str(_SCRIPT_DIR / "accord_inter_modeles.py"),
             "--dirs"] + dirs +
            ["--output", accord_dir],
            f"Accord {corpus_name}",
        )


# ════════════════════════════════════════════════════════════════════════
# ETAPE 6 : COMPARAISON STATISTIQUE SMS vs PHILOSOPHIE
# ════════════════════════════════════════════════════════════════════════
def step_compare(models: list[str], output_dir: str):
    section("ETAPE 6 : COMPARAISON STATISTIQUE SMS vs PHILOSOPHIE")

    try:
        import numpy as np
        from scipy import stats as sp_stats
    except ImportError:
        print("  ! scipy non installe, comparaison statistique impossible.")
        return

    compare_dir = os.path.join(output_dir, "comparaison_corpora")
    os.makedirs(compare_dir, exist_ok=True)

    # Collect syntactic structure data per corpus
    def load_structures(corpus_name: str) -> dict[str, list[float]]:
        """Load all structure counts from all models for a corpus."""
        all_vals: dict[str, list[float]] = {feat: [] for feat in STRUCTURE_FEATURES}
        for model_name in models:
            struct_csv = os.path.join(
                output_dir, corpus_name, model_name,
                "structures_syntaxiques.csv",
            )
            if not os.path.isfile(struct_csv):
                continue
            try:
                import pandas as pd
                df = pd.read_csv(struct_csv)
                for feat in STRUCTURE_FEATURES:
                    col = f"nombre_{feat}"
                    if col in df.columns:
                        vals = df[col].dropna().tolist()
                        all_vals[feat].extend(vals)
            except Exception as e:
                print(f"  ! Erreur lecture {struct_csv}: {e}")
        return all_vals

    sms_data = load_structures("SMS")
    philo_data = load_structures("Philosophie")

    # Statistical tests (Mann-Whitney U)
    rows = []
    print(f"\n  {'Feature':<35} {'SMS moy':>10} {'Philo moy':>10} "
          f"{'U stat':>10} {'p-value':>12} {'Signif':>8}")
    print("  " + "-" * 90)

    for feat in STRUCTURE_FEATURES:
        sms_vals = sms_data.get(feat, [])
        philo_vals = philo_data.get(feat, [])

        if len(sms_vals) < 5 or len(philo_vals) < 5:
            print(f"  {feat:<35} {'N/A':>10} {'N/A':>10} "
                  f"{'':>10} {'':>12} {'trop peu':>8}")
            continue

        sms_mean = float(np.mean(sms_vals))
        philo_mean = float(np.mean(philo_vals))

        try:
            u_stat, p_value = sp_stats.mannwhitneyu(
                sms_vals, philo_vals, alternative="two-sided"
            )
        except ValueError:
            u_stat, p_value = 0.0, 1.0

        # Effect size (rank-biserial correlation)
        n1, n2 = len(sms_vals), len(philo_vals)
        effect_size = 1 - (2 * u_stat) / (n1 * n2) if n1 * n2 > 0 else 0.0

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 \
              else "*" if p_value < 0.05 else ""

        print(f"  {feat:<35} {sms_mean:>10.2f} {philo_mean:>10.2f} "
              f"{u_stat:>10.0f} {p_value:>12.2e} {sig:>8}")

        rows.append({
            "feature": feat,
            "sms_n": len(sms_vals),
            "sms_mean": round(sms_mean, 4),
            "sms_std": round(float(np.std(sms_vals)), 4),
            "philo_n": len(philo_vals),
            "philo_mean": round(philo_mean, 4),
            "philo_std": round(float(np.std(philo_vals)), 4),
            "mann_whitney_U": round(u_stat, 2),
            "p_value": p_value,
            "effect_size": round(effect_size, 4),
            "significatif": sig,
        })

    # Export CSV
    if rows:
        csv_out = os.path.join(compare_dir, "comparaison_syntaxique.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  -> {csv_out}")

    # Human-readable summary
    summary_path = os.path.join(compare_dir, "resume_comparaison.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("COMPARAISON STATISTIQUE : SMS vs PHILOSOPHIE\n")
        f.write("=" * 60 + "\n\n")

        f.write("Modeles utilises :\n")
        for m in models:
            affinity = MODEL_AFFINITY.get(m, "?")
            f.write(f"  - {m:<20} affinite : {affinity}\n")

        f.write(f"\nFeatures significativement differentes (p < 0.05) :\n")
        f.write("-" * 60 + "\n")

        significant = [r for r in rows if r["p_value"] < 0.05]
        if significant:
            significant.sort(key=lambda r: r["p_value"])
            for r in significant:
                direction = "SMS > Philo" if r["sms_mean"] > r["philo_mean"] \
                            else "Philo > SMS"
                f.write(
                    f"  {r['feature']:<35} p={r['p_value']:.2e}  "
                    f"effect={r['effect_size']:.3f}  ({direction})\n"
                )
        else:
            f.write("  Aucune feature significativement differente.\n")

        f.write(f"\nFeatures NON significatives (p >= 0.05) :\n")
        f.write("-" * 60 + "\n")
        non_sig = [r for r in rows if r["p_value"] >= 0.05]
        for r in non_sig:
            f.write(f"  {r['feature']:<35} p={r['p_value']:.2e}\n")

    print(f"  -> {summary_path}")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchestrateur multi-modeles / multi-corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python orchestrator.py\n"
            "  python orchestrator.py --output-dir /content/results\n"
            "  python orchestrator.py --models gsd fsmb stanza\n"
            "  python orchestrator.py --skip-download\n"
        ),
    )
    p.add_argument(
        "--output-dir", "-o",
        default=str(_REPO_ROOT / "results"),
        help="Dossier racine des resultats (defaut : <repo>/results).",
    )
    p.add_argument(
        "--models", "-m", nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help=f"Modeles a utiliser (defaut : tous). Choix : {', '.join(ALL_MODELS)}.",
    )
    p.add_argument(
        "--models-dir",
        default=str(_REPO_ROOT / "models"),
        help="Dossier de stockage des modeles (defaut : <repo>/models).",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Ne pas telecharger les modeles (suppose deja presents).",
    )
    p.add_argument(
        "--skip-parse", action="store_true",
        help="Ne pas relancer le parsing (utilise les resultats existants).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    models = args.models
    models_dir = args.models_dir
    t0 = time.time()

    print(f"\n{BANNER}")
    print(f"  ORCHESTRATEUR MULTI-MODELES / MULTI-CORPUS")
    print(BANNER)
    print(f"  Modeles    : {', '.join(models)}")
    print(f"  Sortie     : {output_dir}")
    print(f"  Modeles    : {models_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Prepare philosophy corpus
    ok = step_prepare_corpus()
    if not ok:
        print("  ! Echec preparation corpus. Arret.")
        sys.exit(1)

    # Step 2: Download models
    if not args.skip_download:
        step_download_models(models, models_dir)

    # Step 3: Parse
    if not args.skip_parse:
        step_parse(models, output_dir, models_dir)

    # Step 4: Syntactic structures
    step_structures(models, output_dir)

    # Step 5: Inter-model agreement
    step_accord(models, output_dir)

    # Step 6: Statistical comparison
    step_compare(models, output_dir)

    # Final summary
    elapsed = time.time() - t0
    section("TERMINE")
    print(f"  Duree totale : {elapsed / 60:.1f} minutes")
    print(f"  Resultats    : {output_dir}/")
    print(f"\n  Structure :")
    for corpus_name in CORPORA:
        print(f"    {corpus_name}/")
        for model_name in models:
            d = os.path.join(output_dir, corpus_name, model_name)
            status = "OK" if os.path.isdir(d) else "manquant"
            print(f"      {model_name:<20} [{status}]")
    print(f"    accord_inter_modeles/")
    print(f"    comparaison_corpora/")
    print(BANNER)


if __name__ == "__main__":
    main()
