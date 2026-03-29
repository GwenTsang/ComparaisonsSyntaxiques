#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Orchestrateur multi-modeles / multi-corpus

  Lance tous les modeles de parsing (HopsParser + Stanza) sur les deux
  corpus (SMS et Philosophie), puis execute les analyses syntaxiques.

  Usage :
    python orchestrator.py
    python orchestrator.py --output-dir /content/results
    python orchestrator.py --models gsd fsmb stanza
    python orchestrator.py --skip-download
==========================================================================
"""

import argparse
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


def section(title: str):
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


def run_script(args: list[str], label: str) -> bool:
    """Lance un script Python en sous-processus (isolation VRAM)."""
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["LOGURU_LEVEL"] = "ERROR"

    cmd = [sys.executable] + args
    print(f"\n  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
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
    print(BANNER)


if __name__ == "__main__":
    main()
