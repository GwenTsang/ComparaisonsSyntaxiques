#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
==========================================================================
  Orchestrateur 3-way : lance les deux analyses en parallèle

    • compare_distances_3way.py   (distances de dépendance)
    • compare_structures_3way.py  (structures syntaxiques)

  Usage :
    python run_all_3way.py
    python run_all_3way.py --results-dir ../results
    python run_all_3way.py --sequential       # un à la fois
==========================================================================
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

BANNER = "=" * 90

SCRIPTS = [
    {
        "name": "Distances syntaxiques 3-way",
        "path": str(_SCRIPT_DIR / "compare_distances_3way.py"),
    },
    {
        "name": "Structures syntaxiques 3-way",
        "path": str(_SCRIPT_DIR / "compare_structures_3way.py"),
    },
]


def _run_script(script_info: dict, extra_args: list[str]) -> dict:
    """
    Run a single analysis script as a subprocess.
    Returns a dict with name, return code, stdout, stderr, and elapsed time.
    """
    name = script_info["name"]
    path = script_info["path"]

    cmd = [sys.executable, path] + extra_args

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_SCRIPT_DIR),
    )
    elapsed = time.perf_counter() - t0

    return {
        "name": name,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Orchestre les deux analyses 3-way en parallèle."
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=None,
        help="Dossier racine des résultats (transmis aux sous-scripts).",
    )
    parser.add_argument(
        "--conllu-path", "-c",
        default=None,
        help="Chemin du fichier .conllu Le Monde (transmis aux sous-scripts).",
    )
    parser.add_argument(
        "--sequential", "-s",
        action="store_true",
        help="Exécuter les scripts séquentiellement au lieu de simultanément.",
    )
    args = parser.parse_args()

    # Build extra args to forward to both sub-scripts
    extra_args: list[str] = []
    if args.results_dir:
        extra_args += ["--results-dir", args.results_dir]
    if args.conllu_path:
        extra_args += ["--conllu-path", args.conllu_path]

    print(f"\n{BANNER}")
    print("  ORCHESTRATEUR — ANALYSES 3-WAY SIMULTANÉES")
    print(BANNER)
    print(f"  Mode          : {'sequentiel' if args.sequential else 'parallele'}")
    print(f"  Scripts       : {len(SCRIPTS)}")
    for s in SCRIPTS:
        print(f"    - {s['name']}")
    print(f"  Args transmis : {extra_args or '(defauts)'}")
    print(BANNER)

    t_global = time.perf_counter()
    results: list[dict] = []

    if args.sequential:
        # ── Sequential execution ──
        for script in SCRIPTS:
            print(f"\n  > Lancement : {script['name']} ...")
            res = _run_script(script, extra_args)
            results.append(res)
            # Stream output immediately
            if res["stdout"]:
                print(res["stdout"])
            if res["stderr"]:
                print(res["stderr"], file=sys.stderr)
    else:
        # ── Parallel execution ──
        print(f"\n  > Lancement simultane de {len(SCRIPTS)} analyses ...\n")
        with ProcessPoolExecutor(max_workers=len(SCRIPTS)) as executor:
            futures = {
                executor.submit(_run_script, script, extra_args): script
                for script in SCRIPTS
            }
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                # Print output as each script finishes
                print(f"\n{'-' * 90}")
                print(f"  [OK] Termine : {res['name']}  ({res['elapsed']:.1f}s)")
                print(f"{'-' * 90}")
                if res["stdout"]:
                    print(res["stdout"])
                if res["stderr"]:
                    print(res["stderr"], file=sys.stderr)

    total_elapsed = time.perf_counter() - t_global

    # ── Summary ──
    print(f"\n{BANNER}")
    print("  RESUME DE L'EXECUTION")
    print(BANNER)

    all_ok = True
    for res in results:
        status = "[OK]" if res["returncode"] == 0 else f"[ERREUR] (code {res['returncode']})"
        if res["returncode"] != 0:
            all_ok = False
        print(f"  {status:20s}  {res['name']:<35s}  {res['elapsed']:.1f}s")

    print(f"\n  Temps total : {total_elapsed:.1f}s")
    if all_ok:
        print("  Toutes les analyses ont reussi.")
    else:
        print("  /!\\ Certaines analyses ont echoue. Voir les logs ci-dessus.")
    print(BANNER)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
