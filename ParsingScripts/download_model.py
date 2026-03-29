#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Téléchargement automatisé des modèles de parsing

  Modèles supportés :
    • camembertav2-base-gsd       (DeBERTa v2, HuggingFace)
    • camembertav2-base-fsmb      (DeBERTa v2, HuggingFace)
    • camembertav2-base-sequoia   (DeBERTa v2, HuggingFace)
    • camembertav2-base-rhapsodie (DeBERTa v2, HuggingFace)
    • zenodo-spoken               (RoBERTa, Zenodo)

  Usage :
    python download_model.py --list
    python download_model.py --model fsmb
    python download_model.py --model all --output-dir ./models
==========================================================================
"""

import argparse
import os
import pathlib
import sys
import tarfile
import tempfile
import urllib.request

# ════════════════════════════════════════════════════════════════════════
# REGISTRE DES MODÈLES
# ════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "gsd": {
        "full_name": "camembertav2-base-gsd",
        "repo_id": "almanach/camembertav2-base-gsd",
        "architecture": "DeBERTa v2",
        "source": "HuggingFace",
        "method": "huggingface",
    },
    "fsmb": {
        "full_name": "camembertav2-base-fsmb",
        "repo_id": "almanach/camembertav2-base-fsmb",
        "architecture": "DeBERTa v2",
        "source": "HuggingFace",
        "method": "huggingface",
    },
    "sequoia": {
        "full_name": "camembertav2-base-sequoia",
        "repo_id": "almanach/camembertav2-base-sequoia",
        "architecture": "DeBERTa v2",
        "source": "HuggingFace",
        "method": "huggingface",
    },
    "rhapsodie": {
        "full_name": "camembertav2-base-rhapsodie",
        "repo_id": "almanach/camembertav2-base-rhapsodie",
        "architecture": "DeBERTa v2",
        "source": "HuggingFace",
        "method": "huggingface",
    },
    "zenodo-spoken": {
        "full_name": "UD_all_spoken_French-camembert",
        "architecture": "RoBERTa (CamemBERT)",
        "source": "Zenodo",
        "url": (
            "https://zenodo.org/record/7703346/files/"
            "UD_all_spoken_French-camembert.tar.xz?download=1"
        ),
        "method": "http_tarxz",
    },
}

BANNER = "=" * 72


# ════════════════════════════════════════════════════════════════════════
# FONCTIONS DE TÉLÉCHARGEMENT
# ════════════════════════════════════════════════════════════════════════

def download_hf_model(name: str, repo_id: str, output_dir: str) -> str:
    """
    Télécharge un modèle depuis HuggingFace via huggingface_hub.
    Contrairement à git clone, cela gère correctement les fichiers LFS.
    """
    dest = os.path.join(output_dir, MODEL_REGISTRY[name]["full_name"])

    if os.path.isdir(dest):
        print(f"  ℹ Modèle déjà présent : {dest}")
        return dest

    print(f"\n  Téléchargement de {name} depuis HuggingFace…")
    print(f"  Repo : {repo_id}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "  ✗ huggingface-hub non installé.\n"
            "    Installez-le : pip install huggingface-hub"
        )

    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        local_dir_use_symlinks=False,
    )

    print(f"  ✓ {name} téléchargé → {dest}")
    return dest


def download_zenodo_model(name: str, url: str, output_dir: str) -> str:
    """Télécharge et extrait un modèle depuis Zenodo (tar.xz)."""
    expected_dir = os.path.join(
        output_dir, MODEL_REGISTRY[name]["full_name"]
    )

    if os.path.isdir(expected_dir):
        print(f"  ℹ Modèle déjà présent : {expected_dir}")
        return expected_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Téléchargement de {name} depuis Zenodo…")
    print(f"  URL : {url}")

    # Téléchargement avec barre de progression
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.xz")
    os.close(tmp_fd)

    try:
        _download_with_progress(url, tmp_path)

        print(f"  Extraction…")
        with tarfile.open(tmp_path, "r:xz") as tar:
            tar.extractall(path=output_dir)

        print(f"  ✓ {name} extrait → {output_dir}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Chercher le dossier extrait
    if os.path.isdir(expected_dir):
        return expected_dir

    # Fallback : trouver le premier nouveau dossier
    for entry in os.listdir(output_dir):
        candidate = os.path.join(output_dir, entry)
        if os.path.isdir(candidate) and entry != ".":
            print(f"  ℹ Dossier extrait détecté : {candidate}")
            return candidate

    return output_dir


def _download_with_progress(url: str, dest_path: str):
    """Télécharge un fichier avec une barre de progression simple."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "HopsColab/1.0")

    with urllib.request.urlopen(req) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None

        downloaded = 0
        block_size = 1024 * 1024  # 1 Mo

        if total:
            total_mb = total / (1024 * 1024)
            print(f"  Taille : {total_mb:.1f} Mo")

        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    dl_mb = downloaded / (1024 * 1024)
                    print(
                        f"    {dl_mb:.1f} / {total_mb:.1f} Mo  ({pct:.0f}%)",
                        end="\r",
                    )
        if total:
            print()  # Nouvelle ligne après la barre de progression


# ════════════════════════════════════════════════════════════════════════
# INTERFACE PUBLIQUE
# ════════════════════════════════════════════════════════════════════════

def list_models():
    """Affiche les modèles disponibles."""
    print(f"\n{BANNER}")
    print("  Modèles disponibles")
    print(BANNER)
    print(
        f"\n  {'Nom court':<18} {'Architecture':<22} {'Source':<14} "
        f"{'Nom complet'}"
    )
    print("  " + "-" * 70)
    for short, info in MODEL_REGISTRY.items():
        print(
            f"  {short:<18} {info['architecture']:<22} "
            f"{info['source']:<14} {info['full_name']}"
        )
    print()


def download_model(name: str, output_dir: str) -> str:
    """
    Télécharge un modèle par son nom court.
    Retourne le chemin du dossier du modèle.
    """
    if name not in MODEL_REGISTRY:
        sys.exit(
            f"  ✗ Modèle inconnu : '{name}'\n"
            f"  Modèles disponibles : {', '.join(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[name]
    method = info["method"]

    if method == "huggingface":
        return download_hf_model(name, info["repo_id"], output_dir)
    elif method == "http_tarxz":
        return download_zenodo_model(name, info["url"], output_dir)
    else:
        sys.exit(f"  ✗ Méthode de téléchargement inconnue : {method}")


def get_model_path(name: str, output_dir: str) -> str:
    """
    Retourne le chemin du modèle s'il existe déjà,
    sinon le télécharge et retourne le chemin.
    """
    if name not in MODEL_REGISTRY:
        sys.exit(
            f"  ✗ Modèle inconnu : '{name}'\n"
            f"  Modèles disponibles : {', '.join(MODEL_REGISTRY.keys())}"
        )

    expected = os.path.join(output_dir, MODEL_REGISTRY[name]["full_name"])
    if os.path.isdir(expected):
        return expected

    return download_model(name, output_dir)


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent
_DEFAULT_MODEL_DIR = str(_REPO_ROOT / "models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Télécharge les modèles de parsing pour HopsColab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python download_model.py --list\n"
            "  python download_model.py --model fsmb\n"
            "  python download_model.py --model all\n"
            "  python download_model.py --model gsd --output-dir /data/models\n"
        ),
    )
    p.add_argument(
        "--model", "-m",
        help=(
            "Nom court du modèle à télécharger "
            f"({', '.join(MODEL_REGISTRY.keys())}, ou 'all')."
        ),
    )
    p.add_argument(
        "--list", "-l", action="store_true",
        help="Affiche la liste des modèles disponibles.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=_DEFAULT_MODEL_DIR,
        help=f"Dossier de destination (défaut : {_DEFAULT_MODEL_DIR}).",
    )
    args = p.parse_args()

    if not args.list and not args.model:
        p.error("Spécifiez --model NOM ou --list.")

    return args


def main():
    args = parse_args()

    if args.list:
        list_models()
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "all":
        print(f"\n{BANNER}")
        print("  Téléchargement de tous les modèles")
        print(BANNER)
        for name in MODEL_REGISTRY:
            path = download_model(name, args.output_dir)
            print(f"  → {path}\n")
    else:
        path = download_model(args.model, args.output_dir)
        print(f"\n  → Modèle disponible dans : {path}")


if __name__ == "__main__":
    main()
