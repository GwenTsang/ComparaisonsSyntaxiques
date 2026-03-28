#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Préparation d'un corpus à partir de fichiers TXT

  Lit tous les fichiers .txt d'un dossier, segmente les paragraphes en
  phrases, et produit un CSV compatible avec le pipeline de parsing
  (CamembertaHOPS.py, structures_syntaxiques.py, etc.).

  Usage :
    python prepare_corpus.py --input-dir ../Corpus/Copies
    python prepare_corpus.py --input-dir ../Corpus/Copies --output ../Corpus/philosophie.csv
==========================================================================
"""

import argparse
import csv
import os
import pathlib
import re
import sys

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

BANNER = "=" * 72


# ════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ════════════════════════════════════════════════════════════════════════

def split_into_sentences(text: str) -> list[str]:
    """
    Segmente un texte en phrases. Gère les abréviations courantes
    et les guillemets.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    # Séparer sur ponctuation forte suivie d'un espace et d'une majuscule
    parts = re.split(r'(?<=[.!?…])\s+(?=[A-ZÀ-ÖÙ-Ý«\"])', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def read_txt_file(filepath: str) -> list[str]:
    """
    Lit un fichier TXT et retourne les paragraphes non vides.
    Un paragraphe = un bloc de texte séparé par une ou plusieurs lignes vides.
    """
    with open(filepath, encoding="utf-8") as fh:
        content = fh.read()

    # Supprimer les astérisques (italiques/gras Markdown)
    content = content.replace("*", "")

    # Séparer en paragraphes (séparés par des lignes vides)
    raw_paragraphs = re.split(r"\n\s*\n", content)
    paragraphs = []
    for para in raw_paragraphs:
        # Joindre les lignes internes du paragraphe
        cleaned = " ".join(para.split())
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prépare un corpus de fichiers TXT pour le pipeline de parsing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python prepare_corpus.py --input-dir ../Corpus/Copies\n"
            "  python prepare_corpus.py --input-dir ./textes --output ./corpus.csv\n"
            "  python prepare_corpus.py --input-dir ./textes --mode paragraphs\n"
        ),
    )
    p.add_argument(
        "--input-dir", "-i", required=True,
        help="Dossier contenant les fichiers .txt.",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Chemin du fichier CSV de sortie (défaut : <input-dir>/corpus.csv).",
    )
    p.add_argument(
        "--column", "-c", default="Texte",
        help="Nom de la colonne texte dans le CSV (défaut : Texte).",
    )
    p.add_argument(
        "--mode", "-m", default="sentences",
        choices=["sentences", "paragraphs"],
        help=(
            "Mode de découpage : 'sentences' (une phrase par ligne, défaut) "
            "ou 'paragraphs' (un paragraphe par ligne)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_csv = args.output or os.path.join(input_dir, "corpus.csv")
    column_name = args.column
    mode = args.mode

    if not os.path.isdir(input_dir):
        sys.exit(f"  ✗ Dossier introuvable : {input_dir}")

    # ── Lister les fichiers TXT ──
    txt_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".txt")
    ])

    if not txt_files:
        sys.exit(f"  ✗ Aucun fichier .txt trouvé dans {input_dir}")

    print(f"\n{BANNER}")
    print(f"  Préparation du corpus")
    print(BANNER)
    print(f"  Dossier source : {input_dir}")
    print(f"  Fichiers TXT   : {len(txt_files)}")
    print(f"  Mode           : {mode}")
    print(f"  Colonne        : {column_name}")

    # ── Extraction ──
    rows = []
    for txt_file in txt_files:
        filepath = os.path.join(input_dir, txt_file)
        source = os.path.splitext(txt_file)[0]

        paragraphs = read_txt_file(filepath)
        print(f"\n  {txt_file} : {len(paragraphs)} paragraphe(s)")

        if mode == "sentences":
            for para in paragraphs:
                sentences = split_into_sentences(para)
                for sent in sentences:
                    rows.append({
                        column_name: sent,
                        "source_fichier": source,
                    })
            n_sents = sum(
                1 for r in rows if r["source_fichier"] == source
            )
            print(f"    → {n_sents} phrase(s)")
        else:
            for para in paragraphs:
                rows.append({
                    column_name: para,
                    "source_fichier": source,
                })

    # ── Export CSV ──
    fieldnames = [column_name, "source_fichier"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{BANNER}")
    print(f"  ✓ {len(rows)} ligne(s) exportées → {output_csv}")
    print(BANNER)


if __name__ == "__main__":
    main()
