#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Inspection manuelle du corpus avec spacy_stanza

  Utilise le pipeline spacy_stanza (modele Stanza fr) pour parser des
  phrases et visualiser les arbres de dependance via displacy.

  Modes d'utilisation :
    1) Texte libre en argument
    2) Lecture d'un CSV de corpus (SMS ou Philosophie)
    3) Filtrage automatique des phrases a longues dependances
       (pour identifier les erreurs potentielles du modele)

  Usage :
    # Inspecter un texte libre
    python inspect_corpus.py --text "Le chat mange la souris."

    # Inspecter le corpus SMS (colonne SMS)
    python inspect_corpus.py --csv Corpus/1000_SMS_transcodage.csv \
           --column SMS --rows 1-10

    # Filtrer les phrases avec la dep. max la plus elevee
    python inspect_corpus.py --csv Corpus/philosophie.csv \
           --column Texte --top-deps 15

    # Lancer le serveur displacy interactif
    python inspect_corpus.py --text "..." --serve
==========================================================================
"""

import argparse
import os
import pathlib
import sys
import warnings

# Suppress noisy logs before any model import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

BANNER = "=" * 76


# ── model setup ───────────────────────────────────────────────────────

def load_nlp():
    """Load the spacy_stanza pipeline for French."""
    import torch
    
    # PyTorch 2.6+ workaround: monkey-patch torch.load to default to weights_only=False
    # because Stanza models require arbitrary code execution during unpickling.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    import spacy_stanza

    print("  Chargement du pipeline spacy_stanza (fr) ...")
    nlp = spacy_stanza.load_pipeline("fr", verbose=False)
    print("  Pipeline charge.")
    return nlp


# ── analysis helpers ──────────────────────────────────────────────────

def analyse_doc(doc):
    """Return basic dependency statistics for a spaCy Doc."""
    distances = []
    for token in doc:
        if token.head != token:
            dist = abs(token.i - token.head.i)
            distances.append(dist)

    if not distances:
        return {"max_dep": 0, "mean_dep": 0.0, "n_tokens": len(doc)}

    return {
        "max_dep": max(distances),
        "mean_dep": sum(distances) / len(distances),
        "n_tokens": len(doc),
    }


def render_html(doc, output_path: str, options: dict | None = None):
    """Render dependency tree to an HTML file."""
    from spacy import displacy

    opts = {"distance": 100, "compact": True, "bg": "#1a1a2e", "color": "#eee"}
    if options:
        opts.update(options)

    html = displacy.render(doc, style="dep", page=True, options=opts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def render_serve(doc, port: int = 5000):
    """Launch the displacy interactive server."""
    from spacy import displacy

    opts = {"distance": 100, "compact": True}
    print(f"\n  Serveur displacy sur http://localhost:{port}")
    print("  Ctrl+C pour arreter.\n")
    displacy.serve(doc, style="dep", port=port, options=opts)


# ── reading corpus ───────────────────────────────────────────────────

def read_texts_from_csv(
    csv_path: str,
    column: str,
    row_range: str | None = None,
) -> list[tuple[int, str]]:
    """Read texts from a CSV corpus file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    column : str
        Column name containing the text.
    row_range : str, optional
        Row range in the form "start-end" (1-indexed, inclusive).

    Returns
    -------
    list[tuple[int, str]]
        List of (row_index, text).
    """
    import pandas as pd

    # Try comma first, then semicolon
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        if column not in df.columns:
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")

    if column not in df.columns:
        print(f"  ! Colonne '{column}' introuvable. "
              f"Colonnes disponibles : {list(df.columns)}")
        sys.exit(1)

    texts = df[column].dropna().astype(str)

    if row_range:
        parts = row_range.split("-")
        start = int(parts[0]) - 1  # 0-indexed
        end = int(parts[1]) if len(parts) > 1 else start + 1
        texts = texts.iloc[start:end]

    return [(i + 1, t) for i, t in zip(texts.index, texts)]


# ── main modes ────────────────────────────────────────────────────────

def mode_text(nlp, text: str, output_dir: str, serve: bool):
    """Inspect a single text."""
    doc = nlp(text)
    stats = analyse_doc(doc)

    print(f"\n  Texte  : {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"  Tokens : {stats['n_tokens']}")
    print(f"  Dist. dep. max  : {stats['max_dep']}")
    print(f"  Dist. dep. moy  : {stats['mean_dep']:.2f}")

    # Print dependency arcs
    print(f"\n  {'Token':<20} {'DEP':<12} {'HEAD':<20} {'Distance':>10}")
    print("  " + "-" * 65)
    for token in doc:
        if token.head != token:
            dist = abs(token.i - token.head.i)
            marker = " <<<" if dist >= 10 else ""
            print(f"  {token.text:<20} {token.dep_:<12} "
                  f"{token.head.text:<20} {dist:>10}{marker}")

    if serve:
        render_serve(doc)
    else:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, "inspection.html")
        render_html(doc, html_path)
        print(f"\n  -> Arbre de dependance : {html_path}")


def mode_csv(
    nlp,
    csv_path: str,
    column: str,
    row_range: str | None,
    top_deps: int | None,
    output_dir: str,
    serve: bool,
):
    """Inspect texts from a CSV corpus."""
    texts = read_texts_from_csv(csv_path, column, row_range)

    if top_deps:
        # Parse all texts and rank by max dependency distance
        print(f"\n  Analyse de {len(texts)} textes pour trouver "
              f"les {top_deps} avec les plus longues dependances ...")
        scored: list[tuple[int, str, int, float]] = []
        for idx, text in texts:
            doc = nlp(text)
            stats = analyse_doc(doc)
            scored.append((idx, text, stats["max_dep"], stats["mean_dep"]))
        scored.sort(key=lambda x: x[2], reverse=True)
        texts = [(s[0], s[1]) for s in scored[:top_deps]]

        # Print ranking
        print(f"\n  {'Rang':>5} {'Ligne':>6} {'Max dep':>8} {'Moy dep':>8}  Texte")
        print("  " + "-" * 90)
        for rank, (idx, txt, mx, mn) in enumerate(scored[:top_deps], 1):
            preview = txt[:60].replace("\n", " ")
            print(f"  {rank:>5} {idx:>6} {mx:>8} {mn:>8.2f}  {preview}...")

    os.makedirs(output_dir, exist_ok=True)

    for i, (idx, text) in enumerate(texts):
        print(f"\n  ── Texte {idx} ──")
        doc = nlp(text)
        stats = analyse_doc(doc)

        print(f"  Tokens : {stats['n_tokens']}")
        print(f"  Dist. dep. max  : {stats['max_dep']}")
        print(f"  Dist. dep. moy  : {stats['mean_dep']:.2f}")

        # Highlight long arcs (distance >= 10)
        long_arcs = [
            (t.text, t.dep_, t.head.text, abs(t.i - t.head.i))
            for t in doc if t.head != t and abs(t.i - t.head.i) >= 10
        ]
        if long_arcs:
            print(f"\n  Arcs longs (distance >= 10) :")
            print(f"  {'Token':<20} {'DEP':<12} {'HEAD':<20} {'Dist':>6}")
            print("  " + "-" * 60)
            for tok, dep, head, dist in sorted(long_arcs, key=lambda x: -x[3]):
                print(f"  {tok:<20} {dep:<12} {head:<20} {dist:>6}")

        if serve and i == 0:
            render_serve(doc)
        else:
            html_path = os.path.join(output_dir, f"inspection_{idx:04d}.html")
            render_html(doc, html_path)
            print(f"  -> {html_path}")

    print(f"\n  {len(texts)} fichier(s) HTML genere(s) dans {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspection manuelle du corpus avec spacy_stanza.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            '  python inspect_corpus.py --text "Le chat mange la souris."\n'
            "  python inspect_corpus.py --csv Corpus/1000_SMS_transcodage.csv "
            "--column SMS --rows 1-10\n"
            "  python inspect_corpus.py --csv Corpus/philosophie.csv "
            "--column Texte --top-deps 15\n"
            "  python inspect_corpus.py --text \"...\" --serve\n"
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--text", "-t",
        help="Texte libre a analyser.",
    )
    src.add_argument(
        "--csv",
        help="Fichier CSV de corpus a inspecter.",
    )
    p.add_argument(
        "--column", "-c",
        default="Texte",
        help="Nom de la colonne contenant le texte (defaut : Texte).",
    )
    p.add_argument(
        "--rows",
        help="Plage de lignes a inspecter, ex. '1-10' (1-indexed, inclusif).",
    )
    p.add_argument(
        "--top-deps",
        type=int,
        help="Afficher les N textes avec les plus longues dependances.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Dossier de sortie pour les fichiers HTML "
             "(defaut : results/inspection/).",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Lancer le serveur displacy interactif "
             "(au lieu de generer un HTML).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or str(_REPO_ROOT / "results" / "inspection")

    print(f"\n{BANNER}")
    print("  INSPECTION MANUELLE DU CORPUS (spacy_stanza)")
    print(BANNER)

    nlp = load_nlp()

    if args.text:
        mode_text(nlp, args.text, output_dir, args.serve)
    else:
        mode_csv(
            nlp,
            args.csv,
            args.column,
            args.rows,
            args.top_deps,
            output_dir,
            args.serve,
        )

    print(f"\n{BANNER}\n")


if __name__ == "__main__":
    main()
