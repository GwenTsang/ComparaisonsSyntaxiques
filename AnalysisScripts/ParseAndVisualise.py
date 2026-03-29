#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ParseAndVisualise.py — Inspection manuelle du corpus avec spacy_stanza.

Utilise le pipeline spacy_stanza (modele Stanza fr) pour parser des
phrases et visualiser les arbres de dependance via displacy.

Modes :
  1) Texte libre    --text "..."
  2) Corpus CSV     --csv <fichier> --column <col> [--rows 1-10]
  3) Filtrage       --csv ... --top-deps 15

Pour les resultats pre-calcules (CoNLL-U), utilisez serve_conllu.py.

Usage :
    python AnalysisScripts/ParseAndVisualise.py --text "Le chat mange la souris."
    python AnalysisScripts/ParseAndVisualise.py --csv Corpus/1000_SMS_transcodage.csv --column SMS --rows 1-10
    python AnalysisScripts/ParseAndVisualise.py --csv Corpus/philosophie.csv --column Texte --top-deps 15
    python AnalysisScripts/ParseAndVisualise.py --text "..." --serve
"""

import argparse
import os
import pathlib
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BANNER = "=" * 76


# ── pipeline ──────────────────────────────────────────────────────────

def load_nlp():
    import torch

    _orig = torch.load
    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)
    torch.load = _patched

    import spacy_stanza
    print("  Chargement du pipeline spacy_stanza (fr) ...")
    nlp = spacy_stanza.load_pipeline("fr", verbose=False)
    print("  Pipeline charge.")
    return nlp


# ── helpers ───────────────────────────────────────────────────────────

def analyse_doc(doc) -> dict:
    distances = [abs(t.i - t.head.i) for t in doc if t.head != t]
    if not distances:
        return {"max_dep": 0, "mean_dep": 0.0, "n_tokens": len(doc)}
    return {
        "max_dep" : max(distances),
        "mean_dep": sum(distances) / len(distances),
        "n_tokens": len(doc),
    }


def render_html(doc, path: str, options: dict | None = None) -> str:
    from spacy import displacy
    opts = {"distance": 100, "compact": True, "bg": "#1a1a2e", "color": "#eee"}
    if options:
        opts.update(options)
    html = displacy.render(doc, style="dep", page=True, options=opts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def render_serve(doc, port: int = 5000):
    from spacy import displacy
    print(f"\n  Serveur displacy sur http://localhost:{port}")
    print("  Ctrl+C pour arreter.\n")
    displacy.serve(doc, style="dep", port=port, options={"distance": 100, "compact": True})


# ── CSV reader ────────────────────────────────────────────────────────

def read_csv(csv_path: str, column: str, row_range: str | None = None) -> list[tuple[int, str]]:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        if column not in df.columns:
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")

    if column not in df.columns:
        print(f"  Colonne '{column}' introuvable. Disponibles : {list(df.columns)}")
        sys.exit(1)

    texts = df[column].dropna().astype(str)
    if row_range:
        parts = row_range.split("-")
        s = int(parts[0]) - 1
        e = int(parts[1]) if len(parts) > 1 else s + 1
        texts = texts.iloc[s:e]
    return [(i + 1, t) for i, t in zip(texts.index, texts)]


# ── modes ─────────────────────────────────────────────────────────────

def mode_text(nlp, text: str, output_dir: str, serve: bool):
    doc   = nlp(text)
    stats = analyse_doc(doc)

    print(f"\n  Texte  : {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"  Tokens : {stats['n_tokens']}")
    print(f"  Dep. max : {stats['max_dep']}  |  Dep. moy : {stats['mean_dep']:.2f}")

    print(f"\n  {'Token':<20} {'DEP':<12} {'HEAD':<20} {'Dist':>8}")
    print("  " + "-" * 62)
    for t in doc:
        if t.head != t:
            dist   = abs(t.i - t.head.i)
            marker = " <<<" if dist >= 10 else ""
            print(f"  {t.text:<20} {t.dep_:<12} {t.head.text:<20} {dist:>8}{marker}")

    if serve:
        render_serve(doc)
    else:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "inspection.html")
        render_html(doc, path)
        print(f"\n  -> {path}")


def mode_csv(nlp, csv_path: str, column: str, row_range: str | None,
             top_deps: int | None, output_dir: str, serve: bool):
    texts = read_csv(csv_path, column, row_range)

    if top_deps:
        print(f"\n  Analyse de {len(texts)} textes ...")
        scored = []
        for idx, text in texts:
            doc   = nlp(text)
            stats = analyse_doc(doc)
            scored.append((idx, text, stats["max_dep"], stats["mean_dep"]))
        scored.sort(key=lambda x: x[2], reverse=True)
        texts = [(s[0], s[1]) for s in scored[:top_deps]]
        print(f"\n  {'Rang':>5} {'Ligne':>6} {'Max':>6} {'Moy':>6}  Texte")
        print("  " + "-" * 80)
        for rank, (idx, txt, mx, mn) in enumerate(scored[:top_deps], 1):
            print(f"  {rank:>5} {idx:>6} {mx:>6} {mn:>6.2f}  {txt[:60].replace(chr(10), ' ')}...")

    os.makedirs(output_dir, exist_ok=True)
    for i, (idx, text) in enumerate(texts):
        print(f"\n  -- Texte {idx} --")
        doc   = nlp(text)
        stats = analyse_doc(doc)
        print(f"  Tokens : {stats['n_tokens']}  |  Dep. max : {stats['max_dep']}"
              f"  |  Dep. moy : {stats['mean_dep']:.2f}")

        long_arcs = [
            (t.text, t.dep_, t.head.text, abs(t.i - t.head.i))
            for t in doc if t.head != t and abs(t.i - t.head.i) >= 10
        ]
        if long_arcs:
            print(f"\n  Arcs longs (>= 10) :")
            for tok, dep, head, dist in sorted(long_arcs, key=lambda x: -x[3]):
                print(f"    {tok:<20} {dep:<12} {head:<20} {dist:>6}")

        if serve and i == 0:
            render_serve(doc)
        else:
            path = os.path.join(output_dir, f"inspection_{idx:04d}.html")
            render_html(doc, path)
            print(f"  -> {path}")

    print(f"\n  {len(texts)} fichier(s) HTML dans {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspection du corpus avec spacy_stanza + displacy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", "-t", help="Texte libre a analyser.")
    src.add_argument("--csv",        help="Fichier CSV de corpus.")
    p.add_argument("--column", "-c", default="Texte",
                   help="Colonne texte dans le CSV (defaut : Texte).")
    p.add_argument("--rows",         help="Plage de lignes CSV, ex. '1-10'.")
    p.add_argument("--top-deps", type=int,
                   help="Afficher les N textes avec les plus longues dependances.")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Dossier de sortie HTML (defaut : results/inspection/).")
    p.add_argument("--serve", action="store_true",
                   help="Lancer le serveur displacy interactif.")
    return p.parse_args()


def main():
    args      = parse_args()
    out_dir   = args.output_dir or str(_REPO_ROOT / "results" / "inspection")

    print(f"\n{BANNER}")
    print("  INSPECTION MANUELLE DU CORPUS (spacy_stanza)")
    print(BANNER)

    nlp = load_nlp()

    if args.text:
        mode_text(nlp, args.text, out_dir, args.serve)
    else:
        mode_csv(nlp, args.csv, args.column, args.rows,
                 args.top_deps, out_dir, args.serve)

    print(f"\n{BANNER}\n")


if __name__ == "__main__":
    main()
