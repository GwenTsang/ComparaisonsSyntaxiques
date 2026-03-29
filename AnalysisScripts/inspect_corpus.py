#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Inspection manuelle du corpus avec spacy_stanza / ConLL-U

  Modes d'utilisation :
    1) Texte libre en argument  (re-parse via Stanza)
    2) Lecture d'un CSV de corpus (SMS ou Philosophie)  (re-parse via Stanza)
    3) Filtrage automatique des phrases a longues dependances
       (pour identifier les erreurs potentielles du modele)
    4) Lecture directe d'un fichier CoNLL-U pre-compute  (sans Stanza)

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

    # Visualiser un fichier CoNLL-U pre-compute (ex: resultats HopsParser)
    python inspect_corpus.py --conllu results/SMS/fsmb/output_SMS.conllu

    # Visualiser uniquement les phrases 5 a 10 d'un CoNLL-U
    python inspect_corpus.py --conllu results/SMS/gsd/output_SMS.conllu \
           --sent-range 5-10

    # Serveur interactif a partir d'un CoNLL-U
    python inspect_corpus.py --conllu results/SMS/stanza/output_SMS.conllu \
           --sent-range 1-3 --serve
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


# ── reading corpus (CSV) ─────────────────────────────────────────────

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


# ── reading CoNLL-U ───────────────────────────────────────────────────

# Known non-UD tagsets (POS or deprel) — display a warning but still render
_NON_UD_DEPRELS = {
    "suj", "a_obj", "obj.p", "obj.cpl", "aff", "ponct", "mod",
    "seg", "dep.coord", "p_obj", "obj.cpl",
}


def _parse_sent_range(sent_range: str | None) -> tuple[int, int]:
    """Return (start_0indexed, end_exclusive) from a '1-10' style string."""
    if not sent_range:
        return (0, 10 ** 9)
    parts = sent_range.split("-")
    start = int(parts[0]) - 1
    end = int(parts[1]) if len(parts) > 1 else start + 1
    return (start, end)


def read_conllu(
    conllu_path: str,
    sent_range: str | None = None,
) -> list[dict]:
    """Parse a CoNLL-U file into a list of sentence dicts.

    Each sentence dict has:
      - 'text'   : str   (from # text = ... comment, or reconstructed)
      - 'tokens' : list of token dicts with keys
                   id, form, lemma, upos, head, deprel
      - 'idx'    : int   (1-indexed sentence number in the file)
      - 'has_deps' : bool (False if all head/deprel fields are '_')
      - 'non_ud_rels' : set of deprel values not in UD

    Parameters
    ----------
    conllu_path : str
        Path to the CoNLL-U file.
    sent_range : str, optional
        Sentence range '5-10' (1-indexed, inclusive).

    Returns
    -------
    list[dict]
    """
    start, end = _parse_sent_range(sent_range)
    sentences = []
    current_tokens: list[dict] = []
    current_meta: dict = {}
    sent_global_idx = 0  # 0-indexed counter across file

    def _flush():
        nonlocal sent_global_idx
        if not current_tokens:
            return
        sent_global_idx += 1
        # Apply range filter (1-indexed)
        if not (start <= sent_global_idx - 1 < end):
            current_tokens.clear()
            current_meta.clear()
            return

        text = current_meta.get("text", " ".join(t["form"] for t in current_tokens))
        has_deps = any(
            t["head"] != "_" and t["deprel"] != "_"
            for t in current_tokens
        )
        non_ud = {
            t["deprel"] for t in current_tokens
            if t["deprel"] in _NON_UD_DEPRELS
        }
        sentences.append({
            "text": text,
            "tokens": list(current_tokens),
            "idx": sent_global_idx,
            "has_deps": has_deps,
            "non_ud_rels": non_ud,
        })
        current_tokens.clear()
        current_meta.clear()

    with open(conllu_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line:
                _flush()
                continue
            if line.startswith("#"):
                # Parse metadata comments
                if line.startswith("# text") and "=" in line:
                    current_meta["text"] = line.split("=", 1)[1].strip()
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                continue  # malformed line
            token_id = parts[0]
            # Skip multi-word tokens (e.g. "1-2") and empty nodes ("1.1")
            if "-" in token_id or "." in token_id:
                continue
            current_tokens.append({
                "id"     : int(token_id),
                "form"   : parts[1],
                "lemma"  : parts[2],
                "upos"   : parts[3],
                "head"   : parts[6],
                "deprel" : parts[7],
            })
    _flush()  # last sentence if file does not end with blank line
    return sentences


def conllu_to_displacy_data(sentence: dict) -> dict:
    """Convert a CoNLL-U sentence dict to displacy manual rendering data.

    displacy 'manual' format:
      {
        'words'  : [{'text': ..., 'tag': ...}, ...],
        'arcs'   : [{'start': int, 'end': int, 'label': str, 'dir': str}, ...],
      }
    Indices are 0-based in displacy.
    """
    tokens = sentence["tokens"]
    words = [
        {"text": t["form"], "tag": t["upos"] if t["upos"] != "_" else ""}
        for t in tokens
    ]
    arcs = []
    for t in tokens:
        if t["head"] == "_" or t["deprel"] == "_":
            continue
        head_id = int(t["head"])
        if head_id == 0:
            continue  # root arc — displacy handles root separately
        dep_idx  = t["id"] - 1   # 0-based
        head_idx = head_id - 1   # 0-based
        if dep_idx < head_idx:
            arcs.append({"start": dep_idx,  "end": head_idx, "label": t["deprel"], "dir": "right"})
        else:
            arcs.append({"start": head_idx, "end": dep_idx,  "label": t["deprel"], "dir": "left"})
    return {"words": words, "arcs": arcs}


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
        print(f"\n  -- Texte {idx} --")
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


# ── CoNLL-U mode ─────────────────────────────────────────────────────

def _check_conllu_compatibility(conllu_path: str, sentences: list[dict]) -> None:
    """Print compatibility warnings for a CoNLL-U file."""
    path_lower = conllu_path.lower()

    # Detect input files (no annotations)
    no_deps = [s for s in sentences if not s["has_deps"]]
    if no_deps:
        print(
            f"\n  [!] ATTENTION : {len(no_deps)}/{len(sentences)} phrase(s) "
            "n'ont PAS d'annotations de dependance (champs HEAD/DEPREL = '_').\n"
            "     => Ce fichier est probablement un fichier INPUT (avant parsing).\n"
            "     => Utilisez le fichier 'output_*.conllu' correspondant."
        )

    # Detect non-UD tagset (e.g. fsmb / FTB annotation)
    all_non_ud = set()
    for s in sentences:
        all_non_ud |= s["non_ud_rels"]
    if all_non_ud:
        print(
            f"\n  [i] TAGSET non-UD detecte : relations {sorted(all_non_ud)}.\n"
            "     Ce fichier utilise probablement l'annotation FTB/SEM (modele fsmb).\n"
            "     Les arbres seront affiches correctement mais les labels\n"
            "     differeront des conventions Universal Dependencies (UD)."
        )

    if not no_deps and not all_non_ud:
        print("\n  [OK] Fichier CoNLL-U valide et compatible UD.")


def mode_conllu(
    conllu_path: str,
    sent_range: str | None,
    output_dir: str,
    serve: bool,
) -> None:
    """Visualise pre-computed CoNLL-U dependency trees via displacy."""
    from spacy import displacy

    print(f"\n  Lecture de : {conllu_path}")
    sentences = read_conllu(conllu_path, sent_range)

    if not sentences:
        print("  ! Aucune phrase trouvee (verifier --sent-range ou le fichier).")
        return

    print(f"  {len(sentences)} phrase(s) chargee(s).")
    _check_conllu_compatibility(conllu_path, sentences)

    opts = {"distance": 100, "compact": True, "bg": "#1a1a2e", "color": "#eee"}

    if serve:
        # displacy.serve expects spaCy Doc objects; use manual mode instead
        manual_data = [conllu_to_displacy_data(s) for s in sentences]
        print(f"\n  Serveur displacy (manuel) sur http://localhost:5000")
        print("  Ctrl+C pour arreter.\n")
        displacy.serve(
            manual_data,
            style="dep",
            manual=True,
            options={"distance": 100, "compact": True},
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    for s in sentences:
        stats = analyse_doc_from_tokens(s["tokens"])
        print(f"\n  -- Phrase {s['idx']} : {s['text'][:80]}")
        print(f"     Tokens : {stats['n_tokens']} | "
              f"Dep. max : {stats['max_dep']} | "
              f"Dep. moy : {stats['mean_dep']:.2f}")
        if not s["has_deps"]:
            print("     [SKIP] Pas d'annotations de dependance.")
            continue

        data = conllu_to_displacy_data(s)
        html = displacy.render(
            data,
            style="dep",
            page=True,
            manual=True,
            options=opts,
        )
        fname = f"conllu_{s['idx']:04d}.html"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"     -> {fpath}")

    print(f"\n  {len(sentences)} fichier(s) HTML genere(s) dans {output_dir}/")


def analyse_doc_from_tokens(tokens: list[dict]) -> dict:
    """Compute dep distance stats from CoNLL-U token dicts."""
    distances = []
    for t in tokens:
        if t["head"] != "_" and int(t["head"]) != 0:
            dist = abs(t["id"] - int(t["head"]))
            distances.append(dist)
    if not distances:
        return {"max_dep": 0, "mean_dep": 0.0, "n_tokens": len(tokens)}
    return {
        "max_dep" : max(distances),
        "mean_dep": sum(distances) / len(distances),
        "n_tokens": len(tokens),
    }


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspection manuelle du corpus (spacy_stanza ou CoNLL-U).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            '  python inspect_corpus.py --text "Le chat mange la souris."\n'
            "  python inspect_corpus.py --csv Corpus/1000_SMS_transcodage.csv "
            "--column SMS --rows 1-10\n"
            "  python inspect_corpus.py --csv Corpus/philosophie.csv "
            "--column Texte --top-deps 15\n"
            "  python inspect_corpus.py --text \"...\" --serve\n"
            "  python inspect_corpus.py "
            "--conllu results/SMS/fsmb/output_SMS.conllu --sent-range 1-5\n"
            "  python inspect_corpus.py "
            "--conllu results/SMS/gsd/output_SMS.conllu --serve\n"
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
    src.add_argument(
        "--conllu",
        help="Fichier CoNLL-U pre-compute a visualiser (sans re-parser via Stanza).",
    )
    p.add_argument(
        "--column", "-c",
        default="Texte",
        help="Nom de la colonne contenant le texte (defaut : Texte). "
             "Utilisé uniquement avec --csv.",
    )
    p.add_argument(
        "--rows",
        help="Plage de lignes CSV a inspecter, ex. '1-10'. "
             "Utilisé uniquement avec --csv.",
    )
    p.add_argument(
        "--sent-range",
        help="Plage de phrases CoNLL-U a inspecter, ex. '1-10'. "
             "Utilisé uniquement avec --conllu.",
    )
    p.add_argument(
        "--top-deps",
        type=int,
        help="Afficher les N textes avec les plus longues dependances. "
             "Utilisé uniquement avec --csv.",
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

    if args.conllu:
        # ── CoNLL-U mode: no Stanza required ──────────────────────────
        print("  INSPECTION CoNLL-U (affichage sans re-parsing)")
        print(BANNER)
        mode_conllu(
            args.conllu,
            args.sent_range,
            output_dir,
            args.serve,
        )
    else:
        # ── Stanza mode ───────────────────────────────────────────────
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
