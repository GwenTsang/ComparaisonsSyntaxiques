#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_conllu.py — Serveur displacy pour fichiers CoNLL-U pre-calcules.

Lance un serveur displacy interactif a partir d'un fichier CoNLL-U
(resultats HopsParser ou Stanza). Aucun modele n'est charge.

Usage :
    python AnalysisScripts/serve_conllu.py results/SMS/fsmb/output_SMS.conllu
    python AnalysisScripts/serve_conllu.py results/SMS/gsd/output_SMS.conllu --sent-range 5-20
    python AnalysisScripts/serve_conllu.py results/SMS/stanza/output_SMS.conllu --port 5001
"""

import argparse
import sys

# ── Known non-UD deprels (FTB/SEM annotation, e.g. fsmb model) ───────
_NON_UD_DEPRELS = {
    "suj", "a_obj", "obj.p", "obj.cpl", "aff", "ponct",
    "mod", "seg", "dep.coord", "p_obj",
}


# ── CoNLL-U parser ────────────────────────────────────────────────────

def _parse_sent_range(sent_range: str | None) -> tuple[int, int]:
    if not sent_range:
        return (0, 10 ** 9)
    parts = sent_range.split("-")
    start = int(parts[0]) - 1
    end   = int(parts[1]) if len(parts) > 1 else start + 1
    return (start, end)


def read_conllu(path: str, sent_range: str | None = None) -> list[dict]:
    """Return a list of sentence dicts from a CoNLL-U file.

    Each dict has: 'text', 'tokens' (list of token dicts), 'idx' (1-based),
    'has_deps' (bool), 'non_ud_rels' (set).
    """
    start, end = _parse_sent_range(sent_range)
    sentences: list[dict] = []
    tokens: list[dict] = []
    meta: dict = {}
    global_idx = 0

    def flush():
        nonlocal global_idx
        if not tokens:
            return
        global_idx += 1
        if not (start <= global_idx - 1 < end):
            tokens.clear(); meta.clear()
            return
        text     = meta.get("text", " ".join(t["form"] for t in tokens))
        has_deps = any(t["head"] != "_" and t["deprel"] != "_" for t in tokens)
        non_ud   = {t["deprel"] for t in tokens if t["deprel"] in _NON_UD_DEPRELS}
        sentences.append({
            "text": text, "tokens": list(tokens),
            "idx": global_idx, "has_deps": has_deps, "non_ud_rels": non_ud,
        })
        tokens.clear(); meta.clear()

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                flush(); continue
            if line.startswith("#"):
                if line.startswith("# text") and "=" in line:
                    meta["text"] = line.split("=", 1)[1].strip()
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            tid = cols[0]
            if "-" in tid or "." in tid:   # multi-word / empty node
                continue
            tokens.append({
                "id": int(tid), "form": cols[1], "lemma": cols[2],
                "upos": cols[3], "head": cols[6], "deprel": cols[7],
            })
    flush()
    return sentences


# ── displacy conversion ───────────────────────────────────────────────

def to_displacy(sentence: dict) -> dict:
    """Convert one CoNLL-U sentence to displacy manual-render format."""
    words = [
        {"text": t["form"], "tag": t["upos"] if t["upos"] != "_" else ""}
        for t in sentence["tokens"]
    ]
    arcs = []
    for t in sentence["tokens"]:
        if t["head"] == "_" or t["deprel"] == "_":
            continue
        head_id = int(t["head"])
        if head_id == 0:
            continue
        di = t["id"] - 1
        hi = head_id - 1
        if di < hi:
            arcs.append({"start": di, "end": hi, "label": t["deprel"], "dir": "right"})
        else:
            arcs.append({"start": hi, "end": di, "label": t["deprel"], "dir": "left"})
    return {"words": words, "arcs": arcs}


# ── compatibility check ───────────────────────────────────────────────

def check_and_warn(path: str, sentences: list[dict]) -> None:
    no_deps = [s for s in sentences if not s["has_deps"]]
    if no_deps:
        print(
            f"\n[!] {len(no_deps)}/{len(sentences)} phrase(s) sans annotations"
            " de dependance (HEAD/DEPREL = '_')."
            "\n    => Ce fichier est probablement un INPUT (avant parsing)."
            "\n    => Utilisez le fichier 'output_*.conllu' correspondant."
        )
    all_non_ud: set[str] = set()
    for s in sentences:
        all_non_ud |= s["non_ud_rels"]
    if all_non_ud:
        print(
            f"\n[i] Tagset non-UD detecte : {sorted(all_non_ud)}"
            "\n    Annotation FTB/SEM (modele fsmb) — labels differents de l'UD."
            "\n    Les arbres s'affichent normalement."
        )
    if not no_deps and not all_non_ud:
        print("[OK] Fichier CoNLL-U valide et compatible UD.")


# ── main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serveur displacy interactif pour un fichier CoNLL-U.",
    )
    p.add_argument("conllu", help="Chemin vers le fichier CoNLL-U.")
    p.add_argument(
        "--sent-range", default=None,
        help="Plage de phrases a afficher, ex. '1-20' (1-indexed, inclusif)."
             " Par defaut : toutes les phrases.",
    )
    p.add_argument(
        "--port", type=int, default=5000,
        help="Port du serveur displacy (defaut : 5000).",
    )
    return p.parse_args()


def main() -> None:
    from spacy import displacy

    args = parse_args()

    print(f"\nLecture de : {args.conllu}")
    sentences = read_conllu(args.conllu, args.sent_range)

    if not sentences:
        print("Aucune phrase trouvee. Verifiez --sent-range ou le fichier.")
        sys.exit(1)

    print(f"{len(sentences)} phrase(s) chargee(s).")
    check_and_warn(args.conllu, sentences)

    # Filtrer les phrases sans annotations (fichiers input)
    to_serve = [s for s in sentences if s["has_deps"]]
    if not to_serve:
        print("\nAucune phrase avec annotations. Rien a afficher.")
        sys.exit(1)

    manual_data = [to_displacy(s) for s in to_serve]

    print(f"\nServeur displacy sur http://localhost:{args.port}")
    print("Ctrl+C pour arreter.\n")
    displacy.serve(
        manual_data,
        style="dep",
        manual=True,
        port=args.port,
        options={"distance": 100, "compact": True},
    )


if __name__ == "__main__":
    main()
