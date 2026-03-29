#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Analyse des structures syntaxiques à partir de fichiers CoNLL-U

  Fonctionne avec les sorties de tout parser conforme au format CoNLL-U
  (Stanza, HopsParser, UDPipe, …).

  Structures détectées :
    - Subordonnées relatives (qui, que, dont, prép. + lequel)
    - Complétives (ccomp / xcomp)
    - Hypothétiques (si + SCONJ)
    - Gérondives (participe présent adverbial)
    - Incises (parataxis sans conjonction)
    - Propositions coordonnées (CCONJ + conj)
==========================================================================
"""

import argparse
import csv
import glob
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# ════════════════════════════════════════════════════════════════════════
# FEATURES
# ════════════════════════════════════════════════════════════════════════
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

# Column names in the output CSV (count + proportion for each feature)
CSV_COLUMNS_PER_FEATURE = []
for feat in STRUCTURE_FEATURES:
    CSV_COLUMNS_PER_FEATURE.append(f"nombre_{feat}")
    CSV_COLUMNS_PER_FEATURE.append(f"proportion_{feat}")


# ════════════════════════════════════════════════════════════════════════
# CONLL-U READER  (extended: includes lemma + feats)
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ConlluWord:
    id: int
    form: str
    lemma: str
    upos: str
    feats: str      # raw FEATS string, e.g. "PronType=Rel|Number=Sing"
    head: int
    deprel: str


@dataclass
class ConlluSentence:
    words: list[ConlluWord] = field(default_factory=list)
    comments: dict = field(default_factory=dict)


def read_conllu(filepath: str) -> list[ConlluSentence]:
    """Parse un fichier CoNLL-U et renvoie une liste de phrases."""
    sentences = []
    cur_words: list[ConlluWord] = []
    cur_com: dict = {}

    with open(filepath, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            # Skip hopsparser banners
            if line.startswith("[hops]"):
                continue
            if line.startswith("#"):
                m = re.match(r"#\s*(\S+)\s*=\s*(.*)", line)
                if m:
                    cur_com[m.group(1)] = m.group(2).strip()
                continue
            if not line.strip():
                if cur_words:
                    sentences.append(ConlluSentence(
                        words=list(cur_words), comments=dict(cur_com)
                    ))
                    cur_words, cur_com = [], {}
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            # Skip multi-word tokens and empty nodes
            if "-" in parts[0] or "." in parts[0]:
                continue
            try:
                cur_words.append(ConlluWord(
                    id=int(parts[0]),
                    form=parts[1],
                    lemma=parts[2] if parts[2] != "_" else parts[1].lower(),
                    upos=parts[3] if parts[3] != "_" else "X",
                    feats=parts[5] if len(parts) > 5 and parts[5] != "_" else "",
                    head=int(parts[6]) if parts[6] != "_" else 0,
                    deprel=parts[7] if parts[7] != "_" else "dep",
                ))
            except ValueError:
                continue

    if cur_words:
        sentences.append(ConlluSentence(
            words=list(cur_words), comments=dict(cur_com)
        ))
    return sentences


# ════════════════════════════════════════════════════════════════════════
# DÉTECTION DES STRUCTURES SYNTAXIQUES
# ════════════════════════════════════════════════════════════════════════
def detect_structures(sentences: list[ConlluSentence]) -> dict:
    """
    Détecte les structures syntaxiques dans une liste de phrases CoNLL-U.
    Retourne un dict { feature_name: count }.
    """
    counts = {feat: 0 for feat in STRUCTURE_FEATURES}

    for sent in sentences:
        words = sent.words
        prep_trigger = False      # ADP vu juste avant → prép. + lequel?
        rel_clause_trigger = False
        cconj_trigger = False

        for i, w in enumerate(words):
            # ── Relatives introduites par un pronom relatif ──
            is_rel_pronoun = False
            lem = w.lemma.lower() if w.lemma else ""
            
            if "PronType=Rel" in w.feats:
                is_rel_pronoun = True
            elif w.upos == "PRON" and lem in ("qui", "que", "qu'", "dont", "lequel", "laquelle", "lesquels", "lesquelles", "auquel", "auxquels", "auxquelles", "duquel", "desquels", "desquelles"):
                is_rel_pronoun = True

            if is_rel_pronoun and w.id != 1:
                if lem == "qui":
                    if w.deprel in ("nsubj", "nsubj:pass"):
                        counts["subordonnees_qui"] += 1
                        rel_clause_trigger = True
                elif lem in ("que", "qu'"):
                    if w.deprel == "obj":
                        counts["subordonnees_que"] += 1
                        rel_clause_trigger = True
                elif lem == "dont":
                    counts["subordonnees_dont"] += 1
                    rel_clause_trigger = True
                elif lem in ("lequel", "laquelle", "lesquels", "lesquelles", "auquel", "auxquels", "auxquelles", "duquel", "desquels", "desquelles"):
                    counts["subordonnees_prep_lequel"] += 1
                    rel_clause_trigger = True
                elif lem == "_":
                    # Fallback UD_FTB (où le lemme est manquant)
                    if w.deprel in ("nsubj", "nsubj:pass"):
                        counts["subordonnees_qui"] += 1
                        rel_clause_trigger = True
                    elif w.deprel == "obj":
                        counts["subordonnees_que"] += 1
                        rel_clause_trigger = True
                    else:
                        is_preceded_by_adp = (i > 0 and words[i-1].upos == "ADP")
                        if is_preceded_by_adp:
                            counts["subordonnees_prep_lequel"] += 1
                            rel_clause_trigger = True
                        elif w.deprel in ("iobj", "nmod", "obl"):
                            counts["subordonnees_dont"] += 1
                            rel_clause_trigger = True

            # ── Complétives (ccomp / xcomp) ──
            if w.upos == "VERB" and w.deprel in ("ccomp", "xcomp"):
                counts["completives"] += 1

            # ── Hypothétiques (si + SCONJ) ──
            if w.lemma == "si" and w.upos == "SCONJ":
                counts["hypothetiques"] += 1

            # ── Gérondif (participe présent à valeur adverbiale) ──
            if (w.upos == "VERB"
                    and "VerbForm=Part" in w.feats
                    and "Tense=Pres" in w.feats
                    and w.deprel == "advcl"):
                counts["gerondif"] += 1

            # ── Conjonction de coordination ──
            if w.upos == "CCONJ":
                cconj_trigger = True

            # ── Incises (parataxis / conj sans CCONJ) ──
            if (w.upos in ("VERB", "NOUN")
                    and w.deprel in ("conj", "parataxis")
                    and not cconj_trigger):
                counts["incises"] += 1
            elif w.upos == "VERB" and w.deprel == "acl:relcl":
                if not rel_clause_trigger:
                    counts["incises"] += 1
                else:
                    rel_clause_trigger = False

            # ── Propositions coordonnées (CCONJ + conj verbal) ──
            if cconj_trigger and w.upos == "VERB":
                if w.deprel == "conj":
                    counts["propositions_coordonnees"] += 1
                cconj_trigger = False

    return counts


def compute_features(sentences: list[ConlluSentence]) -> dict:
    """
    Renvoie un dict avec count et proportion pour chaque structure.
    """
    counts = detect_structures(sentences)
    n_sents = max(len(sentences), 1)

    result = {}
    for feat in STRUCTURE_FEATURES:
        c = counts[feat]
        result[f"nombre_{feat}"] = c
        result[f"proportion_{feat}"] = round(c / n_sents, 4)
    return result


# ════════════════════════════════════════════════════════════════════════
# REGROUPEMENT PAR TEXT_IDX  (utilisé par les scripts de parsing)
# ════════════════════════════════════════════════════════════════════════
def group_sentences_by_text_idx(sentences: list[ConlluSentence]) -> dict[int, list[ConlluSentence]]:
    """
    Regroupe les phrases par leur commentaire `# text_idx = N`.
    Si absent, chaque phrase est son propre groupe.
    """
    groups: dict[int, list[ConlluSentence]] = {}
    fallback_idx = 0
    for sent in sentences:
        tidx_str = sent.comments.get("text_idx")
        if tidx_str is not None:
            try:
                tidx = int(tidx_str)
            except ValueError:
                tidx = fallback_idx
                fallback_idx += 1
        else:
            tidx = fallback_idx
            fallback_idx += 1
        groups.setdefault(tidx, []).append(sent)
    return groups


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════


def discover_columns(input_dir: str) -> list[str]:
    """Auto-discover column names from output_*.conllu filenames."""
    pattern = os.path.join(input_dir, "output_*.conllu")
    cols = sorted(
        os.path.basename(f).replace("output_", "").replace(".conllu", "")
        for f in glob.glob(pattern)
    )
    return cols


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse des structures syntaxiques à partir de fichiers CoNLL-U.",
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Dossier contenant les fichiers output_<col>.conllu d'un parser.",
    )
    p.add_argument(
        "--output-csv", default=None,
        help="Chemin du CSV de sortie (défaut : <input-dir>/structures_syntaxiques.csv).",
    )
    p.add_argument(
        "--columns", nargs="+", default=None,
        help="Colonnes à analyser (défaut : auto-détection depuis output_*.conllu).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_csv = args.output_csv or os.path.join(input_dir, "structures_syntaxiques.csv")

    print(f"  Dossier d'entrée : {input_dir}")
    print(f"  CSV de sortie    : {output_csv}")

    # ── Discover or use explicit columns ──
    if args.columns:
        cols = args.columns
    else:
        cols = discover_columns(input_dir)
    if not cols:
        sys.exit("  ✗ Aucun fichier output_*.conllu trouvé.")
    print(f"  Colonnes         : {', '.join(cols)}")

    all_rows = []

    for col in cols:
        conllu_path = os.path.join(input_dir, f"output_{col}.conllu")
        if not os.path.exists(conllu_path):
            print(f"  ⚠ {conllu_path} introuvable → colonne {col} ignorée.")
            continue

        sentences = read_conllu(conllu_path)
        groups = group_sentences_by_text_idx(sentences)
        print(f"  {col}: {len(sentences)} phrases, {len(groups)} textes")

        for tidx in sorted(groups.keys()):
            sents = groups[tidx]
            feats = compute_features(sents)
            row = {"texte_id": tidx + 1, "colonne": col}
            row.update(feats)
            all_rows.append(row)

    if not all_rows:
        sys.exit("  ✗ Aucun fichier CoNLL-U trouvé.")

    # ── Export CSV ──
    fieldnames = ["texte_id", "colonne"] + CSV_COLUMNS_PER_FEATURE
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  ✓ {len(all_rows)} lignes exportées → {output_csv}")

    # ── Résumé rapide ──
    print(f"\n  Résumé (moyennes sur l'ensemble) :")
    for feat in STRUCTURE_FEATURES:
        key = f"nombre_{feat}"
        vals = [r[key] for r in all_rows]
        avg = sum(vals) / len(vals) if vals else 0
        total = sum(vals)
        print(f"    {feat:<35}  total={total:>5}  moy={avg:.2f}")


if __name__ == "__main__":
    main()
