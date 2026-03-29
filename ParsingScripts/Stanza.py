#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Parsing avec Stanza

  Sortie : <output_dir>/
             ├── resultats_par_sms.csv
             ├── output_<col>.conllu
             └── (logs terminaux)
==========================================================================
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOGURU_LEVEL"] = "ERROR"

import pathlib
import re
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import stanza

# ── PyTorch 2.6+ compatibility ──
# Stanza's pretrained embeddings use numpy pickle serialization, which fails
# with torch.load(weights_only=True). Restore the old default.
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": kwargs.get("weights_only", False)}
)

warnings.filterwarnings("ignore")

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
DEFAULT_COLS = ["SMS", "Transcodage_1", "Transcodage_2"]
DEFAULT_NICE = {
    "SMS":            "SMS brut",
    "Transcodage_1":  "Transcodage 1 (profs)",
    "Transcodage_2":  "Transcodage 2 (étudiants)",
}

_PRENOMS = [
    "Paul", "Marie", "Jean", "Sophie", "Pierre",
    "Julie", "Thomas", "Claire", "Nicolas", "Anne",
]
PRE_MAP = {f"<PRE_{i}>": nom for i, nom in enumerate(_PRENOMS, start=1)}

FKEYS = [
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

DIVERGENCE_KEYS = [
    "profondeur_arbre_max",
    "distance_dependance_max",
    "distance_dependance_moy",
    "nb_phrases",
]

LANG = "fr"
BATCH_SIZE = 64
BANNER = "=" * 76


# ════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════════════
def section(title: str):
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    for tag, prenom in PRE_MAP.items():
        text = text.replace(tag, prenom)
    text = re.sub(r"<PRE_\d+>", "Paul", text)
    return text


# ════════════════════════════════════════════════════════════════════════
# EXTRACTEURS SYNTAXIQUES
# ════════════════════════════════════════════════════════════════════════
def _tree_depth(sentence) -> int:
    children = defaultdict(list)
    root_id = None
    for w in sentence.words:
        if w.head == 0:
            root_id = w.id
        else:
            children[w.head].append(w.id)
    if root_id is None:
        return 0
    stack = [(root_id, 1)]
    max_depth = 1
    while stack:
        node, d = stack.pop()
        if d > max_depth:
            max_depth = d
        for child in children.get(node, []):
            stack.append((child, d + 1))
    return max_depth


def _dep_distances(sentence) -> list:
    return [abs(w.id - w.head) for w in sentence.words if w.head != 0]


def extract_features(doc) -> dict:
    depths = [_tree_depth(s) for s in doc.sentences]
    dists_per_sent = [_dep_distances(s) for s in doc.sentences]
    all_dists = [d for sent_dists in dists_per_sent for d in sent_dists]
    dep_counts = [len(d) for d in dists_per_sent]

    return {
        "profondeur_arbre_max":       max(depths, default=0),
        "profondeur_arbre_moy":       float(np.mean(depths))     if depths    else 0.0,
        "distance_dependance_max":    max(all_dists, default=0),
        "distance_dependance_moy":    float(np.mean(all_dists))  if all_dists else 0.0,
        "distance_dependance_var":    float(np.var(all_dists))   if all_dists else 0.0,
        "nb_dependances_moy_phrase":  float(np.mean(dep_counts)) if dep_counts else 0.0,
        "nb_dependances_var_phrase":  float(np.var(dep_counts))  if dep_counts else 0.0,
        "nb_phrases":                 len(doc.sentences),
        "nb_tokens":                  sum(len(s.words) for s in doc.sentences),
    }


# ════════════════════════════════════════════════════════════════════════
# CONLL-U EXPORT
# ════════════════════════════════════════════════════════════════════════
def export_conllu(docs: list, texts: list[str], output_path: str):
    """
    Exporte une liste de Documents Stanza au format CoNLL-U.
    Ajoute les commentaires text_idx et sent_idx pour compatibilité
    avec structures_syntaxiques.py.
    """
    with open(output_path, "w", encoding="utf-8") as fh:
        global_sent_idx = 0
        for tidx, doc in enumerate(docs):
            if doc is None:
                continue
            for sidx, sent in enumerate(doc.sentences):
                fh.write(f"# text_idx = {tidx}\n")
                fh.write(f"# sent_idx = {sidx}\n")
                sent_text = sent.text if hasattr(sent, 'text') and sent.text else ""
                fh.write(f"# text = {sent_text}\n")
                for word in sent.words:
                    wid = word.id
                    form = word.text
                    lemma = word.lemma if word.lemma else "_"
                    upos = word.upos if word.upos else "_"
                    xpos = word.xpos if word.xpos else "_"
                    feats = word.feats if word.feats else "_"
                    head = word.head if word.head is not None else 0
                    deprel = word.deprel if word.deprel else "_"
                    fh.write(
                        f"{wid}\t{form}\t{lemma}\t{upos}\t{xpos}\t"
                        f"{feats}\t{head}\t{deprel}\t_\t_\n"
                    )
                fh.write("\n")
                global_sent_idx += 1


# ════════════════════════════════════════════════════════════════════════
# ARGUMENTS CLI
# ════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parsing avec Stanza",
    )
    p.add_argument(
        "--csv", required=True,
        help="Chemin vers le fichier CSV du corpus.",
    )
    p.add_argument(
        "--output", default=str(_REPO_ROOT / "resultats_stanza"),
        help="Dossier de sortie (défaut : <repo>/resultats_stanza).",
    )
    p.add_argument(
        "--columns", nargs="+", default=None,
        help=(
            "Colonnes texte à analyser (défaut : SMS Transcodage_1 "
            "Transcodage_2). Pour un corpus mono-colonne : --columns Texte."
        ),
    )
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    CSV_PATH   = args.csv
    OUTPUT_DIR = args.output

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── Détection GPU ──
    if torch.cuda.is_available():
        GPU_DEVICE = 0
        GPU_NAME = torch.cuda.get_device_name(GPU_DEVICE)
        USE_GPU = True
        print(f"  \u2713 GPU : {GPU_NAME}")
    else:
        GPU_DEVICE = -1
        USE_GPU = False
        print("  \u26a0 Pas de GPU CUDA \u2192 CPU")

    # 1) CHARGEMENT DU CORPUS
    section("1) CHARGEMENT DU CORPUS")

    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for sep in (None, ";", ",", "\t"):
            try:
                kw = {"encoding": enc}
                if sep is None:
                    kw["sep"] = sep
                    kw["engine"] = "python"
                else:
                    kw["sep"] = sep
                candidate = pd.read_csv(CSV_PATH, **kw)
                if len(candidate.columns) >= 1:
                    df = candidate
                    print(f"  Fichier lu avec encodage={enc}, s\u00e9parateur={repr(sep)}")
                    break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        sys.exit(f"  \u2717 Impossible de lire {CSV_PATH}")

    # \u2500\u2500 R\u00e9solution des colonnes \u2500\u2500
    if args.columns:
        COLS = args.columns
    elif all(c in df.columns for c in DEFAULT_COLS):
        COLS = DEFAULT_COLS
    else:
        COLS = [c for c in df.columns if df[c].dtype == "object"]
        if not COLS:
            sys.exit("  \u2717 Aucune colonne texte d\u00e9tect\u00e9e dans le CSV.")
        print(f"  \u2139 Colonnes auto-d\u00e9tect\u00e9es : {COLS}")

    NICE = {}
    for col in COLS:
        NICE[col] = DEFAULT_NICE.get(col, col)

    N = len(df)
    print(f"  {N} lignes charg\u00e9es.")

    texts = {col: df[col].apply(clean_text).tolist() for col in COLS}
    for col in COLS:
        non_vide = sum(1 for t in texts[col] if t)
        print(f"  {NICE[col]:<30} : {non_vide} textes non vides")

    # ──────────────────────────────────────────────────────────────
    # 2) INITIALISATION DE STANZA
    # ──────────────────────────────────────────────────────────────
    section("2) INITIALISATION DE STANZA")

    device_label = f"GPU {GPU_NAME}" if USE_GPU else "CPU"
    print(f"  Langue       : {LANG}")
    print(f"  Device       : {device_label}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Processeurs  : tokenize, mwt, pos, lemma, depparse")

    stanza.download(LANG, verbose=False)
    nlp = stanza.Pipeline(
        LANG,
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=USE_GPU,
        device=f"cuda:{GPU_DEVICE}" if USE_GPU else "cpu",
        tokenize_batch_size=BATCH_SIZE,
        pos_batch_size=BATCH_SIZE,
        depparse_batch_size=BATCH_SIZE,
        lemma_batch_size=BATCH_SIZE,
        mwt_batch_size=BATCH_SIZE,
        verbose=False,
    )

    if USE_GPU:
        mem_alloc = torch.cuda.memory_allocated(GPU_DEVICE) / 1024**2
    else:
        print("  ✓ Pipeline prête (CPU).")

    # ──────────────────────────────────────────────────────────────
    # 3) ANALYSE SYNTAXIQUE (PAR BATCH GPU)
    # ──────────────────────────────────────────────────────────────
    section("3) ANALYSE SYNTAXIQUE (batch GPU)")

    feats  = {col: [None] * N for col in COLS}
    docs_per_col = {col: [None] * N for col in COLS}
    errors = {col: [] for col in COLS}

    total = N * len(COLS)
    print(f"  {total} textes a analyser ({N} x {len(COLS)} colonnes)")
    print(f"  Traitement par batch de {BATCH_SIZE} textes via bulk_process()\n")

    done = 0
    for col in COLS:
        t_col = time.time()
        print(f"  -- {NICE[col]} --")

        text_list = texts[col]
        batch_starts = list(range(0, N, BATCH_SIZE))

        for b_start in batch_starts:
            b_end = min(b_start + BATCH_SIZE, N)
            batch_texts = text_list[b_start:b_end]

            indexed_non_empty = [
                (j, t) for j, t in enumerate(batch_texts) if t
            ]

            if indexed_non_empty:
                local_indices, non_empty = zip(*indexed_non_empty)
                try:
                    batch_docs = nlp.bulk_process(list(non_empty))
                except Exception as exc:
                    print(f"    ! Erreur batch [{b_start}:{b_end}] : {exc}")
                    print(f"      -> Fallback un par un")
                    batch_docs = []
                    for txt in non_empty:
                        try:
                            batch_docs.append(nlp(txt))
                        except Exception as e2:
                            batch_docs.append(None)
                            errors[col].append((b_start, str(e2)))

                for local_j, doc in zip(local_indices, batch_docs):
                    global_i = b_start + local_j
                    if doc is not None:
                        feats[col][global_i] = extract_features(doc)
                        docs_per_col[col][global_i] = doc

            done += (b_end - b_start)
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"    {done:>5}/{total}  "
                  f"({done / total * 100:5.1f} %)  "
                  f"ecoule {elapsed / 60:.1f} min  "
                  f"restant ~{eta / 60:.1f} min", end="\r")

        dt_col = time.time() - t_col
        print(f"\n    OK {NICE[col]} : {dt_col:.1f} s")

        if USE_GPU:
            mem = torch.cuda.memory_allocated(GPU_DEVICE) / 1024**2
            mem_max = torch.cuda.max_memory_allocated(GPU_DEVICE) / 1024**2
            print(f"      VRAM : {mem:.0f} Mo (pic : {mem_max:.0f} Mo)")

    elapsed_total = time.time() - t0
    print(f"\n  OK Analyse terminee en {elapsed_total / 60:.1f} minutes.")
    total_errors = sum(len(v) for v in errors.values())
    if total_errors:
        print(f"  ! {total_errors} erreur(s) rencontree(s).")

    # ──────────────────────────────────────────────────────────────
    # 3b) EXPORT CONLL-U
    # ──────────────────────────────────────────────────────────────
    section("3b) EXPORT CONLL-U")
    for col in COLS:
        col_docs = docs_per_col[col]
        conllu_path = os.path.join(OUTPUT_DIR, f"output_{col}.conllu")
        export_conllu(col_docs, texts[col], conllu_path)
        n_docs = sum(1 for d in col_docs if d is not None)
        print(f"  {col} : {n_docs} documents -> {conllu_path}")

    # ──────────────────────────────────────────────────────────────
    # 4) EXPORT DES RESULTATS
    # ──────────────────────────────────────────────────────────────
    section("4) EXPORT")

    rows = []
    for i in range(N):
        row = {"sms_id": i + 1}
        for col in COLS:
            row[f"{col}_texte"] = texts[col][i]
            if feats[col][i] is not None:
                for k in FKEYS:
                    row[f"{col}_{k}"] = feats[col][i][k]
            else:
                for k in FKEYS:
                    row[f"{col}_{k}"] = np.nan
        if len(COLS) > 1 and all(feats[c][i] is not None for c in COLS):
            row["divergence_score"] = sum(
                max(feats[c][i][k] for c in COLS) - min(feats[c][i][k] for c in COLS)
                for k in DIVERGENCE_KEYS
            )
        else:
            row["divergence_score"] = np.nan
        rows.append(row)

    csv_out = os.path.join(OUTPUT_DIR, "resultats_par_sms.csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False, encoding="utf-8")
    print(f"  -> {csv_out}")

    # -- Resume --
    elapsed = time.time() - t0
    print(f"\n{BANNER}")
    print(f"  OK Parsing termine en {elapsed / 60:.1f} minutes.")
    print(f"  Resultats dans : {OUTPUT_DIR}/")
    print(BANNER)


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()