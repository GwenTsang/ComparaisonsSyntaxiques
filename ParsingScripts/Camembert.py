#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Parsing avec HopsParser

  Entrée  : CSV contenant les colonnes SMS, Transcodage_1, Transcodage_2
  Modèle  : dossier contenant config.json + weights.pt (hopsparser)
  Sortie  : <output_dir>/
              ├── resultats_par_sms.csv
              ├── input_<col>.conllu
              └── output_<col>.conllu
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

# ── Auto-detect hopsparser package from the cloned repository ──
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

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

BANNER = "_" * 76
BATCH_SIZE = 64


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
# TOKENISATION ET SEGMENTATION
# ════════════════════════════════════════════════════════════════════════
def split_into_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    parts = re.split(r'(?<=[.!?…])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def tokenize_fr(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


# ════════════════════════════════════════════════════════════════════════
# CONLL-U : CONSTRUCTION / LECTURE
# ════════════════════════════════════════════════════════════════════════
def build_conllu(text_list: list[str]):
    """Construit un fichier CoNLL-U. Retourne (conllu_string, sentence_map)."""
    lines, smap = [], []
    for tidx, text in enumerate(text_list):
        if not text:
            continue
        sents = split_into_sentences(text)
        for sidx, sent in enumerate(sents):
            toks = tokenize_fr(sent)
            if not toks:
                continue
            smap.append((tidx, sidx))
            lines.append(f"# text_idx = {tidx}")
            lines.append(f"# sent_idx = {sidx}")
            safe = sent.replace("\n", " ").replace("\r", " ")
            lines.append(f"# text = {safe}")
            for i, tok in enumerate(toks, 1):
                lines.append(f"{i}\t{tok}\t_\t_\t_\t_\t_\t_\t_\t_")
            lines.append("")
    return "\n".join(lines) + "\n", smap


def parse_conllu(filepath: str) -> list[dict]:
    """Lit un fichier CoNLL-U. Retourne une liste de phrases."""
    sentences, cur_words, cur_com = [], [], {}
    with open(filepath, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if line.startswith("[hops]"):
                continue
            if line.startswith("#"):
                m = re.match(r"#\s*(\S+)\s*=\s*(.*)", line)
                if m:
                    cur_com[m.group(1)] = m.group(2).strip()
                continue
            if not line.strip():
                if cur_words:
                    sentences.append({"words": list(cur_words),
                                      "comments": dict(cur_com)})
                    cur_words, cur_com = [], {}
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            if "-" in parts[0] or "." in parts[0]:
                continue
            try:
                cur_words.append({
                    "id":     int(parts[0]),
                    "form":   parts[1],
                    "upos":   parts[3] if parts[3] != "_" else "X",
                    "head":   int(parts[6]) if parts[6] != "_" else 0,
                    "deprel": parts[7] if parts[7] != "_" else "dep",
                })
            except ValueError:
                continue
    if cur_words:
        sentences.append({"words": list(cur_words), "comments": dict(cur_com)})
    return sentences


def group_by_text(parsed: list[dict], smap: list[tuple], N: int) -> dict:
    out = defaultdict(list)
    for sent, (tidx, _) in zip(parsed, smap):
        out[tidx].append(sent)
    return dict(out)


# ════════════════════════════════════════════════════════════════════════
# HOPSPARSER : EXÉCUTION IN-PROCESS
# ════════════════════════════════════════════════════════════════════════
def run_hops_inprocess(parser_model, inp: str, outp: str,
                       batch_size: int = 64) -> tuple[bool, str]:
    try:
        from hopsparser.utils import smart_open
        with smart_open(inp) as in_stream, smart_open(outp, "w") as out_stream:
            for tree in parser_model.parse(inpt=in_stream, batch_size=batch_size):
                out_stream.write(tree.to_conllu())
                out_stream.write("\n\n")
        return True, ""
    except Exception as e:
        return False, str(e)


# ════════════════════════════════════════════════════════════════════════
# EXTRACTION DE FEATURES
# ════════════════════════════════════════════════════════════════════════
def _tree_depth(words: list[dict]) -> int:
    children = defaultdict(list)
    root = None
    for w in words:
        if w["head"] == 0:
            root = w["id"]
        else:
            children[w["head"]].append(w["id"])
    if root is None:
        return 0
    stack, mx = [(root, 1)], 1
    while stack:
        node, d = stack.pop()
        if d > mx:
            mx = d
        for c in children.get(node, []):
            stack.append((c, d + 1))
    return mx


def _dep_dists(words: list[dict]) -> list[int]:
    return [abs(w["id"] - w["head"]) for w in words if w["head"] != 0]


def extract_features(sents: list[dict]) -> dict | None:
    if not sents:
        return None
    depths     = [_tree_depth(s["words"]) for s in sents]
    dists_per  = [_dep_dists(s["words"]) for s in sents]
    all_dists  = [d for ds in dists_per for d in ds]
    dep_counts = [len(d) for d in dists_per]
    return {
        "profondeur_arbre_max":       max(depths, default=0),
        "profondeur_arbre_moy":       float(np.mean(depths))     if depths     else 0.0,
        "distance_dependance_max":    max(all_dists, default=0),
        "distance_dependance_moy":    float(np.mean(all_dists))  if all_dists  else 0.0,
        "distance_dependance_var":    float(np.var(all_dists))   if all_dists  else 0.0,
        "nb_dependances_moy_phrase":  float(np.mean(dep_counts)) if dep_counts else 0.0,
        "nb_dependances_var_phrase":  float(np.var(dep_counts))  if dep_counts else 0.0,
        "nb_phrases":                 len(sents),
        "nb_tokens":                  sum(len(s["words"]) for s in sents),
    }


# ════════════════════════════════════════════════════════════════════════
# ARGUMENTS CLI
# ════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    from download_model import MODEL_REGISTRY

    model_names = ", ".join(MODEL_REGISTRY.keys())
    p = argparse.ArgumentParser(
        description="Parsing avec HopsParser (CamemBERT v2 / DeBERTa)",
    )

    # ── Source du modèle (chemin direct OU nom pour auto-download) ──
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        help="Chemin vers le dossier du modèle hopsparser.",
    )
    model_group.add_argument(
        "--model-name",
        help=(
            f"Nom court du modèle à télécharger automatiquement "
            f"({model_names})."
        ),
    )

    p.add_argument(
        "--csv",
        default=str(_REPO_ROOT / "Corpus" / "1000_SMS_transcodage.csv"),
        help="Chemin vers le fichier CSV du corpus.",
    )
    p.add_argument(
        "--output", default=None,
        help="Dossier de sortie (défaut : <nom_du_modèle>-HOPS).",
    )
    p.add_argument(
        "--columns", nargs="+", default=None,
        help=(
            "Colonnes texte à analyser (défaut : SMS Transcodage_1 "
            "Transcodage_2). Pour un corpus mono-colonne, utilisez "
            "par ex. --columns Texte."
        ),
    )
    p.add_argument(
        "--models-dir",
        default=str(_REPO_ROOT / "models"),
        help="Dossier où stocker les modèles téléchargés (défaut : <repo>/models).",
    )
    args = p.parse_args()

    # ── Auto-download si --model-name ──
    if args.model_name:
        from download_model import get_model_path
        args.model = get_model_path(args.model_name, args.models_dir)
        print(f"  ✓ Modèle résolu : {args.model}")

    if args.output is None:
        model_name = pathlib.Path(args.model).resolve().name
        args.output = str(
            pathlib.Path(args.model).resolve().parent / f"{model_name}-HOPS"
        )
    return args


# ════════════════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    CSV_PATH   = args.csv
    OUTPUT_DIR = args.output
    MODEL_PATH = args.model

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── Détection GPU ──
    if torch.cuda.is_available():
        device   = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✓ GPU : {gpu_name}  ({gpu_vram:.1f} Go)")
    else:
        device = "cpu"
        print("  ⚠ Pas de GPU CUDA → CPU")

    # ──────────────────────────────────────────────────────────────
    # 1) CHARGEMENT DU CORPUS
    # ──────────────────────────────────────────────────────────────
    section("1) CHARGEMENT DU CORPUS")

    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for sep in (None, ";", ",", "\t"):
            try:
                kw = {"encoding": enc}
                if sep is None:
                    kw["sep"], kw["engine"] = sep, "python"
                else:
                    kw["sep"] = sep
                cand = pd.read_csv(CSV_PATH, **kw)
                if len(cand.columns) >= 1:
                    df = cand
                    print(f"  Lu avec enc={enc}, sep={repr(sep)}")
                    break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        sys.exit(f"  ✗ Impossible de lire {CSV_PATH}")

    # ── Résolution des colonnes ──
    if args.columns:
        COLS = args.columns
    elif all(c in df.columns for c in DEFAULT_COLS):
        COLS = DEFAULT_COLS
    else:
        # Auto-détection : prendre toutes les colonnes texte du CSV
        COLS = [c for c in df.columns if df[c].dtype == "object"]
        if not COLS:
            sys.exit("  ✗ Aucune colonne texte détectée dans le CSV.")
        print(f"  ℹ Colonnes auto-détectées : {COLS}")

    NICE = {}
    for col in COLS:
        NICE[col] = DEFAULT_NICE.get(col, col)

    N = len(df)
    print(f"  {N} lignes chargées.")
    texts = {col: df[col].apply(clean_text).tolist() for col in COLS}
    for col in COLS:
        nv = sum(1 for t in texts[col] if t)
        print(f"  {NICE[col]:<30} : {nv} textes non vides")

    # ──────────────────────────────────────────────────────────────
    # 2) CHARGEMENT DU MODÈLE
    # ──────────────────────────────────────────────────────────────
    section("2) CHARGEMENT DU MODÈLE")

    if not os.path.isdir(MODEL_PATH):
        sys.exit(f"  ✗ Dossier modèle introuvable : {MODEL_PATH}")

    # Auto-detect model/ subdirectory
    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        candidate = os.path.join(MODEL_PATH, "model")
        if os.path.isfile(os.path.join(candidate, "config.json")):
            print(f"  ℹ config.json trouvé dans {candidate}")
            MODEL_PATH = candidate
        else:
            sys.exit(f"  ✗ config.json introuvable dans {MODEL_PATH} ni dans {candidate}")

    print(f"  ✓ Modèle           : {MODEL_PATH}")
    print(f"  Device             : {device}")

    from hopsparser.parser import BiAffineParser
    print(f"  Chargement du modèle en mémoire…")
    t_load = time.time()
    parser_model = BiAffineParser.load(MODEL_PATH)
    parser_model = parser_model.to(device)
    parser_model.eval()
    
    # ──────────────────────────────────────────────────────────────
    # 3) ANALYSE SYNTAXIQUE (HopsParser, in-process)
    # ──────────────────────────────────────────────────────────────
    section("3) ANALYSE SYNTAXIQUE (HopsParser, in-process)")

    feats = {col: [None] * N for col in COLS}

    for col in COLS:
        tc = time.time()
        print(f"\n  ── {NICE[col]} ──")

        conllu_str, smap = build_conllu(texts[col])
        n_sents = len(smap)
        print(f"    {n_sents} phrases tokenisées")
        if n_sents == 0:
            print(f"    ⚠ Rien à analyser")
            continue

        inp = os.path.join(OUTPUT_DIR, f"input_{col}.conllu")
        outp = os.path.join(OUTPUT_DIR, f"output_{col}.conllu")
        with open(inp, "w", encoding="utf-8") as fh:
            fh.write(conllu_str)

        print(f"    Lancement parsing in-process (device={device}, batch_size={BATCH_SIZE}) …")
        ok, stderr = run_hops_inprocess(parser_model, inp, outp, batch_size=BATCH_SIZE)

        if ok and os.path.exists(outp):
            parsed = parse_conllu(outp)
            print(f"    {len(parsed)} phrases en sortie (attendu {n_sents})")

            grp = group_by_text(parsed, smap, N)
            for tidx, sents in grp.items():
                feats[col][tidx] = extract_features(sents)
        else:
            print(f"    ✗ Échec du parsing : {stderr[:500]}")
            print(f"    → Tentative phrase par phrase (batch_size=1) …")

            from hopsparser.deptree import DepGraph

            fallback_out = os.path.join(OUTPUT_DIR, f"_fallback_{col}.conllu")
            n_ok, n_fail = 0, 0

            with open(inp, encoding="utf-8") as in_f:
                trees = list(DepGraph.read_conll(in_f))

            with open(fallback_out, "w", encoding="utf-8") as out_f:
                for tree in trees:
                    try:
                        for result_tree in parser_model.parse(
                            inpt=[tree.to_conllu() + "\n\n"],
                            batch_size=1,
                        ):
                            out_f.write(result_tree.to_conllu())
                            out_f.write("\n\n")
                            n_ok += 1
                    except Exception as e:
                        n_fail += 1
                        if n_fail <= 5:
                            print(f"      ⚠ Échec phrase : {str(e)[:100]}")

            print(f"      Résultat fallback : {n_ok} OK, {n_fail} échouées")

            if n_ok > 0:
                parsed = parse_conllu(fallback_out)
                grp = group_by_text(parsed, smap[:len(parsed)], N)
                for tidx, sents in grp.items():
                    feats[col][tidx] = extract_features(sents)

            if os.path.exists(fallback_out):
                import shutil
                shutil.move(fallback_out, outp)

        dt = time.time() - tc
        ok_n = sum(1 for f in feats[col] if f is not None)
        print(f"    ✓ {ok_n}/{N} textes analysés en {dt:.1f} s")
        if torch.cuda.is_available():
            print(f"      VRAM alloc : "
                  f"{torch.cuda.memory_allocated(0)/1024**2:.0f} Mo  "
                  f"(pic {torch.cuda.max_memory_allocated(0)/1024**2:.0f} Mo)")

    elapsed = time.time() - t0
    print(f"\n  ✓ Analyse terminée en {elapsed/60:.1f} min.")

    # ──────────────────────────────────────────────────────────────
    # 4) EXPORT DES RÉSULTATS
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
        if all(feats[c][i] is not None for c in COLS):
            row["divergence_score"] = sum(
                max(feats[c][i][k] for c in COLS)
                - min(feats[c][i][k] for c in COLS)
                for k in DIVERGENCE_KEYS
            )
        else:
            row["divergence_score"] = np.nan
        rows.append(row)

    csv_out = os.path.join(OUTPUT_DIR, "resultats_par_sms.csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False, encoding="utf-8")
    print(f"  → {csv_out}")

    # ── Résumé ──
    print(f"\n{BANNER}")
    print(f"  ✓ Parsing terminé en {elapsed / 60:.1f} minutes.")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(0) / 1024**2
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  Pic VRAM : {peak:.0f} Mo / {vram:.0f} Mo")
    print(f"  Résultats dans : {OUTPUT_DIR}/")
    print(BANNER)


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()