#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
  Benchmark du processeur syntaxique HopsParser
  (CamemBERT v2 / DeBERTa, fine-tuné sur UD French GSD)
  sur un corpus de 1000 SMS

  + Comparaison automatique avec les résultats Stanza (PREV_RESULTS)
==========================================================================

Entrée  : ./1000_SMS_transcodage.csv
Modèle  : ./model  (hopsparser, camembert-v2-base-gsd)
Sortie  : ./camembertav2-base-gsd-hopsparser/
            ├── resultats_par_sms.csv
            ├── statistiques_agregees.csv
            ├── tests_wilcoxon.csv
            ├── distributions_deprel_upos.csv
            ├── exemples_desaccords.txt
            ├── comparaison_stanza_vs_hops.csv
            ├── input_<col>.conllu          (entrées envoyées au parser)
            └── output_<col>.conllu         (sorties du parser)
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
CSV_PATH     = "/content/1000_SMS_transcodage.csv"
OUTPUT_DIR   = "/content/camembertav2-base-gsd-hopsparser"
PREV_RESULTS = "/content/resultats_stanza"
MODEL_PATH   = "./model"

COLS = ["SMS", "Transcodage_1", "Transcodage_2"]
NICE = {
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

BANNER   = "=" * 76
COL_W    = 28

# Batch size for in-process parsing (number of CoNLL-U sentences per batch).
# 32 is safe for a T4 (14 GB VRAM) with DeBERTa-base.
BATCH_SIZE = 32


# ════════════════════════════════════════════════════════════════════════
# UTILITAIRES D'AFFICHAGE
# ════════════════════════════════════════════════════════════════════════
def section(title: str):
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


# ════════════════════════════════════════════════════════════════════════
# NETTOYAGE TEXTE
# ════════════════════════════════════════════════════════════════════════
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
    """Segmentation en phrases (heuristique simple pour SMS)."""
    if not text or not text.strip():
        return []
    text = text.strip()
    parts = re.split(r'(?<=[.!?…])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def tokenize_fr(text: str) -> list[str]:
    """Tokeniseur regex simple."""
    if not text or not text.strip():
        return []
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


# ════════════════════════════════════════════════════════════════════════
# CONLL-U  :  CONSTRUCTION / PARSING
# ════════════════════════════════════════════════════════════════════════
def build_conllu(text_list: list[str]):
    """
    Construit un fichier CoNLL-U à partir d'une liste de textes.
    Retourne (conllu_string, sentence_map).
    sentence_map : liste de (text_idx, sent_local_idx) par phrase.
    """
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
            lines.append("")          # ligne vide = fin de phrase
    # Ensure the file ends with a newline after the last blank line
    return "\n".join(lines) + "\n", smap


def parse_conllu(filepath: str) -> list[dict]:
    """
    Lit un fichier CoNLL-U produit par hopsparser.
    Retourne une liste de phrases, chaque phrase = dict(words=[…], comments={…}).
    """
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
    """Regroupe les phrases parsées par text_idx d'origine."""
    out = defaultdict(list)
    for sent, (tidx, _) in zip(parsed, smap):
        out[tidx].append(sent)
    return dict(out)


# ════════════════════════════════════════════════════════════════════════
# HOPSPARSER  :  EXÉCUTION IN-PROCESS
# ════════════════════════════════════════════════════════════════════════
def run_hops_inprocess(parser_model, inp: str, outp: str,
                       batch_size: int = 32) -> tuple[bool, str]:
    """
    Lance le parsing in-process en utilisant l'API Python de hopsparser.
    parser_model : instance de BiAffineParser déjà chargée et sur le bon device.
    """
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
# EXTRACTION DE FEATURES (depuis CoNLL-U parsé)
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


def deprel_cnt(sents):
    return Counter(w["deprel"] for s in sents for w in s["words"])


def upos_cnt(sents):
    return Counter(w["upos"] for s in sents for w in s["words"])


def readable_parse(sents, max_s=4) -> str:
    lines = []
    for i, s in enumerate(sents[:max_s]):
        wm = {w["id"]: w["form"] for w in s["words"]}
        arcs = [f"{w['form']}/{w['upos']} →"
                f"{wm.get(w['head'],'ROOT') if w['head']!=0 else 'ROOT'}"
                f"({w['deprel']})"
                for w in s["words"]]
        lines.append(f"     Phrase {i+1}: " + "  ".join(arcs))
    if len(sents) > max_s:
        lines.append(f"     … +{len(sents)-max_s} phrase(s) omise(s)")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# JENSEN-SHANNON
# ════════════════════════════════════════════════════════════════════════
def js_div(c1: Counter, c2: Counter) -> float:
    keys = sorted(set(c1) | set(c2))
    if not keys:
        return 0.0
    v1 = np.array([c1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([c2.get(k, 0) for k in keys], dtype=float)
    if v1.sum() == 0 or v2.sum() == 0:
        return 1.0
    return float(jensenshannon(v1 / v1.sum(), v2 / v2.sum()))


# ════════════════════════════════════════════════════════════════════════
# COMPARAISON STANZA  vs  HOPSPARSER
# ════════════════════════════════════════════════════════════════════════
def compare_with_stanza():
    section("10) COMPARAISON STANZA  vs  HOPSPARSER")

    stanza_csv = os.path.join(PREV_RESULTS, "resultats_par_sms.csv")
    hops_csv   = os.path.join(OUTPUT_DIR,   "resultats_par_sms.csv")

    if not os.path.exists(stanza_csv):
        print(f"  ⚠ {stanza_csv} introuvable → comparaison impossible.")
        return

    dfs = pd.read_csv(stanza_csv)
    dfh = pd.read_csv(hops_csv)
    n   = min(len(dfs), len(dfh))
    dfs, dfh = dfs.iloc[:n].copy(), dfh.iloc[:n].copy()

    print(f"\n  {n} SMS comparés")
    print(f"  ⚠ Rappel : Stanza utilise son propre tokeniseur neuronal,")
    print(f"    tandis que HopsParser reçoit ici une tokenisation regex.")
    print(f"    Les écarts sur nb_tokens reflètent aussi cette différence.\n")

    # ── A) Moyennes côte à côte ──
    comparisons_labels = [
        ("SMS ↔ T1", "SMS", "Transcodage_1"),
        ("SMS ↔ T2", "SMS", "Transcodage_2"),
        ("T1  ↔ T2", "Transcodage_1", "Transcodage_2"),
    ]

    print(f"  {'Feature':<34} {'Corpus':<22} {'Stanza':>9} {'Hops':>9} {'Δ(H−S)':>9}")
    print("  " + "─" * 88)

    comp_rows = []
    for col in COLS:
        for k in FKEYS:
            cn = f"{col}_{k}"
            if cn in dfs.columns and cn in dfh.columns:
                sv = dfs[cn].dropna()
                hv = dfh[cn].dropna()
                if len(sv) and len(hv):
                    sm, hm = sv.mean(), hv.mean()
                    d = hm - sm
                    print(f"  {k:<34} {NICE[col]:<22} {sm:>9.3f} {hm:>9.3f} {d:>+9.3f}")
                    comp_rows.append({"corpus": col, "feature": k,
                                      "stanza_mean": sm, "hops_mean": hm, "delta": d})
        print()

    # ── B) Corrélations Pearson ──
    print(f"\n  Corrélations Pearson (Stanza ↔ HopsParser) :")
    print(f"  {'Feature':<34} {'Corpus':<22} {'r':>8} {'p':>12}")
    print("  " + "─" * 80)
    for col in COLS:
        for k in FKEYS:
            cn = f"{col}_{k}"
            if cn in dfs.columns and cn in dfh.columns:
                mask = dfs[cn].notna() & dfh[cn].notna()
                if mask.sum() > 10:
                    r, p = sp_stats.pearsonr(dfs.loc[mask, cn], dfh.loc[mask, cn])
                    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                    print(f"  {k:<34} {NICE[col]:<22} {r:>8.4f} {p:>12.2e} {star}")

    # ── C) Robustesse SMS → T1 ──
    print(f"\n\n  Robustesse : écart moyen |feat(SMS) − feat(T1)| par parser")
    print(f"  (Plus petit = parser plus stable face au langage SMS)\n")
    print(f"  {'Feature':<34} {'Stanza':>12} {'Hops':>12} {'Plus stable':>14}")
    print("  " + "─" * 76)

    rob_s, rob_h = 0, 0
    for k in FKEYS:
        cs, ct = f"SMS_{k}", f"Transcodage_1_{k}"
        if all(c in dfs.columns for c in [cs, ct]) and \
           all(c in dfh.columns for c in [cs, ct]):
            mask = dfs[cs].notna() & dfs[ct].notna() & dfh[cs].notna() & dfh[ct].notna()
            if mask.sum() > 10:
                gs = (dfs.loc[mask, cs] - dfs.loc[mask, ct]).abs().mean()
                gh = (dfh.loc[mask, cs] - dfh.loc[mask, ct]).abs().mean()
                better = "Stanza" if gs < gh else "HopsParser" if gh < gs else "="
                if gs < gh:   rob_s += 1
                elif gh < gs: rob_h += 1
                print(f"  {k:<34} {gs:>12.3f} {gh:>12.3f} {better:>14}")

    print(f"\n  Score robustesse : Stanza={rob_s}  HopsParser={rob_h} "
          f"(sur {len(FKEYS)} features)")

    # ── D) Top 10 désaccords inter-parsers (SMS brut) ──
    print(f"\n\n  Top 10 SMS où les deux parsers divergent le plus (corpus SMS brut) :\n")
    scores = []
    for i in range(n):
        sc = 0
        for k in FKEYS:
            cn = f"SMS_{k}"
            if cn in dfs.columns and cn in dfh.columns:
                sv, hv = dfs.iloc[i][cn], dfh.iloc[i][cn]
                if pd.notna(sv) and pd.notna(hv):
                    sc += abs(sv - hv)
        scores.append((i, sc))
    scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (idx, sc) in enumerate(scores[:10], start=1):
        txt_col = "SMS_texte"
        txt = dfh.iloc[idx].get(txt_col, "N/A") if txt_col in dfh.columns else "N/A"
        print(f"  {rank:>2}. SMS n°{idx+1}  (Σ|Δ| = {sc:.2f})")
        print(f"      {str(txt)[:120]}")
        for k in ["profondeur_arbre_max", "distance_dependance_max",
                   "distance_dependance_moy", "nb_tokens"]:
            cn = f"SMS_{k}"
            sv = dfs.iloc[idx].get(cn, np.nan)
            hv = dfh.iloc[idx].get(cn, np.nan)
            print(f"      {k}: Stanza={sv}  Hops={hv}")
        print()

    # ── E) Export CSV comparaison ──
    comp_csv = os.path.join(OUTPUT_DIR, "comparaison_stanza_vs_hops.csv")
    pd.DataFrame(comp_rows).to_csv(comp_csv, index=False, encoding="utf-8")
    print(f"  → {comp_csv}")

    # ── F) Verdict ──
    print(f"\n  {'─'*60}")
    print(f"  VERDICT PRÉLIMINAIRE")
    print(f"  {'─'*60}")
    print(f"  • Robustesse SMS→T1 : "
          f"{'Stanza' if rob_s > rob_h else 'HopsParser' if rob_h > rob_s else 'Égalité'} "
          f"devant ({rob_s} vs {rob_h})")
    print(f"  • Examiner les exemples de désaccord ci-dessus pour")
    print(f"    juger la plausibilité linguistique de chaque parser.")
    print(f"  • ⚠ Sans gold standard, aucun verdict définitif.")
    print(f"  {'─'*60}")


# ════════════════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ════════════════════════════════════════════════════════════════════════
def main():
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
                if all(c in cand.columns for c in COLS):
                    df = cand
                    print(f"  Lu avec enc={enc}, sep={repr(sep)}")
                    break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        sys.exit(f"  ✗ Impossible de lire {CSV_PATH}")

    N = len(df)
    print(f"  {N} SMS chargés.")
    texts = {col: df[col].apply(clean_text).tolist() for col in COLS}
    for col in COLS:
        nv = sum(1 for t in texts[col] if t)
        print(f"  {NICE[col]:<30} : {nv} textes non vides")

    # ──────────────────────────────────────────────────────────────
    # 2) VÉRIFICATION ENVIRONNEMENT + CHARGEMENT MODÈLE
    # ──────────────────────────────────────────────────────────────
    section("2) CHARGEMENT DU MODÈLE")

    if not os.path.isdir(MODEL_PATH):
        sys.exit(f"  ✗ Dossier modèle introuvable : {MODEL_PATH}")
    print(f"  ✓ Modèle           : {MODEL_PATH}")
    print(f"  Device             : {device}")

    # Chargement in-process (une seule fois !)
    from hopsparser.parser import BiAffineParser
    print(f"  Chargement du modèle en mémoire…")
    t_load = time.time()
    parser_model = BiAffineParser.load(MODEL_PATH)
    parser_model = parser_model.to(device)
    parser_model.eval()
    print(f"  ✓ Modèle chargé en {time.time() - t_load:.1f} s")

    if torch.cuda.is_available():
        print(f"      VRAM après chargement : "
              f"{torch.cuda.memory_allocated(0)/1024**2:.0f} Mo")

    if os.path.exists(PREV_RESULTS):
        print(f"  ✓ Résultats Stanza : {PREV_RESULTS}")
    else:
        print(f"  ⚠ Résultats Stanza non trouvés ({PREV_RESULTS})")
        print(f"    → la comparaison finale sera omise.")

    # ──────────────────────────────────────────────────────────────
    # 3) ANALYSE SYNTAXIQUE  (HopsParser, in-process)
    # ──────────────────────────────────────────────────────────────
    section("3) ANALYSE SYNTAXIQUE (HopsParser, in-process)")

    feats      = {col: [None]*N for col in COLS}
    parsed_all = {col: [None]*N for col in COLS}
    deprel_gl  = {col: Counter() for col in COLS}
    upos_gl    = {col: Counter() for col in COLS}

    for col in COLS:
        tc = time.time()
        print(f"\n  ── {NICE[col]} ──")

        # Construire le CoNLL-U d'entrée
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

        # Parsing in-process (batch_size contrôlé)
        print(f"    Lancement parsing in-process (device={device}, batch_size={BATCH_SIZE}) …")
        ok, stderr = run_hops_inprocess(parser_model, inp, outp, batch_size=BATCH_SIZE)

        if ok and os.path.exists(outp):
            parsed = parse_conllu(outp)
            print(f"    {len(parsed)} phrases en sortie (attendu {n_sents})")

            grp = group_by_text(parsed, smap, N)
            for tidx, sents in grp.items():
                parsed_all[col][tidx] = sents
                feats[col][tidx]      = extract_features(sents)
                deprel_gl[col]       += deprel_cnt(sents)
                upos_gl[col]         += upos_cnt(sents)
        else:
            print(f"    ✗ Échec du parsing : {stderr[:500]}")
            print(f"    → Tentative phrase par phrase (batch_size=1) …")

            # Fallback: parse sentence par sentence
            from hopsparser.utils import smart_open
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
                # Re-align with smap: since we parse all trees in order,
                # successes correspond to the first n_ok entries in smap
                grp = group_by_text(parsed, smap[:len(parsed)], N)
                for tidx, sents in grp.items():
                    parsed_all[col][tidx] = sents
                    feats[col][tidx]      = extract_features(sents)
                    deprel_gl[col]       += deprel_cnt(sents)
                    upos_gl[col]         += upos_cnt(sents)

            if os.path.exists(fallback_out):
                # Copy fallback output as the main output
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
    # 4) STATISTIQUES DESCRIPTIVES
    # ──────────────────────────────────────────────────────────────
    section("4) STATISTIQUES DESCRIPTIVES  (moyenne ± σ)")

    arrays = {
        col: {k: [feats[col][i][k]
                   for i in range(N) if feats[col][i] is not None]
              for k in FKEYS}
        for col in COLS
    }

    hdr = f"  {'Feature':<34}" + "".join(f"{NICE[c]:>{COL_W}}" for c in COLS)
    print(f"\n{hdr}\n  " + "─" * (34 + COL_W * len(COLS)))

    stats_rows = []
    for k in FKEYS:
        rs = f"  {k:<34}"
        rd = {"feature": k}
        for col in COLS:
            a = arrays[col][k]
            if a:
                m, s = np.mean(a), np.std(a)
                rs += f"  {m:>9.3f} ± {s:<9.3f}"
                rd[f"{col}_mean"]   = m
                rd[f"{col}_std"]    = s
                rd[f"{col}_median"] = float(np.median(a))
            else:
                rs += f"  {'N/A':>{COL_W-2}}"
        print(rs)
        stats_rows.append(rd)

    # ──────────────────────────────────────────────────────────────
    # 5) TESTS DE WILCOXON
    # ──────────────────────────────────────────────────────────────
    section("5) TESTS DE WILCOXON (signed-rank, appariés)")

    comparisons = [
        ("SMS ↔ T1", "SMS", "Transcodage_1"),
        ("SMS ↔ T2", "SMS", "Transcodage_2"),
        ("T1  ↔ T2", "Transcodage_1", "Transcodage_2"),
    ]

    print(f"\n  {'Paire':<12} {'Feature':<34} {'Δ moy':>9} "
          f"{'W':>12} {'p':>12} {'sig':>5}")
    print("  " + "─" * 88)

    wilcoxon_rows = []
    for pn, ca, cb in comparisons:
        for k in FKEYS:
            va = [feats[ca][i][k] for i in range(N)
                  if feats[ca][i] and feats[cb][i]]
            vb = [feats[cb][i][k] for i in range(N)
                  if feats[ca][i] and feats[cb][i]]
            if len(va) < 20:
                continue
            aa, ab = np.array(va), np.array(vb)
            delta = float(np.mean(aa - ab))
            try:
                w, p = sp_stats.wilcoxon(aa, ab)
                sig = "***" if p < .001 else "**" if p < .01 \
                      else "*" if p < .05 else ""
            except Exception:
                w, p, sig = np.nan, np.nan, "?"
            print(f"  {pn:<12} {k:<34} {delta:>+9.3f} "
                  f"{w:>12.0f} {p:>12.2e} {sig:>5}")
            wilcoxon_rows.append({"paire": pn, "feature": k,
                                  "delta_moyen": delta, "W": w,
                                  "p_value": p, "significatif": sig})

    # ──────────────────────────────────────────────────────────────
    # 6) DISTRIBUTIONS DEPREL & UPOS
    # ──────────────────────────────────────────────────────────────
    for label, gcounts in [
        ("RELATIONS DE DÉPENDANCE (deprel)", deprel_gl),
        ("CATÉGORIES MORPHO-SYNTAXIQUES (UPOS)", upos_gl),
    ]:
        section(f"6) {label}")
        all_tags = sorted(
            set().union(*(gcounts[c].keys() for c in COLS)),
            key=lambda t: gcounts["SMS"].get(t, 0), reverse=True)
        totals = {c: max(sum(gcounts[c].values()), 1) for c in COLS}

        hdr = f"  {'Tag':<16}" + "".join(f"{NICE[c]:>{COL_W}}" for c in COLS)
        print(f"\n{hdr}\n  " + "─" * (16 + COL_W * len(COLS)))
        for tag in all_tags[:17]:
            rs = f"  {tag:<16}"
            for col in COLS:
                n_ = gcounts[col].get(tag, 0)
                rs += f"  {n_:>7} ({n_/totals[col]*100:>5.1f} %)"
            print(rs)

        print()
        for pn, ca, cb in comparisons:
            print(f"  JS({pn:<12}) = {js_div(gcounts[ca], gcounts[cb]):.5f}")

    # ──────────────────────────────────────────────────────────────
    # 7) TOP 20 DÉSACCORDS (entre les 3 versions d'un même SMS)
    # ──────────────────────────────────────────────────────────────
    section("7) TOP 20 DÉSACCORDS (SMS vs T1 vs T2)")

    divs = []
    for i in range(N):
        if all(feats[c][i] for c in COLS):
            sc = sum(max(feats[c][i][k] for c in COLS)
                     - min(feats[c][i][k] for c in COLS)
                     for k in DIVERGENCE_KEYS)
            divs.append((i, sc))
    divs.sort(key=lambda x: x[1], reverse=True)

    rpt = os.path.join(OUTPUT_DIR, "exemples_desaccords.txt")
    n_ex = min(20, len(divs))

    with open(rpt, "w", encoding="utf-8") as fout:
        fout.write("EXEMPLES DE DÉSACCORDS – HOPSPARSER\n")
        fout.write(f"Top {n_ex} (divergence = Σ[max−min] sur {DIVERGENCE_KEYS})\n")
        fout.write(BANNER + "\n\n")

        for rank, (idx, sc) in enumerate(divs[:n_ex], start=1):
            blk = [f"{'─'*66}",
                   f"  #{rank} — SMS n°{idx+1}  (div = {sc:.2f})",
                   f"{'─'*66}"]
            for col in COLS:
                f_ = feats[col][idx]
                blk.append(f"\n  [{NICE[col]}]")
                blk.append(f"  Texte : {texts[col][idx]}")
                blk.append(
                    f"  → prof={f_['profondeur_arbre_max']}  "
                    f"dist_max={f_['distance_dependance_max']}  "
                    f"dist_moy={f_['distance_dependance_moy']:.2f}  "
                    f"#phr={f_['nb_phrases']}  #tok={f_['nb_tokens']}")
                if parsed_all[col][idx]:
                    blk.append(readable_parse(parsed_all[col][idx]))
            blk.append("")
            txt = "\n".join(blk)
            fout.write(txt + "\n")
            if rank <= 5:
                print(txt)

    if n_ex > 5:
        print(f"\n  … {n_ex-5} exemples supplémentaires → {rpt}")

    # ──────────────────────────────────────────────────────────────
    # 8) FOCUS T1 ↔ T2
    # ──────────────────────────────────────────────────────────────
    section("8) FOCUS : TRANSCODAGE 1 ↔ TRANSCODAGE 2")

    vp = [(i, texts["Transcodage_1"][i], texts["Transcodage_2"][i])
          for i in range(N)
          if texts["Transcodage_1"][i] and texts["Transcodage_2"][i]]
    same = sum(1 for _, a, b in vp if a == b)
    diff = len(vp) - same

    print(f"\n  Paires valides     : {len(vp)}")
    print(f"  Identiques         : {same}  ({same/max(len(vp),1)*100:.1f} %)")
    print(f"  Différentes        : {diff}  ({diff/max(len(vp),1)*100:.1f} %)")

    t12d = []
    for i, a, b in vp:
        if a != b and feats["Transcodage_1"][i] and feats["Transcodage_2"][i]:
            d = sum(abs(feats["Transcodage_1"][i][k]
                        - feats["Transcodage_2"][i][k])
                    for k in DIVERGENCE_KEYS)
            t12d.append((i, d))
    t12d.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 10 divergences T1 ≠ T2 :\n")
    for rk, (idx, d) in enumerate(t12d[:10], start=1):
        r1 = feats["Transcodage_1"][idx]
        r2 = feats["Transcodage_2"][idx]
        print(f"  {rk:>2}. SMS n°{idx+1} (Δ={d:.2f})")
        print(f"      T1: {texts['Transcodage_1'][idx]}")
        print(f"      T2: {texts['Transcodage_2'][idx]}")
        print(f"      T1 → prof={r1['profondeur_arbre_max']}  "
              f"dist_max={r1['distance_dependance_max']}  "
              f"dist_moy={r1['distance_dependance_moy']:.2f}  "
              f"#phr={r1['nb_phrases']}")
        print(f"      T2 → prof={r2['profondeur_arbre_max']}  "
              f"dist_max={r2['distance_dependance_max']}  "
              f"dist_moy={r2['distance_dependance_moy']:.2f}  "
              f"#phr={r2['nb_phrases']}")
        print()

    # Déterminisme
    dis = sum(1 for i, a, b in vp
              if a == b
              and feats["Transcodage_1"][i] and feats["Transcodage_2"][i]
              and any(feats["Transcodage_1"][i][k]
                      != feats["Transcodage_2"][i][k] for k in FKEYS))
    print(f"  Déterminisme (texte T1 = T2 ⇒ analyse identique) :")
    if dis == 0:
        print(f"  ✓ Aucun désaccord sur les {same} textes identiques.")
    else:
        print(f"  ⚠ {dis} désaccord(s) inattendu(s) !")

    # ──────────────────────────────────────────────────────────────
    # 9) EXPORT DES RÉSULTATS
    # ──────────────────────────────────────────────────────────────
    section("9) EXPORT DES FICHIERS")

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

    csv_detail = os.path.join(OUTPUT_DIR, "resultats_par_sms.csv")
    pd.DataFrame(rows).to_csv(csv_detail, index=False, encoding="utf-8")
    print(f"  → {csv_detail}")

    csv_stats = os.path.join(OUTPUT_DIR, "statistiques_agregees.csv")
    pd.DataFrame(stats_rows).to_csv(csv_stats, index=False, encoding="utf-8")
    print(f"  → {csv_stats}")

    csv_wilcoxon = os.path.join(OUTPUT_DIR, "tests_wilcoxon.csv")
    pd.DataFrame(wilcoxon_rows).to_csv(csv_wilcoxon, index=False, encoding="utf-8")
    print(f"  → {csv_wilcoxon}")

    dist_rows = []
    for label_type, gcounts in [("deprel", deprel_gl), ("upos", upos_gl)]:
        all_tags = sorted(set().union(*(gcounts[c].keys() for c in COLS)))
        totals = {c: max(sum(gcounts[c].values()), 1) for c in COLS}
        for tag in all_tags:
            row = {"type": label_type, "tag": tag}
            for col in COLS:
                row[f"{col}_count"] = gcounts[col].get(tag, 0)
                row[f"{col}_pct"]   = gcounts[col].get(tag, 0) / totals[col] * 100
            dist_rows.append(row)
    csv_dist = os.path.join(OUTPUT_DIR, "distributions_deprel_upos.csv")
    pd.DataFrame(dist_rows).to_csv(csv_dist, index=False, encoding="utf-8")
    print(f"  → {csv_dist}")

    print(f"  → {rpt}")

    # ──────────────────────────────────────────────────────────────
    # 10) COMPARAISON STANZA vs HOPSPARSER
    # ──────────────────────────────────────────────────────────────
    if os.path.exists(PREV_RESULTS):
        compare_with_stanza()
    else:
        print(f"\n  ⚠ Dossier {PREV_RESULTS} absent → comparaison omise.")

    # ──────────────────────────────────────────────────────────────
    # RÉSUMÉ FINAL
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    print(f"\n{BANNER}")
    print(f"  ✓ Analyse complète en {elapsed / 60:.1f} minutes.")
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