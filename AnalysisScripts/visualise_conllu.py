#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualise_conllu.py — Visualisation displacy pour fichiers CoNLL-U pre-calcules.

Exporte un rendu displacy en fichier HTML a partir d'un fichier CoNLL-U
(resultats HopsParser ou Stanza). Aucun modele n'est charge, evitant
ainsi les problemes de ports inaccessibles sur Google Colab.

Usage :
    python AnalysisScripts/visualise_conllu.py results/SMS/fsmb/output_SMS.conllu
    python AnalysisScripts/visualise_conllu.py results/SMS/gsd/output_SMS.conllu --sent-range 5-20 --output mon_rendu.html
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

def to_displacy(sentence: dict, highlight_chain: list[int] | None = None) -> dict:
    """Convert one CoNLL-U sentence to displacy manual-render format.
    
    Si highlight_chain est specifie (liste d'IDs de tokens), 
    seuls les arcs reliant ces tokens de maniere contigue seront conserves.
    """
    words = []
    for t in sentence["tokens"]:
        tag = ""
        if t["upos"] != "_":
            if highlight_chain is None or t["id"] in highlight_chain:
                tag = t["upos"]
        words.append({"text": t["form"], "tag": tag})
        
    # Filter valid edges if we only want to show a specific chain
    chain_edges = set()
    if highlight_chain:
        for i in range(len(highlight_chain) - 1):
            gov = highlight_chain[i]
            dep = highlight_chain[i+1]
            chain_edges.add((gov, dep))
            chain_edges.add((dep, gov)) # to match unordered
            
    arcs = []
    for t in sentence["tokens"]:
        if t["head"] == "_" or t["deprel"] == "_":
            continue
        head_id = int(t["head"])
        if head_id == 0:
            continue
            
        di = t["id"]
        hi = head_id
        
        # If highlighting a chain, skip arcs not in the chain
        if highlight_chain and (di, hi) not in chain_edges:
            continue
            
        # Displacy uses 0-based indexing
        di_idx = di - 1
        hi_idx = hi - 1
        
        if di_idx < hi_idx:
            arcs.append({"start": di_idx, "end": hi_idx, "label": t["deprel"], "dir": "right"})
        else:
            arcs.append({"start": hi_idx, "end": di_idx, "label": t["deprel"], "dir": "left"})
            
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


# ── chains identification ─────────────────────────────────────────────

def get_chains_of_length(tokens: list[dict], target_length: int) -> list[list[int]]:
    """Retourne toutes les chaines (listes d'IDs) de la longueur specifiee."""
    children = {t["id"]: [] for t in tokens}
    for t in tokens:
        if t["head"] != "_" and t["deprel"] != "_":
            head_id = int(t["head"])
            if head_id != 0 and head_id in children:
                children[head_id].append(t["id"])

    def get_paths(node_id: int) -> list[list[int]]:
        if not children.get(node_id):
            return [[node_id]]
        paths = []
        for child_id in children[node_id]:
            for p in get_paths(child_id):
                paths.append([node_id] + p)
        return paths

    all_chains = []
    # On cherche a partir de chaque noeud pour trouver des sous-chaines aussi
    for start_node in children.keys():
        for p in get_paths(start_node):
            # p est une liste de noeuds. target_length est le nombre de liens (arcs)
            if len(p) - 1 >= target_length:
                # On recupere exactement la sous-chaine de bonne longueur
                # au cas ou le chemin est plus long
                for i in range(len(p) - target_length):
                    all_chains.append(p[i : i + target_length + 1])
                    
    return all_chains

def extract_special_chains(sentences: list[dict], lengths: list[int] = [7, 5, 4], max_per_length: int = 2) -> list[tuple[dict, list[int], int]]:
    """Extrait des exemples de phrases pour les longueurs specifiees."""
    results = []
    found_counts = {l: 0 for l in lengths}
    
    # On traque les textes deja ajoutes pour eviter les doublons de phrases
    added_texts = set()
    
    for s in sentences:
        if all(c >= max_per_length for c in found_counts.values()):
            break
            
        for length in lengths:
            if found_counts[length] >= max_per_length:
                continue
                
            chains = get_chains_of_length(s["tokens"], length)
            if chains and s["text"] not in added_texts:
                results.append((s, chains[0], length))
                found_counts[length] += 1
                added_texts.add(s["text"])
                break  # On passe a la phrase suivante
                
    return results



# ── main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Génère une visualisation HTML pour un fichier CoNLL-U.",
    )
    p.add_argument("conllu", help="Chemin vers le fichier CoNLL-U.")
    p.add_argument(
        "--sent-range", default=None,
        help="Plage de phrases a afficher, ex. '1-20' (1-indexed, inclusif)."
             " Par defaut : toutes les phrases.",
    )
    p.add_argument(
        "--output", default="visualisation_arbres.html",
        help="Chemin du fichier HTML exporté (defaut: visualisation_arbres.html).",
    )
    p.add_argument(
        "--find-chains", action="store_true",
        help="Recherche et isole deux chaines de dependances de "
             "4, 5 et 7 niveaux au lieu de tout afficher.",
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

    manual_data = []
    html_blocks = []
    
    if args.find_chains:
        print("\nRecherche de chaines specifiques (7, 5 et 4 niveaux)...")
        examples = extract_special_chains(to_serve, lengths=[7, 5, 4], max_per_length=2)
        
        for sent, chain, length in examples:
            print(f" - Trouve chaine de {length} niveaux : {sent['text'][:60]}...")
            
            # Preparation de la presentation HTML
            chain_words = [t["form"] for t in sent["tokens"] if t["id"] in chain]
            chain_rels = []
            for i in range(len(chain) - 1):
                dep = [t for t in sent["tokens"] if t["id"] == chain[i+1]][0]
                chain_rels.append(dep["deprel"])
                
            desc_html = (
                f"<div style='margin-bottom: 2rem; padding: 1rem; border-left: 5px solid #ff4b4b; background-color: #ffeaea;'>"
                f"<h2 style='margin-top: 0; color: #cc0000;'>Chaîne de {length} niveaux</h2>"
                f"<p><strong>Motifs :</strong> {' → '.join(chain_rels)}</p>"
                f"<p><strong>Mots concernés :</strong> {' → '.join(chain_words)}</p>"
                f"</div>"
            )
            
            displacy_data = to_displacy(sent, highlight_chain=chain)
            svg = displacy.render(
                [displacy_data],
                style="dep",
                page=False,
                manual=True,
                jupyter=False,
                options={"distance": 75, "compact": False, "bg": "#f9f9f9", "color": "#cc0000", "word_spacing": 25},
            )
            html_blocks.append(desc_html + svg + "<hr style='margin: 3rem 0; border: 1px solid #ddd;'/>")
            
        if not html_blocks:
            print("Aucune chaine trouvee dans le corpus/la plage donnee.")
            sys.exit(0)
            
        css_inject = "<style>.displacy-word, .displacy-tag { fill: #000000 !important; }</style>"
        final_html = f"<html><head><meta charset='utf-8'><title>Chaines syntaxiques</title>{css_inject}</head><body style='font-family: sans-serif; margin: 2rem;'>{''.join(html_blocks)}</body></html>"

            
    else:
        manual_data = [to_displacy(s) for s in to_serve]
        print(f"\nGénèration du rendu HTML des arbres...")
        final_html = displacy.render(
            manual_data,
            style="dep",
            page=True,
            manual=True,
            jupyter=False,
            options={"distance": 100, "compact": True, "bg": "#f9f9f9", "color": "#000"},
        )
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"[SUCCESS] Visualisation sauvegardée dans : {args.output}")
    print("Telechargez et ouvrez ce fichier dans votre navigateur pour voir les arbres.")


if __name__ == "__main__":
    main()
