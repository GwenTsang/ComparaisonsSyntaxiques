# Pipeline d'Analyse et de Comparaison Syntaxique

Nous proposons une analyse lexicométrique et syntaxique de corpus de trois différents genres très distincts : le genre SMS, le genre philosophique, et le genre journalistique.

Plus précisément :

* Un corpus de 13 dissertations de philosophie rédigées lors des concours du CAPES et de l’agrégation (provenant de [ce repository](https://github.com/GwenTsang/concours-de-philosophie)).
* Un corpus de 1000 SMS dans trois versions : version brute, version transcrite par des étudiants et version transcrite par des chercheurs (téléchargé depuis [88milsms.huma-num.fr](http://88milsms.huma-num.fr/corpus_en.html)).
* Un corpus d’articles du journal “Le Monde” déjà annoté en syntaxe de dépendances (téléchargeable depuis [UD_French-FTB](https://universaldependencies.org/treebanks/fr_ftb/) / [GitHub](https://github.com/UniversalDependencies/UD_French-FTB)).

### Préalables méthodologiques

Nous nous sommes aperçus que les parsers du site [Texttokids](https://texttokids.ortolang.fr) ne fonctionnaient pas extrêmement bien sur le langage SMS.

après avoir consulté le code source des parsers proposés sur ce site [à ce lien](https://gitlab.huma-num.fr/texttokids/ttkwp3-2025/-/tree/main/text_complexity/server/src/processor/syntaxe), on s'en est inspiré pour reproduire les deux mêmes tâches (complexité dans les dépendances syntaxiques et complexité dans les structures syntaxiques), mais avec d'autres modèles :
Voici le tableau avec la colonne **“Lien corpus”** ajoutée :

| Nom                      | Genre textuel du corpus d’entraînement                                                         | Lien modèle                                                                 | Lien corpus                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `camembertav2-gsd`       | Français standard écrit (web, Wikipédia)                                                       | [huggingface.co/almanach/camembertav2-base-gsd](huggingface.co/almanach/camembertav2-base-gsd) | [github.com/UniversalDependencies/UD_French-GSD](github.com/UniversalDependencies/UD_French-GSD) |
| `camembertav2-fsmb`      | Français des réseaux sociaux (informel, bruité)                                                | [huggingface.co/almanach/camembertav2-base-fsmb](huggingface.co/almanach/camembertav2-base-fsmb) | gitlab.inria.fr/seddah/french-social-media-bank                                               |
| `camembertav2-sequoia`   | Corpus multi-domaines : Wikipédia, Europarl, presse, médical (wiki, news, medical, nonfiction) | [huggingface.co/almanach/camembertav2-sequoia](huggingface.co/almanach/camembertav2-sequoia)     | [github.com/UniversalDependencies/UD_French-Sequoia](github.com/UniversalDependencies/UD_French-Sequoia) |
| `camembertav2-rhapsodie` | Français parlé annoté (oral linguistique structuré)                                            | [huggingface.co/almanach/camembertav2-base-rhapsodie](huggingface.co/almanach/camembertav2-base-rhapsodie) | [github.com/UniversalDependencies/UD_French-Rhapsodie](github.com/UniversalDependencies/UD_French-Rhapsodie) |
| `spoken-french`          | Français oral (agrégat UD : ParisStories + Rhapsodie ; spontané + annoté)                      | [hopsparser.readthedocs.io/en/latest/models.html](hopsparser.readthedocs.io/en/latest/models.html) | (agrégat de corpus UD, dont Rhapsodie et ParisStories) |

Pour pallier cela, nous avons utilisé certains parsers ayant été entraînés sur des corpus assez proches, par exemple [camembertav2-base-fsmb](https://huggingface.co/almanach/camembertav2-base-fsmb) qui a été entraîné sur le corpus FSMB (French Social Media Bank, cf. [Seddah (2012)](https://inria.hal.science/hal-00780895)).

Le projet utilise deux pipelines principales :

1. **Un orchestrateur de parsing multi-modèles** (`ParsingScripts/orchestrator.py`) qui transforme les textes bruts en représentations sous forme d'arbres de dépendances.

2. **Un orchestrateur d'analyse comparative 3-way** (`AnalysisScripts/run_all_3way.py`) qui permet de mesurer et quantifier la complexité syntaxique entre différents genres textuels.

---

## Architecture du projet

```text
HopsColab/
├── Corpus/                # Corpus bruts et préparés (csv, txt)
│   ├── Copies/            # Copies de concours de philosophie
│   ├── philosophie.csv    # Corpus de philosophie converti pour le parsing
│   └── ...
├── UD_FTB/                # French TreeBank (corpus de référence Le Monde)
├── models/                # Modèles pré-entraînés téléchargés localement
├── results/               # Résultats bruts, annotations CoNLL-U
│   └── ...
├── ParsingScripts/        # Pipeline 1 : Parsing et annotation
│   ├── orchestrator.py    # Point d'entrée principal pour le parsing
│   └── ...
└── AnalysisScripts/       # Pipeline 2 : Analyse stat et comparaisons
    ├── run_all_3way.py    # Point d'entrée pour les comparaisons 3-way
    └── ...
```


## Pipeline 1 : Parsing et Extraction (Orchestrateur Multi-Modèles)

Située dans `ParsingScripts`, cette pipeline s'occupe de la transformation du texte en objets syntaxiques quantifiables. 
Elle s'exécute via la commande `python ParsingScripts/orchestrator.py`.

Elle se déroule en 5 étapes automatisées :

1. **Préparation des corpus (`prepare_corpus.py`)** : 
   Convertit les fichiers texte bruts (ex: les copies de philosophie dans `Corpus/Copies/`) en format CSV utilisable par les modèles (`Corpus/philosophie.csv`). Le corpus SMS est déjà formaté.
   
2. **Téléchargement des Modèles (`download_model.py`)** : 
   Télécharge les 5 modèles de parsing HopsParser basés sur des architectures Transformers de pointe (GSD, FSMB, Sequoia, Rhapsodie, Zenodo-Spoken) ainsi que le modèle Stanza.
   
3. **Parsing Multi-Modèles (`Camembert.py` / `Stanza.py`)** : 
   Chaque modèle analyse chaque corpus. Cette étape produit des arbres de dépendance au format standard **CoNLL-U** (ex: `output_0.conllu`) et un récapitulatif brut `resultats_par_sms.csv` pour chaque combinaison (Modèle / Corpus).
   
4. **Extraction des Structures (`structures_syntaxiques.py`)** : 
   À partir des fichiers CoNLL-U générés, la pipeline extrait des traits grammaticaux spécifiques (Ratio Noms/Verbes, taux de subordination, distribution POS, etc.) pour chaque phrase.
   
5. **Accord Inter-Modèles (`accord_inter_modeles.py`)** : 
   Calcule la robustesse des annotations sur les différents corpus pour mesurer le niveau d'accord entre les différents parseurs (Alpha de Krippendorff).

---

## Pipeline 2 : Analyse Comparative 3-Way

Située dans `AnalysisScripts`, cette pipeline prend les résultats générés par le parsing et les confronte avec un troisième corpus de référence ("Le Monde" via UD_FTB).
Elle s'exécute via la commande `python AnalysisScripts/run_all_3way.py`.

Cette étape vise à différencier trois registres de langue :
- **SMS** : Oralité transcrite, expression courte et informelle.
- **Le Monde** : Journalisme, écrit formel, informatif.
- **Philosophie** : Vraies copies rédigées pendant les concours de philosophie.

L'orchestrateur lance en parallèle deux analyses :

1. **Analyse des Distances Syntaxiques (`compare_distances_3way.py`)** :
   - Évalue la **complexité structurelle** des phrases : profondeur moyenne et maximale des arbres de dépendance, distance moyenne et variance entre un mot et son noeud gouverneur.
   - Les résultats sont pondérés selon l'affinité des modèles vis-à-vis du corpus (ex: le modèle _FSMB_, spécialisé dans les réseaux sociaux, fait autorité sur les SMS ; le modèle _GSD_ sur la philosophie).

2. **Analyse des Structures Morphosyntaxiques (`compare_structures_3way.py`)** :
   - Compare la **stratégie discursive** de chaque genre :
     - Complexité phrastique (Subordination vs Coordination / Juxtaposition).
     - Densité de l'information (Ratio Noms/Verbes, densité de modificateurs adjectivaux).
     - Indices de l'oralité (Présence d'interjections, effacements syntaxiques atypiques).

Les rapports comparatifs (textuels et données CSV) sont exportés dans `results/comparaison_distances` et `results/comparaison_structures`.


## Comment lancer la pipeline ?

### 1. Pré-requis et Installation

Assurez-vous de disposer de Python 3.10+ et installez les dépendances :
```bash
pip install -r requirements.txt
pip install -e "."
```

### 2. Lancement complet de l'orchestrateur de parsing

Cette étape s'exécute de préférence sur un GPU car elle charge et exécute 6 modèles d'apprentissage profond (1 modèle RoBERTa, 4 modèles DeBERTa et Stanza) sur des centaines de phrases combinées.

```bash
# Lancer toute la pipeline d'annotation
python ParsingScripts/orchestrator.py

# (Optionnel) Ne lancer que sur un sous-ensemble de modèles pour aller plus vite
python ParsingScripts/orchestrator.py --models fsmb gsd stanza
```

### 3. Lancement de l'analyse comparative

Une fois l'étape de parsing terminée :

```bash
# Générer les rapports de comparaison 3-way
python AnalysisScripts/run_all_3way.py

# Générer de manière séquentielle (pour une lecture directe dans la console)
python AnalysisScripts/run_all_3way.py --sequential
```

Tous les résultats, tableaux comparatifs complets et rapports d'analyses se trouveront dans le répertoire `results/`.
