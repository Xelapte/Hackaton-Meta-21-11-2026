# Tâche 2 (Model Equality) : index des fichiers (état au 21/11, 21h30)

Énoncé : `docs/references/enonce_tache2_model_equality.pdf` ; article MET : `docs/references/MET_Gao_2410.20247v2.pdf`.
Budget : 100 réponses par endpoint ep- pour toute la compétition. **Un seul membre envoie aux ep-, avec `collecte.py`.**

## Collecte
| Fichier | Auteur | Rôle |
|---|---|---|
| `collecte.py` | huang25 | collecte avec garde-fous : plafond par endpoint, vérification du budget serveur, `--confirmer` obligatoire pour les ep-, reprise |
| `sonde_ref.py` | huang25 | sonde des endpoints ref- seulement (verrou : refuse les ep-) |
| **`prompts_ep_phaseA.csv`** | huang25 | **jeu à envoyer aux ep- (phase A)** : 10 prompts × 3 tirages = 30 réponses/endpoint, 256 tokens. Choisi sur les mesures ref- : code 0,70, arithmétique 0,60 (seul type qui voit la LoRA 0,3), explication 0,53, continuation allemande ~0,6, identité (familles) |
| `prompts_phaseA_v2.csv` | huang25 | v2 mesurée en répétition (instructions et continuation française retirées : 0,15 et 0,05) |
| `prompts_phaseA.csv` | huang25 | v1, remplacé par la v2 |
| `batterie.csv`, `batterie2.csv` | huang25 | batteries de sonde (prompts courts ; continuations et copies) |
| `batterie_longue.csv` | bouaboud1 | 30 prompts longs et neutres |

## Analyse
| Fichier | Auteur | Rôle |
|---|---|---|
| `score_local.py` | bouaboud1 | vraisemblance en boîte blanche : modèle de base local contre fp8 simulé (repère fp8) |
| `logprobs_local.py`, `analyse_logprobs.py` | bouaboud1 | statistiques par token sous le modèle local |
| `analyse_prompts.py` | huang25 | « un contre le centre » : quel prompt révèle quelle variante |
| `eval_sonde.py` | huang25 | AUC par type de prompt, validation par moitiés |
| `analyse.py` | kit | baseline TF-IDF + regroupement à deux seuils → CSV de soumission |
| `mmd_test.py` | kit | test MMD de MET |

Réponses (hors git) : `$OFFICIEL/model_equality/output/huang25/` (`sonde_ref.jsonl`, `repetition_ref*.jsonl`) et
`$OFFICIEL/model_equality/reference_answers_equipe.jsonl` (bouaboud1).
