# Tâche 1 : index des scripts (état au 21/11, 21h30)

Méthode retenue et protocole : `docs/RECHERCHE_WATERMARK.md`. Meilleure soumission : n° 1412, H3b, public 0,5042.

## Retenu (chaîne de la soumission)
| Script | Rôle |
|---|---|
| `protocole.py` + `splits.json` | plis gelés (ne plus les modifier), références triviales, critère de gain robuste |
| `rectangles.py` | recherche de rectangle par décodage (grille grossière) → `output/rect/` |
| `recherche_fine.py` | recherche fine des petits rectangles (H8b), avec reprise → `output/rect_fin/` |
| `regle.py` | règle de décision H3b (+ `--fin`, `--repli carte|unet`) et prédiction du test (`--predire`) |
| `baseline.py` | modes `window`, `direct`, `cartes` (cartes par position → `cartes_feats.npz`), `unet` (`--plis_geles`) |

## Outils d'analyse
| Script | Rôle |
|---|---|
| `analyse_donnees.py` | statistiques des masques, exemples PNG |
| `analyse_erreurs.py` | analyse d'erreurs de H3b avec oracles |
| `eval_cartes.py` | AUC et mIoU d'une carte seule sur les plis gelés |
| `tests_decodeurs.py` | tests rapides des décodeurs : oracle (H3a), invariance (H4), occlusion (H6a), masque (H7a), alignement (H8a) |

## Essais rejetés (gardés pour la traçabilité, ne pas utiliser)
| Script | Pourquoi |
|---|---|
| `explorer_cartes.py`, `explorer_reembed.py` | explorations du début (remplacées par `baseline.py --mode cartes`) |
| `postproc.py` | plancher sur les plis non gelés (avant le protocole) |
| `synthetique.py` | synthétique v1 (supports du test, trous = original) : sans gain ; **importé par `synth_v2.py`, ne pas supprimer** |
| `synth_v2.py`, `detecteur.py` | détecteur appris v2 : écart de domaine (SS 0,33, PS 0,25 sur le vrai train) |

Sorties volumineuses (hors git) : `$OFFICIEL/Hackathon_Watermark_Localisation_Task/output/` — `rect/`, `rect_fin/`, `cartes_feats.npz`,
`soumis/` (fichiers soumis), `recherche/` (PNG) ; essais rejetés : `synth/` (7 Go), `synth_v2/`, `det_v2/`, `test_*`.
