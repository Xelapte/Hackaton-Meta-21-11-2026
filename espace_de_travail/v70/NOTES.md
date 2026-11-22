# v70 — Unlearning : suite de la famille « tout plus hésitant » (v47 p_*)

Repris par Claude (Opus) à 9h45 le 22/11, après Fable. Aucun fichier de v47 modifié : les scripts `v47/unlearn.py` et
`v47/etape2.py` sont appelés tels quels, sorties dans v70/.

## Point de départ (leaderboard public)
- Record : `v47/p_7_7` = **0.3144** (δ classes partielles 7, δ des 9 classes lourdes 7, étape 2 marge 16,5, copies +2,5).
- `v47/e_50b` = 0.3016 : la cible « d'ensemble » (moyenne avec 5 professeurs dé-mémorisés, `--teacher_mix 0.5`) bat la cible seule
  (hb_comb 0.2937) d'environ +0,008.
- Déjà prêts dans v47 et mis en tête de la file d'envoi (`lb_file.txt`) à 9h48 : `p_8_8`, puis `p_7_6`.

## Runs v70 (GPU 1, après wm62_tm_L5, ~9 min chacun)
| Nom | δ partielles | δ lourdes | Cible d'ensemble | Marge étape 2 | Score public |
|---|---|---|---|---|---|
| ep_7_7 | 7 | 7 | oui | 16,5 | |
| ep_7_7b | 7 | 7 | oui | 17,2 (copies +3,2, réglage de e_50b) | |

Choix de l'ordre d'envoi selon le score de p_8_8 : s'il bat p_7_7, on pousse vers 9/9 ; sinon on reste à 7/7 avec la cible d'ensemble.

## Résultats des fichiers v47 envoyés
- p_8_8 : 0.3068 ; p_7_6 : 0.3112 (record p_7_7 0.3144 inchangé) → 7/7 est le sommet ; p_9_9 et ep_8_8 retirés de la file à 9h57.
- Ordre d'envoi à 10h03 : p_6_9, pc1, hb_tout6, hhg (puis ep_7_7, ep_7_7b dès qu'ils sont prêts, en tête).

## Résultats v70 (22/11)
| Fichier | Recette | Public |
|---|---|---|
| ep_7_7 | p_7_7 + cible d'ensemble, étape 2 marge 16,5 / +2,5 | 0,2956 |
| ep_7_7b | idem, étape 2 marge 17,2 / +3,2 | 0,3076 |
| **p77_m172** | **base v47/s1_7_7 (celle du record), étape 2 seule, marge 17,2 / +3,2** | **0,3236** (record, 11h42) |
| p77_m176 | idem, marge 17,6 / +3,6 (ensemble m/s 1,91/4,49) | 0,3174 (11h51) : moins bien, le sommet est vers 17,2 |
| p77_m180 | idem, marge 18,0 / +4,0 (ensemble m/s 1,95/4,60) | non soumis |

Observation : le réglage d'étape 2 « marge 17,2 / copies +3,2 » a gagné +0,011 (e_50 → e_50b), +0,012 (ep_7_7 → ep_7_7b) et
+0,009 (p_7_7 → p77_m172) sur trois bases différentes. La cible d'ensemble, elle, perd sur la base 7/7 (0,3144 → 0,2956).
