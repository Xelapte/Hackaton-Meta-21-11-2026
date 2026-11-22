# v62 — Watermark : passes longues + ensemble + décision finale

## Idée
v61 (`seg_synth_new.py`, copié ici en `seg_synth.py`, seul changement : sorties dans v62/) n'entraînait que 4–5 min sur le synthétique,
et la courbe montait encore (PixelSeal 0,577 → 0,621 entre les pas 500 et 1000). Ici :
- **passes longues** : 30 min de synthétique, réserve de 3000 images, reprise des derniers poids v61 (r3 > r2 > r1), affinage CV 45 s / pli ;
- **ensemble** : les plis de CV sont les mêmes dans toutes les passes (rang du nom % 5), donc on peut moyenner les probabilités
  hors pli de plusieurs passes sans fuite (`decide62.py --runs methode=v61/r2,v62/L5`) ;
- **décision** (`decide62.py`) : seuil, ou « mIoU attendu maximal » par image (idée de v60/decide.py), choisi en validation imbriquée.

## Jobs (file de l'équipe, déposés à 9h25)
| GPU | Nom | Script | Durée |
|---|---|---|---|
| 0 | wm62_ps_L5 | run_long_pixelseal_s5.sh | ~38 min |
| 1 | wm62_tm_L5 | run_long_trustmark_s5.sh | ~38 min |
| 2 | wm62_ss_L5 | run_long_stegastamp_s5.sh | ~38 min |
| 3 | wm62_ss_L7 | run_long_stegastamp_s7.sh | ~38 min |

## Résultats
- decide62 sur v61 (r2/r2/r1) : seuil imbriqué 0,6432, emiou imbriqué 0,6453 (+0,002 : négligeable, pas soumis seul).
- Passes r3 (v61, +330 s) : PixelSeal CV 0,6431 → 0,6562 ; TrustMark 0,6426 → 0,6513 ; StegaStamp r2 0,6481 → **0,7019**.
- Ensemble r2+r3 + emiou : PixelSeal 0,6602, TrustMark 0,6557 ; StegaStamp r2 seule (seuil 0,3) 0,7009 → global imbriqué **0,6722**.
- Passes longues : SS L7 CV 0,7797 (r3 0,7262) ; PS L5 CV 0,6780 (r3 0,6562).

## Soumissions
| n° | Fichier | Local (imbriqué) | Public |
|---|---|---|---|
| 4258 | `subs/v62_m.pt` (meilleur ensemble par méthode : PS 4 passes, TM L11 + T17, SS 4 passes ; TTA ; emiou) | 0,7506 | **0,7621** (finale, 11h56) |
| 4204 | `subs/v62_k.pt` (PS = v62 L5 + v63 L9, P13, Q25 ; TM = v63 L11 + T17 ; SS = v63 D13, D15, E23 ; TTA miroir moyennée ; emiou) | 0,7501 | 0,7610 |
| 4155 | `subs/v62_g.pt` (PS L5+L9+P13 ; TM L5 + L11 TTA ; SS D13 + D15) | 0,7425 | 0,7549 |
| 4080 | `subs/v62_f.pt` (PS = v62 L5 + v63 L9, TM, SS = v62 L5) | 0,7236 | **0,7400** |
| 4027 | `subs/v62_d.pt` (PS, TM, SS = passes longues v62 L5, emiou) | 0,7197 | **0,7363** |
| 3974 | `subs/v62_c.pt` (PS v62 L5 emiou, TM v61 r2+r3 emiou, SS v62 L7 seuil 0,45) | 0,7051 | **0,7252** |
| 3931 | `subs/v62_b.pt` (idem, mais SS = v61 r3 seule, seuil 0,3 : CV 0,7262) | 0,6807 | **0,6936** |
| 3894 | `subs/v62_a.pt` (PS r2+r3 emiou, TM r2+r3 emiou, SS r2 seuil 0,3) | 0,6722 | **0,6814** |

## Rapport pour l'équipe
- **Réutilisé** : `v61/seg_synth_new.py` et `v61/assemble.py` (copies), poids v61 comme initialisation ; idée « emiou » de `v60/decide.py`.
- **Créé** : `decide62.py`, `run_long_*.sh`.
- **Changé** : rien hors de v62/ (sauf les fichiers `v61/GO_R3_*` qui activent les passes r3 déjà en file).

## Fin de matinée (v63 : passes profondes, TTA miroir, 22/11 10h30–11h40)
- `v63/seg_synth.py` = v62 + prédiction TTA miroir enregistrée à part (`train_oof_flip`, `test_cv_flip`) ; `decide62.py` : suffixe `+f`.
- CV hors pli (seuil 0,5) : PS L9 0,6893, P13 0,6783, Q25 0,6914 ; TM L11 0,6915, T17 0,6932 ; SS D13 0,8280, D15 0,8191, E21 0,8146, E23 0,8201.
- TTA miroir (moyenne) : PS +0,001 à +0,007, TM +0,005 à +0,006, SS +0,003 à +0,006.
- Test A/B de l'affinage long (135 s au lieu de 45 s par pli, SS) : pli 0 à 0,778 contre 0,787 → arrêté, pas de gain.
- Ensembles retenus (imbriqué) : PS 0,7036 (4 passes), TM 0,7073 (L11 + T17 ; ajouter L5 baisse à 0,7052), SS 0,8394–0,8410.
- `v62_l.pt` (ensembles complets) : 0,7499 en local, pas mieux que `v62_k.pt` (0,7501) → non soumis.
