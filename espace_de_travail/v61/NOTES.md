# v61 — Watermark : segmentation apprise sur des données synthétiques

## Idée
On fabrique nous-mêmes des exemples avec la vérité terrain (à la « Watermark Anything ») :
1. fond = une image d'une **autre** méthode (donc sans le tatouage étudié) ;
2. une forme tirée des **vrais masques** du train (rectangle entier, rectangle troué, tache d'objet), augmentée (symétries, translation) ;
3. on y incruste le tatouage avec l'**embedder officiel** (message aléatoire) : le masque est gratuit ;
4. un réseau de segmentation (encodeur = l'extracteur officiel de la méthode, déjà sensible au tatouage, + petit décodeur FPN à deux échelles) apprend le masque 128 × 128.

Phase A : synthétique seul (270–330 s). Phase B : affinage sur les vraies images en CV 5 plis (pli = rang du nom % 5), prédictions hors pli = score honnête, test = moyenne des 5 modèles.
Aucune sélection de modèle sur les vraies images (les évaluations « [A] » du journal sont de la surveillance seulement).

## Résultats (mIoU officiel, vrai train, hors pli)

| Méthode | Passe | Synthétique seul | CV (hors pli) | Seuil |
|---|---|---|---|---|
| PixelSeal | r2 | 0,6333 | 0,6431 | 0,5 |
| TrustMark | r2 | 0,6290 | 0,6426 | 0,45 |
| StegaStamp | r1 | 0,5283 | 0,6481 | 0,3 |
| **Global** | | | **0,6446** | |

Références : H3b soumis (n° 1412) 0,5065 local / 0,5042 public ; meilleur v60 (U-Net empileur + emiou) 0,5457 local.

## Soumissions
| n° | Fichier | Local | Public |
|---|---|---|---|
| 3826 | `subs/v61_a.pt` (assemble.py, r2/r2/r1, variante cv, seuil par méthode) | 0,6446 | **0,6538** (meilleur de l'équipe, +0,15) |

## Rapport pour l'équipe
- **Réutilisé** : `wmloc` (embedders, extracteurs, `miou_image`) du kit officiel, en lecture seule ; les vrais masques du train pour les formes.
- **Créé** : `seg_synth.py` (r1), `seg_synth_new.py` (r2 : nouveau générateur de formes + modèle à deux échelles), `assemble.py` (choix variante/seuil, écriture de la soumission), `run_*.sh`.
- **Changé** : rien dans le dossier officiel ni dans `depot-coequipier`.
- **Mesuré** : tableau ci-dessus. La passe r3 (reprise des poids r2, +330 s, graine 1) a été activée à 9h27 (fichiers `GO_R3_*`) ; suite en v62.
