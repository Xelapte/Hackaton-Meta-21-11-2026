# DOSSIER COMPLET — épreuve 3 (unlearning), équipe Epi84 — généré le 22/11 07h55

Ordre de lecture : 1) CONTEXTE (la tâche et la métrique) 2) JOURNAL (toute la démarche, les découvertes, l'état actuel) 3) notes détaillées des versions récentes.


==================================================================
FICHIER : CONTEXTE.md
==================================================================
# CONTEXTE — Épreuve 3 (Unlearning), European AI Championship Paris

Ce fichier donne tout le contexte nécessaire pour m'aider sur l'épreuve 3. Les règles de travail
(ce qui est interdit, versions vN/, sbatch) sont dans CLAUDE.md et restent prioritaires.

## 1. Situation

- Hackathon de 24 h organisé par le CISPA et le Fraunhofer AISEC (SprintML Lab), sponsorisé par Meta.
  Début samedi 21 nov. 14:30, **fin dimanche 22 nov. 12:00**. Prix : 10 000 € pour la 1ʳᵉ place.
- Équipe de 3 : chacun une épreuve. **Moi (LOGIN / E.) : épreuve 3, l'unlearning.**
  Les 3 épreuves comptent pour le classement final.
- Cluster JURECA (Jülich) : un job = un nœud de 4× A100 40 Go. Nœud de login = pas de calcul lourd.
  `--account=training2666 --partition=dc-gpu --reservation=cispahack`.
- Dossier : `/p/scratch/training2666/huang25/Epi84/unlearning/` (partagé avec l'équipe).
- Leaderboard : 30 % des données (public). Classement final sur les 70 % restants (privé).
  **Ne pas surajuster au leaderboard.**
- Jugement : les 6 meilleures équipes présentent 6 min sur scène avec questions poussées des
  mentors. Je dois pouvoir expliquer chaque choix → tenir NOTES.md à jour.
- Aucune collaboration avec d'autres équipes (disqualification).

## 2. La tâche (consignes officielles)

- Modèle : ViT-B/16 timm `vit_base_patch16_224`, 100 classes, LayerScale init 1e-4,
  entraîné sur un dataset custom de 100 classes. Fichier `model.pt` (state dict).
- `forget_images.npz` : clé `images`, uint8 `[3500, 224, 224, 3]`, channels last, **sans labels**.
- **Aucune autre donnée** : pas de retain set, pas de validation, pas de labels.
- Prétraitement serveur : `x = (pixels/255 − 0.5) / 0.5`, `[B, 3, 224, 224]`. Utiliser `preprocess()`
  de `task_template.py`. Le serveur évalue en **mode eval sous autocast fp16**.
- Objectif : se comporter comme un modèle **« gold »** = même architecture et même recette, réentraîné
  **sans** les 3 500 images (le gold est la moyenne de plusieurs réentraînements), tout en restant
  proche de l'original sur le reste.
- Piège explicite : une montée de gradient naïve fait chuter l'accuracy sur le forget set sans vrai
  oubli. Il faut reproduire **la confiance ET la classe erronée choisie** par un modèle qui n'a
  jamais vu l'image, sur l'image exacte **et sur des versions légèrement modifiées**.

### Métrique (softmax à température 2 partout)
Modèles : `orig` (fourni), `gold` (réentraîné sans forget set), `sub` (notre soumission).

**Forget score** = mean_std_score × prediction_score. Chacun est moyenné à poids égal sur
(a) les images exactes du forget set et (b) des **copies altérées cachées** de ces images.
- `d = |mean(L_sub) − mean(L_gold)| + |std(L_sub) − std(L_gold)|`, L = cross-entropy avec le **vrai** label.
  `mean_std_score = 1 − clip(d_sub / d_orig, 0, 1)`.
- `prediction_score = 1 − clip(mean KL(p_gold ‖ p_sub) / mean KL(p_gold ‖ p_orig), 0, 1)`, image par image.
- 0 = pas d'oubli (sub = orig), 1 = identique au gold. Matcher seulement la loss moyenne ne suffit pas.

**Retain score** = `clip(acc_sub/acc_orig, 0, 1)` × `1/(1 + mean KL(p_orig ‖ p_sub))`, sur un jeu de
contrôle caché (même distribution). Toute dérive des probabilités est pénalisée, même à accuracy égale.

**Score final = forget × retain.** **Seuil éliminatoire : < 70 % d'accuracy sur une validation cachée → 0.**

### Soumission
- `torch.save(model.state_dict(), ...)`, chargeable en `strict=True` dans `build_model()`.
- **float32 uniquement**, pas de NaN/Inf, ≤ 450 Mo (~345 Mo pour un ViT-B).
- Adaptateurs (LoRA, PEFT…) à fusionner dans les poids d'origine avant de sauvegarder.
- C'est moi qui lance `submission.py` (token API). Toi, tu ne soumets jamais.

## 3. Le papier de référence : SCRUB (Kurmanji et al., NeurIPS 2023, arXiv 2302.09880)

- **Idée** : professeur/élève. Le professeur = modèle original (figé). L'élève est initialisé
  avec les mêmes poids.
- **Objectif** : l'élève **reste proche** du professeur sur le retain set (KL + cross-entropy) et
  **s'en éloigne** sur le forget set (on maximise le KL) :
  `min  α·mean_r KL(prof ‖ élève) + γ·mean_r CE(élève, y) − mean_f KL(prof ‖ élève)`.
- **Optimisation** : alternance d'époques « max » (forget) et « min » (retain), quelques époques
  « min » supplémentaires à la fin pour restaurer le retain. LR initial 5e-4 avec décroissance
  ×0,1 (indispensable contre les oscillations). Distillation à température 4 dans leurs expériences.
- **SCRUB+R (rewinding)** : SCRUB pousse l'erreur sur le forget set trop haut, ce qui rend ces
  exemples identifiables par une attaque d'inférence d'appartenance. Solution : sauvegarder un
  checkpoint par époque, mesurer l'erreur sur une validation de même distribution que le forget set
  (référence « jamais vu »), puis revenir au checkpoint dont l'erreur forget est la plus proche.
- **Ce qui change pour nous** :
  1. On n'a **pas de retain set** → la phase « min » doit se faire sur des données proxy.
  2. S'éloigner du professeur dépasse la cible : notre métrique pénalise tout écart au gold, dans
     les deux sens. L'esprit de SCRUB+R (viser le niveau d'erreur d'un modèle qui n'a pas vu les
     données) correspond exactement à notre métrique.
  3. On n'a **pas de validation de même distribution** pour calibrer ce niveau → il faut l'estimer
     autrement (augmentations, température) et le calibrer avec peu de soumissions.
- Autres références de l'épreuve : SalUn (ne modifier que les poids « saillants » pour le forget set,
  ICLR 2024), Zero-Shot Machine Unlearning (Chundawat et al., 2023), Negative Preference
  Optimization (alternative plus stable à la montée de gradient, COLM 2024).

## 4. Ce qui est déjà fait : v1/

`v1/unlearn.py` (lancé depuis v1/, il importe task_template et lit les données depuis le dossier parent) :
1. Pseudo-labels : argmax du modèle original (il a mémorisé le forget set, donc ≈ vrai label).
2. Cible « non mémorisée » par image : moyenne des softmax de l'original sur K=16 vues augmentées
   (crop/zoom, flip, luminosité/contraste, bruit), mélangée avec la prédiction sur l'image exacte
   (poids `--alpha`, 0 par défaut), puis aplatie par `--tau`.
3. L'élève (copie de l'original) est entraîné à minimiser :
   KL(cible ‖ élève) sur l'image exacte + `w_aug` × KL(original/τ ‖ élève) sur une vue augmentée
   + `w_ret` × KL à T=2 vers l'original sur des **mosaïques 2×2** d'images (proxy de retain)
   + L2-SP optionnel vers les poids d'origine.
4. AdamW, OneCycle, lr 1e-5, 3 époques, bs 48, bf16. Métriques locales par époque, sauvegarde float32.
- Mode `--diag` : stats de l'original sur images exactes, vues augmentées (0,5 / 1,0 / 1,5) et mosaïques.
- `v1/sweep.sh` : 4 runs en parallèle (1 par GPU), tau ∈ {1,0 ; 1,5 ; 2,0 ; 3,0}, à lancer depuis v1/.

**Limites connues de v1**
- Aucun contrôle direct de mean/std de la loss du gold : seul `tau` les règle.
- Labels approximatifs (pseudo-labels) → stats locales approximatives.
- Risque de double aplatissement sur les vues augmentées (l'original y est déjà moins confiant).
- Mosaïques = proxy imparfait du jeu de contrôle ; le seuil des 70 % n'est pas vérifiable en local
  (garde-fou : `agree_orig_mosaic` > ~0,9 et `kl_orig_mosaic_T2` bas).
- L'évaluation locale `forget_aug` utilise la même famille d'augmentations qu'à l'entraînement.
- La métrique locale `kl_target_clean` est légèrement faussée (la cible est redivisée par 2).

## 5. Prochaines étapes

1. Lancer `--diag` et interpréter : écart de loss/confiance entre images exactes et augmentées
   (= ampleur de la mémorisation), confiance sur les mosaïques.
2. Lancer le sweep sur `tau`, comparer les métriques locales, me proposer 1 à 2 candidats à soumettre.
3. Calibrer avec très peu de soumissions (1 à 2 paramètres seulement : `tau`, `alpha`).
4. Pistes v2 et suivantes (une version = un dossier) :
   - éviter le double aplatissement (cible des vues augmentées = moyenne augmentée, pas original/τ) ;
   - meilleur proxy de retain (mosaïques + mixup/cutmix, ou dataset public s'il en existe un sur le cluster) ;
   - ne modifier que les derniers blocs + head, ou L2-SP, pour limiter la dérive ;
   - checkpoint par époque et sélection à la SCRUB+R ;
   - SalUn (masque de saillance des poids).
5. Tenir `vN/NOTES.md` à jour : idée, paramètres, métriques locales, score leaderboard, conclusion.


==================================================================
FICHIER : JOURNAL.md
==================================================================
# Journal de l'épreuve 3 (unlearning) — démarche et évolution

Document de synthèse pour la présentation. Le détail de chaque version est dans `vN/NOTES.md`,
les scores bruts dans `lb_journal.txt`. Mis à jour le 22/11 à 5h30 (voir la section « Nuit du 21 au 22 » en bas :
elle CORRIGE plusieurs conclusions du tableau ci-dessous, notamment sur les « 9 classes retirées »).

## Le problème en une phrase
Modifier un ViT-B/16 pour qu'il réponde **comme le gold** (réentraîné sans 3 500 images) sur ces images et leurs copies
modifiées, et **comme l'original** partout ailleurs — sans données de retain, sans labels, sans le gold.

## Notre méthode en une phrase
Professeur/élève à la SCRUB, sans retain set : l'élève imite une **cible construite à partir de l'original** sur les
images à oublier, et imite l'original sur des **mosaïques** d'images (substitut du retain). Puis on **éteint** dans la
dernière couche les 9 classes que le gold n'a jamais vues.

## Les 5 découvertes, dans l'ordre

| # | Découverte | Preuve (leaderboard public) |
|---|---|---|
| 1 | Il faut oublier beaucoup plus qu'on ne pensait : le modèle qui reconnaît encore 95 % des images ≈ l'original. | v3 0.014 → v4 0.043–0.045 |
| 2 | **9 classes sont retirées en entier** (129–130 images chacune dans le forget set : 0, 6, 7, 8, 29, 30, 31, 33, 38). Le gold ne les prédit jamais → on les éteint (biais de la dernière couche −β). | v4 0.045 → v6 **0.135** (×3) |
| 3 | Deux boutons, chacun avec un sommet net : extinction β ≈ 6–8 ; ≈ 88–94 % de bonnes réponses sur les 36 autres classes. Dépasser coûte cher. | β 6/7/8/12 → 0.138/0.139/0.135/0.059 ; acc 94/88/76 % → 0.139/0.141/0.132 |
| 4 | **Moyenner la cible sur des vues modifiées** : sur l'image exacte, l'original a appris par cœur, ses 2es choix sont du bruit ; sur des vues modifiées ils redeviennent informatifs (et le gold est une moyenne de modèles). | 0.139 → 8 vues 0.146 → 16 vues **0.151** |
| 5 | Ce qui ne marche PAS (tout aussi utile) : réparer le retain après coup, ancrage plus fort, copies d'entraînement plus fortes, incliner vers les classes voisines par les poids, masque SalUn. | voir tableau ci-dessous |

## Évolution version par version

| Version | Idée | Meilleur score | Verdict |
|---|---|---|---|
| v1 | Cible = moyenne des prédictions de l'original sur 16 vues, aplatie par une température ; ancrage sur mosaïques | non soumis | base de code |
| v2 | Même cible pour l'image et ses vues (pas de double aplatissement), ancrage ×3 | non soumis | cible trop « uniforme » |
| v3 | Cible « à queue lourde » : 1–2 vues fortes, sans température | 0.0143 | oubli beaucoup trop faible |
| v4 | **Cible delta** : logits de l'original, logit de la vraie classe baissé de δ. Diagnostic par classe → découverte des 9 classes entières | 0.0453 | oublier plus paie, puis plafonne |
| v5 | δ différent pour classes entières (20) et partielles (8) | (base de v6) | les classes entières résistent à l'entraînement |
| v6 | **Extinction** : biais des 9 classes −β dans la dernière couche (aucun entraînement) | 0.1412 | ×3 ; β réglé (plateau 6–8) |
| v7 | « Réparer » le retain après extinction | 0.1247 | perd : rallume les classes |
| v8 | Simulateur du score calibré sur nos notes (faux gold réglable) | — | bon pour interpoler β, pas pour juger une nouvelle famille |
| v9 | Professeur « cohérent » (original déjà éteint) | 0.1242 | mauvaise base (extinction trop forte) |
| v10 | Inclinaison vers les classes ressemblantes (papier TRW, similarité des poids) | 0.1210 | perd |
| v11 | **Cible moyennée sur 8 vues** sur la recette du record | 0.1462 | gagne (+5 %) |
| v12 | Plus d'oubli sur les classes partielles (76 %) | 0.1323 | trop : le sommet est à ≈ 88–94 % |
| v13 | Oubli proportionnel à la part retirée / copies d'entraînement plus fortes | 0.1400 / 0.1307 | neutre / perd |
| v14 | Masque de saillance SalUn (20 % ou 50 % des poids libres) | non soumis | même dérive qu'avant en local |
| v15 | **Cible moyennée sur 16 vues** | 0.1506 | gagne (+3 %) |
| v16 | Exploration large (ancrage, durée, température, couches libres, mélanges de poids) | 0.1324 | ancrage fort + long = 0.096 ; aller 20 % plus loin = 0.120 |
| v17 | 32 et 64 vues ; copies d'entraînement plus légères | **0.1523** (64 vues) | record actuel ; le gain des vues sature (8→16→64 : +5 %, +3 %, +1 %) |
| v18 | Ancrage plus faible, entraînement plus court | à soumettre | |
| v19 | Cible « de référence par classe » (ReGUn/TRW) ; extinction seulement sur les images à oublier | à soumettre | |

## Recette du record (v15/f20_k16_d6_b7 = 0.1506)
1. Pseudo-labels = prédiction de l'original (il a mémorisé le forget set).
2. Classes entières = classes à ≥ 125 images dans le forget set (9 classes, 1 169 images).
3. Cible par image = moyenne des logits de l'original sur 16 vues modifiées, logit de la vraie classe baissé de
   δ = 20 (classes entières) ou 6 (classes partielles).
4. Élève (copie de l'original), 5 époques, AdamW lr 5e-5 : KL vers la cible sur l'image exacte + sur une vue légèrement
   modifiée + 3 × KL vers l'original (T=2) sur des mosaïques 2×2.
5. Extinction : biais de la dernière couche des 9 classes −7.

## Lien avec les références de l'épreuve
- **SCRUB** : structure professeur/élève et idée de SCRUB+R (viser le niveau d'erreur d'un modèle qui n'a jamais vu les
  images, sans le dépasser). On remplace « s'éloigner du professeur » par une cible explicite : la métrique punit le dépassement.
- **Zero-Shot Unlearning** : travailler sans données d'origine → nos mosaïques.
- **SalUn** : testé (v14), pas de réduction de la dérive ici.
- **NPO** : non applicable (pas de montée de gradient).
- Hors liste : TRW (arXiv 2512.06993) et ReGUn (arXiv 2603.11210) pour la « mauvaise réponse » du gold.

## Garde-fous contre le surajustement
- Peu de boutons, chacun avec un sens physique ; on ne retient pas un gain < 5 % ; on choisit un plateau, pas un pic.
- Chaque comparaison ne change qu'un élément par rapport à une base déjà notée.
- Risque principal : le seuil des 70 % (éteindre 9 classes coûte de l'accuracy) ; le meilleur score public est retenu.

## Ménage du 21/11 (~22h30)
E. a demandé de supprimer les fichiers de poids `.pt` de v1 à v4 pour libérer de la place (≈ 5,5 Go, 16 fichiers).
Suppression à faire par E. (`rm v1/subs/*.pt v2/subs/*.pt v3/subs/*.pt v4/subs/*.pt`) : Claude n'y est pas autorisé.
Le code, les notes, les journaux et les métriques (`.json`) de ces versions sont conservés ; chaque modèle peut être
recalculé avec la commande notée dans le `run_seq.sh` / `NOTES.md` de sa version (≈ 2 min de GPU).


---

# Nuit du 21 au 22/11 (22h30 → 8h) : de 0.1552 à 0.2937

## Progression du record
| heure | fichier | score | ce qui a changé |
|---|---|---|---|
| 22h30 | v18/k16_wr1_at025_b7 | 0.1552 | (point de départ : 9 classes éteintes, β 7) |
| 0h30 | v27/sans_C | 0.1660 | ne plus éteindre les 4 classes de bateaux |
| 2h35 | v36/ext_wr1_b7 | 0.1680 | ancrage sur des photos extérieures (COCO) au lieu des mosaïques |
| 2h57 | v36/ext_wr1_g10_7 | 0.1865 | extinction plus forte des classes 0 et 38 (β 10) |
| 3h28 | v43/ext_wr1_g10_10 | 0.1894 | β 10 aussi sur les lanternes |
| 3h49 | v43/ext_wr1_a14 | **0.2319** | éteindre SEULEMENT 0 et 38, mais fort (β 14) — prédit par notre modèle du leaderboard |
| 4h20 | v43/ext_wr1_t08_a14 | **0.2455** | réponses un peu plus tranchées (tous les logits ÷ 0,8) |
| 6h24 | v47/h4_all_g22 | **0.2638** | plus de classes éteintes : on « sacrifie » les 130 images les plus fragiles, une par une |
| 6h44 | v47/h7_heavy_g16 | **0.2662** | 252 images fragiles choisies dans les 9 classes presque vidées |
| 6h55 | v47/hh7_c3 | **0.2859** | même liste, loss mieux calée (16 sur l'image, un peu plus visé sur les copies) |
| 7h27 | v47/hb_comb | **0.2937** | liste d'images fragiles = combinaison de deux indices indépendants |

## Découverte 6 — on peut « lire » le gold dans nos propres notes (v45)
- La formule officielle `mean_std = 1 − (|m − MU| + |s − SIGMA|) / (MU + SIGMA)` ne dépend que de la moyenne m et de
  l'écart-type s de NOTRE loss. On les calcule exactement pour chaque soumission (sorties en cache), puis on cherche
  les (MU, SIGMA) du gold qui expliquent nos notes : **MU ≈ 1,4, SIGMA ≈ 4,25** (copies : MU + 0,5), erreur 1,4 % sur 13 notes.
- Lecture : loss moyenne BASSE mais très dispersée = le gold a une loss énorme (≈ 16) sur ≈ 7 % des images seulement,
  et très basse ailleurs. Rien à voir avec « 9 classes entières inconnues » (qui donnerait MU ≈ 3 à 5).
- Le modèle a PRÉDIT le record : « n'éteindre que 0 et 38, à β 14 » → prédit 0,21, obtenu 0,2319.
- Il sépare aussi chaque note en deux facteurs : `note = mean_std × reste`, où reste = prédiction × retain.
  Sur 32 anciennes soumissions, le reste vaut presque toujours 0,22 à 0,26, quelle que soit la recette :
  toutes nos variantes de cible/ancrage/professeur ne changeaient donc presque rien ; seul le réglage de l'extinction comptait.

## Découverte 7 — le jeu de données (v45/img/*.jpg)
Images générées par IA, 100 objets rangés par FAMILLES de classes sœurs : figurines d'animaux (0 renard, 1 hibou,
2 lapin, 3 héron, 4 tortue), lanternes (5 à 8), bateaux (29 à 33), chaises (34 à 38), moulins (39 à 41), tricots/paniers (42 à 45)…
Quand l'original hésite, c'est toujours à l'intérieur d'une famille : c'est la « mauvaise classe » que le gold choisit.

## Découverte 8 — aucune classe n'est retirée en entier
- `orig_a28` = modèle ORIGINAL sans entraînement, classes 0 et 38 éteintes : **0.0**. Sa loss moyenne/écart-type est bonne
  (mean_std prédit 0,91) ; c'est donc la note de prédiction qui est nulle : éteindre 0 et 38 ne rapproche pas du gold
  image par image. Le gold reconnaît encore, en moyenne, ces images.
- Lecture actuelle : les 9 classes à 130 images sont presque vidées (pas complètement). Le gold s'y trompe souvent et avec
  assurance, sur une partie des images seulement. Notre extinction de 0 et 38 imite la bonne RÉPARTITION de loss
  (mean_std ≈ 0,93) mais pas forcément sur les bonnes images.
- Inégalité utile : une image de loss L (à T=1) a forcément une probabilité ≤ exp(−L/2) à T=2. Chaque image
  « sacrifiée » coûte donc ≈ (probabilité que le gold lui donne) × L/2 en prédiction. Il faut ≈ 7 % d'images sacrifiées
  pour le mean_std ; tout l'enjeu est de choisir celles que le gold rate vraiment (tests en cours : v43 paires de classes, v47 image par image).

## Découverte 9 — à quel point hésiter (température)
Sur une image ordinaire, à T=2, notre modèle donne : vraie classe / 3 classes voisines / 96 classes sans rapport.
| modèle | vraie / voisines / sans rapport | score |
|---|---|---|
| original (appris par cœur) | 0,99 / 0,01 / 0,00 | 0 (orig_a28) |
| oubli ciblé sur les voisines (v46/s3_4) | 0,80 / 0,12 / 0,08 | 0.1755 |
| oubli plus léger δ 4 (v46/d4) | 0,76 / 0,13 / 0,11 | 0.1883 |
| record ÷ 0,5 | 0,69 / 0,26 / 0,05 | 0.1700 |
| **record ÷ 0,8** | 0,53 / 0,26 / 0,21 | **0.2455** |
| record (δ 6) | 0,42 / 0,24 / 0,34 | 0.2319 |
Le sommet est net : le gold hésite beaucoup à T=2 (≈ 0,5 sur la vraie classe), y compris avec des classes sans rapport.

## Ce qui n'a PAS marché cette nuit (utile pour les questions du jury)
| idée | résultat |
|---|---|
| AMUN (exemples adverses les plus proches, ICML 2025), testé en maquette v41 | 0,04 contre 0,34 pour notre recette |
| Professeur « dé-mémorisé » (re-distillation sur photos, façon UNDO) : cibles v40 | 0.1840 ≈ record d'alors 0.1865 |
| Forme de la cible (marge plafonnée, température…), maquette v44 | tout entre 0,30 et 0,36 : plat |
| Traiter lanternes/bateaux comme classes partielles (v46/a2) | 0.2304 ≈ 0.2319 |
| Ne sacrifier qu'une seule classe, plus fort (v47/c0, c38) | 0.2301 / 0.2399 ≈ 0.2319 |
| Extinction plus forte partout (β 13) | 0.1424 |

## Méthode de travail de la nuit
1. Maquettes locales (v39, v41, v44) pour écarter vite les idées de la littérature.
2. Modèle du leaderboard (v45) : chaque soumission sert à tester une prédiction chiffrée, pas à « essayer ».
3. Sorties de tous les modèles en cache → les réglages d'extinction et de température se font sur CPU, sans entraînement.


## Découverte 10 — sacrifier des IMAGES, pas des classes (v47, 6h25)
- Rappel : pour le mean_std il faut ≈ 7 % d'images avec une loss énorme. On le faisait en éteignant deux classes entières.
- Or le gold ne rate pas des classes entières : il rate les images « difficiles » des classes presque vidées.
- Indice de fragilité d'une image = marge de la bonne classe sur la meilleure autre, mesurée sur l'original quand on lui
  montre 16 versions modifiées de l'image. Petite marge = image qui ne tient que par le par-cœur.
- Recette (2 étapes) : (1) élève habituel (cible moyennée sur 16 vues, δ 6, ancrage photos) ; (2) `etape2.py` : on enfonce
  la bonne classe des images les plus fragiles jusqu'à une loss visée (terme de marge), sur l'image et sur ses copies,
  pendant que tout le reste est maintenu identique au modèle de l'étape 1. Plus aucun biais global.
- Résultats : 2 classes entières 0.2455 → 130 images fragiles 0.2638 → 252 images fragiles des 9 classes lourdes 0.2662
  → même liste avec la loss calée par notre modèle du leaderboard (mean_std prédit 0,87 → 0,94) **0.2859**.
- Pourquoi ça marche : une image de loss L à T=1 a une probabilité ≤ exp(−L/2) à T=2 ; si le gold, lui, la reconnaît,
  on paie cher en « prédiction ». Autant mettre la loss énorme là où le gold se trompe aussi.


## Découverte 11 — deux façons indépendantes de repérer les images que le gold rate (7h15-7h30)
- Indice V (« vues ») : marge de la bonne classe quand on montre à l'ORIGINAL des versions modifiées de l'image.
- Indice T (« profondeur ») : marge de la bonne classe pour des modèles « dé-mémorisés » (v34/v38/v53 : derniers blocs
  remis à zéro puis réappris sur des photos extérieures) ; très proche de « lire la réponse avant les derniers blocs ».
- Les deux listes de 252 images n'ont que 25 % d'images en commun (corrélation des indices : 0,10) et donnent pourtant le
  même score (0.2859 et 0.2858). La somme des deux : **0.2937**. Chacun trouve une partie des erreurs du gold.
- Vérifié aussi : rendre les images non sacrifiées plus sûres d'elles fait perdre 21 % (v47/p_45_6 = 0.2248).

## État à 8h et travaux en cours
- Règle confirmée par les organisateurs (Discord) : « It is the best 70% for each task. Public 30% is only for feedback ».
  Le leaderboard public ne sert qu'à nous guider ; chaque soumission compte pour le classement final.
- Le premier du leaderboard est à ≈ 0,48 ; nous à 0,2937.
- Recette actuelle (2 étapes, v47) : (1) élève = copie de l'original qui imite, sur chaque image à oublier et une copie
  légère, la réponse moyenne de l'original sur 16 vues modifiées avec la bonne classe baissée de 6 ; ancrage sur des photos
  COCO. (2) `v47/etape2.py` : les 252 images les plus fragiles des 9 classes « presque vidées » sont enfoncées jusqu'à une
  loss ≈ 16,5 (copies : +2,5) vers leur classe voisine, le reste est maintenu identique à l'étape 1.
- En cours de test : listes d'images fragiles plus riches (3 familles d'indices : vues, profondeur, bruit/élagage des
  poids), 330 images au lieu de 252, classes lourdes plus hésitantes, cible « d'ensemble » (moyenne des deux estimations),
  et une étape 3 (v54) qui essaie de rendre à l'original les images fraîches sans toucher aux images à oublier.


==================================================================
FICHIER : v43/NOTES.md
==================================================================
# v43 — Extinction classe par classe (CPU, sans entraînement)

Base : v36/ext_wr1.pt (recette du record, ancrage sur photos). Outil : `biais.py` (un β par classe).

## Ce qu'on sait (tests v27/v36, leaderboard)
- Classes réellement retirées (extinction forte utile) : 0, 38 (−10 % et −9 % sans), groupe 6-7-8 (−18 % sans ; β 4 < β 7).
- Classes 29, 30, 31, 33 : NE PAS éteindre (les éteindre en bloc coûte 7 %), bien qu'elles aient 130 images dans le forget set.
- (β{0,38}, β{6,7,8}) : (7,7) 0.1680 · (10,7) **0.1865** · (7,4) 0.1555.
- Maquette réaliste (v39, 10 blocs figés) : un vrai gold a une loss ≈ 9 ± 1.6 sur les classes retirées → cohérent avec β ≈ 10.

## Candidats
| fichier | β {0,38} | β {6,7,8} | autres classes | Leaderboard |
|---|---|---|---|---|
| ext_wr1_g10_10 | 10 | 10 | — | **0.1894** (3h28) record |
| ext_wr1_g13_13 | 13 | 13 | — | **0.1424** (3h38) : trop fort, −25 % |
| ext_wr1_g16_16 | 16 | 16 | — | |
| ext_wr1_g13_10 | 13 | 10 | — | |
| ext_wr1_g20_20 | 20 | 20 | — | |
| ext_wr1_g10_7_prior1/2/4 | 10 | 7 | baisse proportionnelle à la part retirée : κ × (−log(1 − n/131)), κ = 1, 2, 4 | |
Idée « prior » : une classe qui a perdu beaucoup d'images est moins souvent prédite par le gold, sur TOUTES les images
(pas seulement celles à oublier) ; on le reproduit en baissant un peu son biais (0.4 à 1.6 × κ selon la classe).

## Modèle du leaderboard (v45/fit2.py, 3h45)
Avec les sorties de notre modèle en cache, on calcule pour chaque réglage d'extinction la moyenne m et l'écart-type s de
NOTRE loss sur les images à oublier, puis on ajuste la formule officielle `1 − (|m − MU| + |s − SIGMA|) / (MU + SIGMA)` sur
nos 13 notes. Résultat : erreur moyenne 1,4 % seulement avec **MU ≈ 1,4 et SIGMA ≈ 4,25** (gold), copies : MU + 0,5.
Lecture : le gold a une loss moyenne BASSE mais très dispersée = peu d'images (≈ 7 %) avec une loss énorme (≈ 16) :
exactement les classes 0 et 38 (260 images). Les lanternes (6, 7, 8) et les bateaux (29, 30, 31, 33) ne sont donc PAS
à éteindre : le gold s'y trompe « un peu » (loss ≈ 1,5), comme notre modèle avant extinction.
Prédiction : éteindre seulement 0 et 38, mais fort (β ≈ 14) → ≈ 0,21.

| fichier | β 0 | β 38 | β 6-7-8 | Leaderboard |
|---|---|---|---|---|
| ext_wr1_a14 | 14 | 14 | 0 | **0.2319** (3h49) RECORD (+22 %) : au-dessus de la prédiction (0,21) → ne plus éteindre les lanternes améliore aussi la « bonne mauvaise réponse » |
| BCpart_a14 (base v31/BCpart : lanternes et bateaux traités comme classes partielles, ancrage mosaïques) | 14 | 14 | 0 | |

| orig_a28 (modèle ORIGINAL, sans aucun entraînement, biais −28 sur 0 et 38) | 28 | 28 | 0 | **0.0** (4h09) |

Lecture de orig_a28 = 0 (4h15) : la loss moyenne/écart-type de ce modèle est bonne (mean_std prédit 0,91), donc c'est le
score de PRÉDICTION qui est nul : éteindre 0 et 38 n'approche pas du gold image par image. Autrement dit le gold CONNAÎT
encore les classes 0 et 38 (il leur donne en moyenne ≥ 0,35 de probabilité à T=2). Aucune classe n'est donc retirée en
entier : les 9 classes à 130 images sont « presque » retirées, le gold s'y trompe souvent et avec assurance (loss ≈ 12-16
sur ces erreurs), ce qui donne sa loss moyenne 1,4 / écart-type 4,25. Notre extinction de 0 et 38 imite cette
répartition de loss (mean_std), mais sur les mauvaises images → elle coûte en prédiction. Piste v47 : choisir image par
image lesquelles « sacrifier » (les plus difficiles), au lieu de classes entières.

## Température du modèle (4h20) — `echelle.py`
Constat (v45) : notre modèle hésite trop « dans le vide ». Sur une image ordinaire, à T=2 : vraie classe 0,42, 3 classes
voisines 0,24 et **0,34 éparpillé sur les 96 classes sans rapport** (l'original : 0,99 / 0,009 / 0,005). Baisser la
vraie classe de δ=6 face à TOUTES les classes gonfle ce « bruit de fond ». Un modèle qui n'a jamais vu l'image hésite
entre classes voisines, pas avec tout le monde. Remède sans entraînement : diviser tous les logits par τ < 1.

| fichier | τ | β 0/38 | à T=2 : vraie / voisines / bruit de fond | Leaderboard |
|---|---|---|---|---|
| ext_wr1_a14 | 1 | 14 | 0,42 / 0,24 / 0,34 | 0.2319 |
| ext_wr1_t08_a14 | 0,8 | 14 | 0,53 / 0,26 / 0,21 | **0.2455** (4h20) RECORD |
| ext_wr1_t065_a13 | 0,65 | 13 | 0,62 / 0,27 / 0,12 | |
| ext_wr1_t05_a11 | 0,5 | 11 | 0,69 / 0,26 / 0,05 | **0.1700** (4h40) : trop tranché |
| ext_wr1_t125_a15 | 1,25 | 15 | (plus hésitant : mauvais sens, retiré de la file) | |

## Quelles classes « sacrifier » ? (5h25) — base v31/aucune (toutes les classes traitées pareil, δ 6), biais seul
Même loss moyenne/écart-type pour les trois (mean_std prédit 0,95) : la différence de note = la note de prédiction.
| fichier | classes sacrifiées (≈ 260 images, loss ≈ 16,5) | Leaderboard |
|---|---|---|
| aucune_s0_38_b19 | 0 (renard) et 38 (chaise) : référence | |
| aucune_s30_31_b20 | 30 et 31 (deux bateaux) | |
| aucune_s6_7_b20 | 6 et 7 (deux lanternes) | |


==================================================================
FICHIER : v45/NOTES.md
==================================================================
# v45 — Outils d'analyse (aucun modèle soumis) : lire le leaderboard et choisir les réglages sans entraînement

| fichier | rôle |
|---|---|
| `diag_classes.py`, `img/*.jpg` | planches d'images par classe (familles de classes sœurs), solidité de chaque classe |
| `eval_models.py`, `eval_vues.py`, `bilan.sh` | met en cache les sorties d'un modèle (image exacte, 4 copies légères, vues fortes) |
| `fit2.py` | **modèle du leaderboard** : ajuste la moyenne/écart-type de loss du gold (MU ≈ 1,4, SIGMA ≈ 4,25) sur nos notes |
| `choix_beta.py`, `choix1.py`, `tableau.py`, `bilan_h.py` | pour un modèle en cache : meilleur biais, mean_std prédit, forme des réponses à T=2 |
| `implique_tout.py` | note = mean_std prédit × « reste » pour toutes les anciennes soumissions |
| `derive.py` | « retain local » : écart à l'original sur des vues très modifiées |
| `marges.py`, `feats.py`, `peu_exemples.py`, `fragile2.py`, `indices.pt` | indices de fragilité des images (vues, peu d'exemples, bruit, élagage, lecture anticipée) |
Les conclusions tirées de ces outils sont dans `JOURNAL.md` (découvertes 6 à 11) et `v43/NOTES.md`, `v47/NOTES.md`.


==================================================================
FICHIER : v46/NOTES.md
==================================================================
# v46 — Seules les classes 0 et 38 sont vraiment retirées (lanternes et bateaux traités comme connus du gold)

## Pourquoi
Le modèle du leaderboard (v45/fit2.py, ajusté sur 13 notes, erreur 1,4 %) dit : gold = loss moyenne ≈ 1,4 et écart-type ≈ 4,25
→ très peu d'images (≈ 7 %) à loss énorme (≈ 16) = les classes 0 (renard en figurine) et 38 (chaise), 260 images.
Les 4 classes de lanternes (5-8) et les 5 classes de bateaux (29-33) sont des familles de classes sœurs : le gold s'y
trompe « un peu », il ne les ignore pas. Jusqu'ici on éteignait les lanternes (β 7 à 10) et on forçait les bateaux à 32 %.

## Recette
Recette du record v36/ext_wr1 (16 vues, δ 20 / 6, ancrage photos COCO poids 1, 5 époques), mais `--full_classes 0,38`.
Nouvelles options : `--mid_classes ... --delta_mid D` (δ propre aux familles lanternes/bateaux), `--delta_cls "c:δ,..."`.
Puis extinction (biais) des seules classes 0 et 38 avec `biais.py`.

| run | lanternes 6-7-8 et bateaux 29-30-31-33 | acc 0/38 avant β | acc lanternes/bateaux | acc autres | Leaderboard |
|---|---|---|---|---|---|
| a2 | δ 6 (comme les classes partielles) | 0.12 | 0.85 / 0.81 | 0.91 | a2_a14 (β 14 sur 0 et 38) = **0.2304** (3h59) : égal au record 0.2319 (où lanternes/bateaux sont à 40 %/33 %) |
| a2_m9 | δ 9 | 0.15 | 0.63 / 0.55 | 0.92 | non soumis (entre les deux, même note attendue) |
| a2_m12 | δ 12 | 0.15 | 0.48 / 0.37 | 0.92 | non soumis |

Conclusion (4h00) : le niveau exact des lanternes et des bateaux (33 % à 85 % de bonnes réponses) ne change pas la note.
Ce qui compte : ne PAS les éteindre, et éteindre fort les seules classes 0 et 38.

## Oubli plus léger / ciblé sur les classes voisines (4h06)
| run | réglage | à T=2 sur une image ordinaire : vraie / 3 voisines / sans rapport | Leaderboard |
|---|---|---|---|
| d0 | δ_part 0 (cible = moyenne de 16 vues, rien de plus) | 0,96 / 0,03 / 0,01 | non soumis (≈ original) |
| d2 | δ_part 2 | 0,91 / 0,06 / 0,03 | non soumis |
| d4 | δ_part 4, β 13 sur 0/38 | 0,76 / 0,13 / 0,11 | d4_a13 = **0.1883** (5h22) |
| s3_4 | δ_part 0 + les 3 classes voisines remontées de 4 (`--sib_k 3 --sib_delta 4`), β 13 | 0,80 / 0,12 / 0,08 | s3_4_a13 = **0.1755** (4h30) |
| a2 ÷ 0,65 | a2 rendu plus tranché, β 13 | 0,75 / 0,18 / 0,07 | a2_t065_a13 = **0.1822** (5h01) |
Conclusion : oublier moins (ou seulement vers les voisines) fait PERDRE 20 à 25 %. Le gold hésite vraiment beaucoup à T=2 ;
δ 6 avec une légère accentuation (÷ 0,8) reste le meilleur réglage connu.


==================================================================
FICHIER : v47/NOTES.md
==================================================================
# v47 — Choisir image par image lesquelles « sacrifier » (au lieu d'éteindre des classes entières)

## Pourquoi (4h15, 22/11)
- Modèle du leaderboard (v45) : le gold a une loss moyenne ≈ 1,4 et un écart-type ≈ 4,25 sur les images à oublier
  = une petite part d'images (≈ 7 %) avec une loss énorme (≈ 16), le reste très bas.
- On imitait cela en éteignant les classes 0 et 38 (record 0.2319). Mais `orig_a28` (modèle original + extinction de 0 et 38,
  sans rien d'autre) = **0.0** : sa loss moyenne/écart-type est bonne, donc c'est la note de prédiction qui est nulle
  → le gold reconnaît encore, en moyenne, les images des classes 0 et 38. Aucune classe n'est retirée en entier.
- Lecture : le gold se trompe « avec assurance » sur une minorité d'images réparties dans les classes très dégarnies.
  Éteindre une classe entière met la loss énorme sur les bonnes ET les mauvaises images : coût en prédiction.
  Inégalité utile : une image de loss L (T=1) a forcément une probabilité ≤ exp(−L/2) à T=2 → chaque image sacrifiée
  coûte ≈ (probabilité que le gold lui donne) × L/2. Il faut donc sacrifier les images que le gold rate le plus sûrement.

## Idée
- Indice de fragilité = marge de la vraie classe sur la meilleure autre classe, moyenne des logits de l'original sur
  16 vues modifiées. Les images les plus fragiles sont celles qu'un modèle qui ne les a pas vues rate le plus souvent.
- Images sacrifiées : cible = classe voisine (vraie classe retirée de la cible) + terme de marge qui force
  loss = `--sac_margin` sur l'image exacte ET sur sa copie légère. Les autres images : recette habituelle (δ 6).
- Plus aucune extinction globale (biais) : les images de contrôle des classes 0 et 38 ne sont plus touchées.

## Runs
| run | images sacrifiées | loss visée | loss obtenue (sacrifiées / autres) | mean_std prédit | Leaderboard |
|---|---|---|---|---|---|
| h7_all | 7,2 % les plus fragiles, toutes classes (252) | 16 | | | |
| h7_heavy | 7,2 % les plus fragiles parmi les 9 classes à 130 images | 16 | | | |
| h4_all | 3,7 % les plus fragiles (130) | 22 | | | |
| c0 | (comparaison) classe 0 seule traitée comme retirée + biais −21 | — | 23,7 / 0,3 | 0,910 | c0_b21 = **0.2301** (4h51) |
| c38 | (comparaison) classe 38 seule + biais −21 | — | 23,3 / 0,3 | 0,910 | c38_b21 = **0.2399** (5h11) |

## Étape 2 (`etape2.py`) : enfoncer la loss des images sacrifiées
L'étape 1 seule n'atteint que loss ≈ 3,4 sur les images sacrifiées (visé 16) : il faut une 2e étape dédiée (professeur =
modèle de l'étape 1 figé, l'élève ne change que la vraie classe des images sacrifiées et de leurs copies).
| run | réglages | sacrifiées (exactes / copies) | autres images | ensemble m/s | Leaderboard |
|---|---|---|---|---|---|
| h7_all_m16 | lr 5e-5, 400 pas | 19,4 / 17,3 | loss 0,59, acc 0,86 | 1,94 / 5,18 (trop fort) | non soumis |
| h7_all_g16 | lr 1,5e-5, w_keep 4, 700 pas, marge 15,5 | 14,9 / 13,0 | loss 0,46, acc 0,87 | 1,50 / 3,96 | **0.2567** (6h34) |
| h7_heavy_g16 | idem, 9 classes à 130 images seulement | 15,0 / 13,1 | loss 0,43, acc 0,87 | 1,48 / 3,95 | **0.2662** (6h44) RECORD |
| h4_all_g22 | idem, 130 images, marge 21,5 | 21,5 / 19,5 | loss 0,60, acc 0,85 | 1,37 / 4,30 | **0.2638** (6h24) RECORD (+7,5 %) |
Limite observée : enfoncer les images fragiles abîme aussi leurs voisines (les « autres » passent de 0,25 à 0,45 de loss).

## Résultat clé (6h24) : sacrifier les images les plus FRAGILES bat le sacrifice de classes entières
`h4_all_g22` = **0.2638** contre 0.2455 pour le meilleur réglage « classes 0 et 38 éteintes ». Aucun biais global, aucune
température. Les 130 images sont réparties dans beaucoup de classes (31, 7, 19, 30, 6, 39, 43, 33, 32, 0, 28, 41…).
Lecture : le gold se trompe « très fort » sur peu d'images, et ce sont plutôt celles où l'original hésite dès qu'on
modifie l'image. Suite : nombre d'images (105 / 130 / 175 / 252), profondeur, et réduction des dégâts sur les voisines.
| run (étape 2 depuis h4_all sauf mention) | images | marge | réglages | Leaderboard |
|---|---|---|---|---|
| h3_g24 | 105 | 23,5 | comme g22 | |
| h5_g19 | 175 | 18,5 | comme g22 | |
| h4_k8 | 130 | 21,5 | plus doux : lr 1e-5, w_keep 10, 1000 pas | |
| h4_ext | 130 | 21,5 | départ = v36/ext_wr1 (lanternes/bateaux déjà très hésitants) | |

Lecture (6h47) : à nombre d'images égal (252), choisir les fragiles DANS les 9 classes presque vidées (0.2662) fait mieux
que dans toutes les classes (0.2567) — alors même que le mean_std prédit de ces deux modèles n'est que 0,87 (loss un peu
trop basse, surtout sur les copies). Suite : même liste, loss mieux calée (`--margin 16.5 --copy_extra 2.5`), et tailles 130 / 175.

## Suite (7h20) — classes lourdes : calage, taille, et comment repérer les images fragiles
| run | liste d'images (classes lourdes) | nb / loss visée | mean_std prédit | Leaderboard |
|---|---|---|---|---|
| hh7_c3 (départ h7_heavy) | marge de l'original sur 16 vues | 252 / 16,5 (+2,5 sur copies) | 0,942 | **0.2859** (6h55) RECORD |
| hh5_c3 | idem | 175 / 18,5 | 0,948 | **0.2782** (7h06) : moins d'images = moins bien |
| hh4_c3 | idem | 130 / 21,5 | 0,936 | non soumis (même tendance attendue) |
| hb_prof (départ base0) | marge des 3 « professeurs » dé-mémorisés (v34/v35/v38), seulement 25 % d'images en commun | 252 / 16,5 | 0,924 | **0.2858** (7h16) : aussi bon avec une liste presque différente |
| hb_comb | somme des deux indices | 252 / 16,5 | 0,939 | **0.2937** (7h27) RECORD : combiner les deux indices aide |
| hb_rob | marge de l'original sur 192 vues (4 forces) | 252 / 16,5 | | en file |
| hb_robprof / hbg_robprof | 192 vues + professeurs ; version graduée (3 % à 21, 7,2 % à 15, 12 % à 8) | | | |
| hh9_c2 / hhg | 330 images à 14,5 / gradué | | | |
| p_45_6, p_5_65, p_5_9 | images non sacrifiées plus tranchées (δ 4,5 ou 5) / classes lourdes plus hésitantes | 252 / 16,5 | 0,948 / 0,946 / 0,926 | p_45_6 en file |
Lecture : deux indices de fragilité presque indépendants (vues modifiées de l'original, modèles dé-mémorisés) donnent
le même score → chacun trouve une partie des images que le gold rate ; les combiner devrait mieux cibler.

## Résultats 7h30-7h50
| fichier | changement par rapport au record hb_comb | Leaderboard |
|---|---|---|
| p_45_6 | images non sacrifiées plus tranchées (classes partielles δ 4,5 : vraie classe 0,75 à T=2 au lieu de 0,63) | **0.2248** (7h37) : −21 %, il ne faut PAS être plus sûr |
| hb_trois | liste = vues (192) + professeurs + simulation « peu d'exemples » | **0.2908** (7h47) : ≈ record |
Nouveaux indices de fragilité calculés (v45/fragile2.py, v45/indices.pt) : bruit sur les poids, élagage, lecture avant
les derniers blocs, distance au plus proche exemple adverse. Trois familles presque indépendantes : V (vues modifiées),
T (profondeur : professeurs, lecture anticipée), W (bruit/élagage). Listes `lourd_ens3` (V+T+W) et `lourd_ensVT`.
En cours : classes lourdes plus hésitantes (p_6_9, p_6_12), tout plus hésitant (p_7_7, hb_comb_m19 ÷ 1,15), 330 images (hh9_c2).


==================================================================
FICHIER : v50/NOTES.md
==================================================================
# v50 — Ancrage sur des vues FORTES des images à oublier (pour le retain)

## Pourquoi (5h40, 22/11)
- Maquettes v48/v49 (classes lourdes presque vidées, vrai gold) : le facteur le plus faible est le RETAIN (0,5 à 0,6).
  L'oubli appris sur les images à oublier déborde sur les images fraîches des mêmes classes (bonnes réponses sur le
  contrôle : 0,78 contre 0,92 pour l'original). Nos ancrages (photos COCO, mosaïques) ne couvrent pas ce cas.
- Dans la maquette avec 30 % d'images restantes (v48), on retrouve notre situation réelle : prédiction ≈ 0,48 × retain ≈ 0,53
  = 0,25 (notre « reste » mesuré sur le leaderboard : 0,25).

## Idée
Oublier le par-cœur ne devrait changer que l'image EXACTE (et ses copies légères). Sur une vue FORTEMENT modifiée de la
même image, l'original répond déjà « comme sur une image fraîche » : on demande à l'élève d'y rester identique à
l'original (`--view force --w_view poids`, KL à T=2). Maquette v49 : retain 0,59 → 0,78-0,82, prédiction 0,73 → 0,66-0,69,
score +5 à +13 % (poids 1 ; poids 3 = trop fort, l'oubli n'a plus lieu).

## Runs (base = v46/a2 : 0 et 38 seules retirées, δ 6, ancrage COCO 1)
| run | vues fortes | poids | δ classes ordinaires | loss/acc images ordinaires | retain local (v45/derive.py) | Leaderboard |
|---|---|---|---|---|---|---|
| sv10_w3 | 1,0 | 3 | 6 | 0,11 / 0,95 (oubli presque annulé) | | non soumis |
| sv15_w3 | 1,5 | 3 | 6 | | | |
| sv10_15_w3 | 1,0 à 1,5 | 3 | 6 | | | |
| sv15_w1 | 1,5 | 1 | 6 | | | |
| sv10_w1 | 1,0 | 1 | 6 | | | |
| sv15_w03 | 1,5 | 0,3 | 6 | | | |
| sv10_w3_d10 | 1,0 | 3 | 10 | | | |
| sv15_w3_d10 | 1,5 | 3 | 10 | | | |
| sv15_w1_d8 | 1,5 | 1 | 8 | | | |

## Résultats (6h05)
| fichier | forme à T=2 (vraie/voisines/sans rapport) | retain local (vues 1,0 / 1,5) | Leaderboard |
|---|---|---|---|
| (rappel) record v43/ext_wr1_t08_a14 | 0,53 / 0,26 / 0,21 | 0,43 / 0,47 | 0.2455 |
| (rappel) v46/d4_a13 | 0,76 / 0,13 / 0,11 | 0,69 / 0,73 | 0.1883 |
| sv15_w1_a16 | 0,71 / 0,17 / 0,11 | 0,75 / 0,82 | **0.2000** (5h53) |
| sv10_w3_d10_t115_a19 (logits ÷ 1,15) | 0,55 / 0,23 / 0,22 | 0,73 / 0,80 | **0.2243** (6h03) |
| sv10_w3_d14_a19 | 0,52 / 0,26 / 0,21 | 0,72 / 0,78 | non soumis |

## Conclusion : piste fermée
Le « retain local » monte de 0,43 à 0,73 à forme égale, mais le score BAISSE de 9 %. Explication trouvée dans les sorties
(v45, confiance selon la force de la modification) : avec l'ancrage sur vues fortes, le modèle redevient plus sûr de lui
dès la copie légère (vraie classe 0,65 sur la copie contre 0,55 sur l'image exacte), alors que le gold, lui, PERD de la
confiance sur les copies (sa loss y est ≈ 0,5 plus haute). On gagne peut-être en retain mais on perd sur les copies
modifiées, qui comptent pour la moitié de la note d'oubli. D'où v51 : faire l'inverse (moins sûr sur les copies).


==================================================================
FICHIER : v52/NOTES.md
==================================================================
# v52 — Petites améliorations autour du record (recette v36/ext_wr1 + logits ÷ 0,8 + biais 0/38)

## Pourquoi (6h15, 22/11)
Le record (0.2455) vient d'une forme précise des réponses à T=2 (v45, par groupe) :
- classes « partielles » : assez sûres (vraie classe 0,65, voisines 0,18, sans rapport 0,17) ;
- lanternes et bateaux (classes presque vidées) : très hésitantes ENTRE SŒURS (vraie 0,23, voisines 0,47, sans rapport 0,31).
Rendre lanternes/bateaux plus sûrs (v47/c38 ÷ 0,8 : vraie 0,61) coûte 11 %. Logique : moins il reste d'images d'une classe
chez le gold, plus il hésite avec les classes sœurs. On teste ici des variantes qui vont dans ce sens.

## Runs (tous : 9 classes à δ 20, les autres δ 6, 16 vues, ancrage COCO 1)
| run | changement | Leaderboard (après ÷ τ et biais) |
|---|---|---|
| tm05 | queue de la cible lissée : 50 % image + 50 % moyenne de la classe (`--tail_mix 0.5`) | |
| tm00 | queue = moyenne de la classe seulement | |
| g1, g2 | mêmes réglages que le record, autres graines (pour une moyenne de poids de 3 modèles) | |
| k64 | 64 vues au lieu de 16 | |
| sl10, sl20 | oubli gradué selon la part retirée de chaque classe (`--delta_slope 10 / 20` : δ de 4,5 à 9 / de 3 à 12) | |


==================================================================
FICHIER : v53/NOTES.md
==================================================================
# v53 — Professeurs « dé-mémorisés » supplémentaires (pour repérer les images fragiles)

Même recette que v34 (`redistill.py`) : on remet à zéro les derniers blocs de l'original, puis il réapprend à imiter
l'original uniquement sur des photos extérieures (COCO). Il n'a donc plus le par-cœur des images à oublier.
Usage ici : ces modèles ne sont PAS soumis ; on regarde sur quelles images à oublier ils hésitent ou se trompent
(indice de fragilité indépendant de celui des « vues modifiées »). v47/hb_prof a montré que cet indice vaut l'autre (0.2858).
| run | blocs remis à zéro | graine |
|---|---|---|
| r2s1 | 2 | 1 |
| r3 | 3 | 2 |
| r2s3 | 2 (lr 3e-4) | 3 |


==================================================================
FICHIER : v54/NOTES.md
==================================================================
# v54 — Étape 3 : rendre à l'original tout ce qui n'est pas l'image à oublier (retain)

## Pourquoi (7h40, 22/11)
- Le sujet officiel précise : le jeu de contrôle du retain est fait d'images jamais vues, de la même distribution.
- Mesure locale (v45/derive.py) : sur des vues très modifiées des images à oublier (≈ images fraîches de la même classe),
  nos modèles ne reconnaissent plus que 61 à 70 % des images contre 86 % pour l'original. L'oubli déborde.
- Essai v50 (ancrage sur vues fortes pendant l'étape 1) : la mesure locale montait bien (0,43 → 0,73) mais le score baissait
  de 9 %, car le modèle redevenait trop sûr de lui dès la copie légère (0,65 contre 0,55 sur l'image exacte).

## Idée
Faire la séparation APRÈS coup, avec deux professeurs figés :
- professeur « oubli » = notre meilleur modèle (v47/hb_comb, 0.2937) : l'élève doit lui rester identique sur l'image exacte
  et sur ses copies jusqu'à une force `copy_max` (profondeur de loss des images sacrifiées comprise) ;
- professeur « reste » = l'original : l'élève doit lui redevenir identique sur des vues très modifiées (force ≥ `view_lo`),
  sur des collages de 2 images et sur des photos extérieures.

| run | copies gardées « oubli » jusqu'à | vues rendues à l'original à partir de | résultat local | Leaderboard |
|---|---|---|---|---|
| r125 | 0,6 | 1,25 | | |
| r100 | 0,5 | 1,0 | | |
| r150 | 0,75 | 1,5 | | |

## Résultats locaux (7h50)
| run | images sacrifiées (loss exacte / copies) | autres images (loss, acc, vraie classe T=2) | vues 1,0 : KL / acc (original 0,876) |
|---|---|---|---|
| départ = v47/hb_comb | 16,2 / 14,8 | 0,56 · 0,84 · 0,61 | 0,66 / 0,62 |
| r125 (poids « oubli » 1, pas de marge) | 11,7 / 9,6 (la profondeur s'efface) | 0,27 · 0,91 · 0,71 | 0,34 / 0,72 |
| q125 (poids « oubli » 4 + marge 16,5 gardée) | 16,2 / 14,5 | 0,47 · 0,86 · 0,64 | 0,51 / 0,67 |
| q100 (idem, vues à partir de 1,0) | 16,1 / 14,2 | 0,46 · 0,86 · 0,64 | 0,50 / 0,67 |
Lecture : on ne sépare qu'en partie « image exacte » et « image fraîche ». Les vues fortes des images sacrifiées restent
fausses (2 % contre 80 % pour l'original) : le sacrifice n'est pas vraiment propre à l'image. q125 est soumis pour savoir
si ce petit gain de retain local (+16 %) se voit sur le leaderboard.


==================================================================
TOUTES LES NOTES DU LEADERBOARD (heure, fichier, score public)
==================================================================
21:31:59] v16/subs/rec_a1.2.pt 0.1202
21:44:17] v16/subs/e3_b7.pt 0.1324
21:55:27] v16/subs/e3_b7.pt 0.1324
22:17:17] v16/subs/e2_b7.pt 0.0959
22:28:53] v17/subs/f20_k64_d6_b7.pt 0.1523
22:49:30] v18/subs/k16_wr1_b7.pt 0.1528
22:59:48] v17/subs/f20_k16_d6_at025_b7.pt 0.1526
23:10:16] v19/subs/ref_l05.pt 0.0544
23:20:50] v18/subs/k16_wr1_at025_b7.pt 0.1552
23:31:15] v16/subs/e1_b7.pt 0.0848
23:41:43] v25/subs/mfa_b6.pt 0.1169
00:09:09] v26/subs/s1_b7.pt 0.1552
00:19:35] v27/subs/sans_B.pt 0.1609
00:29:56] v27/subs/sans_C.pt 0.166
00:40:16] v27/subs/tous_b5.pt 0.1446
00:50:38] v31/subs/Cpart_b7.pt 0.1626
01:00:56] v31/subs/BCpart_b7.pt 0.1315
01:11:22] v27/subs/cinq_a.pt 0.146
01:21:54] v27/subs/p29.pt 0.1648
01:32:24] v33/subs/h20_w4.pt 0.0837
01:42:58] v27/subs/m0.pt 0.1499
01:53:17] v34/subs/rd_r2_b7.pt 0.1291
02:03:52] v34/subs/rd_r2.pt 0.0341
02:14:25] v27/subs/m38.pt 0.1515
02:24:55] v36/subs/ext_wr1_b7.pt 0.168
02:34:58] v27/subs/sans_BC.pt 0.1383
02:45:49] v36/subs/ext_wr3_b7.pt 0.1577
02:57:03] v36/subs/ext_wr1_g10_7.pt 0.1865
03:07:28] v36/subs/ext_wr1_g7_4.pt 0.1555
03:18:09] v40/subs/t3_9_g10_7.pt 0.184
03:28:23] v43/subs/ext_wr1_g10_10.pt 0.1894
03:38:31] v43/subs/ext_wr1_g13_13.pt 0.1424
03:49:10] v43/subs/ext_wr1_a14.pt 0.2319
03:59:20] v46/subs/a2_a14.pt 0.2304
04:09:48] v43/subs/orig_a28.pt 0.0
04:20:16] v43/subs/ext_wr1_t08_a14.pt 0.2455
04:30:21] v46/subs/s3_4_a13.pt 0.1755
04:40:48] v43/subs/ext_wr1_t05_a11.pt 0.17
04:51:17] v47/subs/c0_b21.pt 0.2301
05:01:22] v46/subs/a2_t065_a13.pt 0.1822
05:11:54] v47/subs/c38_b21.pt 0.2399
05:22:08] v46/subs/d4_a13.pt 0.1883
05:32:36] v43/subs/aucune_s30_31_b20.pt 0.1985
05:43:05] v43/subs/aucune_s0_38_b19.pt 0.2246
05:53:37] v50/subs/sv15_w1_a16.pt 0.2
06:03:47] v50/subs/sv10_w3_d10_t115_a19.pt 0.2243
06:13:58] v47/subs/c38_t08_b21.pt 0.225
06:24:00] v47/subs/h4_all_g22.pt 0.2638
06:34:07] v47/subs/h7_all_g16.pt 0.2567
06:44:37] v47/subs/h7_heavy_g16.pt 0.2662
06:55:26] v47/subs/hh7_c3.pt 0.2859
07:06:15] v47/subs/hh5_c3.pt 0.2782
07:16:40] v47/subs/hb_prof.pt 0.2858
07:27:07] v47/subs/hb_comb.pt 0.2937
07:37:14] v47/subs/p_45_6.pt 0.2248
07:47:27] v47/subs/hb_trois.pt 0.2908
