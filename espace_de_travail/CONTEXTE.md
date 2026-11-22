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
