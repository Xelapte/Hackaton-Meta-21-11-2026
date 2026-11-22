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
| hb_ens3 | liste = trois familles d'indices (vues + profondeur + bruit/élagage) | **0.2844** (7h58) : mean_std prédit 0,913 → même « reste » (0,31) que hb_comb ; la 3e famille n'apporte rien |
| e_50 | cible « d'ensemble » à l'étape 1 : moyenne (en probabilités) de la cible habituelle et de 5 modèles dé-mémorisés | **0.2909** (8h19) : mean_std prédit 0,912 seulement → « reste » 0,319 contre 0,313 pour hb_comb (+2 %) ; relancé avec une loss mieux calée (e_50b) |
| hb_comb_pr05 | record + « décalage de fréquence » (biais de chaque classe baissé de 0,5 × log(150 / images restantes)) | **0.2868** (8h30) : −2 %, piste fermée |
| e_t2_50 | cible d'ensemble, moyenne faite à T=2 | **0.2720** (8h50) : mean_std prédit 0,943 → « reste » 0,288 ; forme trop pointue (0,63 / 0,22 / 0,15), fermé (e_t2_r retiré de la file) |
| e_50b | e_50 avec étape 2 recalée (marge 17,2, copies +3,2) | **0.3016** (9h00) = **NOUVEAU RECORD** : mean_std prédit 0,935 → « reste » 0,3226 (+3 % sur hb_comb). Forme à T=2 des images non sacrifiées : 0,53 vraie / 0,24 sœurs / 0,23 autres (hb_comb : 0,61 / 0,23 / 0,16) : les professeurs déplacent de la masse vers les classes « sans rapport », image par image |
| hh9_c2 | base0, 330 images à 14,5 (au lieu de 252 à 16,5) | **0.2843** (9h11) : moins bien, on reste à 252 |
| e_70 / e_70b / e_30b | part des professeurs 0,7 (étape 2 d'origine) / 0,7 recalée (17,3 ; +3,4) / 0,3 recalée (17,0 ; +3,0) | en file |
