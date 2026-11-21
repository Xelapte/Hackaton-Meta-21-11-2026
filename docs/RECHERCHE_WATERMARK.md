# Tâche 1 (localisation du tatouage) : journal de recherche

## Règles fixées avant les expériences (21/11, 18h15)

- **Plis gelés** : `watermark/splits.json`, 5 plis stratifiés par méthode × `full_watermark`, graine 0
  (`python watermark/protocole.py --creer`, refuse de les recréer). Toutes les mesures se font dessus.
- **Score rapporté** : mIoU officiel sur le pli tenu à l'écart, moyenné sur les 5 plis, ± écart-type entre plis,
  avec le détail par méthode. Tout réglage (seuil, paramètre de post-traitement) se fait sur les 4 autres plis.
- **Gain retenu** seulement s'il est > 0,02, > écart-type du gain entre plis, et positif sur ≥ 4 plis sur 5
  (`protocole.gain_robuste`). Sinon : « pas de gain démontré ».
- **Peu de réglages libres** : au plus 2 configurations par hypothèse ; hyperparamètres (époques, largeur)
  fixés avant le run ; chaque configuration essayée est comptée ci-dessous. Ordre de grandeur du biais de
  sélection : meilleur de N essais ≈ +0,005 × √(2 ln N) (≈ +0,013 pour N = 30), couvert par le seuil de 0,02.
- **Données synthétiques** : supports pris dans le train seulement (aucune image du test) ; recette unique
  calée sur les statistiques du train (surfaces, rapports, types), pas sur le score ; validation sur les vraies images.
- **Test** : prédiction finale seulement. **Score public** : un seul regard, à la fin, comme contrôle indépendant.
- Pas d'augmentation miroir / rotation tant qu'elle n'est pas validée (H4).
- **Ressources** : GPU 2 (watermark) et GPU 3, attribué à la tâche 1 par huang25 à 18h15. Supports du synthétique : train seulement (pas de téléchargement).

## Références (protocole)

| Méthode | mIoU (5 plis) | PixelSeal | TrustMark | StegaStamp |
|---|---|---|---|---|
| Tout à 0 | 0,2518 ± 0,0040 | 0,2530 | 0,2484 | 0,2540 |
| Meilleur de tout à 0 / tout à 1, par méthode | 0,2529 ± 0,0038 | 0,2530 | 0,2516 | 0,2540 |
| Classe majoritaire devinée (moyenne de la carte) | 0,4040 ± 0,0175 | 0,4300 | 0,4488 | 0,3332 |
| *Oracle classe majoritaire (borne, connaît le masque)* | *0,4547 ± 0,0048* | *0,4516* | *0,4523* | *0,4601* |

## Analyse des données (`watermark/analyse_donnees.py`, PNG dans `output/recherche/`)

- `full_watermark` sépare deux régimes. **Partiel** : zone tatouée petite (part q10/50/90 = 0,01/0,04/0,23),
  une composante en général, souvent un rectangle plein (rectangularité médiane 0,97 ; q10 0,4 : objets transplantés).
  **Full** : part 0,77/0,94/0,99, 1 à 7 trous propres en forme d'objet.
- Les trous des images full ont un contenu **régénéré** (objet retiré par inpainting), pas l'original.
- La part tatouée est bimodale : 35 % des images < 10 %, 33 % > 90 %.
- `full_watermark` n'existe pas dans le test : le régime doit être prédit.

## Hypothèses et résultats

| # | Hypothèse | Configurations | mIoU (5 plis) | PS | TM | SS | Conclusion |
|---|---|---|---|---|---|---|---|
| H3a | Décoder la vraie boîte de la zone tatouée (images partielles) se détache du hasard | 1 (test) | — | médiane 0,55 | 0,95 | 0,61 | AUC vraie boîte / hasard ≈ 0,80 ; une boîte décalée de 25 % retombe au hasard : signal très précis en position. Viable. |
| H4 | Le tatouage survit au miroir / aux rotations | 1 (test) | — | survit : miroir H, rot 90 | survit : miroir H | rien | **Rejetée** : augmentation miroir/rotation interdite (le U-Net `--aug` soumis la subissait). |
| H3b | Recherche de rectangle par décodage (grille 48–512 px — 64–512 pour SS, fixé pour le coût —, puis raffinement) ; règle : image entière < t_full et meilleure boîte ≥ t_boîte → masque = boîte, sinon carte H1 ; 2 seuils par méthode réglés dans les plis | 1 | **0,5065 ± 0,0092** (carte H1 : ~0,436) | 0,5018 | 0,5480 | 0,4698 | **Gardée** : gain sur PS+TM +0,057 ± 0,012 (5/5 plis) ; sur SS 0,357 → 0,470. Gain venu des images partielles (TM 0,473 → 0,641, PS 0,498 → 0,559, SS 0,216 → ~0,45). |
| H5a | U-Net + 10 cartes, **sans** augmentation, plis gelés, 40 époques (fixées d'avance), seuil par méthode réglé dans les plis | 1 | 0,4691 ± 0,0081 | 0,4613 | 0,4718 | 0,4743 | Comme le U-Net soumis (0,4705) : l'augmentation nuisible ne coûtait rien au total. Bien meilleur que la carte pour SS. |
| H3c | H3b + menus discrets réglés dans les plis : trous des images full par cote z propre à l'image (A), boîte ∩ carte (B), « tout à 0 » sans boîte | 1 (3 menus) | +0,0003 sur H3b | | | | **Rejetée** : pas de gain (les plis choisissent presque toujours la carte). |
| H3b-unet | H3b avec les probabilités hors pli de H5a comme repli (au lieu de la carte) | 1 | **0,5208 ± 0,0090** | 0,5013 | 0,5556 | 0,5056 | **Gain probable mais sous le seuil** : +0,014 ± 0,007, 5/5 plis positifs, < 0,02. Réserve : les probabilités hors pli d'un pli viennent de modèles entraînés sur les autres plis (petite fuite au second ordre lors du réglage des seuils). Décision laissée à l'équipe. |
| H6a | Trous des images full par occlusion : flouter une boîte et mesurer la variation de la marge de décodage de l'image entière | 1 (test, 15 images / méthode, CPU) | — | AUC 0,67 | AUC 0,80 | AUC 0,72 | **Rejetée** : effet minuscule (médiane −0,1 % à −3 %) devant la dispersion ; une recherche parmi des centaines de boîtes ne trouverait pas le trou. |
| H1 | Cartes par position (corrélation au message connu) : une carte seule + un seuil par méthode | 13 cartes évaluées | meilleures : PS `ps_512` 0,4715 ± 0,008 ; TM `tm_245` 0,4649 ± 0,007 ; SS `ss_16` 0,3634 ± 0,024 | AUC 0,89 (0,94 à 256 px) | AUC 0,91 | AUC 0,66 | Gardée (PS/TM). Fenêtres : PS 0,453, TM 0,281, SS 0,255. AUC élevée mais portée par les grandes zones. |

## Analyse d'erreurs de H3b (`watermark/analyse_erreurs.py`, repli U-Net, 510 images)

- **Régime** (full / partiel) : presque parfait ; oracle régime parfait −0,003 → pas un levier.
- **Boîtes** (images partielles) : IoU médiane de la boîte choisie PS 0,015, TM 0,29, SS 0,000. Oracle « meilleure des 20
  boîtes enregistrées » +0,035 ; oracle « boîte vraie » **+0,153** → le levier est la recherche, pas le classement.
- **Par forme de zone** : TrustMark trouve **84 %** des rectangles (score de la meilleure boîte 1,00) et 0 % des objets non
  rectangulaires (transplantés). PixelSeal et StegaStamp ne trouvent que 33 % / 5 % des rectangles, petits (médiane 1,3 % / 2,2 %
  de l'image, ~60–80 px) : la grille (≥ 48 / 64 px, pas d'une demi-taille) est trop grossière, un décalage de 25 % suffit
  à tomber au hasard (H3a).
- **Images full** : U-Net ≥ tout à 1 (SS 0,524 contre 0,456) ; borne « trous parfaits » +0,26 (hors de portée : H6a).
- **H7a — décodage masqué** (garder l'image à 512 px, flouter / griser hors de la boîte vraie, pour respecter l'échelle des objets
  transplantés) : bits justes ≈ hasard pour les 3 méthodes (0,50–0,52) → **rejetée**.
- **Piste restante** : recherche fine des petits rectangles PS et SS (boîtes dès 32 px, pas d'un quart, raffinement serré),
  même règle à 2 seuils, sans paramètre nouveau ; gain estimé +0,01 à +0,03 ; plusieurs heures de GPU (run de nuit).

## Essais avant le protocole (plis non gelés, à ne pas comparer directement)

- Mode direct (17h20) : 0,4339 (seuils et résolution réglés sur tout le train : optimiste).
- U-Net + 10 cartes, `--aug` (miroir / rotation, non validé), 5 plis aléatoires : 0,4705 ; soumis (n° 1136) : public 0,4804.
- U-Net + synthétique v1 (supports pris aussi dans le test, trous = original remis) : 0,4691 → écarté.
- Canal de détection PixelSeal (0,33), re-tatouage PixelSeal / TrustMark (~0,27) : rejetés.

**Compteur de configurations essayées : ~38** (résolutions, tailles de bloc, 3 U-Net, synthétique v1, 13 cartes).
