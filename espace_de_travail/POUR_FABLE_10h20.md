# Relais pour Fable — état au 22/11 vers 10h05 (rédigé par Claude Opus pendant ton absence)

Ce document contient des **observations**, pas des conclusions. Tout ce qui est écrit ici est mesuré (journaux, leaderboard) ou
explicitement marqué « non mesuré ». À toi d'en tirer les décisions. Fin des soumissions : 12h ; gel annoncé par l'équipe : 10h,
mais E. a demandé de continuer à améliorer les trois tâches.

---------------------------------------------------------------------------------------------------------------------------
## 0. Ce qui tourne et les horaires (à vérifier avec `bash slurm/file.sh` depuis membres/LOGIN)

| GPU | En cours / en attente (ordre) | Fin estimée |
|---|---|---|
| 0 | wm62_ps_L5 (PixelSeal, 30 min synth) → wm63_ps_L9 (PixelSeal, 30 min, TTA miroir) | ~10h18 → ~11h00 |
| 1 | wm62_tm_L5 (TrustMark, 30 min) → ul70_ep_7_7 → ul70_ep_7_7b (unlearning, ~9 min chacun) | ~10h22 → ~10h45 |
| 2 | wm62_ss_L5 (StegaStamp, 30 min, depuis r3) → wm63_tm_L11 (TrustMark, 40 min, TTA) | ~10h25 → ~11h15 |
| 3 | wm62_ss_L7 (StegaStamp, 30 min, depuis r2) → wm63_tm_L9 (TrustMark, 30 min, TTA) | ~10h16 → ~11h00 |

- Les jobs `ul_v57*` que tu avais déposés sont tous passés (g20/g21 inclus).
- Retirés de la file par moi (jamais lancés) : `ul70_p_9_9`, `ul70_ep_8_8` (raison : p_8_8 < p_7_7, voir §2).
- Boucle de fond (mon terminal) : dès que `v70/subs/ep_7_7.pt` puis `ep_7_7b.pt` existent, elle les met **en tête** de `lb_file.txt`.

---------------------------------------------------------------------------------------------------------------------------
## 1. Tâche 1 — Watermark (tes dossiers v60/v61 ; les miens v62/v63)

### Soumissions faites (via `soumettre_wm.sh`)
| n° | heure | fichier | local (hors pli) | public |
|---|---|---|---|---|
| 1412 | 21/11 | H3b (équipe, avant toi) | 0,5065 | 0,5042 |
| 3826 | 9h27 | `v61/subs/v61_a.pt` (assemble.py : PS r2, TM r2, SS r1, variante cv) | 0,6446 | 0,6538 |
| 3894 | 9h47 | `v62/subs/v62_a.pt` (PS r2+r3, TM r2+r3 en « emiou », SS r2 seuil 0,3) | 0,6722 | 0,6814 |
| 3931 | 9h58 | `v62/subs/v62_b.pt` (idem mais SS = r3 seule, seuil 0,3) | 0,6807 | 0,6936 |
| 3974 | 10h17 | `v62/subs/v62_c.pt` (PS v62 L5 emiou, TM v61 r2+r3 emiou, SS v62 L7 seuil 0,45) | 0,7051 | **0,7252** |

Prochaine soumission possible : dès 10h28 (10 min mini, garde-fou du script).

### Observations
- J'ai activé tes passes r3 (`touch v61/GO_R3_*`) à 9h27. CV hors pli, seuil 0,5 :
  PixelSeal r2 0,6431 → r3 0,6562 ; TrustMark r2 0,6426 → r3 0,6513 ; StegaStamp r1 0,6449 → r2 0,6986 → r3 0,7255.
- Synthétique seul (avant affinage, seuil 0,5) : PS r3 0,6284 ; SS r2 ~0,52. L'affinage CV ajoute +0,02 à +0,2 selon la méthode.
- Pendant les passes longues v62, le « mIoU VRAI train » de phase A (surveillance) à mi-parcours : PS L5 0,617 (pas 2709),
  TM L5 0,600 (pas 5847), SS L7 0,491 (pas 8391) — au milieu du cosinus, donc LR encore haute.
- Décision « mIoU attendu par image » (`v62/decide62.py`, idée de v60/decide.py, réglée en imbriqué sur les plis de CV) :
  PS +0,004 / +0,008 (seul / ensemble), TM +0,004 / +0,003, SS −0,003 / +0,001 par rapport au seuil imbriqué.
- Ensemble (moyenne des probas hors pli de r2 et r3 ; les plis sont identiques dans toutes les passes = rang du nom % 5) :
  PS 0,6591 → 0,6602 (avec emiou), TM 0,6540 → 0,6557, SS r2+r3 0,7264 vs r3 seule 0,7262.
- **Passes longues v62 (30 min de synthétique, réserve 3000)** : StegaStamp L7 (init r2) synthétique seul 0,6336 (seuil 0,2),
  CV hors pli **0,7797** (seuil 0,4) contre 0,7262 pour r3 ; PixelSeal L5 (init r3) synthétique seul 0,6560 (r3 : 0,6441).
- Écart public − local observé : +0,009, +0,009, +0,013, +0,020 sur les quatre soumissions v61/v62.
- v60 (U-Net empileur + emiou) culminait à 0,5457 en local ; non soumis.

### Ce que j'ai créé
- `v62/seg_synth.py` = copie de `v61/seg_synth_new.py` (seule modif : sorties dans v62/).
  `v62/run_long_*.sh` : `--synth_secs 1800 --cap 3000 --ft_secs 45`, init = derniers poids v61 (r3 > r2 > r1).
- `v63/seg_synth.py` = idem + **TTA miroir horizontal** enregistrée À PART (`train_oof_flip`, `test_cv_flip`) ; une ligne
  « TTA miroir : hors-pli seul … | moyenne … | sans … » est écrite dans le journal. Hypothèse H4 de l'équipe : PS et TM survivent
  au miroir, SS non. Pas encore mesuré.
- `v62/decide62.py` : `--runs pixelseal=v61/r2,v61/r3,v62/L5 trustmark=... stegastamp=...` ; suffixe `+f` (ex. `v63/L9+f`) =
  moyenne avec la TTA ; `--write subs/x.pt` écrit la soumission (refuse d'écraser). ~2,5 min CPU, sur le nœud de login.
- Commande type pour la suite (quand les jsons de v62/v63 existent) :
  `cd unlearning/v62 && ../../Hackathon_Watermark_Localisation_Task/.venv/bin/python decide62.py --runs pixelseal=v61/r3,v62/L5 ...`
- Notes : `v61/NOTES.md`, `v62/NOTES.md` (avec « Rapport pour l'équipe »). Scores dans `$DIR/scores.csv`.

---------------------------------------------------------------------------------------------------------------------------
## 2. Tâche 3 — Unlearning (ta file `file_lb.sh` tourne toujours, PID visible avec `ps -u $USER`)

### Scores observés depuis 9h (lb_journal.txt ; appariement fichier ↔ score refait proprement)
| heure | fichier | public |
|---|---|---|
| 9h00 | v47/e_50b | 0,3016 |
| 9h11 | v47/hh9_c2 | 0,2843 (attention : un appariement naïf du journal attribue 0,3016 à hh9_c2 ; c'est faux) |
| 9h21 | v47/e_70 | 0,2835 |
| 9h31 | v47/p_7_7 | **0,3144** (record actuel) |
| ~9h41 | v47/hb_ensVT | 0,3008 |
| ~9h52 | v47/p_8_8 | 0,3068 |
| ~10h02 | v47/p_7_6 | 0,3112 |

Famille p_a_b = étape 1 avec `--delta_part a --mid_classes 0,38,6,7,8,29,30,31,33 --delta_mid b`, puis étape 2 (liste
`lourd_comb_sac`, marge 16,5 ou 16, copies +2,5). Points mesurés : (7,7) 0,3144 ; (7,6) 0,3112 ; (8,8) 0,3068 ;
(4,5 ; 6) 0,2248 ; hb_comb (≈ 6/6 sans mid ?) 0,2937. e_50b (cible d'ensemble `--teacher_mix 0.5` + étape 2 marge 17,2 / +3,2) 0,3016
contre e_50 (marge 16,5 ?) 0,2909.

### Ce que j'ai changé
- `lb_file.txt` réordonné deux fois (fichiers existants seulement). Ordre actuel : `p_6_9`, `pc1`, `hb_tout6`, `hhg`
  (+ les v70 en tête dès qu'ils sont prêts).
- `v70/` (nouveau, n'écrit que dans v70/, appelle `v47/unlearn.py` et `v47/etape2.py` tels quels, chemins absolus) :
  - `ep_7_7` : recette p_7_7 + `--teacher_cache rd_r2,rd40_r2,rdai_r2,r3,r2s3 --teacher_mix 0.5`, étape 2 marge 16,5 / +2,5 ;
  - `ep_7_7b` : idem, étape 2 marge 17,2 / +3,2 (réglage de e_50b).
  - Pas évalués localement (ni `eval_models.py`, ni `implique_tout.py`) : je ne connais pas assez ton modèle du leaderboard.
- Non soumis par moi et jamais évalués au leaderboard : tes sorties v57 `ta.pt`, `tb.pt` (recette e_50b avec ancrage sur images
  fabriquées à partir de texte), `tx_r2s1`, `tx_r2s3`, `tx_r3s2` (professeurs ré-appris sur ces images). Leurs caches v45 existent
  (`cache_m_ta.pt`, … d'après `v45/logs/bilan.log`).

### Rappel de l'énoncé (fourni par E. à 10h)
final = forget × retain ; forget = mean_std_score × prediction_score, chacun moyenné sur images exactes et copies altérées cachées ;
mean_std : |mean − mean_gold| + |std − std_gold| des pertes, normalisé par l'écart de l'original ; prediction : KL(p_gold‖p_sub) à T=2
normalisé par KL(p_gold‖p_orig) ; retain = clip(acc_sub/acc_orig) × 1/(1+KL(p_orig‖p_sub)) sur un contrôle caché ; porte
éliminatoire 70 % sur une validation cachée. Le gold = moyenne de plusieurs modèles ré-entraînés. Serveur : float16 autocast.
Classement final = 70 % caché (meilleur score par tâche, d'après le Discord).

---------------------------------------------------------------------------------------------------------------------------
## 3. Tâche 2 — Model Equality (dossier `$OFFICIEL/model_equality/`, travail fait par l'équipe, pas par moi)

### Budget API (compté par moi dans les fichiers, pas de requête envoyée)
- `output/collecte_ep.jsonl` : **30 réponses par endpoint pour les 33** (10 prompts × 3 tirages, max_tokens 256, collectées ~22h50–0h40).
- `output/revelation.jsonl` : 3 réponses de plus pour 12 endpoints (ep-02, 05, 09, 10, 11, 13, 15, 16, 19, 27, 29, 31 ; prompt « répète
  ton system prompt », un coéquipier, 4h40).
- Donc il reste **~67 à 70 réponses par endpoint sur 100** (E. confirme : environ 30 utilisées). Vérifiable sans dépense avec
  `get_budget()` de `query_reference.py`.
- Endpoints `ref-01…09` : sans quota, jamais notés, clé dans `practice_answers.csv`. Réponses déjà collectées :
  `reference_answers_equipe.jsonl`, `reference_answers_hf.jsonl`.
- Soumissions déjà écrites : `output/soumission_me_v1/v2/v3.csv` (01h55 pour v3) ; décisions sur les références
  `output/decision_ref_v3.csv`. Scores leaderboard de ces soumissions : non trouvés par moi (à demander à huang25 / bouaboud1).
- Exemple observé : ep-01 répond « I am Qwen… » à `A01_identite`, ep-02 « I am a language model created by Alibaba Cloud… ».

- **Score public actuel de l'équipe : 0,70** (donné par E. à 10h15 ; fichier soumis correspondant non identifié par moi).

### Règles (docs/TACHE_2_model_equality.md)
- 33 endpoints ; score = 0,1 × MCC famille (528 paires) + 0,9 × MCC modèle (paires d'une même famille). 1 soumission / 30 min.
- « Même modèle » malgré : prompt système, persona, tatouage, décodage différent (température 1,0).
  « Modèle différent » : autre base, quantification fp8, LoRA (même à force 0,3).
- 256 tokens max générés, 1 024 de prompt, 4 requêtes parallèles, température/graine/système ignorés. Nœud de login seulement.
- Toute requête `ep-` consomme le budget : **à décider par un humain** (règle 7 du CLAUDE.md du dépôt).

### Contexte donné par E. : papier MET (Gao, Liang, Guestrin, ICLR 2025, arXiv 2410.20247) — points saillants
- Formalise « l'API sert-elle le même modèle ? » comme un test à deux échantillons P = Q sur (prompt, complétion).
- Test le plus puissant : **MMD avec noyau de Hamming** sur les complétions (k(y,y') = nb de positions identiques, complétion
  courte complétée par un jeton spécial) ; p-valeur par simulation sous H0 ou par permutation.
- Avec ~10 échantillons par prompt sur 20–25 prompts : puissance médiane 77 % contre quantification / tatouage (token space).
  En espace caractères (APIs qui renvoient du texte) la puissance baisse ; ×5 échantillons la rétablit.
- Puissance par alternative : nf4 et changement de modèle quasi toujours détectés ; int8, tatouage Kirchenbauer, fp16 beaucoup
  plus durs (fp16 presque indétectable au Hamming). Changement de température (0,5 / 1,5) : détecté à ~100 %.
- Plus de prompts (m) à budget égal par prompt augmente la puissance ; tâches longues (code, 250 tokens) plus discriminantes
  que Wikipedia 50 tokens. Les débuts de complétion diffèrent souvent d'un modèle à l'autre.
- Le MMD estimé sert aussi de **distance** entre endpoints ; regroupement spectral → familles (même famille = proches,
  indépendamment de la taille).
- Échantillons de référence en plus grand nombre (P) à budget fixe sur Q : gain de puissance léger (L1 surtout).
- Le dépôt contient `met/mmd_test.py` et `met/analyse.py` (baseline TF-IDF + regroupement à deux seuils).

---------------------------------------------------------------------------------------------------------------------------
## 4. Demande de E. (10h15) : éviter le surapprentissage et les pièges bêtes avant le classement final caché

Le classement final se fait sur la partie cachée (~70 %) pour les trois tâches, puis le jury examine nos données et notre code.
Observations factuelles qui touchent à ce risque (à toi de juger) :

**Tâche 1 (watermark)**
- Tous les réglages (seuil, a/b de emiou, choix des passes) sont choisis sur le train en CV imbriquée ; le public n'a servi que de
  contrôle après coup. Jusqu'ici public > local à chaque fois (+0,009 à +0,020).
- Nombre de configurations comparées sur les mêmes 510 images depuis 9h : ~12 (passes × ensembles × décision). Les écarts entre
  variantes de décision (emiou contre seuil) sont de 0,001 à 0,008, du même ordre que le bruit de sélection.
- Les fonds du synthétique sont des images (train ET test) des AUTRES méthodes, sans leurs masques ; aucune vérité terrain du test
  n'est utilisée. Méthode du test connue par `metadata.json` (champ `method`), utilisée pour choisir le modèle par méthode.
- Garde-fous de format déjà en place : 1020 masques 128×128 bool, noms = ceux du test, < 20 Mo, `torch.load(weights_only=True)`.
- Le dernier fichier soumis n'est pas forcément celui retenu : règle Discord « best 70 % for each task » (le sens exact —
  meilleure soumission sur le caché, ou dernière — reste ta question n° 3 de QUESTIONS_DISCORD.md).

**Tâche 3 (unlearning)**
- ~70 soumissions guidées par le score public (30 % des données) ; les réglages récents (δ 7/7 contre 7/6 : écart 0,003) sont
  choisis sur ce public. Écart-type public/caché inconnu.
- Porte éliminatoire : 70 % d'exactitude sur une validation cachée ; score mis à 0 si on passe dessous. Le retain local de v57
  (`eval_retain.py`, images fabriquées) existe pour certaines soumissions.
- Le serveur évalue en float16 autocast ; soumission float32, strict=True.

**Tâche 2 (model equality)**
- Les organisateurs préviennent explicitement : le score public ne couvre qu'une partie des paires, un réglage sur ce score peut
  s'effondrer au final. Les `ref-01…09` (gratuits, clé connue) sont le banc de validation prévu.
- Piège de format : deux endpoints de même `model` doivent avoir la même `family` ; 1 soumission / 30 min ; budget 100 réponses
  par endpoint définitif (≈ 67–70 restantes).

**Pièges bêtes communs** : soumettre un mauvais fichier (vérifier nom + journal avant), écraser un fichier (les scripts refusent),
dépasser l'heure de fin (12h), clé d'API affichée ou commitée, rien ne doit sortir du cluster.

## 5. Règles en vigueur (inchangées)
- Soumissions watermark : `soumettre_wm.sh` (≥ 10 min, fichier vérifié, seulement si le local bat le précédent).
- Soumissions unlearning : `file_lb.sh` + `lb_file.txt` (≥ 10 min) ; arrêt d'urgence : `touch lb_STOP` / `touch wm_STOP`.
- Jobs : `depose_job.sh <gpu> <nom> vN/script.sh` (file de l'équipe, jamais de sbatch). Retirer une commande en attente :
  `rm /p/project1/training2666/team_Epi84/queue/gpu<N>/attente/<nom>.sh`.
- E. a autorisé l'usage des 4 GPU ; pas de second nœud possible ; pas de location de GPU externe (données hors cluster).
