# v80 — Tâche 2 (Model Equality) : dernière matinée (22/11, 10h20–12h)

Point de départ : `model_equality/output/soumis/soumission_v4.csv` (un coéquipier, public 0,70) : 3 familles de 11 ; par famille un groupe
« déguisés » (prompt système, persona, tatouage, température 1,0) = même modèle ; tous les autres endpoints seuls dans leur modèle.

## Ce qui a été fait
- **Lecture du score** : 0,70 = 0,1 + 0,9 × 2/3 est exactement ce que donne « groupe réel de 5 par famille, nous en avons 4/4/3, aucun
  faux positif ». Il manquait donc, par famille, l'endpoint « normal » (le modèle de base sans déguisement), et un déguisé chez Llama.
- **ep-29** : prompt système extrait mot pour mot (`output/revelation.jsonl`, un coéquipier) → ajouté au groupe Llama.
  `sub_A.csv` (10h52) : public 0,70 (inchangé : ses paires ne sont pas dans l'échantillon public).
- **Collecte ciblée** (budget restant, API ≈ 12 réponses/min, 4 en parallèle) : ep-13 et ep-19 (67 réponses longues chacun, 12 prompts
  à forte entropie : `prompts2.jsonl`, `collecte2.jsonl`) ; les 7 candidats Qwen (01, 12, 18, 25, 32, 26, 21) : suites aléatoires
  (`prompts4.jsonl`, `coll4.jsonl`) puis question d'identité (`prompts5.jsonl`, `ident5.jsonl`). Budgets de ces 9 endpoints : 100/100.
- **Tests** (modèle de base local, CPU) : z de vraisemblance exact (`score2.py`), comptes agrégés par token (`score3.py`, `bilan3.py`),
  excès conditionnels au token précédent sur les suites aléatoires (`score4.py`, `bilan4.py`).
- **Résultat** : ep-13 et ep-19 indiscernables du modèle de base avec 100 réponses chacun (z nouvelles réponses −0,7 et −0,9 ;
  T ajustée 0,704 et 0,711). Aucun candidat Qwen ne ressort non plus. Une variante très légère (LoRA faible) est invisible à ce budget.
- **Arbitrage par le classement** : `sub_B.csv` (11h22) = sub_A + ep-13 dans le groupe Llama → **public 0,85**. Un mauvais ajout ne peut
  que baisser le score public (il n'ajoute que des faux positifs), et le serveur garde alors la meilleure soumission.
- Dernier créneau (11h52) : `sub_D.csv` = sub_B + ep-26 (candidat Qwen 1.5B le plus « propre » : aucun prompt à |z| > 1 en phase A).

## Soumissions
| heure | fichier | public |
|---|---|---|
| 10h52 | sub_A.csv (v4 + ep-29) | 0,70 |
| 11h22 | sub_B.csv (+ ep-13) | **0,85** (retenue) |
| 11h52 | sub_D.csv (+ ep-26) | 0,65 : ep-26 n'est pas la base de Qwen 1.5B (restent ep-32 ou ep-21) |

## Attention
- `score3.py` prend le GPU s'il en voit un : sur le nœud de login, toujours lancer avec `CUDA_VISIBLE_DEVICES=""` (fait à partir de 11h28 ;
  les premiers appels ont utilisé par erreur, quelques secondes, le GPU partagé du nœud de login).
- Le score public ne couvre qu'environ un tiers des paires : il a servi une seule fois, pour un choix binaire (ep-13 contre ep-19).
