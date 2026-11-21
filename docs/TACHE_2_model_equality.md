# Tâche 2 : Model Equality (quels endpoints servent le même modèle ?)

Dossier officiel : `$OFFICIEL/model_equality/` (venv : `--venv met`). Fichiers fournis : `task_template.py`, `query_reference.py`, `submission.py`, `score_practice.py`, `practice_answers.csv`.

## Le problème
- **33 endpoints** `ep-01` à `ep-33`. Pour chacun, trouvez sa **famille** (même modèle de base) et, pour chaque paire d'une même famille, s'ils servent **le même modèle**.
- **Différent** (autre identité) : autre modèle de base, autre **quantification** (ex. fp8), **adaptateur LoRA** (même à force 0,3).
- **Même modèle** malgré les apparences : **prompt système**, **persona**, **tatouage**, autre **stratégie de décodage** (ex. échantillonnage à température 1,0).
- **Score** : `0,1 × MCC famille (528 paires) + 0,9 × MCC modèle (paires d'une même famille)`, MCC négatif ramené à 0. La famille seule rapporte au plus 0,1.
- **Soumission** : CSV `endpoint,family,model`. Deux endpoints de même label `model` doivent avoir le même label `family`. **Une soumission toutes les 30 minutes**. Le score affiché ne porte que sur une partie des paires ; le classement final utilise les autres.

## L'API (dans `task_template.py`)
- `POST http://34.147.229.55/33-model-equality/v1/chat/completions`, en-tête `X-API-Key: <clé d'équipe>`. JSON : `model` (ex. `"ep-07"`), `max_tokens`, `messages` (un seul message `user`).
- **Budget : 100 réponses par endpoint pour toute la compétition**, 256 tokens générés au plus, 1 024 tokens de prompt, 4 requêtes en parallèle. Température, graine et messages système sont ignorés ou refusés.
- Les réponses s'ajoutent à `answers.jsonl` ; relancer reprend là où on s'était arrêté.
- **Endpoints de référence `ref-01` à `ref-09` : sans quota, jamais notés**, avec la réponse dans `practice_answers.csv`. Ils utilisent les mêmes « déguisements et pièges » que les vrais :

| ref | modèle | piège |
|---|---|---|
| 01, 03, 05, 06, 08 | S | normal, prompt système, persona, tatouage, température 1,0 |
| 02 | T1 | quantifié fp8 |
| 04, 07, 09 | T3, T4, T2 | LoRA à force 0,7, 1,5 et 0,3 |

## Où ça tourne
- **Sur le nœud de login**, pas dans les files GPU : l'API demande internet, et les nœuds de calcul n'en ont pas. Le travail est léger (requêtes HTTP + analyse CPU).
- **La clé d'API ne va jamais dans git.** Mettez-la dans une variable : remplacez dans les copies officielles `API_KEY = "YOUR_API_KEY_HERE"` par `API_KEY = os.environ["TEAM_API_KEY"]` (avec `import os`), puis `read -rs TEAM_API_KEY && export TEAM_API_KEY`.

## Le cycle de travail
1. **Mettre au point sur les références, gratuitement.** Écrivez des prompts dans `reference_prompts.txt`, lancez `python query_reference.py`, puis :
   ```bash
   cd $OFFICIEL/model_equality && source .venv/bin/activate
   python <votre clone>/met/analyse.py --answers reference_answers.jsonl \
       --prefix ref- --key practice_answers.csv --show
   ```
   `analyse.py` (baseline : similarité TF-IDF de caractères entre réponses aux mêmes prompts, puis regroupement hiérarchique à deux seuils) cherche les meilleurs seuils sur les 9 références. Le score de `score_practice.py` porte sur le modèle seul.
2. **Choisir le plan de requêtes avant de dépenser le budget** : combien de prompts, combien de répétitions (ex. 40 prompts × 2, en gardant 20 réponses de réserve par endpoint). Décision d'équipe.
3. **Interroger les vrais endpoints** avec `task_template.py` (`QUERY = True`, vos prompts dans `prompts.txt`) : un seul membre lance, pour ne pas dépasser le budget.
4. **Produire la soumission** : `python met/analyse.py --answers answers.jsonl --prefix ep- --family_thr X --model_thr Y --out submission.csv`, puis `python submission.py` (30 min entre deux).

## Pistes pour battre la baseline
- **Prompts discriminants** : réponses longues et factuelles (connaissances, code, raisonnement), consignes de format précises. Le style change avec le prompt système et la persona, pas les connaissances ni les tics du modèle.
- **Les pièges « même modèle »** (persona, prompt système, tatouage, température) changent beaucoup le texte : une similarité de surface les sépare à tort. Testez des caractéristiques robustes au style : contenu factuel, nombres, structure, longueur relative.
- **Les pièges « modèle différent »** (fp8, LoRA 0,3) changent peu le texte : il faut des prompts où de petites différences de poids changent la réponse (questions difficiles, longues générations).
- Le test MMD du papier (`met/mmd_test.py`, fiches 2 et 7) s'applique si vous avez plusieurs réponses par prompt (répétitions).
