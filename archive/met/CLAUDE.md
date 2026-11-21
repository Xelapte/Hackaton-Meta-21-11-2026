# Tâche Model Equality Testing (GPU 0)

> **Contexte du papier, pas la tâche réelle.** La tâche du hackathon est décrite dans `docs/TACHE_2_model_equality.md`.

Le papier : *Model Equality Testing: Which model is this API serving?*, arXiv 2410.20247. Code officiel : github.com/i-gao/model-equality-testing, cloné dans `$DIR/repos/model-equality-testing`.

## Le principe

- La question : une API sert-elle vraiment le modèle annoncé, ou une version modifiée (quantifiée, tatouée, ou un autre modèle) ?
- C'est un **test à deux échantillons** :
  - P = des réponses du modèle de référence, généré soi-même ;
  - Q = des réponses de l'API, pour les mêmes prompts ;
  - H0 : P et Q ont la même distribution.
- **La statistique** : la MMD avec un **noyau de Hamming**, qui mesure la proportion de positions où deux réponses ont le même token. Elle se calcule **prompt par prompt**, sans biais, puis se moyenne.
- **La p-value** vient d'un test de permutation : on mélange les étiquettes P et Q **à l'intérieur de chaque prompt**, B fois. On rejette H0 si p < α.
- **L'hypothèse composite** : la référence peut être en fp32 ou en fp16. On ne rejette que si le test rejette contre les deux.
- **Plusieurs APIs testées** : appliquer une correction de Bonferroni.
- **Les réglages du papier** : 10 réponses par prompt, 50 tokens, température 1. Avec Llama-3 8B, la puissance vaut 1,00 contre le tatouage, 0,93 contre nf4 et 0,79 contre int8.
- **En caractères** (réponses d'API en texte) : passer à 50 réponses par prompt remonte la puissance à environ 1,00.

## Le code

| Fichier | Rôle |
|---|---|
| `met/mmd_test.py` | `pad`, `hamming_gram`, `mmd_prompt` (sans biais) et `mmd_test(DP, DQ, B, seed)`. Testé : 5 % de faux positifs sous H0. |
| `met/generate_vllm.py` | Génère avec vLLM : `--dtype float32/bfloat16/float16`, `--seed` obligatoire, `--gpu_mem`. Sortie `.npz` avec un padding de −1. |
| `met/generate_hf.py` | Génère avec transformers : `--variant int8`, `nf4`, `watermark` (biais 2,5, contexte 1, lefthash), `fp16` ou `fp32`. |
| `met/compare.py` | Compare deux fichiers `.npz`, ou `.json` de textes comparés en caractères. Écrit **la p-value** comme score : plus bas = différence détectée. |
| `met/run.sh` | P et P' en fp32 (graines 1 et 2), puis Q pour nf4, int8 et watermark, puis les comparaisons. |
| `met/prompts_exemple.jsonl` | Le format des prompts : `{"id": "...", "text": "..."}` sur chaque ligne. |

Le meilleur score se lit avec `python -m common.scores --bas met`.

## Les pièges

- **Une graine différente pour chaque fichier.** Avec la même graine, on obtient les mêmes réponses, et le test n'a plus de sens.
- **Le contrôle P contre P'** doit donner p > α la plupart du temps. Sinon, le problème vient du pipeline (graines, gabarit de chat, décodage), pas des variantes.
- **Le décodage** est imposé : température 1, top-p 1, pas de top-k. Le `generation_config.json` des modèles instruct fixe souvent d'autres valeurs.
- **Le gabarit de chat** est appliqué en texte, puis la tokenisation se fait avec `add_special_tokens=False`, pour éviter un double BOS. Le gabarit de Llama-3.1 insère la date du jour.
- **Le tatouage** de transformers a un biais de 2,0 par défaut. Le papier utilise 2,5, que le code fixe explicitement.
- **Un 8B en fp32 prend environ 32 Go** de poids. vLLM réserve 90 % du GPU : rien d'autre ne doit tourner sur ce GPU pendant ce temps. Sur un GPU de 40 Go, la place pour le cache est juste : monte `--gpu_mem` à 0,95 si besoin.
- **Les modèles soumis à accès** : Llama demande l'approbation de Meta, Mistral demande d'accepter le partage de ses coordonnées. Le téléchargement (`hf download`) se fait uniquement sur le nœud de login. Sur le nœud GPU, `HF_HUB_OFFLINE=1`.

## Les leviers

Plus de prompts, plus de réponses par prompt, des réponses plus longues, une référence en fp16 si l'hypothèse composite le permet. Pour comparer des API, le texte en caractères est plus robuste que les tokens. Côté calcul, utilise le GPU 3 pour une deuxième instance indépendante : un 8B tient sur un GPU, donc pas besoin de répartir un modèle sur deux GPU.

## À adapter dès qu'on connaît l'énoncé

Le format des réponses à tester, la métrique (puissance ? exactitude de détection ? taux de faux positifs imposé ?) et le budget de requêtes. Convertis les données des tests dans le format de `compare.py`, puis règle le nombre d'échantillons en fonction du budget.
