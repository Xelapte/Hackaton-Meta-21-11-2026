# Tâche SCRUB (GPU 1)

> **Contexte du papier, pas la tâche réelle.** La tâche du hackathon est décrite dans `docs/TACHE_3_unlearning.md`.

Le papier : *Towards Unbounded Machine Unlearning* (SCRUB), arXiv 2302.09880. Code officiel : github.com/Meghdad92/SCRUB, cloné dans `$DIR/repos/SCRUB`. Il est fait surtout de notebooks et importe wandb : avec `WANDB_MODE=disabled`, il fonctionne sans internet.

## Le principe

- Un modèle est entraîné sur D. On veut qu'il **oublie** un sous-ensemble D_f (le forget set) tout en restant bon sur D_r (le retain set), sans tout réentraîner.
- **Le schéma enseignant–élève** : l'enseignant est le modèle original, figé. L'élève part d'une copie de l'enseignant. On note d(x) = KL(p_enseignant(x) ‖ p_élève(x)).
- **L'objectif** : min α·KL sur D_r + γ·CE sur D_r − KL sur D_f. Les étapes alternent :
  - l'étape « max » éloigne l'élève de l'enseignant sur D_f ;
  - l'étape « min » le garde proche sur D_r.
  
  On fait `max_steps` étapes max au début, puis uniquement des étapes min.
- **SCRUB+R (rembobinage)** : dans le scénario vie privée, trop oublier se repère aussi (effet Streisand). On revient au point de sauvegarde dont l'erreur sur D_f est la plus proche de celle d'un modèle de référence.
- **Les trois scénarios du papier :**
  - **RB**, retirer un biais : maximiser l'erreur sur D_f ;
  - **RC**, résoudre une confusion ;
  - **UP**, vie privée : imiter le modèle réentraîné sans D_f, avec une attaque d'appartenance proche de 50 %.
- **L'attaque d'appartenance (MIA) de base** : une régression logistique sur les pertes, bornées à ±400, avec validation croisée en 5 plis. Elle compare D_f à des exemples de test de la même classe. LiRA est plus forte mais demande des centaines de modèles fantômes.

## Le code

| Fichier | Rôle |
|---|---|
| `scrub/cifar.py` | `resnet18_cifar` (conv1 3×3, pas de maxpool), `load_cifar(root, augment, fake)` avec `download=False`, `split_forget`, `loader`, `train` (SGD 0,1, cosinus, bf16), `predict`, `error`. |
| `scrub/scrub.py` | `kl_teacher_student`, `scrub(teacher, forget_loader, retain_loader, max_steps, n_epochs, alpha, gamma, lr)` qui renvoie `(élève, points de sauvegarde)`, `rewind`, `basic_mia`. |
| `scrub/train_original.py` | Le modèle original, ou le réentraînement de référence (`--retrain --mode class/selective`). Enregistre le modèle **à la fin seulement**. |
| `scrub/run_scrub.py` | Évalue l'original, la référence, SCRUB et SCRUB+R : erreurs test, retain et forget, et MIA. Score écrit = **erreur sur D_f**. |
| `scrub/run.sh` | Original et réentraînement en parallèle (sautés s'ils existent déjà), puis SCRUB en mode classe et en mode sélectif. |

Le test sans GPU : `python scrub/train_original.py --fake --epochs 1 --out /tmp/t.pt`, puis `python scrub/run_scrub.py --fake --original /tmp/t.pt --n_epochs 2 --max_steps 1 --bs_forget 16 --bs_retain 32`.

## Les réglages du papier (tableau 3)

- **Oubli d'une classe :** `--max_steps 2 --n_epochs 3 --bs_forget 512 --bs_retain 128`.
- **Oubli sélectif de 100 exemples :** `--max_steps 5 --n_epochs 5 --bs_forget 16 --bs_retain 64`.

## Les pièges

- **Le score par défaut (l'erreur sur D_f) ne convient qu'au scénario RB.** Pour la vie privée, la cible est une erreur sur D_f et une MIA **proches de celles du réentraînement**, pas les plus hautes possibles. Change l'appel à `log_score` dans `run_scrub.py` selon la métrique des tests.
- Le `scrub()` du kit utilise Adam avec un pas d'apprentissage constant. Le papier le fait décroître.
- La MIA de `run_scrub.py` réserve la moitié des images de test de la classe visée au rembobinage, et l'autre moitié à l'attaque.
- L'erreur sur D_r est mesurée sur 5 000 exemples tirés au hasard, pour aller vite.
- Les chargeurs de données utilisent 8 processus chacun. Avec plusieurs runs en parallèle, c'est le processeur qui sature.
- CIFAR est chargé avec `download=False` : les données doivent déjà être dans `$DIR/data`, téléchargées sur le nœud de login.

## Les leviers

`alpha`, `gamma`, `lr`, `max_steps`, `n_epochs`, les tailles de batch, le rembobinage, la qualité du modèle original (nombre d'époques, augmentation de données), et un balayage de 3 ou 4 runs en parallèle sur le GPU 1 : un ResNet-18 CIFAR n'utilise que quelques Go.
