# v81_pitch — le pitch de 6 minutes (Epi84)

- `pitch_script.md` : le texte à lire à voix haute, en français puis en anglais, avec les marques `[Diapo N]`.
- `pitch_Epi84.pptx` : 8 diapositives 16:9, texte en anglais ; les notes de l'orateur sont la version française du script.
- `build_deck.py` : le script qui fabrique le fichier. Textes des diapositives et chiffres : bloc `DATA` en haut.
- `pylib/` : python-pptx installé ici seulement (aucun `.venv` touché). Roues pour Python 3.9 : utiliser `/usr/bin/python3`.

## Régénérer (2 secondes, nœud de login, pas de GPU)

```bash
cd /p/scratch/training2666/huang25/Epi84/unlearning/v81_pitch
/usr/bin/python3 build_deck.py          # réécrit pitch_Epi84.pptx
```

Le script relit le fichier écrit et affiche : les titres, le nombre de mots et la durée estimée du script par
diapositive (150 mots/min, réglable avec `--debit`), les trois scores affichés, et un avertissement « DÉBORDE »
si un texte est trop long pour sa boîte (estimation).

## Mettre à jour un score avant midi

Dans `DATA`, chaque tâche (`task1`, `task2`, `task3`) commence par un bloc `>>> SCORES À METTRE À JOUR` :

| Champ | Rôle |
|---|---|
| `steps` | Les barres du graphique : `[étiquette, score public LB]`. Ajouter une ligne pour un nouveau score. |
| `headline` | Laisser `None` : le grand chiffre = le meilleur score de `steps` (diapositive de résultat, titre, conclusion). Mettre un texte pour forcer une valeur. |
| `headline_caption` | La ligne sous le grand chiffre : **changer l'heure** (« public LB (11:23) »). |
| `result_lines` | Les trois lignes courtes à côté du graphique (score local, limite, etc.). |

Le script parlé contient ses propres chiffres : après un changement de score, corriger aussi `pitch_script.md`
(français et anglais), puis relancer `build_deck.py` pour que les notes suivent.

## Changer le texte parlé

Modifier `pitch_script.md`. Les notes de la diapositive N = le texte français placé sous `[Diapo N]`
(les titres `#` sont ignorés ; la partie après `## English version` n'est pas lue).

## Ce qui est mesuré, ce qui ne l'est pas

- Tous les scores affichés viennent du leaderboard public ; le classement final se fait sur une partie cachée.
- Tâche 1 : les heures des 6e et 7e soumissions (0,7363 et 0,7400) ne sont pas connues du script : étiquettes
  « 6th run », « 7th run », à remplacer par l'heure si on veut.
- Tâche 2 : ce que la v4 (0,70) a changé n'est pas détaillé, ni sur la diapositive ni dans le script parlé.
- Le fichier n'a pas été rendu en image par PowerPoint ou LibreOffice (absents du nœud de login) : **l'ouvrir une
  fois dans PowerPoint avant de présenter**. Polices : Georgia (titres) et Calibri (texte). Les graphiques sont
  natifs (clic droit, « Modifier les données »).
