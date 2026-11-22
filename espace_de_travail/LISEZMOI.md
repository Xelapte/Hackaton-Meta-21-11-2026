# Espace de travail — tâches 1, 2, 3 et pitch

European AI Championship (Paris, 21–22 novembre 2026), équipe Epi84. Ce dossier est une copie des fichiers texte de
l'espace de travail du cluster (`$OFFICIEL/unlearning/`) : code, scripts de lancement, notes et petits résultats.
Chaque idée a son dossier `vN/` avec un `NOTES.md` (idée, paramètres, mesures locales, score au classement).

## Scores finaux (classement public, à la clôture le 22/11 à 12h)

| Tâche | Score public | Fichier soumis | Où lire la méthode |
|---|---|---|---|
| 1. Localisation du tatouage | 0,7621 (local, validation croisée : 0,7506) | `v62/subs/v62_m.pt` | `v61/NOTES.md`, `v62/NOTES.md` |
| 2. Model Equality | 0,85 | `v80/sub_B.csv` | `v80/NOTES.md` |
| 3. Unlearning | 0,3236 | `v70/subs/p77_m172.pt` | `JOURNAL.md`, `v70/NOTES.md` |

Le classement final se fait sur une partie cachée des tests : ces scores publics ne sont qu'une indication.

## Où est quoi

| Dossier | Contenu |
|---|---|
| `v1/` … `v57/`, `v70/` | Unlearning : professeur/élève sans jeu de retain, cible moyennée sur des vues, images « fragiles ». Fil conducteur dans `JOURNAL.md` et `RESUME_COMPLET.md`. |
| `v60/` … `v63/` | Tatouage : segmentation apprise sur des données synthétiques fabriquées avec les embedders officiels (`v61/seg_synth_new.py`), passes longues, TTA miroir, ensembles et décision finale (`v62/decide62.py`). |
| `v80/` | Model Equality : collecte ciblée (`collecte2.py`, `coll4.py`, `ident5.py`), tests de vraisemblance sous le modèle de base local (`score2.py`, `score3.py`, `score4.py`), soumissions `sub_*.csv`. |
| `v81_pitch/` | Présentation : `pitch_Epi84.pptx`, texte à lire `pitch_script.md`, générateur `build_deck.py`. |
| racine | Scripts de soumission et de file d'attente (`soumettre_lb.sh`, `soumettre_wm.sh`, `file_lb.sh`, `depose_job.sh`), journaux des scores (`lb_journal.txt`), consignes données à l'agent (`CONSIGNES_AGENT.md`). |

## Ce qui n'est pas ici

Les poids (`.pt`), les données (`.npz`, `.npy`), les réponses des API (`.jsonl`), les journaux d'exécution (`.log`) et les
environnements Python restent sur le cluster, conformément aux règles du dépôt. Les fichiers du kit des organisateurs
(`task_template.py`, `submission.py`) ne sont pas recopiés.

## Anonymisation

Les prénoms et l'identifiant de cluster de l'auteur sont remplacés (`E.`, `LOGIN`, « l'équipe », `depot-coequipier`).
Pour relancer un script sur le cluster, il faut donc rétablir ces chemins. Aucune clé d'API ni jeton ne figure dans ces
fichiers : les scripts lisent la clé dans un fichier local qui n'est pas dans le dépôt.
