# Contexte
Hackathon European AI Championship (CISPA). Tâche : machine unlearning d'un ViT-B/16
(timm vit_base_patch16_224, 100 classes). Lire task_template.py pour la métrique exacte.
Cible = imiter un modèle "gold" réentraîné sans les 3500 images de forget_images.npz
(PAS une accuracy nulle), tout en restant proche du modèle original ailleurs.
Seuil éliminatoire : 70 % d'accuracy sur une validation cachée.

# Règles absolues
- Dossier PARTAGÉ avec l'équipe : ne JAMAIS supprimer, déplacer ou écraser un fichier existant
  hors du dossier de version en cours (v1/, v2/...).
- Ne JAMAIS modifier : model.pt, forget_images.npz, task_template.py, submission.py, ../.venv,
  ni rien dans les autres tâches.
- Ne JAMAIS installer de paquet (pip/uv) dans .venv sans me demander.
- Nœud de login = pas de calcul lourd. GPU : UNIQUEMENT via la file de l équipe (GPU 1 en priorité ; les GPU 0, 2, 3 aussi s ils sont libres — autorisé par E. le 22/11 à 1h37 ; slurm/soumettre.sh, lancé depuis membres/LOGIN), jamais de sbatch.
  (--account=training2666 --partition=dc-gpu --reservation=cispahack). Un job = un nœud de 4 A100.
- Soumissions (autorisé par E. le 21/11 à ~21h15, pendant son absence) : Claude peut lancer submission.py
  lui-même, une soumission toutes les 10 min au plus, dans l'ordre convenu puis selon les résultats.
  Commande : TEAM_API_KEY=$(cat ~/.hack_api_key) SUBMISSION_FILE=<fichier> .venv/bin/python submission.py
  Ne jamais afficher, copier ni écrire la clé ailleurs. Vérifier chaque fichier avant (strict=True, float32, fini).
  Noter chaque score dans le NOTES.md de la version. Ne jamais modifier submission.py.
- Watermark (épreuve « Localize Image Watermark », dossier ../Hackathon_Watermark_Localisation_Task) : tâche reprise aussi par
  E. le 22/11 à ~9h05, jusqu'à nouvel ordre. Claude et ses agents LISENT ce dossier (et ../depot-coequipier) sans jamais rien y
  modifier, et n'écrivent que dans unlearning/v60/, v61/… (un NOTES.md par dossier, avec une section « Rapport pour l'équipe » :
  ce qui a été réutilisé, créé, changé, mesuré). Mêmes règles GPU (file de l'équipe, jamais de sbatch).
  Soumissions watermark (autorisé par E. le 22/11 à ~9h06) : Claude peut soumettre lui-même avec la fonction submit() de
  submission_template.py (fichier inchangé), via unlearning/soumettre_wm.sh, au plus une toutes les 10 min, et seulement si le
  score local (mIoU sur le train, mesuré de la même façon) bat l'approche précédente. Même clé ~/.hack_api_key, jamais affichée,
  copiée ni écrite ailleurs. Vérifier le fichier avant (1020 masques 128×128 en 0/1, noms complets et uniques, < 20 Mo).
  Noter chaque score dans le NOTES.md du dossier.
- Ne rien envoyer d'autre sur internet (pas de upload, pas de git push) : seules exceptions = submission.py (unlearning) et
  submit() de submission_template.py (watermark) vers le serveur du hackathon.
- Aucune collaboration avec d'autres équipes.

# Méthode de travail
- Chaque nouvelle idée = nouveau dossier : cp -r vN vN+1, puis modifier la copie.
  Ne jamais modifier une version déjà lancée.
- Sorties dans vN/subs/ et vN/logs/.
- Soumission : state_dict float32, strict=True avec build_model(), pas de NaN/Inf, < 450 Mo.
- Jobs GPU : pendant l'absence de E. (autorisé le 21/11 à ~21h15), Claude peut déposer lui-même des jobs dans la file
  de l'équipe (slurm/soumettre.sh, GPU libre), à condition qu'ils n'écrivent que dans leur dossier vN/ et qu'ils soient
  décrits dans le NOTES.md de la version. Jamais de sbatch direct.
- Tenir à jour NOTES.md dans le dossier de version : idée, paramètres, métriques locales,
  score leaderboard. (Sert pour la présentation finale.)
- Expliquer le code simplement : je dois pouvoir le présenter au jury.

# Contexte détaillé
Voir CONTEXTE.md (même dossier).
