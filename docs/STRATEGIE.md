# Stratégie de score et planning

## Les principes

1. **Un score sur chaque tâche, vite.** Commence par une solution simple qui passe les tests des trois tâches, avant d'en perfectionner une. Si le classement additionne les tâches, un zéro coûte plus cher que n'importe quelle optimisation.
2. **Lis la métrique avant le papier.** L'énoncé et le code de score des tests disent ce qui compte. Note tout ce qui diffère du cadre du papier : données, attaques, budget, métrique.
3. **Une commande par tâche reproduit le score** et écrit une ligne dans le journal partagé (`$DIR/scores.csv`).
4. **Investis là où le gain par heure est le plus grand.** Toutes les deux à trois heures, regarde le tableau des scores et réattribue le GPU 3 (joker) et les personnes.

## Le planning

| Quand | Phase | Ce qu'on fait |
|---|---|---|
| Sam. 12h–13h | Mise en place | Lire les énoncés et la métrique, le coordinateur lance `slurm/test.sh` puis `slurm/noeud.sh`, confirmer les responsables. |
| 13h–16h | Baselines | **Objectif : un score sur les trois tâches avant 16h.** |
| 16h–22h | Améliorations | Les leviers qui rapportent le plus ; point d'équipe à 19h pour réattribuer le joker. |
| 22h–7h | Nuit | Les longs calculs partent avant de dormir : génération, balayages d'hyperparamètres. Une personne vérifie vers 3h. |
| Dim. 7h–10h | Derniers gains | Exploiter les résultats de la nuit et choisir les meilleures configurations. |
| 10h | **Gel du code** | Plus aucune nouvelle idée. |
| 10h–11h30 | Runs finaux | Lancer les versions retenues sur les tests et vérifier les soumissions. |
| 11h30–12h | Marge | Imprévus, slides. |
| 12h–15h | Pitch et jury | Démo sur le cluster, ou résultats précalculés. |

## Les décisions déjà prises

- **Les GPU :** 0 pour MET, 1 pour SCRUB, 2 pour Pixel Seal, et 3 pour le joker puis la démo.
- **Pas d'interface web de synchronisation.** On utilise `scores.csv`, `python -m common.scores`, `common.dashboard`, un tableau de tâches et des points d'équipe.
- **La démo du pitch** se prépare seulement si les scores sont assurés, dimanche entre 7h et 10h.

## Le fil conducteur du pitch

Les trois tâches posent la même question : comment **vérifier de l'extérieur** ce qu'on affirme sur un système d'IA ?

- **Pixel Seal :** « cette image a bien été générée par notre modèle ».
- **MET :** « cette API sert bien le modèle annoncé ».
- **SCRUB :** « ces données ont bien été oubliées ».

Pour chaque tâche, le jury veut entendre ce que l'équipe a changé par rapport au papier, et pourquoi ce changement améliore le score.
