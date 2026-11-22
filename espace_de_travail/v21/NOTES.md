# v21 — Maquette réaliste de l'épreuve

Suite de v20 (voir v20/NOTES.md). Défaut de v20 : petit réseau entraîné de zéro, 33 % sur le contrôle → régime irréaliste.
Ici : les 8 premiers blocs du vrai modèle sont gardés (figés), les 4 derniers + la tête sont réentraînés de zéro sur nos
images. « Original maquette » + 3 « gold maquette » + formule officielle. Sert à savoir quelle note (moyenne/dispersion,
prédiction, retain) nous limite, sur quelles images (classes entières ou partielles), et quelle recette gagne.
Résultats : voir logs/replica2.log (résumé à ajouter ici).
