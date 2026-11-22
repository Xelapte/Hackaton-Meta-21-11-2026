# Pitch Epi84 — script à lire à voix haute (6 minutes maximum)

Repères : ouverture 15 s, puis 1 min 45 par tâche, conclusion 20 s. Les marques `[Diapo N]` disent quelle
diapositive est à l'écran. Version française d'abord, version anglaise ensuite (même contenu).
`build_deck.py` recopie la version française dans les notes de chaque diapositive : après une retouche ici,
relancer `/usr/bin/python3 build_deck.py`.

## Version française

### Ouverture (15 s)

[Diapo 1]

Bonjour, nous sommes l'équipe Epi84. Nous avons travaillé sur trois tâches : trouver où se cache un tatouage dans une image, reconnaître quel modèle répond derrière une API, et faire oublier des images à un modèle. Une même idée nous a guidés : mesurer honnêtement.

### Tâche 1 — localiser le tatouage (1 min 45)

[Diapo 2]

Première tâche. On nous donne une image, et il faut dire, pixel par pixel, où elle porte un tatouage invisible. Il y a trois méthodes de tatouage, et seulement 170 images d'exemple pour chacune.

Notre idée : fabriquer nous-mêmes les exemples. C'est comme avoir le tampon officiel. Si c'est nous qui tamponnons la feuille, nous savons exactement où est la marque.

Concrètement, on prend une image qui ne porte pas le tatouage étudié. On choisit une zone, avec une forme qui ressemble aux vrais exemples. On y pose le tatouage avec l'outil officiel des organisateurs. On obtient une image et sa réponse exacte, autant de fois qu'on veut. Un réseau de segmentation, c'est-à-dire un réseau qui colorie les pixels tatoués, apprend sur ces images fabriquées. Ensuite, on l'ajuste sur les vraies.

[Diapo 3]

Le résultat. Hier soir, nous étions à 0,50 sur le classement public. Ce matin : 0,65, puis 0,73, et maintenant 0,76. Ce qui a le plus payé est très simple : entraîner plus longtemps sur nos images fabriquées. Sur StegaStamp, notre score local est passé de 0,65 à 0,84.

Tous nos réglages sont choisis sur des images mises de côté, que le modèle n'a pas vues à l'entraînement, jamais sur le classement. À chaque soumission, le score public est resté à environ 0,02 près de notre score local.

La leçon : nos idées plus savantes n'ont rien apporté. De bonnes données et du temps d'entraînement, voilà ce qui a marché.

### Tâche 2 — quel modèle derrière l'API (1 min 45)

[Diapo 4]

Deuxième tâche. Il fallait dire quelles API cachent le même modèle. Le piège : un déguisement, que ce soit un tatouage, une consigne cachée ou la température, garde le même modèle mais change beaucoup le texte, alors qu'une quantification change le modèle mais presque pas le texte. Comparer les réponses ne marche donc pas.

Sur les endpoints de pratique, on a vu la structure : la base et ses quatre déguisements forment un groupe, les variantes sont seules. Et le score punit plus une fausse fusion qu'un oubli.

Alors on a changé de question : non pas « qui se ressemble ? », mais « qui porte une preuve de déguisement ? ». Trois traces : des réponses identiques pour le tatouage, des mots rares pour la température, une fausse identité pour la consigne cachée. Sans preuve, l'endpoint reste seul.

[Diapo 5]

Résultat : quatre déguisements dans chaque famille, comme dans la pratique, et un score passé de 0,40 à 0,70. Ensuite, on a demandé aux endpoints de recracher leurs consignes : ep-29 l'a fait, notre intuition ep-27 non. La preuve a corrigé l'intuition.

Restait la base de chaque famille. On l'a cherchée avec un test de vraisemblance sous le modèle ouvert. Pour Llama, deux candidats restaient indiscernables, même avec cent réponses chacun. Une seule soumission a tranché, et le score est monté à 0,85.

Limite : pour les deux familles Qwen, on n'identifie pas encore la base. Une variante très légère reste invisible avec cent réponses.

### Tâche 3 — faire oublier des images (1 min 45)

[Diapo 6]

Troisième tâche. Un modèle de vision a appris sur des images, et il doit en oublier 3 500. Le juge est un modèle de référence, réentraîné sans ces images : il faut lui ressembler, image par image. Mais nous n'avons ni ce modèle, ni les autres images d'entraînement, ni les étiquettes.

Notre idée : oublier, ce n'est pas tout rater. Pensez à un élève qui a appris des corrigés par cœur. Pour faire comme s'il ne les avait jamais vus, il ne doit pas répondre n'importe quoi : il doit hésiter là où un autre élève hésiterait.

On part du modèle d'origine : c'est le professeur. Pour chaque image à oublier, on lui montre seize versions légèrement modifiées et on fait la moyenne de ses réponses. On baisse un peu la bonne réponse, et un élève apprend à imiter cette cible. Ailleurs, sur des photos extérieures, il imite le professeur. Enfin, on ne pousse vers l'erreur que les images fragiles, celles que le modèle de référence rate probablement.

[Diapo 7]

Le résultat : partis de 0,014, nous sommes à 0,3236 sur le classement public. Éteindre des catégories entières nous a menés à 0,15. Viser des images précises plutôt que des catégories, à 0,29, puis 0,32.

Soyons honnêtes. À 8 heures ce matin, le premier était autour de 0,48 ; à la clôture, nous étions seizièmes sur cette tâche. Nos progrès de ce matin sont petits : plusieurs variantes sans gain, puis un seul réglage qui nous fait passer de 0,314 à 0,324. Et avec environ soixante-quinze soumissions, nous avons réglé notre méthode sur le classement public. Le risque est de trop coller à ce classement, et nous le disons.

### Conclusion (20 s)

[Diapo 8]

Pour finir, trois leçons. Quand on a l'outil officiel, on peut fabriquer ses propres données. Quand les modèles sont ouverts, on peut tester exactement, et connaître la limite du test. Et il faut comprendre ce que la métrique récompense avant d'optimiser. Partout, nous avons essayé de dire ce qui est mesuré et ce qui ne l'est pas. Merci.

---

## English version

### Opening (15 s)

[Slide 1]

Hello, we are team Epi84. We worked on three tasks: finding where a watermark hides in an image, telling which model answers behind an API, and making a model forget images. One idea guided us throughout: measure honestly.

### Task 1 — locating the watermark (1 min 45)

[Slide 2]

First task. We are given an image, and we must say, pixel by pixel, where it carries an invisible watermark. There are three watermarking methods, and only 170 example images for each.

Our idea: make the examples ourselves. It is like holding the official stamp. If we are the ones stamping the sheet, we know exactly where the mark is.

In practice, we take an image that does not carry the watermark we study. We pick a region, with a shape that looks like the real examples. We put the watermark there with the organisers' official tool. We get an image and its exact answer, as many times as we want. A segmentation network, that is, a network that colours the watermarked pixels, learns from these home-made images. Then we adjust it on the real ones.

[Slide 3]

The result. Last night we were at 0.50 on the public leaderboard. This morning: 0.65, then 0.73, and now 0.76. What paid most is very simple: training longer on our home-made images. On StegaStamp, our local score went from 0.65 to 0.84.

All our settings are chosen on images set aside, which the model did not see during training, never on the leaderboard. At every submission, the public score stayed within about 0.02 of our local score.

The lesson: our cleverer ideas brought nothing. Good data and training time are what worked.

### Task 2 — which model is behind the API (1 min 45)

[Slide 4]

Second task. We had to say which APIs hide the same model. The trap: a disguise, whether a watermark, a hidden system prompt or the temperature, keeps the same model but changes the text a lot, while quantisation changes the model but barely changes the text. So comparing answers does not work.

On the practice endpoints we saw the structure: the base model and its four disguises form one group, and the variants stand alone. And the score punishes a wrong merge more than a missed one.

So we changed the question: not "who looks alike?" but "who carries proof of a disguise?". Three traces: identical answers for the watermark, rare words for the temperature, a fake identity for the system prompt. Without proof, an endpoint stays alone.

[Slide 5]

The result: four disguises in each family, as in practice, and a score that went from 0.40 to 0.70. Then we asked the endpoints to recite their instructions: ep-29 did, our hunch ep-27 did not. The proof corrected the hunch.

What remained was the base model of each family. We looked for it with a likelihood test under the open model. For Llama, two candidates stayed indistinguishable, even with one hundred answers each. A single submission settled it, and the score rose to 0.85.

The limit: for the two Qwen families, we still cannot identify the base. A very light variant stays invisible with one hundred answers.

### Task 3 — making a model forget images (1 min 45)

[Slide 6]

Third task. A vision model learned from images, and it must forget 3,500 of them. The judge is a reference model, retrained without those images: we must resemble it, image by image. But we have neither that model, nor the other training images, nor the labels.

Our idea: forgetting is not failing everything. Think of a student who learned answer keys by heart. To act as if he had never seen them, he must not answer at random: he must hesitate where another student would hesitate.

We start from the original model: the teacher. For each image to forget, we show it sixteen slightly modified versions and average its answers. We lower the correct answer a little, and a student learns to imitate this target. Elsewhere, on outside photos, it imitates the teacher. Finally, we push towards error only the fragile images, those the reference model probably gets wrong.

[Slide 7]

The result: starting from 0.014, we are at 0.3236 on the public leaderboard. Switching off whole categories took us to 0.15. Targeting specific images rather than categories, to 0.29, then 0.32.

Let us be honest. At 8 this morning the leader was around 0.48; when submissions closed, we were sixteenth on this task. This morning's progress is small: several variants with no gain, then a single setting that took us from 0.314 to 0.324. And with about seventy-five submissions, we tuned our method on the public leaderboard. The risk is fitting that leaderboard too closely, and we say so.

### Closing (20 s)

[Slide 8]

To finish, three lessons. When you hold the official tool, you can make your own data. When models are open, you can test exactly, and know the limit of the test. And you must understand what the metric rewards before optimising. Throughout, we tried to say what is measured and what is not. Thank you.
