## License

This project is licensed under the MIT License - see the LICENSE file for details.


### Nous avons 3 algorithmes différets 


# Puzzle Solver — README POUR MAIN.PY
Nous avons gardé dans cette partie (le main), la détection sans apprentissage.
Script autonome : détecte plusieurs pièces de puzzle sur fond bleu, extrait leurs côtés, les classe (plat/bosse/creux) et associe automatiquement les côtés complémentaires entre pièces.

## Utilisation

1. Placer les photos de pièces (fond bleu uni) dans `./resources/n_pieces_ensembles/`.
2. si besoin en tête de fichier, on peut modifier :
   - `FICHIER_IMAGE` image utilisée pour les tests unitaires (`analyser_piece` seule)
   - `AFFICHER_GRAPHIQUES` `False` pour désactiver tous les `plt.show()`
   - `SAT_MIN`  `VAL_MIN`  `LARGEUR_HUE`  calibration du masque de couleur si le fond n'est pas assez bleu/saturé
3. Pour lancer l'algo depuis le terminal : 

```bash
python main.py
```

## Sortie

- Résultats imprimés en console (coins détectés, segments, correspondances, catégories de côtés).
- `dict_ctrl` : `{piece_id: [{"ctrl": array(15,2), "cat": 0|1|2}, ...]}` (un élément par côté).
- `associations` : liste de tuples `((piece_A, côté_A), (piece_B, côté_B))`.
- Plusieurs fenêtres matplotlib si `AFFICHER_GRAPHIQUES=True` (masque, contour+coins, splines, segments normalisés, schéma final).

# Puzzle Solver — README POUR main_reseau_de_neurones.py

## Utilisation 

Image d'entrée : Vous devez renseigner le chemin vers la photo de la pièce de puzzle à analyser (par défaut reseau_de_neurones/data_photo/photo_test_5.jpg).
Lancement : Exécutez simplement le script de manière classique via la commande python main_reseau_de_neurones.py

## Sortie 

L'exécution du script génère trois types de résultats : des données structurées, des affichages graphiques de contrôle, et des logs dans le terminal.

1. Structure de données (Dictionnaire Python)
Le script génère un dictionnaire global nommé dict_ctrl conçu pour stocker l'intégralité des pièces traitées.
La pièce est identifiée dynamiquement par le nom de son fichier image (ex: "photo_test_5").
Pour chaque pièce, le dictionnaire contient les 4 bords (numérotés de 0 à 3 correspondant à Haut, Droite, Bas, Gauche).
Chaque bord contient un tableau matriciel (array numpy) de dimensions 9x2, stocké sous la clé "ctrl", représentant les 9 points de contrôle mathématiques de la courbe B-spline optimisée de ce côté.

2. Sorties Visuelles (Matplotlib)
Diagnostic IA : Une première fenêtre affiche une "Carte de Chaleur" (Heatmap) en fausses couleurs (magma) montrant exactement où le réseau de neurones a localisé les 4 coins de la pièce.
Contrôle d'Ajustement : Une grille de 4 graphiques (2x2) s'affiche ensuite, détaillant chaque côté normalisé de la pièce. Chaque graphique superpose :
Les points gris : Le contour brut prédit par l'IA.
Les croix rouges : Les 9 points de contrôle B-spline calculés.
La ligne bleue : La courbe B-spline finale lissée et reconstruite.

3. Logs Console
Le terminal affiche la progression étape par étape (Chargement IA, Traitement, Normalisation, Optimisation) et confirme la dimension des arrays stockés dans le dictionnaire final.

# Puzzle Solver — README POUR main_vision_par_ordi.py



## Utilisation

Placer les photos de pièces (fond uni, le mieux c'est de les scanner à l'imprimante) dans detect_coins_vision_par_ordi/puzzle1/, nommées 1_1.jpg, 1_2.jpg, ... (le nombre réel de photos est détecté automatiquement, pas besoin d'en avoir exactement 12).
Si besoin, opencv-python doit être installé pour activer GrabCut (pip install opencv-python) — sinon le script se rabat automatiquement sur le masque HSV brut.
Si besoin en tête de fichier, on peut modifier :

DOSSIER_PUZZLE emplacement du dossier contenant les photos
UTILISER_GRABCUT False pour désactiver GrabCut même si opencv-python est installé
SAT_MIN  VAL_MIN  LARGEUR_HUE  calibration du masque de couleur
FRACTION_DISTANCE_MIN_COINS  SIGMA_HARRIS  réglage de la détection des coins
N_CTRL  DEGRE_SPLINE  SEUIL_PLAT  réglage de l'optimisation B-spline et de la classification bosse/creux/plat


## Sortie

Résultats imprimés en console (coins détectés, segments, pièces ignorées si détection incomplète, catégories de côtés, associations trouvées).
Une figure de vérification par pièce (masque, contour+coins, segments+spline, côtés normalisés) enregistrée en PNG dans detect_coins_vision_par_ordi/puzzle1/verification/.
associations : liste de tuples ((piece_A, côté_A), (piece_B, côté_B)), calculée par l'algorithme hongrois (optimal globalement, contrairement à un appariement glouton).
Fenêtres matplotlib : schéma abstrait des pièces et de leurs associations, puis assemblage géométrique réel (les pièces sont affichées avec leurs vraies photos, tournées et translatées pour que les côtés associés coïncident bord à bord). <= ca ne marche pas vraiment...