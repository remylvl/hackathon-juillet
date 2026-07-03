## License

This project is licensed under the MIT License - see the LICENSE file for details.



# Puzzle Solver — README POUR MAIN.PY
Nous avons gardé dans cette partie (le main), la détection sans apprentissage.
Script autonome : détecte plusieurs pièces de puzzle sur fond bleu, extrait leurs côtés, les classe (plat/bosse/creux) et associe automatiquement les côtés complémentaires entre pièces.

## Installation que nous avons dû faire en plus 

```bash
pip install numpy matplotlib scipy scikit-image
```

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
