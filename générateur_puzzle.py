import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import random

# Générateur et Mélangeur

def generer_bord_aleatoire(cat):
    base_x = np.array([0.0, 0.15, 0.35, 0.35, 0.50, 0.65, 0.65, 0.85, 1.0])
    base_y = np.array([0.0, 0.00, -0.05, 0.25, 0.30, 0.25, -0.05, 0.00, 0.0])
    
    bruit_x = np.random.uniform(-0.03, 0.03, 9)
    bruit_y = np.random.uniform(-0.03, 0.03, 9)
    bruit_x[0] = bruit_x[-1] = bruit_y[0] = bruit_y[-1] = 0.0 
    
    points = np.column_stack((base_x + bruit_x, base_y + bruit_y))
    if cat == 2:
        points[:, 1] = -points[:, 1]
    return points

def inverser_bord_complementaire(points):
    points_inverses = np.zeros_like(points)
    for i in range(9):
        points_inverses[i, 0] = 1.0 - points[8-i, 0]
        points_inverses[i, 1] = -points[8-i, 1]
    return points_inverses

def generer_puzzle_complet(lignes, colonnes):
    dict_ctrl = {}
    for r in range(lignes):
        for c in range(colonnes):
            piece_id = r * colonnes + c
            dict_ctrl[piece_id] = [None, None, None, None]

    bord_plat = np.column_stack((np.linspace(0, 1, 9), np.zeros(9)))

    for r in range(lignes):
        for c in range(colonnes):
            piece_id = r * colonnes + c
            
            # Pièces du haut
            if r == 0:
                dict_ctrl[piece_id][0] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                piece_dessus_id = (r - 1) * colonnes + c
                bord_dessus = dict_ctrl[piece_dessus_id][2]
                dict_ctrl[piece_id][0] = {"ctrl": inverser_bord_complementaire(bord_dessus["ctrl"]), "cat": 3 - bord_dessus["cat"]}

            # Pièces de gauche
            if c == 0:
                dict_ctrl[piece_id][3] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                piece_gauche_id = r * colonnes + (c - 1)
                bord_gauche = dict_ctrl[piece_gauche_id][1]
                dict_ctrl[piece_id][3] = {"ctrl": inverser_bord_complementaire(bord_gauche["ctrl"]), "cat": 3 - bord_gauche["cat"]}

            # Pièce de droite
            if c == colonnes - 1:
                dict_ctrl[piece_id][1] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                cat = np.random.choice([1, 2])
                dict_ctrl[piece_id][1] = {"ctrl": generer_bord_aleatoire(cat), "cat": cat}

            # Pièces du bas
            if r == lignes - 1:
                dict_ctrl[piece_id][2] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                cat = np.random.choice([1, 2])
                dict_ctrl[piece_id][2] = {"ctrl": generer_bord_aleatoire(cat), "cat": cat}

    return dict_ctrl

def melanger_dictionnaire_puzzle(dict_ctrl):
    """
    Pour simuler le mélange des pièces, on leur réattribue des IDs aléatoires. 
    """
    cles_originales = list(dict_ctrl.keys())
    cles_melangees = cles_originales.copy()
    random.shuffle(cles_melangees)
    
    dict_ctrl_melange = {}
    for i, ancienne_cle in enumerate(cles_originales):
        nouvelle_cle = cles_melangees[i]
        dict_ctrl_melange[nouvelle_cle] = dict_ctrl[ancienne_cle]
        
    return dict_ctrl_melange


# Résolution des puzzles

def distance_cotes(ctrlA, ctrlB):
    ctrlB_inverse = np.zeros_like(ctrlB)
    for i in range(9):
        ctrlB_inverse[i, 0] = 1.0 - ctrlB[8-i, 0]
        ctrlB_inverse[i, 1] = -ctrlB[8-i, 1]
    return np.linalg.norm(ctrlA - ctrlB_inverse)

def associer_pieces(dict_ctrl):
    cotes = []
    for pid, cotes_piece in dict_ctrl.items():
        for cid, info in enumerate(cotes_piece):
            if info["cat"] != 0:
                cotes.append((pid, cid))

    distances = {}
    for (pA, cA) in cotes:
        for (pB, cB) in cotes:
            if pA != pB:
                catA = dict_ctrl[pA][cA]["cat"]
                catB = dict_ctrl[pB][cB]["cat"]
                if (catA, catB) in [(1, 2), (2, 1)]:
                    distances[((pA, cA), (pB, cB))] = distance_cotes(
                        dict_ctrl[pA][cA]["ctrl"], dict_ctrl[pB][cB]["ctrl"]
                    )

    associations = []
    associes = set()

    for (pA, cA) in cotes:
        if (pA, cA) in associes: continue
        meilleur, meilleure_dist = None, np.inf

        for (pB, cB) in cotes:
            if (pB, cB) in associes or pA == pB: continue
            key = ((pA, cA), (pB, cB))
            if key in distances and distances[key] < meilleure_dist:
                meilleure_dist = distances[key]
                meilleur = (pB, cB)

        if meilleur is not None:
            associations.append(((pA, cA), meilleur))
            associes.add((pA, cA))
            associes.add(meilleur)

    return associations


# Affichage

def reconstruire_coordonnees(dict_ctrl, associations):
    """
    Parcourt les associations de proche en proche pour déduire les (x, y) de chaque pièce.
    """
    # 1. Construction du graphe 
    adj = {pid: {} for pid in dict_ctrl}
    for (pA, cA), (pB, cB) in associations:
        adj[pA][cA] = pB
        adj[pB][cB] = pA
        
    # 2. Recherche du point de départ (Le coin Haut-Gauche a des bords plats en 0 et 3)
    root = None
    for pid, cotes in dict_ctrl.items():
        if cotes[0]["cat"] == 0 and cotes[3]["cat"] == 0:
            root = pid
            break
            
    if root is None: # Secours si le coin n'est pas trouvé
        root = list(dict_ctrl.keys())[0]

    # 3. Propagation BFS des coordonnées
    positions = {root: (0, 0)} # (ligne, colonne)
    queue = [root]
    
    while queue:
        curr = queue.pop(0)
        r, c = positions[curr]
        
        for cote_curr, voisin in adj[curr].items():
            if voisin not in positions:
                # Si curr est attaché à voisin par son côté droit (1), voisin est en c+1
                if cote_curr == 0: n_r, n_c = r - 1, c
                elif cote_curr == 1: n_r, n_c = r, c + 1
                elif cote_curr == 2: n_r, n_c = r + 1, c
                elif cote_curr == 3: n_r, n_c = r, c - 1
                
                positions[voisin] = (n_r, n_c)
                queue.append(voisin)
                
    # 4. Secours pour les pièces orphelines (si l'algo a raté des connexions)
    orphelines = set(dict_ctrl.keys()) - set(positions.keys())
    offset_c = max([c for r, c in positions.values()] + [0]) + 2
    for pid in list(orphelines):
        positions[pid] = (0, offset_c)
        offset_c += 1
        
    return positions

def dessiner_grille_pieces(ax, dict_ctrl, positions, titre):
    t = np.linspace(0, 1, 100)
    knots = np.concatenate((np.zeros(2), np.linspace(0, 1, 9 - 2 + 1), np.ones(2)))

    for piece_id, (r, c) in positions.items():
        cotes = dict_ctrl[piece_id]
        
        # Un léger offset pour simuler l'assemblage (pour bien voir les lignes bleues)
        for cote_idx, data in enumerate(cotes):
            ctrl = data["ctrl"]
            spline_x = BSpline(knots, ctrl[:, 0], 2)(t)
            spline_y = BSpline(knots, ctrl[:, 1], 2)(t)
            
            if cote_idx == 0:   # Haut
                plot_x = c + spline_x ; plot_y = -r + spline_y
            elif cote_idx == 1: # Droite
                plot_x = c + 1 + spline_y ; plot_y = -r - spline_x
            elif cote_idx == 2: # Bas
                plot_x = c + 1 - spline_x ; plot_y = -r - 1 - spline_y
            elif cote_idx == 3: # Gauche
                plot_x = c - spline_y ; plot_y = -r - 1 + spline_x

            ax.plot(plot_x, plot_y, 'b-', linewidth=1.5)
            
        # Affichage de l'ID au centre de la pièce
        ax.text(c + 0.5, -r - 0.5, str(piece_id), color='red', fontsize=12, 
                ha='center', va='center', fontweight='bold')

    ax.axis('equal')
    ax.axis('off')
    ax.set_title(titre, fontsize=16)

def afficher_comparaison(dict_parfait, pos_parfaites, dict_melange, pos_reconstruites):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    dessiner_grille_pieces(ax1, dict_parfait, pos_parfaites, "État Initial (Génération)")
    dessiner_grille_pieces(ax2, dict_melange, pos_reconstruites, "Reconstruction par l'Algorithme")
    
    plt.tight_layout()
    plt.show()


# EXÉCUTION DU TEST COMPLET

if __name__ == "__main__":
    LIGNES = 3
    COLONNES = 4
    
    print("1. Génération du puzzle original...")
    dict_ctrl_original = generer_puzzle_complet(LIGNES, COLONNES)
    # Positions théoriques parfaites
    pos_originales = {r*COLONNES+c: (r, c) for r in range(LIGNES) for c in range(COLONNES)}
    
    print("2. Mélange total des pièces (simulation de la réalité)...")
    dict_ctrl_melange = melanger_dictionnaire_puzzle(dict_ctrl_original)
    
    print("3. Algorithme de Résolution à l'aveugle...")
    associations = associer_pieces(dict_ctrl_melange)
    print(f" -> {len(associations)} liaisons trouvées par l'algorithme.")
    
    print("4. Propagation BFS des coordonnées pour le dessin...")
    pos_reconstruites = reconstruire_coordonnees(dict_ctrl_melange, associations)
    
    print("5. Affichage comparatif...")
    afficher_comparaison(dict_ctrl_original, pos_originales, dict_ctrl_melange, pos_reconstruites)