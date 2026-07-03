import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import random
import copy


# Générateur et Mélangeur
# Idée générale : on représente chaque côté d'une pièce de puzzle par 9 points de contrôle d'une B-spline (une courbe). Une pièce a donc 4 côtés = 4 listes de 9 points.
# cat = catégorie du côté : 0 = plat (bord du puzzle), 1 = bosse, 2 = creux. Deux côtés ne peuvent s'emboîter que si l'un est une bosse (1) et l'autre un creux (2) complémentaire.

def generer_bord_aleatoire(cat):
    """
    Crée un côté de pièce "à la main" : une forme de base qui ressemble
    à une bosse (une sorte de cloche), à laquelle on ajoute un peu de
    bruit aléatoire pour que chaque pièce soit unique, comme un vrai
    puzzle où deux pièces bosse/creux ne sont jamais parfaitement identiques.
    """
    # Coordonnées x et y "de base" du profil d'une bosse, entre 0 et 1
    base_x = np.array([0.0, 0.15, 0.35, 0.35, 0.50, 0.65, 0.65, 0.85, 1.0])
    base_y = np.array([0.0, 0.00, -0.05, 0.25, 0.30, 0.25, -0.05, 0.00, 0.0])

    # On ajoute un petit bruit aléatoire sur chaque point de contrôle
    bruit_x = np.random.uniform(-0.03, 0.03, 9)
    bruit_y = np.random.uniform(-0.03, 0.03, 9)
    # Les deux extrémités du côté ne doivent pas bouger : elles doivent rester exactement sur les coins de la pièce pour que les côtés se raccordent bien entre eux, c'est pourquoi on enlève le bruit.
    bruit_x[0] = bruit_x[-1] = bruit_y[0] = bruit_y[-1] = 0.0

    points = np.column_stack((base_x + bruit_x, base_y + bruit_y))

    # Si on veut un creux plutôt qu'une bosse, il suffit d'inverser le signe de la coordonnée y (symétrie par rapport à l'axe x)
    if cat == 2:
        points[:, 1] = -points[:, 1]
    return points


def inverser_bord_complementaire(points):
    """
    À partir du côté d'une pièce (par exemple le côté droit de la pièce
    de gauche), calcule le côté complémentaire qui vient s'emboîter
    dessus (le côté gauche de la pièce de droite) :

    - on parcourt les points dans l'ordre inverse (points[8-i]) car les
      deux côtés sont vus dans des sens opposés une fois les pièces mises
      côte à côte
    - on inverse la coordonnée y pour que la bosse devienne un creux
      (et inversement)
    - la coordonnée x est "retournée" (1 - x) pour la même raison
    """
    points_inverses = np.zeros_like(points)
    for i in range(9):
        points_inverses[i, 0] = 1.0 - points[8 - i, 0]
        points_inverses[i, 1] = -points[8 - i, 1]
    return points_inverses


def generer_puzzle_complet(lignes, colonnes):
    """
    Construit un puzzle complet de `lignes` x `colonnes` pièces, cohérent.

    dict_ctrl est le dictionnaire qui stocke toute la géométrie du puzzle :
        dict_ctrl[piece_id] = [cote_haut, cote_droite, cote_bas, cote_gauche] avec pour chaque côté un dict {"ctrl": points_de_controle, "cat": categorie}

    On numérote les pièces ligne par ligne :
        piece_id = r * colonnes + c
    """
    dict_ctrl = {}
    for r in range(lignes):
        for c in range(colonnes):
            piece_id = r * colonnes + c
            dict_ctrl[piece_id] = [None, None, None, None]

    # Un bord plat = une simple ligne droite entre (0,0) et (1,0)
    bord_plat = np.column_stack((np.linspace(0, 1, 9), np.zeros(9)))

    # On construit les pièces dans l'ordre (ligne par ligne, de gauche à droite) : ça permet de toujours réutiliser le côté déjà généré de la pièce du dessus ou de la pièce de gauche pour créer le côté complémentaire, et donc de garantir que le puzzle est cohérent.
    for r in range(lignes):
        for c in range(colonnes):
            piece_id = r * colonnes + c

            # Côté du haut (index 0)
            if r == 0:
                # première ligne : le bord du haut touche l'extérieur du puzzle
                dict_ctrl[piece_id][0] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                # sinon, on récupère le côté BAS de la pièce juste au-dessus (index 2) et on en déduit le côté complémentaire
                piece_dessus_id = (r - 1) * colonnes + c
                bord_dessus = dict_ctrl[piece_dessus_id][2]
                dict_ctrl[piece_id][0] = {
                    "ctrl": inverser_bord_complementaire(bord_dessus["ctrl"]),
                    "cat": 3 - bord_dessus["cat"],  # 1 <-> 2, et 0 reste 0... (ici cat != 0 forcément)
                }

            # Côté de gauche (index 3)
            if c == 0:
                # première colonne : bord extérieur du puzzle
                dict_ctrl[piece_id][3] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                # on récupère le côté DROIT de la pièce voisine de gauche (index 1)
                piece_gauche_id = r * colonnes + (c - 1)
                bord_gauche = dict_ctrl[piece_gauche_id][1]
                dict_ctrl[piece_id][3] = {
                    "ctrl": inverser_bord_complementaire(bord_gauche["ctrl"]),
                    "cat": 3 - bord_gauche["cat"],
                }

            # Côté de droite (index 1) : générée librement, sauf si
            # la pièce est sur le bord droit du puzzle (bord plat)
            if c == colonnes - 1:
                dict_ctrl[piece_id][1] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                cat = np.random.choice([1, 2])
                dict_ctrl[piece_id][1] = {"ctrl": generer_bord_aleatoire(cat), "cat": cat}

            # Côté du bas (index 2) : même logique que le côté droit
            if r == lignes - 1:
                dict_ctrl[piece_id][2] = {"ctrl": bord_plat.copy(), "cat": 0}
            else:
                cat = np.random.choice([1, 2])
                dict_ctrl[piece_id][2] = {"ctrl": generer_bord_aleatoire(cat), "cat": cat}

    return dict_ctrl


def melanger_dictionnaire_puzzle(dict_ctrl):
    """
    Pour simuler le mélange des pièces, on leur réattribue des IDs aléatoires.
    La géométrie de chaque pièce (ses 4 côtés) ne change pas : seul le numéro qui lui est attribué change, comme si on mélangeait les pièces d'un vrai puzzle dans une boîte.
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
# Le but ici est de retrouver, à partir des pièces mélangées, quels côtés vont ensemble puis de reconstituer la position de chaque pièce dans la grille.


def distance_cotes(ctrlA, ctrlB):
    """
    Mesure à quel point deux côtés sont "compatibles" pour s'emboîter.

    On ne peut pas comparer directement ctrlA et ctrlB : ils sont
    parcourus dans des sens opposés une fois les pièces assemblées
    (comme pour inverser_bord_complementaire). On calcule donc d'abord
    le "retourné" de ctrlB, puis on mesure la distance euclidienne
    point par point entre les deux courbes. Plus la distance est
    petite, plus les deux côtés sont susceptibles de s'emboîter parfaitement.
    """
    ctrlB_inverse = np.zeros_like(ctrlB)
    for i in range(9):
        ctrlB_inverse[i, 0] = 1.0 - ctrlB[8 - i, 0]
        ctrlB_inverse[i, 1] = -ctrlB[8 - i, 1]
    return np.linalg.norm(ctrlA - ctrlB_inverse)


def associer_pieces(dict_ctrl):
    """
    Algorithme glouton d'association des côtés :
    pour chaque côté non plat, on cherche le côté compatible
    (bosse <-> creux) le plus proche géométriquement, tant qu'il
    n'est pas déjà associé.

    """
    # 1. On liste tous les côtés "associables" (pas plats) de toutes les pièces
    cotes = []
    for pid, cotes_piece in dict_ctrl.items():
        for cid, info in enumerate(cotes_piece):
            if info["cat"] != 0:
                cotes.append((pid, cid))

    # 2. On calcule la distance entre chaque paire de côtés compatibles

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

    # 3. Pour chaque côté, on choisit le meilleur candidat disponible
    associations = []
    associes = set()  # côtés déjà utilisés dans une association

    for (pA, cA) in cotes:
        if (pA, cA) in associes:
            continue  # ce côté a déjà trouvé son partenaire

        meilleur, meilleure_dist = None, np.inf
        for (pB, cB) in cotes:
            if (pB, cB) in associes or pA == pB:
                continue
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

def visualiser_schema_pieces(dict_ctrl_original, associations):
    """
    Visualisation du résultat de l'association : chaque pièce
    est représentée par un carré numéroté (peu importe sa vraie forme),
    et les associations trouvées par l'algorithme sont tracées en
    pointillés rouges entre les côtés concernés.

    Cette vue sert à vérifier visuellement que l'algorithme a
    bien relié les bons côtés entre eux, indépendamment de la position
    finale des pièces dans la grille.
    """
    dict_ctrl = copy.deepcopy(dict_ctrl_original)
    n_pieces = len(dict_ctrl)

    angle_step = 2 * np.pi / max(1, n_pieces)
    radius = 5

    positions = {}
    for i, pid in enumerate(dict_ctrl.keys()):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[pid] = (x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Schéma Logique des Pièces Mélangées", fontsize=14, fontweight='bold')

    for pid, (x, y) in positions.items():
        # Carré représentant la pièce
        ax.add_patch(plt.Rectangle((x - 1, y - 1), 2, 2, fill=False, linewidth=2))
        ax.text(x, y, str(pid), ha="center", va="center", fontsize=9, color="darkblue")

        # Les 4 côtés du carré, dans le même ordre que dans dict_ctrl :
        # 0 = haut, 1 = droite, 2 = bas, 3 = gauche
        cotes = [
            ((x - 1, y + 1), (x + 1, y + 1)),
            ((x + 1, y + 1), (x + 1, y - 1)),
            ((x - 1, y - 1), (x + 1, y - 1)),
            ((x - 1, y + 1), (x - 1, y - 1)),
        ]

        dict_ctrl[pid].append({"schema_cotes": cotes})

        for i, (p1, p2) in enumerate(cotes):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=1)
            cx = (p1[0] + p2[0]) / 2
            cy = (p1[1] + p2[1]) / 2
            ax.text(cx, cy, f"{i}", fontsize=10, color="black", fontweight='bold')


    for (pA, cA), (pB, cB) in associations:
        cotesA = dict_ctrl[pA][-1]["schema_cotes"]
        cotesB = dict_ctrl[pB][-1]["schema_cotes"]

        p1A, p2A = cotesA[cA]
        p1B, p2B = cotesB[cB]

        cxA = (p1A[0] + p2A[0]) / 2
        cyA = (p1A[1] + p2A[1]) / 2
        cxB = (p1B[0] + p2B[0]) / 2
        cyB = (p1B[1] + p2B[1]) / 2

        ax.plot([cxA, cxB], [cyA, cyB], "r--", linewidth=2)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.show(block=False)


def reconstruire_coordonnees(dict_ctrl, associations):
    """
    Une fois qu'on sait quels côtés vont ensemble, il
    reste à déduire la position (ligne, colonne) de chaque pièce dans
    la grille finale. On fait ça avec un parcours en largeur :
    on part d'une pièce arbirairement choisie et on avance de proche en proche
    grâce aux associations.
    """
    # 1. On construit un graphe d'adjacence : pour chaque pièce, quel est son voisin sur chacun de ses côtés associés ?
    #    adj[pid][cote] = id de la pièce voisine sur ce côté
    adj = {pid: {} for pid in dict_ctrl}
    for (pA, cA), (pB, cB) in associations:
        adj[pA][cA] = pB
        adj[pB][cB] = pA

    # 2. On cherche une pièce de départ : idéalement le coin haut-gauche du puzzle, reconnaissable car ses côtés haut (0) ET gauche (3) sont plats (cat == 0)
    root = None
    for pid, cotes in dict_ctrl.items():
        if cotes[0]["cat"] == 0 and cotes[3]["cat"] == 0:
            root = pid
            break

    if root is None:  # sécurité si jamais aucun coin n'est trouvé
        root = list(dict_ctrl.keys())[0]

    # 3. BFS (parcourt en largeur) : on place la racine en (0, 0), puis pour chaque pièce déjà positionnée, on regarde ses voisines et on en déduit leur position relative selon le côté par lequel elles sont reliées
    positions = {root: (0, 0)}  # (ligne, colonne)
    queue = [root]

    while queue:
        curr = queue.pop(0)
        r, c = positions[curr]

        for cote_curr, voisin in adj[curr].items():
            if voisin not in positions:
                # le côté "cote_curr" de la pièce actuelle nous dit dans quelle direction se trouve la pièce voisine
                if cote_curr == 0:
                    n_r, n_c = r - 1, c  # voisin au-dessus
                elif cote_curr == 1:
                    n_r, n_c = r, c + 1  # voisin à droite
                elif cote_curr == 2:
                    n_r, n_c = r + 1, c  # voisin en dessous
                elif cote_curr == 3:
                    n_r, n_c = r, c - 1  # voisin à gauche

                positions[voisin] = (n_r, n_c)
                queue.append(voisin)

    # 4. Si l'algorithme d'association a raté certaines connexions, certaines pièces restent jamais atteintes par le BFS. On les place quand même quelque part (à droite du puzzle) pour ne pas planter l'affichage, même si leur position n'est pas la bonne.
    orphelines = set(dict_ctrl.keys()) - set(positions.keys())
    offset_c = max([c for r, c in positions.values()] + [0]) + 2
    for pid in list(orphelines):
        positions[pid] = (0, offset_c)
        offset_c += 1

    return positions


def dessiner_grille_pieces(ax, dict_ctrl, positions, titre):
    """
    Dessine toutes les pièces à leur position (r, c) en traçant leurs
    4 côtés comme des courbes B-spline, pour obtenir un rendu qui
    ressemble à un vrai puzzle assemblé.
    """
    t = np.linspace(0, 1, 100)  # paramètre de la courbe, de 0 à 1
    # Vecteur de noeuds (knots) pour une B-spline de degré 2 avec 9
    # points de contrôle. Structure classique : on répète les valeurs
    # aux extrémités pour que la courbe commence et finisse exactement
    # sur le premier et le dernier point de contrôle.
    knots = np.concatenate((np.zeros(2), np.linspace(0, 1, 9 - 2 + 1), np.ones(2)))

    for piece_id, (r, c) in positions.items():
        cotes = dict_ctrl[piece_id]

        for cote_idx, data in enumerate(cotes):
            ctrl = data["ctrl"]
            # On évalue la B-spline correspondant à ce côté pour obtenir une courbe lisse (100 points) plutôt que juste 9 segments
            spline_x = BSpline(knots, ctrl[:, 0], 2)(t)
            spline_y = BSpline(knots, ctrl[:, 1], 2)(t)

            if cote_idx == 0:   # Haut : on va de gauche à droite
                plot_x = c + spline_x
                plot_y = -r + spline_y
            elif cote_idx == 1:  # Droite : on descend, la bosse pointe vers la droite
                plot_x = c + 1 + spline_y
                plot_y = -r - spline_x
            elif cote_idx == 2:  # Bas : on va de droite à gauche
                plot_x = c + 1 - spline_x
                plot_y = -r - 1 - spline_y
            elif cote_idx == 3:  # Gauche : on remonte, la bosse pointe vers la gauche
                plot_x = c - spline_y
                plot_y = -r - 1 + spline_x

            ax.plot(plot_x, plot_y, 'b-', linewidth=1.5)

        # On affiche l'identifiant de la pièce au centre de son carré
        ax.text(c + 0.5, -r - 0.5, str(piece_id), color='red', fontsize=12,
                ha='center', va='center', fontweight='bold')

    ax.axis('equal')
    ax.axis('off')
    ax.set_title(titre, fontsize=16)


def afficher_comparaison(dict_parfait, pos_parfaites, dict_melange, pos_reconstruites):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    dessiner_grille_pieces(ax1, dict_parfait, pos_parfaites, "État Initial")
    dessiner_grille_pieces(ax2, dict_melange, pos_reconstruites, "Reconstruction par l'algorithme après mélange")

    plt.tight_layout()
    plt.show()


# Exécution

if __name__ == "__main__":
    LIGNES = 3
    COLONNES = 4

    print("1. Génération du puzzle original")
    dict_ctrl_original = generer_puzzle_complet(LIGNES, COLONNES)
    pos_originales = {r * COLONNES + c: (r, c) for r in range(LIGNES) for c in range(COLONNES)}

    print("2. Mélange total des pièces")
    dict_ctrl_melange = melanger_dictionnaire_puzzle(dict_ctrl_original)

    print("3. Algorithme de résolution")
    associations = associer_pieces(dict_ctrl_melange)
    print(f" -> {len(associations)} liaisons trouvées par l'algorithme.")

    print("Affichage du Schéma Logique des Pièces")
    visualiser_schema_pieces(dict_ctrl_melange, associations)

    print("4. Propagation BFS des coordonnées pour le dessin")
    pos_reconstruites = reconstruire_coordonnees(dict_ctrl_melange, associations)

    print("5. Affichage")
    afficher_comparaison(dict_ctrl_original, pos_originales, dict_ctrl_melange, pos_reconstruites)
