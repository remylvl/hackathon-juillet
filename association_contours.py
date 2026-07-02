import numpy as np

# On importe la reconnaissance de pièce (segments pivotés)
from elodie.reconnaissance_piece_vf import analyser_piece

# On importe TON optimiseur B-spline
from points_bspline import optimiser_segment


# ============================================================
# 1. CLASSIFICATION BOSSE / CREUX / PLAT
# ============================================================

def classifier_cote(ctrl_points, seuil_plat=0.08):
    """
    Détermine la catégorie du côté :
    0 = plat
    1 = bosse
    2 = creux
    """
    y = ctrl_points[:, 1]
    y_max = np.max(y)
    y_min = np.min(y)

    if abs(y_max) < seuil_plat and abs(y_min) < seuil_plat:
        return 0
    if abs(y_max) > abs(y_min):
        return 1
    return 2


# ============================================================
# 2. CONSTRUCTION DE dict_ctrl À PARTIR DE analyser_piece()
#    + optimiser_segment() (points B-spline optimisés)
# ============================================================

def construire_dict_ctrl(fichier_image):
    """
    Construit dict_ctrl pour une pièce unique à partir de :
        - analyser_piece()  → segments normalisés
        - optimiser_segment() → points de contrôle optimisés
    dict_ctrl[0] = liste de 4 côtés, chacun avec :
        - ctrl : points B-spline optimisés
        - cat  : catégorie (0,1,2)
    """
    data = analyser_piece(fichier_image, afficher=False)

    segments_norm = data["segments"]   # liste de 4 segments pivotés coin-à-coin
    dict_ctrl_piece = []

    for seg_norm in segments_norm:
        #  Ici on utilise TON optimiseur B-spline
        ctrl_opt, _, _ = optimiser_segment(seg_norm)

        # classification
        cat = classifier_cote(ctrl_opt)

        dict_ctrl_piece.append({
            "ctrl": ctrl_opt,
            "cat": cat
        })

    dict_ctrl = {0: dict_ctrl_piece}
    return dict_ctrl


# ============================================================
# 3. UTILISATION DANS association_contours
# ============================================================

dict_ctrl = construire_dict_ctrl("./resources/piece4.jpeg")

print("dict_ctrl construit :")
for cote_id, info in enumerate(dict_ctrl[0]):
    print(f"Côté {cote_id} : catégorie = {info['cat']}")
    print(f"Points de contrôle optimisés :\n{info['ctrl']}\n")

# ------------------------------------------------------------
# 1. Construire la liste des côtés associables (catégorie 1 ou 2)
# ------------------------------------------------------------

cotes = []  # liste de tuples (piece_id, cote_id)
for piece_id, cotes_piece in dict_ctrl.items():
    for cote_id, info in enumerate(cotes_piece):
        if info["cat"] != 0:  # bosse ou creux
            cotes.append((piece_id, cote_id))

# ------------------------------------------------------------
# 2. Calculer la distance entre tous les couples de côtés
# ------------------------------------------------------------

def distance_cotes(ctrlA, ctrlB):
    """Distance entre deux vecteurs de 9 points de contrôle."""
    return np.linalg.norm(ctrlA - ctrlB)

# matrice des distances entre côtés
distances = {}  # clé = ((pieceA,coteA),(pieceB,coteB)), valeur = distance

for (pA, cA) in cotes:
    for (pB, cB) in cotes:
        if pA != pB:
            catA = dict_ctrl[pA][cA]["cat"]
            catB = dict_ctrl[pB][cB]["cat"]

            # bosse ↔ creux uniquement
            if (catA == 1 and catB == 2) or (catA == 2 and catB == 1):
                ctrlA = dict_ctrl[pA][cA]["ctrl"]
                ctrlB = dict_ctrl[pB][cB]["ctrl"]
                distances[((pA, cA), (pB, cB))] = distance_cotes(ctrlA, ctrlB)

# ------------------------------------------------------------
# 3. Association de proche en proche
# ------------------------------------------------------------

associations = []          # liste de couples ((pA,cA),(pB,cB))
associes = set()           # côtés déjà associés

# choisir un côté associable de la pièce 0
cotes_piece0 = [(0, c) for c in range(4) if dict_ctrl[0][c]["cat"] != 0]
if len(cotes_piece0) == 0:
    raise ValueError("La pièce 0 n'a aucun côté associable.")

cote_actuel = cotes_piece0[0]  # on prend le premier côté associable

# boucle principale
while True:
    associes.add(cote_actuel)

    # chercher le meilleur match pour cote_actuel
    meilleur = None
    meilleure_dist = np.inf

    for (pB, cB) in cotes:
        if (pB, cB) not in associes:
            key = (cote_actuel, (pB, cB))
            if key in distances:
                d = distances[key]
                if d < meilleure_dist:
                    meilleure_dist = d
                    meilleur = (pB, cB)

    if meilleur is None:
        break  # plus aucun match possible

    # enregistrer l'association
    associations.append((cote_actuel, meilleur))

    # avancer de proche en proche
    cote_actuel = meilleur

    # si tous les côtés associables sont traités, on arrête
    if len(associes) == len(cotes):
        break

# ------------------------------------------------------------
# Résultat
# ------------------------------------------------------------

print("Associations trouvées :")
for a in associations:
    print(a)
