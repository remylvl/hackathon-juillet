"""
Analyse d'une pièce de puzzle à partir d'une photo + optimisation B-spline
+ construction de dict_ctrl + association de contours.

Ce fichier est autonome.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import ndimage
from scipy.interpolate import splprep, splev, BSpline
from scipy.optimize import least_squares
from skimage.feature import corner_harris, corner_peaks
from skimage import measure
from skimage.color import rgb2hsv
import os
import sys

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

FICHIER_IMAGE = "./resources/piece4.jpeg"
AFFICHER_GRAPHIQUES = True  # passe à False pour désactiver tous les plt.show()

SAT_MIN = 0.20
VAL_MIN = 0.20
LARGEUR_HUE = 0.06

NB_COINS = 4
SEUIL_COURBURE = 50
DISTANCE_MIN_COINS = 200


def construire_dict_ctrl_pour_plusieurs_pieces(dossier_images):
    """
    Analyse toutes les images d'un dossier et construit dict_ctrl
    pour chaque pièce. Affiche les splines au fur et à mesure.
    """
    dict_ctrl = {}
    fichiers = sorted(os.listdir(dossier_images))

    piece_id = 0
    for fichier in fichiers:
        if fichier.lower().endswith((".png", ".jpg", ".jpeg")):
            chemin = os.path.join(dossier_images, fichier)
            print(f"\n--- Analyse de la pièce {piece_id} : {chemin} ---")

            # 1. Analyse de la pièce
            data = analyser_piece(chemin, afficher=False)
            segments_norm = data["segments"]

            dict_ctrl_piece = []

            # 2. Optimisation + affichage spline
            for cote_id, seg_norm in enumerate(segments_norm):
                ctrl_opt, knots, t = optimiser_segment(seg_norm)

                # Affichage spline optimisée
                plt.figure()
                plt.plot(seg_norm[:, 0], seg_norm[:, 1], 'o', label='Segment normalisé')

                spline_x = BSpline(knots, ctrl_opt[:, 0], 2)
                spline_y = BSpline(knots, ctrl_opt[:, 1], 2)
                C = np.vstack((spline_x(t), spline_y(t))).T

                plt.plot(C[:, 0], C[:, 1], '-', label='Spline optimisée')
                plt.plot(ctrl_opt[:, 0], ctrl_opt[:, 1], 'x', label='Points de contrôle optimisés')

                plt.legend()
                plt.axis("equal")
                plt.title(f"Spline optimisée – Pièce {piece_id}, côté {cote_id}")
                plt.show()

                # Classification
                cat = classifier_cote(ctrl_opt)

                dict_ctrl_piece.append({
                    "ctrl": ctrl_opt,
                    "cat": cat
                })

            dict_ctrl[piece_id] = dict_ctrl_piece
            piece_id += 1

    return dict_ctrl


# ----------------------------------------------------------------------
# 1. Chargement de l'image et masque de couleur
# ----------------------------------------------------------------------

def charger_image(chemin):
    img = image.imread(chemin)
    img_hsv = rgb2hsv(img[:, :, :3])
    return img, img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]


def creer_masque_bleu(img_h, img_s, img_v,
                       sat_min=SAT_MIN, val_min=VAL_MIN,
                       largeur_hue=LARGEUR_HUE):
    candidat = (img_s > sat_min) & (img_v > val_min)
    if not np.any(candidat):
        raise ValueError("Aucun pixel suffisamment saturé pour calibrer le masque.")

    h_candidats = img_h[candidat]
    hist, bins = np.histogram(h_candidats, bins=60, range=(0.0, 1.0))
    i_pic = np.argmax(hist)
    h_centre = 0.5 * (bins[i_pic] + bins[i_pic + 1])

    masque = candidat & (np.abs(img_h - h_centre) <= largeur_hue)
    return masque, h_centre


# ----------------------------------------------------------------------
# 2. Nettoyage du masque
# ----------------------------------------------------------------------

def nettoyer_masque(masque):
    masque_propre = ndimage.binary_opening(masque, structure=np.ones((3, 3)))
    masque_propre = ndimage.binary_closing(masque_propre, structure=np.ones((20, 20)))

    labels, nb = ndimage.label(masque_propre)
    tailles = ndimage.sum(masque_propre, labels, range(1, nb + 1))
    if len(tailles) == 0:
        raise ValueError("Aucune composante détectée : ajuste les seuils HSV du masque.")

    plus_grande = np.argmax(tailles) + 1
    masque_piece = labels == plus_grande

    masque_final = ndimage.binary_fill_holes(masque_piece)
    return masque_final


def extraire_bord(masque_final):
    masque_erode = ndimage.binary_erosion(masque_final)
    bord = masque_final & ~masque_erode
    X, Y = np.nonzero(bord)
    return bord, list(X), list(Y)


# ----------------------------------------------------------------------
# 3. Contour et détection des coins
# ----------------------------------------------------------------------

def detecter_coins_harris(masque_final, min_distance=150, threshold_rel=0.1):
    reponse = corner_harris(masque_final.astype(float))
    coins = corner_peaks(reponse, min_distance=min_distance, threshold_rel=threshold_rel)
    return coins


def extraire_contour_principal(masque_final):
    contours = measure.find_contours(masque_final, level=0.5)
    return max(contours, key=len)


def lisser_contour(contour, sigma=7):
    contour_ferme = np.vstack([contour, contour[:1]])
    row_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 0], sigma=sigma, mode="wrap")[:-1]
    col_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 1], sigma=sigma, mode="wrap")[:-1]
    return np.column_stack([row_lisse, col_lisse])


def trouver_indice(contour, point):
    distances = np.sqrt((contour[:, 0] - point[0]) ** 2 + (contour[:, 1] - point[1]) ** 2)
    return np.argmin(distances)


def score_courbure(contour, index, seuil):
    n = len(contour)
    p = contour[index]
    score = 0.0
    for j in range(1, seuil + 1):
        i_gauche = (index - j) % n
        i_droite = (index + j) % n
        score += abs(
            (contour[i_gauche][0] - p[0]) * (contour[i_droite][0] - p[0])
            + (contour[i_gauche][1] - p[1]) * (contour[i_droite][1] - p[1])
        )
    return score


def max_courbure(contour, points, seuil, distance_min, nb_coins=NB_COINS):
    if len(points) == 0:
        return np.array([]), np.array([])

    scores = [score_courbure(contour, index, seuil) for index in points]
    ordre = np.argsort(scores)

    points_filtres = []
    i_filtres = []

    for idx in ordre:
        index = points[idx]
        p = contour[index]
        trop_proche = any(
            np.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) < distance_min
            for q in points_filtres
        )
        if not trop_proche:
            points_filtres.append(p)
            i_filtres.append(index)
        if len(points_filtres) == nb_coins:
            break

    return np.array(points_filtres), np.array(i_filtres)


# ----------------------------------------------------------------------
# 4. Découpage en segments
# ----------------------------------------------------------------------

def extraire_segments(contour, indices):
    segments = []
    n = len(indices)
    for k in range(n):
        i_debut = indices[k]
        i_fin = indices[(k + 1) % n]
        if i_fin > i_debut:
            segment = contour[i_debut:i_fin + 1]
        else:
            segment = np.vstack([contour[i_debut:], contour[:i_fin + 1]])
        segments.append(segment)
    return segments


# ----------------------------------------------------------------------
# 5. Ajustement de splines
# ----------------------------------------------------------------------

def nettoyer_doublons_consecutifs(segment):
    garder = [segment[0]]
    for p in segment[1:]:
        if not np.array_equal(p, garder[-1]):
            garder.append(p)
    return np.array(garder)


def ajuster_spline_segment(segment, lissage=0):
    segment = nettoyer_doublons_consecutifs(segment)
    x = segment[:, 1]
    y = segment[:, 0]
    tck, u = splprep([x, y], s=lissage)
    return tck


# ----------------------------------------------------------------------
# 6. Normalisation coin-à-coin
# ----------------------------------------------------------------------

def normaliser_segment(segment):
    x = segment[:, 1].astype(float)
    y = segment[:, 0].astype(float)

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    px = x - x0
    py = y - y0

    vx, vy = x1 - x0, y1 - y0
    longueur = np.sqrt(vx ** 2 + vy ** 2)
    vx, vy = vx / longueur, vy / longueur

    along = px * vx + py * vy
    height = px * vy - py * vx

    if longueur > 0:
        along = along / longueur
        height = height / longueur

    return np.column_stack([along, height])


# ----------------------------------------------------------------------
# Affichages (optionnels)
# ----------------------------------------------------------------------

def afficher_contour_simplifie(X, Y, contour_simplifie):
    plt.figure()
    plt.plot(X, Y, '.', markersize=1)
    plt.scatter(contour_simplifie[:, 0], contour_simplifie[:, 1], color='red', s=60)
    plt.axis("equal")
    plt.title("Contour simplifié")
    plt.show()


def afficher_segments(segments, coins_finaux):
    plt.figure()
    couleurs = ['red', 'blue', 'green', 'orange']
    for seg, c in zip(segments, couleurs):
        plt.plot(seg[:, 1], seg[:, 0], '.', markersize=2, color=c)
    plt.scatter(coins_finaux[:, 1], coins_finaux[:, 0], color='black', s=80, zorder=5)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.title("Segments entre coins")
    plt.show()


def afficher_splines(segments, splines):
    plt.figure()
    for seg, tck in zip(segments, splines):
        u_fin = np.linspace(0, 1, 200)
        x_fit, y_fit = splev(u_fin, tck)
        plt.plot(seg[:, 1], seg[:, 0], '.', markersize=2, alpha=0.3)
        plt.plot(x_fit, y_fit, '-', linewidth=2)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.title("Splines ajustées")
    plt.show()


def afficher_segments_normalises(segments):
    plt.figure()
    for i, seg in enumerate(segments):
        seg_norm = normaliser_segment(seg)
        plt.plot(seg_norm[:, 0], seg_norm[:, 1], label=f"Segment {i}")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend()
    plt.title("Segments normalisés (repère coin-à-coin)")
    plt.show()


# ----------------------------------------------------------------------
# Pipeline principal : analyser_piece
# ----------------------------------------------------------------------

def analyser_piece(fichier_image=FICHIER_IMAGE, afficher=AFFICHER_GRAPHIQUES):
    img, img_h, img_s, img_v = charger_image(fichier_image)

    masque, h_centre = creer_masque_bleu(img_h, img_s, img_v)
    print(f"Teinte bleue détectée automatiquement : {h_centre:.3f}")

    masque_final = nettoyer_masque(masque)
    _, X, Y = extraire_bord(masque_final)

    coins_harris = detecter_coins_harris(masque_final)
    print(f"{len(coins_harris)} coins détectés (Harris)")

    contour_principal = extraire_contour_principal(masque_final)
    contour_lisse = lisser_contour(contour_principal, sigma=7)
    contour_simplifie = measure.approximate_polygon(contour_lisse, tolerance=10)

    if afficher:
        afficher_contour_simplifie(X, Y, contour_simplifie)

    indices_coins_harris = sorted(trouver_indice(contour_principal, c) for c in coins_harris)
    print("Indices des coins (Harris) dans le contour :", indices_coins_harris)

    indices_points = [trouver_indice(contour_principal, p) for p in contour_simplifie]
    coins_finaux, indices_coins = max_courbure(
        contour_principal, indices_points, SEUIL_COURBURE, DISTANCE_MIN_COINS
    )
    indices_coins = np.sort(indices_coins)
    print(f"{len(coins_finaux)} coins retenus après filtrage par courbure")

    segments = extraire_segments(contour_principal, indices_coins)
    for i, seg in enumerate(segments):
        print(f"Segment {i} : {len(seg)} points")

    if afficher:
        afficher_segments(segments, coins_finaux)

    splines = [ajuster_spline_segment(seg, lissage=len(seg) * 2) for seg in segments]

    if afficher:
        afficher_splines(segments, splines)
        afficher_segments_normalises(segments)

    segments_norm = [normaliser_segment(seg) for seg in segments]

    return {
        "masque_final": masque_final,
        "contour_principal": contour_principal,
        "coins": coins_finaux,
        "segments": segments_norm,
        "splines": splines,
    }


# ----------------------------------------------------------------------
# B-spline : ré-échantillonnage + optimisation
# ----------------------------------------------------------------------

def echantillonner_segment(segment, n=200):
    d = np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1))
    u = np.concatenate([[0], np.cumsum(d)])
    u = u / u[-1]

    from scipy.interpolate import interp1d
    fx = interp1d(u, segment[:, 0])
    fy = interp1d(u, segment[:, 1])

    u_new = np.linspace(0, 1, n)
    return np.column_stack([fx(u_new), fy(u_new)])


def optimiser_segment(segment, n_ctrl=15, degree=2):
    Q = echantillonner_segment(segment, n=200)
    N = len(Q)
    t = np.linspace(0, 1, N)

    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ))

    init_ctrl = Q[np.linspace(0, N - 1, n_ctrl).astype(int)]

    def cost(ctrl_flat):
        ctrl = ctrl_flat.reshape((n_ctrl, 2))
        spline_x = BSpline(knots, ctrl[:, 0], degree)
        spline_y = BSpline(knots, ctrl[:, 1], degree)
        C = np.vstack((spline_x(t), spline_y(t))).T
        return (C - Q).ravel()

    result = least_squares(cost, init_ctrl.ravel())
    return result.x.reshape((n_ctrl, 2)), knots, t


# ----------------------------------------------------------------------
# Classification bosse / creux / plat
# ----------------------------------------------------------------------

def classifier_cote(ctrl_points, seuil_plat=0.08):
    y = ctrl_points[:, 1]
    y_max = np.max(y)
    y_min = np.min(y)

    if abs(y_max) < seuil_plat and abs(y_min) < seuil_plat:
        return 0
    if abs(y_max) > abs(y_min):
        return 1
    return 2


# ----------------------------------------------------------------------
# Construction de dict_ctrl à partir de analyser_piece + optimiser_segment
# ----------------------------------------------------------------------

def construire_dict_ctrl(fichier_image):
    data = analyser_piece(fichier_image, afficher=False)
    segments_norm = data["segments"]

    dict_ctrl_piece = []
    for seg_norm in segments_norm:
        ctrl_opt, _, _ = optimiser_segment(seg_norm)
        cat = classifier_cote(ctrl_opt)
        dict_ctrl_piece.append({"ctrl": ctrl_opt, "cat": cat})

    dict_ctrl = {0: dict_ctrl_piece}
    return dict_ctrl


# ----------------------------------------------------------------------
# Association de contours
# ----------------------------------------------------------------------

def distance_cotes(ctrlA, ctrlB):
    return np.linalg.norm(ctrlA - ctrlB)


def associer_pieces(dict_ctrl):
    cotes = []
    for piece_id, cotes_piece in dict_ctrl.items():
        for cote_id, info in enumerate(cotes_piece):
            if info["cat"] != 0:
                cotes.append((piece_id, cote_id))

    distances = {}
    for (pA, cA) in cotes:
        for (pB, cB) in cotes:
            if pA != pB:  # pièces différentes uniquement
                catA = dict_ctrl[pA][cA]["cat"]
                catB = dict_ctrl[pB][cB]["cat"]
                if (catA == 1 and catB == 2) or (catA == 2 and catB == 1):
                    ctrlA = dict_ctrl[pA][cA]["ctrl"]
                    ctrlB = dict_ctrl[pB][cB]["ctrl"]
                    distances[((pA, cA), (pB, cB))] = distance_cotes(ctrlA, ctrlB)

    associations = []
    associes = set()

    # On commence par la pièce 0
    cotes_piece0 = [(0, c) for c in range(len(dict_ctrl[0])) if dict_ctrl[0][c]["cat"] != 0]
    if len(cotes_piece0) == 0:
        print("La pièce 0 n'a aucun côté associable.")
        return []

    cote_actuel = cotes_piece0[0]

    while True:
        associes.add(cote_actuel)
        meilleur = None
        meilleure_dist = np.inf

        for (pB, cB) in cotes:
            if (pB, cB) not in associes and pB != cote_actuel[0]:
                key = (cote_actuel, (pB, cB))
                if key in distances:
                    d = distances[key]
                    if d < meilleure_dist:
                        meilleure_dist = d
                        meilleur = (pB, cB)

        if meilleur is None:
            break

        associations.append((cote_actuel, meilleur))
        cote_actuel = meilleur

        if len(associes) == len(cotes):
            break

    return associations


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    dossier = "./resources/"   # dossier contenant plusieurs images
    dict_ctrl = construire_dict_ctrl_pour_plusieurs_pieces(dossier)

    print("\n=== dict_ctrl construit pour toutes les pièces ===")
    for piece_id, cotes_piece in dict_ctrl.items():
        print(f"\nPièce {piece_id}:")
        for cote_id, info in enumerate(cotes_piece):
            print(f"  Côté {cote_id} : catégorie = {info['cat']}")

    associations = associer_pieces(dict_ctrl)

    print("\n=== Associations trouvées ===")
    for a in associations:
        print(a)

