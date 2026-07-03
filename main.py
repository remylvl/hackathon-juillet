"""
Analyse d'une pièce de puzzle à partir d'une photo + optimisation B-spline
+ construction de dict_ctrl + association des contours.

Ce fichier est autonome.

Pipeline global :
1. On charge la photo et on isole la pièce (fond coloré) via un masque HSV.
2. On nettoie ce masque (ouverture/fermeture morphologique + remplissage des trous).
3. On extrait le contour de la pièce et on détecte ses 4 coins.
4. On découpe le contour en 4 segments (un par côté).
5. On normalise chaque segment dans un repère "coin à coin" (pour comparer
   des pièces de tailles/orientations différentes).
6. On approxime chaque côté par une B-spline (points de contrôle optimisés
   par moindres carrés).
7. On classe chaque côté (plat / bosse / creux) et on essaie d'associer les
   côtés complémentaires entre plusieurs pièces (reconstruction du puzzle).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import ndimage
from scipy.interpolate import splprep, splev, BSpline
from scipy.optimize import least_squares, linear_sum_assignment
from skimage.feature import corner_harris, corner_peaks
from skimage import measure
from skimage.color import rgb2hsv
import os
import sys
from pathlib import Path

# ============================================================
# Paramètres globaux
# ============================================================

AFFICHER_GRAPHIQUES = True  # passe à False pour désactiver tous les plt.show()

# Seuils HSV utilisés pour repérer les pixels "colorés" (fond de la pièce)
SAT_MIN = 0.20          # saturation minimale pour qu'un pixel soit considéré comme "coloré"
VAL_MIN = 0.20          # luminosité (value) minimale idem
LARGEUR_HUE = 0.06      # demi-largeur de la fenêtre de teinte autour du pic détecté

NB_COINS = 4            # une pièce de puzzle a 4 coins

# --- Anciens seuils fixes (en pixels), conservés uniquement comme valeurs de
# repli si jamais on appelle max_courbure()/nettoyer_masque() en dehors du
# pipeline principal, sans passer par le calcul adaptatif ci-dessous. ---
SEUIL_COURBURE = 50       # nombre de points voisins utilisés pour estimer la courbure locale
DISTANCE_MIN_COINS = 200  # distance minimale (en pixels) entre deux coins retenus

# --- Ratios utilisés pour adapter automatiquement ces seuils à la résolution
# réelle de chaque image (voir `parametres_adaptatifs_masque` et
# `parametres_adaptatifs_coins` plus bas). Exprimés en proportion de la
# taille de l'image / du périmètre du contour, ils évitent de re-calibrer
# les constantes à la main à chaque changement de résolution de photo. ---
RATIO_OUVERTURE_MASQUE = 0.003   # taille du noyau d'ouverture ≈ 0.3 % de la petite dimension
RATIO_FERMETURE_MASQUE = 0.02    # taille du noyau de fermeture ≈ 2 % de la petite dimension
TAILLE_OUVERTURE_MIN = 3
TAILLE_FERMETURE_MIN = 5

RATIO_SEUIL_COURBURE = 0.015     # fenêtre de courbure ≈ 1.5 % du périmètre du contour
RATIO_DISTANCE_MIN_COINS = 0.06  # distance mini entre coins ≈ 6 % du périmètre du contour
SEUIL_COURBURE_MIN = 10
DISTANCE_MIN_COINS_MIN = 20

SIGMA_LISSAGE_CONTOUR = 10  # écart-type du filtre gaussien appliqué au contour brut
RAYON_MINIMA_LOCAUX = 5     # demi-largeur de la fenêtre utilisée pour repérer les minima locaux de courbure

# --- Dossier contenant les images des pièces à analyser ---
# Construit à partir de l'emplacement de CE fichier (et non du dossier
# courant depuis lequel on lance `python main.py`), pour éviter les
# FileNotFoundError liés au dossier de travail : avec un chemin relatif
# comme "./resources/...", ça ne marche que si on lance le script depuis le
# bon dossier. Avec Path(__file__), ça marche depuis n'importe où.
DOSSIER_SCRIPT = Path(__file__).resolve().parent
DOSSIER_IMAGES = DOSSIER_SCRIPT / "resources" / "n_pieces_ensembles"


# ============================================================
# 1. Chargement de l'image et masque de couleur
# ============================================================

def charger_image(chemin):
    """
    Charge une image depuis le disque et la convertit en HSV.

    Retourne :
        img    : image RGB brute (telle que lue par matplotlib)
        h, s, v: les 3 canaux Teinte / Saturation / Valeur, chacun en 2D
    """
    img = image.imread(chemin)
    img_hsv = rgb2hsv(img[:, :, :3])  # on passe du RGB au HSV (on ignore un éventuel canal alpha)
    return img, img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]


def creer_masque_bleu(img_h, img_s, img_v,
                       sat_min=SAT_MIN, val_min=VAL_MIN,
                       largeur_hue=LARGEUR_HUE):
    """
    Construit un masque binaire qui isole le fond coloré de la pièce
    (historiquement bleu, d'où le nom, mais la teinte est en fait
    détectée automatiquement).

    Étapes :
        1. On repère les pixels "candidats" = suffisamment saturés et lumineux.
        2. On construit un histogramme de leur teinte (hue) pour trouver
           la teinte dominante (le pic de l'histogramme).
        3. On garde uniquement les pixels dont la teinte est proche de ce pic.

    Retourne :
        masque   : tableau booléen 2D (True = pixel du fond coloré)
        h_centre : la teinte dominante détectée (utile pour le débogage/affichage)
    """
    candidat = (img_s > sat_min) & (img_v > val_min)
    if not np.any(candidat):
        raise ValueError("Aucun pixel suffisamment saturé pour calibrer le masque.")

    h_candidats = img_h[candidat]
    hist, bins = np.histogram(h_candidats, bins=60, range=(0.0, 1.0))
    i_pic = np.argmax(hist)                       # bin le plus fréquent
    h_centre = 0.5 * (bins[i_pic] + bins[i_pic + 1])  # centre de ce bin

    # On ne garde que les pixels dont la teinte est à moins de `largeur_hue` du pic
    masque = candidat & (np.abs(img_h - h_centre) <= largeur_hue)
    return masque, h_centre


# ============================================================
# 2. Nettoyage du masque
# ============================================================

def parametres_adaptatifs_masque(masque):
    """
    Calcule les tailles de noyaux morphologiques (ouverture/fermeture) en
    fonction de la résolution réelle de l'image, plutôt que d'utiliser des
    tailles fixes en pixels (3x3 / 20x20) calibrées pour une seule résolution.

    Sans cela, des photos prises à une résolution différente feraient soit
    trop de nettoyage (perte de détails), soit pas assez (bruit résiduel).
    """
    dim_min = min(masque.shape)
    taille_ouverture = max(TAILLE_OUVERTURE_MIN, int(round(dim_min * RATIO_OUVERTURE_MASQUE)))
    taille_fermeture = max(TAILLE_FERMETURE_MIN, int(round(dim_min * RATIO_FERMETURE_MASQUE)))
    return taille_ouverture, taille_fermeture


def nettoyer_masque(masque):
    """
    Nettoie le masque brut pour ne garder qu'une seule pièce propre et pleine.

    - Ouverture morphologique : supprime le bruit isolé (petits points parasites).
    - Fermeture morphologique : rebouche les petits trous/fentes dans le contour.
    - On garde uniquement la plus grande composante connexe (= la pièce).
    - On remplit les trous internes (ex : reflets qui cassaient la couleur).

    Les tailles des noyaux morphologiques sont calculées automatiquement en
    fonction de la résolution de l'image (voir `parametres_adaptatifs_masque`),
    ce qui évite d'avoir à recalibrer des constantes fixes à chaque changement
    de taille de photo.
    """
    taille_ouverture, taille_fermeture = parametres_adaptatifs_masque(masque)
    masque_propre = ndimage.binary_opening(masque, structure=np.ones((taille_ouverture, taille_ouverture)))
    masque_propre = ndimage.binary_closing(masque_propre, structure=np.ones((taille_fermeture, taille_fermeture)))

    # Étiquetage des composantes connexes du masque
    labels, nb = ndimage.label(masque_propre)
    tailles = ndimage.sum(masque_propre, labels, range(1, nb + 1))
    if len(tailles) == 0:
        raise ValueError("Aucune composante détectée : ajuste les seuils HSV du masque.")

    # On suppose que la pièce est la plus grande zone colorée détectée
    plus_grande = np.argmax(tailles) + 1
    masque_piece = labels == plus_grande

    # On rebouche les éventuels trous internes (reflets, ombres, etc.)
    masque_final = ndimage.binary_fill_holes(masque_piece)
    return masque_final


def extraire_bord(masque_final):
    """
    Extrait les pixels de bord (frontière) du masque, en soustrayant
    le masque érodé d'un pixel au masque original.

    Retourne :
        bord : masque booléen des pixels de bord
        X, Y : listes des coordonnées (lignes, colonnes) de ces pixels de bord
    """
    masque_erode = ndimage.binary_erosion(masque_final)
    bord = masque_final & ~masque_erode
    X, Y = np.nonzero(bord)
    return bord, list(X), list(Y)


# ============================================================
# 3. Contour et détection des coins
# ============================================================

def detecter_coins_harris(masque_final, min_distance=150, threshold_rel=0.1):
    """
    Détecte des points de coin candidats avec le détecteur de Harris
    (utilisé ici surtout à titre indicatif/diagnostic, en complément
    de la méthode par courbure utilisée plus bas).
    """
    reponse = corner_harris(masque_final.astype(float))
    coins = corner_peaks(reponse, min_distance=min_distance, threshold_rel=threshold_rel)
    return coins


def extraire_contour_principal(masque_final):
    """
    Extrait le contour (liste ordonnée de points) autour du masque, au
    niveau 0.5 (frontière entre 0 et 1), et retourne le plus long contour
    trouvé (celui qui correspond au pourtour de la pièce).
    """
    contours = measure.find_contours(masque_final, level=0.5)
    return max(contours, key=len)


def lisser_contour(contour, sigma=7):
    """
    Lisse le contour (filtre gaussien 1D) séparément sur chaque coordonnée,
    en mode "wrap" car le contour est une courbe fermée.
    """
    contour_ferme = np.vstack([contour, contour[:1]])  # on referme la boucle pour le filtrage
    row_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 0], sigma=sigma, mode="wrap")[:-1]
    col_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 1], sigma=sigma, mode="wrap")[:-1]
    return np.column_stack([row_lisse, col_lisse])


def trouver_indice(contour, point):
    """
    Trouve l'indice, dans `contour`, du point le plus proche du point donné.
    Utile pour faire correspondre un point détecté par une autre méthode
    (ex : coin Harris, point simplifié) à sa position dans le contour original.
    """
    distances = np.sqrt((contour[:, 0] - point[0]) ** 2 + (contour[:, 1] - point[1]) ** 2)
    return np.argmin(distances)


def score_courbure(contour, index, seuil):
    """
    Calcule un score de courbure locale au point d'indice `index` du contour,
    en comparant les vecteurs vers les points situés `seuil` pas avant et
    `seuil` pas après (produit scalaire cumulé sur une fenêtre).

    Attention au sens de lecture du score : sur un tronçon bien droit, les
    vecteurs vers les deux voisins pointent dans des directions quasi
    opposées, ce qui donne un produit scalaire très négatif et donc un score
    (valeur absolue) ÉLEVÉ. À l'inverse, sur un coin bien marqué (angle
    proche de 90°), les deux vecteurs sont quasi perpendiculaires, leur
    produit scalaire est proche de 0, donc le score est FAIBLE.

    → Un coin se traduit par un score bas, pas haut : ce sont les MINIMA de
    cette fonction (le long du contour) qui correspondent aux coins, d'où
    l'usage de `trouver_minima_locaux` ci-dessous plutôt qu'une simple
    recherche de valeurs élevées.
    """
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


def trouver_minima_locaux(scores, rayon=5):
    """
    Renvoie les indices des minima locaux d'une courbe 1D **cyclique**
    (un contour est une boucle fermée, le premier et le dernier point
    sont voisins).

    Un point `i` est un minimum local si son score est le plus petit parmi
    tous les points situés à `rayon` pas de lui de chaque côté (fenêtre
    glissante de largeur `2*rayon + 1`, avec repliement circulaire).

    Cette fonction remplace avantageusement `measure.approximate_polygon`
    comme source de candidats-coins : au lieu de dépendre d'une tolérance
    de simplification de polygone (qui peut décaler ou carrément rater un
    coin selon sa valeur), elle repère directement, de façon purement
    géométrique, tous les creux locaux du score de courbure — c'est-à-dire
    tous les points qui *pourraient* être des coins, y compris des coins
    « secondaires » moins marqués que les 4 coins principaux.
    """
    n = len(scores)
    minima = []
    for i in range(n):
        indices = [(i + k) % n for k in range(-rayon, rayon + 1)]
        if scores[i] == np.min(scores[indices]):
            minima.append(i)
    return np.array(minima, dtype=int)


def parametres_adaptatifs_coins(contour):
    """
    Calcule `seuil_courbure` et `distance_min_coins` proportionnellement au
    périmètre du contour (nombre de points), plutôt que d'utiliser des
    constantes fixes en pixels (SEUIL_COURBURE=50, DISTANCE_MIN_COINS=200).

    Une image à plus haute résolution donne un contour avec beaucoup plus de
    points : avec des seuils fixes, la fenêtre de courbure deviendrait trop
    étroite (bruit) et la distance minimale entre coins trop petite (coins
    dupliqués). En les exprimant en proportion du périmètre, la détection de
    coins reste cohérente quelle que soit la résolution de la photo.
    """
    perimetre = len(contour)
    seuil_courbure = max(SEUIL_COURBURE_MIN, int(round(perimetre * RATIO_SEUIL_COURBURE)))
    distance_min_coins = max(DISTANCE_MIN_COINS_MIN, int(round(perimetre * RATIO_DISTANCE_MIN_COINS)))
    return seuil_courbure, distance_min_coins


def max_courbure(contour, points, seuil, distance_min, nb_coins=NB_COINS, contour_score=None):
    """
    Sélectionne les `nb_coins` points de plus forte courbure (donc de score
    le plus BAS, voir `score_courbure`) parmi une liste de points candidats,
    en imposant une distance minimale entre eux (pour éviter de choisir
    plusieurs points trop proches sur le même coin).

    Version robuste (par rapport à un simple glouton "premier arrivé,
    premier servi") :
      - la sélection n'est pas strictement gloutonne : si un meilleur
        candidat (score plus bas) apparaît près d'un point déjà retenu, il
        peut le REMPLACER au lieu d'être rejeté (`ajouter_candidat`) ;
      - si la distance minimale demandée est trop stricte pour trouver
        `nb_coins` points distincts sur ce contour, on retente avec un seuil
        de distance progressivement réduit (`distance_effective`, sa
        moitié, puis 0), au lieu d'échouer avec moins de coins que prévu ;
      - en tout dernier recours, on complète avec les meilleurs points
        restants même s'ils sont proches d'un point déjà choisi, pour
        garantir `nb_coins` coins en sortie.

    Paramètres
    ----------
    contour : contour utilisé pour lire les COORDONNÉES des points retenus
        (typiquement le contour lissé).
    contour_score : contour utilisé pour CALCULER le score de courbure de
        chaque candidat (généralement identique à `contour`, mais peut être
        différent si l'on veut découpler les deux usages). Si None,
        `contour` est utilisé pour les deux.

    Retourne :
        points_filtres : coordonnées (lignes, colonnes) des coins retenus
        i_filtres      : indices correspondants dans le contour
    """
    if len(points) == 0:
        return np.array([]), np.array([])

    contour_pour_score = contour if contour_score is None else contour_score
    scores = np.array([score_courbure(contour_pour_score, index, seuil) for index in points])
    ordre = np.argsort(scores)  # du score le plus faible (coin marqué) au plus élevé (tout droit)

    # On ne s'impose pas une distance minimale plus grande qu'un quart du
    # périmètre par coin : sur un petit contour, une distance_min fixe
    # pourrait sinon être irréaliste et empêcher de trouver nb_coins points.
    distance_effective = min(
        distance_min,
        max(30, len(contour_pour_score) // max(2 * nb_coins, 1))
    )

    selection = []

    def ajouter_candidat(index, point, score, seuil_distance):
        """Ajoute un candidat, ou remplace le pire candidat voisin si ce
        nouveau point est meilleur (score plus bas) et trop proche de lui."""
        nonlocal selection
        proches = [
            k for k, item in enumerate(selection)
            if np.sqrt((item["point"][0] - point[0]) ** 2 + (item["point"][1] - point[1]) ** 2) < seuil_distance
        ]

        if proches:
            k_pire = max(proches, key=lambda k: selection[k]["score"])
            if score < selection[k_pire]["score"]:
                selection[k_pire] = {"index": index, "point": point, "score": score}
        elif len(selection) < nb_coins:
            selection.append({"index": index, "point": point, "score": score})
        elif score < max(selection, key=lambda item: item["score"])["score"]:
            k_pire = max(range(len(selection)), key=lambda k: selection[k]["score"])
            selection[k_pire] = {"index": index, "point": point, "score": score}

    # On retente avec une distance minimale de plus en plus permissive tant
    # qu'on n'a pas réussi à réunir nb_coins points distincts.
    for seuil_distance in (distance_effective, max(20, distance_effective // 2), 0):
        selection = []
        for pos in ordre:
            index = int(points[pos])
            point = contour[index]
            score = float(scores[pos])
            ajouter_candidat(index, point, score, seuil_distance)
            if len(selection) == nb_coins:
                break
        if len(selection) == nb_coins:
            break

    # Filet de sécurité final : si malgré tout on a moins de nb_coins points
    # (candidats trop peu nombreux), on complète avec les meilleurs restants.
    if len(selection) < nb_coins:
        deja = {item["index"] for item in selection}
        for pos in ordre:
            index = int(points[pos])
            if index in deja:
                continue
            selection.append({"index": index, "point": contour[index], "score": float(scores[pos])})
            if len(selection) == nb_coins:
                break

    selection.sort(key=lambda item: item["index"])  # on remet dans l'ordre le long du contour
    points_filtres = np.array([item["point"] for item in selection])
    i_filtres = np.array([item["index"] for item in selection], dtype=int)
    return points_filtres, i_filtres


# ============================================================
# 4. Découpage en segments
# ============================================================

def extraire_segments(contour, indices):
    """
    Découpe le contour fermé en segments délimités par les indices de coins
    donnés (dans l'ordre). Gère le cas où un segment "boucle" (passe par
    l'indice 0 du tableau, car le contour est cyclique).
    """
    segments = []
    n = len(indices)
    for k in range(n):
        i_debut = indices[k]
        i_fin = indices[(k + 1) % n]
        if i_fin > i_debut:
            segment = contour[i_debut:i_fin + 1]
        else:
            # Le segment "traverse" la fin du tableau : on concatène la fin et le début
            segment = np.vstack([contour[i_debut:], contour[:i_fin + 1]])
        segments.append(segment)
    return segments


# ============================================================
# 5. Ajustement de splines
# ============================================================

def nettoyer_doublons_consecutifs(segment):
    """
    Supprime les points consécutifs strictement identiques dans un segment
    (nécessaire car splprep échoue si deux points consécutifs sont dupliqués).
    """
    garder = [segment[0]]
    for p in segment[1:]:
        if not np.array_equal(p, garder[-1]):
            garder.append(p)
    return np.array(garder)


def ajuster_spline_segment(segment, lissage=0):
    """
    Ajuste une spline paramétrique (via splprep de scipy) sur un segment de points.
    `lissage` (paramètre `s` de splprep) contrôle le compromis fidélité/lissage :
    0 = interpolation exacte, valeur plus grande = courbe plus lisse.
    """
    segment = nettoyer_doublons_consecutifs(segment)
    x = segment[:, 1]
    y = segment[:, 0]
    tck, u = splprep([x, y], s=lissage)
    return tck


# ============================================================
# 6. Normalisation coin-à-coin
# ============================================================

def normaliser_segment(segment):
    """
    Exprime un segment (un côté de la pièce) dans un repère local :
    - l'axe "along" va du premier au dernier point du segment (les deux coins),
      normalisé à une longueur de 1.
    - l'axe "height" mesure l'écart perpendiculaire à cette droite coin-à-coin.

    Cela permet de comparer la forme d'un côté (plat/bosse/creux)
    indépendamment de la taille, position et orientation de la pièce.
    """
    x = segment[:, 1].astype(float)
    y = segment[:, 0].astype(float)

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    px = x - x0
    py = y - y0

    vx, vy = x1 - x0, y1 - y0
    longueur = np.sqrt(vx ** 2 + vy ** 2)
    vx, vy = vx / longueur, vy / longueur

    along = px * vx + py * vy    # projection sur l'axe coin-à-coin
    height = px * vy - py * vx   # écart perpendiculaire (signé)

    if longueur > 0:
        along = along / longueur
        height = height / longueur

    return np.column_stack([along, height])


# ============================================================
# Affichages (optionnels)
# ============================================================

def afficher_contour_simplifie(X, Y, contour_simplifie):
    """Affiche le nuage de points de bord ainsi que le contour simplifié (polygone)."""
    plt.figure()
    plt.plot(X, Y, '.', markersize=1)
    plt.scatter(contour_simplifie[:, 0], contour_simplifie[:, 1], color='red', s=60)
    plt.axis("equal")
    plt.title("Contour simplifié")
    plt.show()


def afficher_segments(segments, coins_finaux):
    """Affiche chaque segment (côté) dans une couleur différente, avec les coins en noir."""
    plt.figure()
    couleurs = ['red', 'blue', 'green', 'orange']
    for seg, c in zip(segments, couleurs):
        plt.plot(seg[:, 1], seg[:, 0], '.', markersize=2, color=c)
    plt.scatter(coins_finaux[:, 1], coins_finaux[:, 0], color='black', s=80, zorder=5)
    plt.axis("equal")
    plt.gca().invert_yaxis()  # convention image : l'axe Y pointe vers le bas
    plt.title("Segments entre coins")
    plt.show()


def afficher_splines(segments, splines):
    """Superpose, pour chaque côté, les points bruts et la spline ajustée."""
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
    """Affiche les côtés normalisés (repère coin-à-coin) pour comparer leurs formes."""
    plt.figure()
    for i, seg in enumerate(segments):
        seg_norm = normaliser_segment(seg)
        plt.plot(seg_norm[:, 0], seg_norm[:, 1], label=f"Segment {i}")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend()
    plt.title("Segments normalisés (repère coin-à-coin)")
    plt.show()


# ============================================================
# Pipeline principal : analyser_piece
# ============================================================

def analyser_piece(fichier_image, afficher=AFFICHER_GRAPHIQUES):
    """
    Pipeline complet d'analyse d'une seule pièce à partir d'un fichier image :
    masque -> nettoyage -> contour -> coins -> segments -> splines -> normalisation.

    Retourne un dictionnaire contenant toutes les données intermédiaires utiles
    (masque final, contour, coins, segments normalisés, splines).
    """
    img, img_h, img_s, img_v = charger_image(fichier_image)

    # --- Masque de couleur (fond de la pièce) ---
    masque, h_centre = creer_masque_bleu(img_h, img_s, img_v)
    print(f"Teinte bleue détectée automatiquement : {h_centre:.3f}")

    masque_final = nettoyer_masque(masque)
    _, X, Y = extraire_bord(masque_final)

    # --- Contour principal, puis LISSAGE ---
    # Point important : le contour brut issu de measure.find_contours() suit
    # le pourtour pixel par pixel, donc il est bruité (marches d'escalier,
    # micro-aspérités). Calculer la courbure directement dessus produit des
    # scores instables, qui peuvent faire "sauter" les coins détectés d'une
    # image à l'autre. On lisse donc le contour AVANT tout calcul de courbure,
    # et on travaille ensuite sur ce contour lissé pour TOUTES les étapes en
    # aval (coins, segments, splines) — c'est ce contour lissé qui est
    # renvoyé comme "contour_principal" dans le résultat.
    contour_brut = extraire_contour_principal(masque_final)
    contour_lisse = lisser_contour(contour_brut, sigma=SIGMA_LISSAGE_CONTOUR)

    # Conservé uniquement à des fins d'affichage/diagnostic (voir plus bas) :
    # ce n'est plus la source des candidats-coins (voir trouver_minima_locaux).
    contour_simplifie = measure.approximate_polygon(contour_lisse, tolerance=10)

    if afficher:
        afficher_contour_simplifie(X, Y, contour_simplifie)

    # --- Détection de coins Harris, utilisés comme candidats supplémentaires ---
    coins_harris = detecter_coins_harris(masque_final)
    print(f"{len(coins_harris)} coins détectés (Harris)")
    indices_harris = [trouver_indice(contour_lisse, c) for c in coins_harris]

    # --- Candidats principaux : minima locaux du score de courbure, calculés
    # sur TOUT le contour lissé (voir trouver_minima_locaux). Cette approche
    # est plus fiable que measure.approximate_polygon : elle ne dépend pas
    # d'une tolérance de simplification qui peut décaler ou rater un coin,
    # et repère directement, point par point, les creux de courbure. ---
    seuil_courbure, distance_min_coins = parametres_adaptatifs_coins(contour_lisse)
    scores_courbure = np.array([
        score_courbure(contour_lisse, i, seuil_courbure) for i in range(len(contour_lisse))
    ])
    indices_minima = trouver_minima_locaux(scores_courbure, rayon=RAYON_MINIMA_LOCAUX)

    # Filet de sécurité : si le lissage ou la géométrie de la pièce ne
    # laissent émerger que très peu de minima locaux (< NB_COINS), on retombe
    # sur l'ensemble de tous les points du contour comme candidats, pour ne
    # jamais se retrouver sans assez de candidats à donner à max_courbure.
    if len(indices_minima) < NB_COINS:
        indices_minima = np.arange(len(contour_lisse), dtype=int)

    # Fusion avec les coins Harris, sans doublons
    indices_points = sorted(set(indices_minima.tolist()) | set(indices_harris))

    # --- Sélection finale des 4 coins par score de courbure ---
    coins_finaux, indices_coins = max_courbure(
        contour_lisse, indices_points, seuil_courbure, distance_min_coins,
        contour_score=contour_lisse
    )
    indices_coins = np.sort(indices_coins)  # on remet les indices dans l'ordre le long du contour
    print(f"{len(coins_finaux)} coins retenus après filtrage par courbure "
          f"(seuil_courbure={seuil_courbure}, distance_min_coins={distance_min_coins})")

    # --- Découpage en 4 segments (côtés) entre coins successifs ---
    # On découpe le contour LISSÉ (et non le contour brut) : les splines
    # ajustées ensuite sur ces segments sont donc elles aussi plus régulières.
    segments = extraire_segments(contour_lisse, indices_coins)
    for i, seg in enumerate(segments):
        print(f"Segment {i} : {len(seg)} points")

    if afficher:
        afficher_segments(segments, coins_finaux)

    # --- Ajustement d'une spline lissée sur chaque côté ---
    splines = [ajuster_spline_segment(seg, lissage=len(seg) * 2) for seg in segments]

    if afficher:
        afficher_splines(segments, splines)
        afficher_segments_normalises(segments)

    # --- Normalisation de chaque côté dans le repère coin-à-coin ---
    segments_norm = [normaliser_segment(seg) for seg in segments]

    return {
        "masque_final": masque_final,
        "contour_principal": contour_lisse,
        "coins": coins_finaux,
        "segments": segments_norm,
        "splines": splines,
    }


# ============================================================
# B-spline : ré-échantillonnage + optimisation
# ============================================================

def echantillonner_segment(segment, n=200):
    """
    Ré-échantillonne un segment de points à intervalles réguliers le long
    de son abscisse curviligne (distance cumulée), pour obtenir `n` points
    uniformément répartis. Nécessaire avant l'optimisation des points de
    contrôle de la B-spline, pour avoir une paramétrisation stable.
    """
    d = np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1))
    u = np.concatenate([[0], np.cumsum(d)])
    u = u / u[-1]  # normalisation entre 0 et 1

    from scipy.interpolate import interp1d
    fx = interp1d(u, segment[:, 0])
    fy = interp1d(u, segment[:, 1])

    u_new = np.linspace(0, 1, n)
    return np.column_stack([fx(u_new), fy(u_new)])


def optimiser_segment(segment, n_ctrl=15, degree=2):
    """
    Trouve, par moindres carrés, les `n_ctrl` points de contrôle d'une
    B-spline de degré `degree` qui approxime au mieux le segment donné.

    Étapes :
        1. Ré-échantillonnage régulier du segment (Q, n=200 points).
        2. Construction d'un vecteur de nœuds (knots) uniforme, avec
           multiplicité `degree` aux extrémités (spline ouverte/clampée).
        3. Initialisation des points de contrôle par sous-échantillonnage
           direct de Q.
        4. Minimisation de l'écart entre la courbe B-spline évaluée en `t`
           et les points cibles Q (least_squares sur les coordonnées des
           points de contrôle).

    Retourne :
        ctrl  : points de contrôle optimisés, shape (n_ctrl, 2)
        knots : vecteur de nœuds utilisé
        t     : paramètres d'évaluation utilisés pour la comparaison
    """
    Q = echantillonner_segment(segment, n=200)
    N = len(Q)
    t = np.linspace(0, 1, N)

    # Vecteur de nœuds : `degree` répétitions à chaque extrémité (spline clampée)
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ))

    # Initialisation : on prend n_ctrl points régulièrement espacés dans Q
    init_ctrl = Q[np.linspace(0, N - 1, n_ctrl).astype(int)]

    def cost(ctrl_flat):
        """Fonction de coût : résidus (C - Q) aplatis, pour least_squares."""
        ctrl = ctrl_flat.reshape((n_ctrl, 2))
        spline_x = BSpline(knots, ctrl[:, 0], degree)
        spline_y = BSpline(knots, ctrl[:, 1], degree)
        C = np.vstack((spline_x(t), spline_y(t))).T
        return (C - Q).ravel()

    result = least_squares(cost, init_ctrl.ravel())
    return result.x.reshape((n_ctrl, 2)), knots, t


# ============================================================
# Création du dictionnaire de contrôle (dict_ctrl)
# ============================================================

def construire_dict_ctrl_pour_plusieurs_pieces(dossier_images):
    """
    Analyse toutes les images d'un dossier et construit dict_ctrl
    pour chaque pièce. Affiche les splines optimisées au fur et à mesure.

    dict_ctrl a la forme :
        {
            piece_id: [
                {"ctrl": points_de_controle, "cat": categorie_du_cote},
                ... (un élément par côté)
            ],
            ...
        }
    """
    dict_ctrl = {}
    fichiers = sorted(os.listdir(dossier_images))

    piece_id = 0
    for fichier in fichiers:
        if fichier.lower().endswith((".png", ".jpg", ".jpeg")):
            chemin = os.path.join(dossier_images, fichier)
            print(f"\n--- Analyse de la pièce {piece_id} : {chemin} ---")

            # 1. Analyse de la pièce (masque, contour, coins, segments normalisés)
            data = analyser_piece(chemin, afficher=False)
            segments_norm = data["segments"]

            dict_ctrl_piece = []

            # 2. Optimisation de la B-spline + affichage, pour chaque côté
            for cote_id, seg_norm in enumerate(segments_norm):
                ctrl_opt, knots, t = optimiser_segment(seg_norm)

                # Affichage de la spline optimisée par-dessus le segment normalisé
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

                # Classification du côté : plat / bosse / creux
                cat = classifier_cote(ctrl_opt)

                dict_ctrl_piece.append({
                    "ctrl": ctrl_opt,
                    "cat": cat
                })

            dict_ctrl[piece_id] = dict_ctrl_piece
            piece_id += 1

    return dict_ctrl


# ============================================================
# Classification bosse / creux / plat
# ============================================================

def classifier_cote(ctrl_points, seuil_plat=0.08):
    """
    Classe un côté de pièce en 3 catégories selon l'amplitude de ses
    points de contrôle sur l'axe "height" (perpendiculaire au côté) :
        0 = plat  (bord extérieur du puzzle, aucune saillie ni creux notable)
        1 = bosse (le côté dépasse vers l'extérieur, y_max dominant)
        2 = creux (le côté rentre vers l'intérieur, y_min dominant)
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
# Construction de dict_ctrl à partir d'une seule pièce
# ============================================================

def construire_dict_ctrl(fichier_image):
    """
    Variante mono-pièce de construire_dict_ctrl_pour_plusieurs_pieces :
    analyse une seule image et retourne un dict_ctrl avec un seul id (0).
    """
    data = analyser_piece(fichier_image, afficher=False)
    segments_norm = data["segments"]

    dict_ctrl_piece = []
    for seg_norm in segments_norm:
        ctrl_opt, _, _ = optimiser_segment(seg_norm)
        cat = classifier_cote(ctrl_opt)
        dict_ctrl_piece.append({"ctrl": ctrl_opt, "cat": cat})

    dict_ctrl = {0: dict_ctrl_piece}
    return dict_ctrl


# ============================================================
# Association de contours (reconstruction du puzzle)
# ============================================================

def distance_cotes(ctrlA, ctrlB):
    """
    Mesure de dissemblance entre deux côtés destinés à s'emboîter
    (une bosse d'un côté, un creux de l'autre).

    Avant de comparer, on remet ctrlB dans le même référentiel géométrique
    que ctrlA :
        - les deux côtés sont parcourus en sens opposés le long de leurs
          contours respectifs (l'un dans le sens horaire, l'autre
          anti-horaire une fois les deux pièces rapprochées) : on inverse
          donc l'ordre des points de contrôle de B ("along" inversé).
        - un creux qui s'emboîte parfaitement dans une bosse a une forme en
          "négatif" : la hauteur (axe perpendiculaire au côté) doit être
          inversée en signe pour que les deux profils se superposent.

    Sans ces deux transformations, on comparait directement deux courbes
    qui ne sont jamais alignées (même une paire parfaitement complémentaire
    aurait une grande distance), ce qui faussait l'association des pièces.

    Plus la distance obtenue est faible, plus les côtés sont complémentaires.
    """
    ctrlB_aligne = ctrlB[::-1].copy()      # on inverse le sens de parcours
    ctrlB_aligne[:, 1] = -ctrlB_aligne[:, 1]  # on inverse le signe de la hauteur (effet miroir)
    return np.linalg.norm(ctrlA - ctrlB_aligne)


# Coût prohibitif utilisé pour empêcher une pièce de s'auto-associer
# (une bosse et un creux appartenant à la même pièce). On ne peut pas
# simplement "retirer" ces cases de la matrice de coût car
# linear_sum_assignment exige une matrice rectangulaire complète : on les
# rend donc juste extrêmement coûteuses pour qu'elles ne soient choisies
# que si vraiment aucune autre option n'existe (et on les filtre après coup).
PENALITE_MEME_PIECE = 1e6


def associer_pieces(dict_ctrl):
    """
    Associe les côtés complémentaires (bosse <-> creux) entre pièces
    différentes, en résolvant un problème d'affectation optimale globale
    via l'algorithme hongrois (`scipy.optimize.linear_sum_assignment`).

    Contrairement à une approche gloutonne (qui associe au fur et à mesure
    le meilleur partenaire encore disponible, au risque de "gâcher" une
    bonne paire trouvée plus tard dans le parcours), l'algorithme hongrois
    minimise la somme totale des distances sur l'ensemble des associations
    en une seule résolution : le résultat est optimal globalement, pas
    seulement localement.

    Le problème est naturellement biparti : on associe l'ensemble des
    côtés "bosse" (catégorie 1) à l'ensemble des côtés "creux"
    (catégorie 2). La matrice de coût contient, pour chaque paire
    (bosse, creux), la distance calculée par `distance_cotes` (qui réaligne
    déjà les deux profils avant de les comparer). Les paires appartenant à
    la même pièce reçoivent un coût prohibitif (`PENALITE_MEME_PIECE`) pour
    ne jamais être retenues, sauf en dernier recours si aucune autre
    option n'existe — auquel cas elles sont filtrées après résolution.

    Retourne une liste de paires ((pieceA, coteA), (pieceB, coteB)), comme
    l'ancienne version gloutonne (même format, compatible avec
    `visualiser_schema_pieces`).
    """
    bosses = []
    creux = []
    for pid, cotes_piece in dict_ctrl.items():
        for cid, info in enumerate(cotes_piece):
            if info["cat"] == 1:
                bosses.append((pid, cid))
            elif info["cat"] == 2:
                creux.append((pid, cid))

    if not bosses or not creux:
        return []

    # Matrice de coût : lignes = bosses, colonnes = creux
    cout = np.empty((len(bosses), len(creux)))
    for i, (pA, cA) in enumerate(bosses):
        for j, (pB, cB) in enumerate(creux):
            if pA == pB:
                cout[i, j] = PENALITE_MEME_PIECE
            else:
                cout[i, j] = distance_cotes(
                    dict_ctrl[pA][cA]["ctrl"],
                    dict_ctrl[pB][cB]["ctrl"]
                )

    # Résolution du problème d'affectation optimale (algorithme hongrois).
    # Fonctionne même si le nombre de bosses diffère du nombre de creux
    # (matrice rectangulaire) : on obtient alors min(len(bosses), len(creux))
    # associations, ce qui est le comportement souhaité.
    lignes, colonnes = linear_sum_assignment(cout)

    associations = []
    for i, j in zip(lignes, colonnes):
        # On écarte les associations "forcées" entre côtés d'une même pièce
        # (n'apparaissent que si le solveur n'avait vraiment aucun autre choix).
        if cout[i, j] >= PENALITE_MEME_PIECE:
            continue
        associations.append((bosses[i], creux[j]))

    return associations


# ============================================================
# Visualisation détaillée des résultats d'une pièce
# ============================================================

def afficher_resultats_piece(resultats, titre="Résultats pièce"):
    """
    Affiche une visualisation claire et compacte (grille 2x2) des résultats
    de l'analyse d'une pièce :
        1. masque final
        2. contour + coins détectés
        3. segments + splines ajustées
        4. segments normalisés (repère coin-à-coin)
    """

    masque = resultats["masque_final"]
    contour = resultats["contour_principal"]
    coins = resultats["coins"]
    segments = resultats["segments"]
    splines = resultats["splines"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(titre, fontsize=16, fontweight="bold")

    # 1. Masque final
    ax = axes[0, 0]
    ax.imshow(masque, cmap="gray")
    ax.set_title("1. Masque final")
    ax.axis("off")

    # 2. Contour + coins
    ax = axes[0, 1]
    ax.plot(contour[:, 1], contour[:, 0], '.', markersize=1, color="gray", alpha=0.5)
    if len(coins) > 0:
        ax.scatter(coins[:, 1], coins[:, 0], color="red", s=80, label="Coins détectés")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("2. Contour + coins")
    ax.legend()

    # 3. Segments + splines
    couleurs = ["red", "blue", "green", "orange", "purple", "brown"]
    ax = axes[1, 0]
    for i, (seg, tck) in enumerate(zip(segments, splines)):
        col = couleurs[i % len(couleurs)]
        ax.plot(seg[:, 1], seg[:, 0], '.', markersize=2, color=col, alpha=0.4)
        u = np.linspace(0, 1, 200)
        x_fit, y_fit = splev(u, tck)
        ax.plot(x_fit, y_fit, '-', linewidth=2, color=col, label=f"Côté {i}")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("3. Segments + splines")
    ax.legend(fontsize=8)

    # 4. Segments normalisés
    ax = axes[1, 1]
    for i, seg in enumerate(segments):
        col = couleurs[i % len(couleurs)]
        seg_norm = normaliser_segment(seg)
        ax.plot(seg_norm[:, 0], seg_norm[:, 1], color=col, label=f"Côté {i}")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_aspect("equal")
    ax.set_title("4. Segments normalisés")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def visualiser_schema_pieces(dict_ctrl, associations):
    """
    Visualisation schématique des pièces et de leurs côtés associés.
    Chaque pièce est représentée par un carré (placé sur un cercle,
    répartition purement esthétique, sans lien avec la position réelle
    dans le puzzle final).
    Chaque côté du carré (haut/droite/bas/gauche) représente un côté de la pièce.
    Les associations trouvées par `associer_pieces` sont tracées en pointillés rouges
    entre les côtés concernés.
    """

    n_pieces = len(dict_ctrl)

    # Placement radial des pièces (juste pour l'affichage, angle égal entre chaque pièce)
    angle_step = 2 * np.pi / max(1, n_pieces)
    radius = 5

    positions = {}
    for pid in range(n_pieces):
        angle = pid * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[pid] = (x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Schéma des pièces et de leurs associations")

    # --- Dessin des pièces ---
    for pid, (x, y) in positions.items():
        # Carré représentant la pièce
        ax.add_patch(plt.Rectangle((x - 1, y - 1), 2, 2,
                                   fill=False, linewidth=2))
        ax.text(x, y, f"Pièce {pid}", ha="center", va="center", fontsize=12)

        # Côtés du carré, dans l'ordre : haut, droite, bas, gauche
        cotes = [
            ((x - 1, y + 1), (x + 1, y + 1)),  # haut
            ((x + 1, y + 1), (x + 1, y - 1)),  # droite
            ((x - 1, y - 1), (x + 1, y - 1)),  # bas
            ((x - 1, y + 1), (x - 1, y - 1)),  # gauche
        ]

        # On stocke la position des côtés directement dans dict_ctrl (dernier élément de la liste)
        dict_ctrl[pid].append({"schema_cotes": cotes})

        # Dessin des côtés + numérotation
        for i, (p1, p2) in enumerate(cotes):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="gray", linewidth=1)
            cx = (p1[0] + p2[0]) / 2
            cy = (p1[1] + p2[1]) / 2
            ax.text(cx, cy, f"{i}", fontsize=8, color="gray")

    # --- Dessin des associations trouvées ---
    for (pA, cA), (pB, cB) in associations:
        cotesA = dict_ctrl[pA][-1]["schema_cotes"]
        cotesB = dict_ctrl[pB][-1]["schema_cotes"]

        p1A, p2A = cotesA[cA]
        p1B, p2B = cotesB[cB]

        # Centre de chaque côté concerné
        cxA = (p1A[0] + p2A[0]) / 2
        cyA = (p1A[1] + p2A[1]) / 2
        cxB = (p1B[0] + p2B[0]) / 2
        cyB = (p1B[1] + p2B[1]) / 2

        # Ligne pointillée reliant les deux côtés associés
        ax.plot([cxA, cxB], [cyA, cyB], "r--", linewidth=2)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()


# ============================================================
# Main : exécution du pipeline complet sur un dossier de pièces
# ============================================================

if __name__ == "__main__":
    dossier = DOSSIER_IMAGES  # ex: <dossier_du_script>/resources/n_pieces_ensemble

    if not dossier.is_dir():
        raise FileNotFoundError(
            f"Dossier introuvable : {dossier}\n"
            f"Crée ce dossier à côté de ce script et mets-y tes photos de pièces "
            f"(.png/.jpg/.jpeg), ou modifie DOSSIER_IMAGES en haut du fichier."
        )

    # 1. Analyse + affichage des résultats pour chaque image du dossier
    fichiers = sorted(os.listdir(dossier))
    for fichier in fichiers:
        if fichier.lower().endswith((".png", ".jpg", ".jpeg")):
            chemin = os.path.join(dossier, fichier)
            print(f"\n--- Analyse de la pièce : {chemin} ---")

            # Analyse de la pièce (sans les affichages intermédiaires détaillés)
            resultats = analyser_piece(chemin, afficher=False)

            # Affichage synthétique (grille 2x2) des résultats
            afficher_resultats_piece(resultats, titre=f"Résultats : {fichier}")

    # 2. Construction de dict_ctrl pour toutes les pièces du dossier
    dict_ctrl = construire_dict_ctrl_pour_plusieurs_pieces(dossier)

    print("\n=== dict_ctrl construit pour toutes les pièces ===")
    for piece_id, cotes_piece in dict_ctrl.items():
        print(f"\nPièce {piece_id}:")
        for cote_id, info in enumerate(cotes_piece):
            print(f"  Côté {cote_id} : catégorie = {info['cat']}")

    # 3. Association des pièces (recherche des côtés complémentaires)
    associations = associer_pieces(dict_ctrl)

    print("\n=== Schéma des pièces et associations ===")
    visualiser_schema_pieces(dict_ctrl, associations)

    print("\n=== Associations trouvées ===")
    for a in associations:
        print(a)
