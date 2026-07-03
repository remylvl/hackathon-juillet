"""
main2.py — Pipeline complet de reconnaissance de puzzle.

Reproduit le comportement du script principal d'origine (analyse de toutes
les pièces d'un dossier, construction de `dict_ctrl` avec optimisation
B-spline + classification bosse/creux/plat de chaque côté, association des
pièces entre elles, puis visualisation du schéma final), mais en s'appuyant
sur le pipeline de détection plus robuste développé dans l'autre script :

  - masque de couleur HSV calibré automatiquement, affiné par GrabCut
    (beaucoup plus robuste aux ombres portées sur le fond) ;
  - détection des coins par "fraction de pièce locale" (un vrai coin à 90°
    a très peu de pièce autour de lui comparé au bout arrondi d'un tenon),
    plutôt que par simple score de courbure ;
  - seuils exprimés en fraction du périmètre du contour (valables quelle
    que soit la résolution de la photo ou la taille de la pièce) ;
  - une figure de vérification à 4 panneaux enregistrée sur disque pour
    chaque pièce, plutôt que des fenêtres matplotlib affichées une à une.

Les pièces analysées sont attendues dans le dossier `elodie/puzzle1/`
(à côté de ce script), nommées `1_1.jpg`, `1_2.jpg`, ..., `1_12.jpg`.

Ce fichier est autonome.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.transforms import Affine2D
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev, BSpline, interp1d
from scipy.optimize import least_squares
from skimage import measure
from skimage.color import rgb2hsv

try:
    import cv2
    _CV2_DISPONIBLE = True
except ImportError:
    _CV2_DISPONIBLE = False


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PUZZLE = os.path.join(DOSSIER_SCRIPT, "elodie", "puzzle1")
DOSSIER_VERIFICATION = os.path.join(DOSSIER_PUZZLE, "verification")

# Les pièces sont nommées 1_1.jpg, 1_2.jpg, ..., 1_{NB_PIECES}.jpg
NB_PIECES = 12

UTILISER_GRABCUT = True     # affine le masque HSV avec GrabCut (robuste aux ombres)
                             # nécessite `pip install opencv-python`

# Paramètres de calibration du masque de couleur
SAT_MIN = 0.20
VAL_MIN = 0.20
LARGEUR_HUE = 0.06

# Paramètres de détection des coins / segments (en fraction du périmètre,
# donc valables quelle que soit la résolution de la photo)
NB_COINS = 4
FRACTION_DISTANCE_MIN_COINS = 0.15
RAYON_FRACTION_COINS = 0.015

# Paramètres de l'optimisation B-spline + classification (repris du script
# "main" d'origine)
N_CTRL = 15
DEGRE_SPLINE = 2
SEUIL_PLAT = 0.08


# ========================================================================
# PARTIE 1 — Chargement image / masque de couleur / GrabCut
# (reprise telle quelle du pipeline robuste de l'autre script)
# ========================================================================

def charger_image(chemin):
    """Charge l'image et renvoie (image RGB, canaux H, S, V)."""
    img = image.imread(chemin)
    img_rgb = img[:, :, :3]
    img_hsv = rgb2hsv(img_rgb)
    return img_rgb, img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]


def creer_masque_couleur(img_h, img_s, img_v, sat_min=SAT_MIN, val_min=VAL_MIN,
                          largeur_hue=LARGEUR_HUE):
    """Construit un masque binaire de la pièce en calibrant automatiquement
    la teinte dominante parmi les pixels suffisamment saturés."""
    candidat = (img_s > sat_min) & (img_v > val_min)
    if not np.any(candidat):
        raise ValueError("Aucun pixel suffisamment saturé pour calibrer le masque.")

    h_candidats = img_h[candidat]
    hist, bins = np.histogram(h_candidats, bins=60, range=(0.0, 1.0))
    i_pic = np.argmax(hist)
    h_centre = 0.5 * (bins[i_pic] + bins[i_pic + 1])

    masque = candidat & (np.abs(img_h - h_centre) <= largeur_hue)
    return masque, h_centre


def creer_masque_grabcut(fichier_image, masque_initial, iterations=5, marge_certaine_fond=40):
    """Affine un masque grossier avec l'algorithme GrabCut, beaucoup plus
    robuste aux ombres portées par la pièce sur le fond qu'un simple
    seuillage de teinte."""
    if not _CV2_DISPONIBLE:
        raise ImportError("GrabCut nécessite opencv-python : `pip install opencv-python`.")

    img_bgr = cv2.imread(fichier_image)
    if img_bgr is None:
        raise ValueError(f"Impossible de lire l'image avec OpenCV : {fichier_image}")

    masque_gc = np.full(img_bgr.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    masque_gc[masque_initial] = cv2.GC_PR_FGD

    # Cœur érodé du masque initial : quasi certainement la pièce.
    noyau = np.ones((15, 15), np.uint8)
    coeur = cv2.erode(masque_initial.astype(np.uint8), noyau, iterations=1).astype(bool)
    masque_gc[coeur] = cv2.GC_FGD

    # Zone loin de la boîte englobante de la pièce : certainement le fond.
    lignes, colonnes = np.where(masque_initial)
    if len(lignes) > 0:
        r_min, r_max = lignes.min(), lignes.max()
        c_min, c_max = colonnes.min(), colonnes.max()
        r0 = max(0, r_min - marge_certaine_fond)
        r1 = min(img_bgr.shape[0], r_max + marge_certaine_fond)
        c0 = max(0, c_min - marge_certaine_fond)
        c1 = min(img_bgr.shape[1], c_max + marge_certaine_fond)

        certain_fond = np.ones(img_bgr.shape[:2], dtype=bool)
        certain_fond[r0:r1, c0:c1] = False
        masque_gc[certain_fond] = cv2.GC_BGD

    modele_fond = np.zeros((1, 65), np.float64)
    modele_avant_plan = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, masque_gc, None, modele_fond, modele_avant_plan,
                iterations, cv2.GC_INIT_WITH_MASK)

    return np.isin(masque_gc, [cv2.GC_FGD, cv2.GC_PR_FGD])


def nettoyer_masque(masque):
    """Enlève le bruit, ne garde que la plus grande composante connexe et
    comble les trous internes."""
    masque_propre = ndimage.binary_opening(masque, structure=np.ones((3, 3)))
    masque_propre = ndimage.binary_closing(masque_propre, structure=np.ones((20, 20)))

    labels, nb = ndimage.label(masque_propre)
    tailles = ndimage.sum(masque_propre, labels, range(1, nb + 1))
    if len(tailles) == 0:
        raise ValueError("Aucune composante détectée : ajuste les seuils HSV du masque.")

    plus_grande = np.argmax(tailles) + 1
    masque_piece = labels == plus_grande

    return ndimage.binary_fill_holes(masque_piece)


def obtenir_masque_piece(fichier_image, img_h, img_s, img_v, utiliser_grabcut=UTILISER_GRABCUT):
    """Masque final de la pièce : seuillage HSV, raffinement optionnel par
    GrabCut, puis nettoyage morphologique."""
    masque_brut, h_centre = creer_masque_couleur(img_h, img_s, img_v)
    print(f"Teinte dominante détectée automatiquement : {h_centre:.3f}")

    if utiliser_grabcut:
        if not _CV2_DISPONIBLE:
            print("⚠ opencv-python n'est pas installé (pip install opencv-python) : "
                  "GrabCut désactivé, utilisation du masque HSV brut.")
        else:
            masque_brut = creer_masque_grabcut(fichier_image, masque_brut)

    return nettoyer_masque(masque_brut)


# ========================================================================
# PARTIE 2 — Contour, détection des coins par fraction de pièce locale
# (reprise telle quelle du pipeline robuste de l'autre script)
# ========================================================================

def extraire_contour_principal(masque_final):
    """Renvoie le plus long contour détecté dans le masque."""
    contours = measure.find_contours(masque_final, level=0.5)
    return max(contours, key=len)


def lisser_contour(contour, sigma=7):
    """Lisse un contour fermé par filtre gaussien (wrap autour de la boucle)."""
    contour_ferme = np.vstack([contour, contour[:1]])
    row_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 0], sigma=sigma, mode="wrap")[:-1]
    col_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 1], sigma=sigma, mode="wrap")[:-1]
    return np.column_stack([row_lisse, col_lisse])


def candidats_convexes(contour, fraction_fenetre=0.02, angle_min=10, fraction_distance=0.03):
    """Repère les points CONVEXES du contour (virage net vers l'extérieur) :
    à la fois les 4 vrais coins ET les pics de chaque tenon."""
    n = len(contour)
    fenetre = max(5, int(n * fraction_fenetre))
    angles = np.zeros(n)
    for i in range(n):
        p = contour[i]
        a = contour[(i - fenetre) % n]
        b = contour[(i + fenetre) % n]
        v1, v2 = p - a, b - p
        angles[i] = np.degrees(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2)))

    angles_bouclees = np.concatenate([angles, angles[:fenetre]])
    pics, _ = find_peaks(-angles_bouclees, height=angle_min,
                          distance=max(5, int(n * fraction_distance)))
    return sorted(pics[pics < n].tolist())


def fraction_piece_locale(masque_final, y, x, rayon):
    """Fraction de pixels appartenant à la pièce (vs fond) dans un disque de
    rayon `rayon` centré en (x, y)."""
    ymin, ymax = max(0, int(y - rayon)), min(masque_final.shape[0], int(y + rayon) + 1)
    xmin, xmax = max(0, int(x - rayon)), min(masque_final.shape[1], int(x + rayon) + 1)
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    disque = (yy - y) ** 2 + (xx - x) ** 2 <= rayon ** 2
    return masque_final[ymin:ymax, xmin:xmax][disque].mean()


def detecter_coins_par_fraction_piece(masque_final, contour, nb_coins=NB_COINS,
                                       rayon_fraction=RAYON_FRACTION_COINS, rayon_min=10,
                                       fraction_distance_min=FRACTION_DISTANCE_MIN_COINS,
                                       nb_tentatives_max=6):
    """Détecte les coins d'une pièce à partir d'une idée géométrique simple
    et robuste : en un vrai coin à 90°, le fond occupe environ 1/4 du
    voisinage local de la pièce ; au bout d'un tenon, le fond occupe
    beaucoup plus. On retient donc, parmi les points convexes du contour,
    ceux ayant la plus petite fraction de pièce locale."""
    n = len(contour)
    rayon = max(rayon_min, n * rayon_fraction)
    candidats = candidats_convexes(contour)

    scores = []
    for i in candidats:
        y, x = contour[i]
        scores.append((i, fraction_piece_locale(masque_final, y, x, rayon)))
    scores.sort(key=lambda t: t[1])  # fraction croissante : coins vifs d'abord

    distance_min = max(10, n * fraction_distance_min)
    coins_indices = []
    for tentative in range(nb_tentatives_max):
        coins_indices = []
        for i, _ in scores:
            p = contour[i]
            trop_proche = any(
                np.linalg.norm(contour[j] - p) < distance_min
                for j in coins_indices
            )
            if not trop_proche:
                coins_indices.append(i)
            if len(coins_indices) == nb_coins:
                break
        if len(coins_indices) >= nb_coins:
            break
        distance_min /= 1.3
    else:
        print(f"⚠ Impossible d'atteindre {nb_coins} coins (obtenu : {len(coins_indices)}).")

    coins_indices = np.array(sorted(coins_indices))
    coins = contour[coins_indices] if len(coins_indices) > 0 else np.array([])

    return coins, coins_indices


def trouver_indice_plus_proche(contour, point_xy):
    """Indice du point du contour le plus proche d'un point (x, y) donné —
    sert de filet de sécurité pour imposer des coins à la main."""
    x, y = point_xy
    distances = np.sqrt((contour[:, 1] - x) ** 2 + (contour[:, 0] - y) ** 2)
    return int(np.argmin(distances))


# ========================================================================
# PARTIE 3 — Découpage en segments + ajustement de spline
# (reprise telle quelle du pipeline robuste de l'autre script)
# ========================================================================

def extraire_segments(contour, indices):
    """Découpe le contour en segments entre coins consécutifs (boucle)."""
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


def nettoyer_doublons_consecutifs(segment):
    """Enlève les points consécutifs identiques (requis par splprep)."""
    garder = [segment[0]]
    for p in segment[1:]:
        if not np.array_equal(p, garder[-1]):
            garder.append(p)
    return np.array(garder)


def ajuster_spline_segment(segment, lissage=0):
    """Ajuste une spline paramétrique (x(u), y(u)) sur un segment, en
    coordonnées image brutes (sert uniquement à l'affichage)."""
    segment = nettoyer_doublons_consecutifs(segment)
    x = segment[:, 1]
    y = segment[:, 0]
    tck, u = splprep([x, y], s=lissage)
    return tck


# ========================================================================
# PARTIE 4 — Normalisation coin-à-coin, optimisation B-spline et
# classification bosse/creux/plat (reprises du script "main" d'origine)
# ========================================================================

def normaliser_segment(segment):
    """Exprime un segment dans un repère où le premier coin est l'origine,
    l'axe (along) relie les deux coins du segment, et l'ensemble est mis à
    l'échelle par la longueur du segment (along, height sans dimension).
    Cette invariance d'échelle est nécessaire pour comparer la forme de
    côtés provenant de pièces/photos différentes."""
    x = segment[:, 1].astype(float)
    y = segment[:, 0].astype(float)

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    px = x - x0
    py = y - y0

    vx, vy = x1 - x0, y1 - y0
    longueur = np.sqrt(vx ** 2 + vy ** 2)
    if longueur == 0:
        raise ValueError(
            "Segment dégénéré : le premier et le dernier point coïncident "
            "(deux coins confondus). Vérifie la détection des coins de la pièce."
        )
    vx, vy = vx / longueur, vy / longueur

    along = px * vx + py * vy
    height = px * vy - py * vx

    along = along / longueur
    height = height / longueur

    return np.column_stack([along, height])


def echantillonner_segment(segment, n=200):
    """Ré-échantillonne un segment à pas curviligne constant (n points)."""
    d = np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1))
    u = np.concatenate([[0], np.cumsum(d)])
    u = u / u[-1]

    fx = interp1d(u, segment[:, 0])
    fy = interp1d(u, segment[:, 1])

    u_new = np.linspace(0, 1, n)
    return np.column_stack([fx(u_new), fy(u_new)])


def optimiser_segment(segment, n_ctrl=N_CTRL, degree=DEGRE_SPLINE):
    """Ajuste par moindres carrés les points de contrôle d'une B-spline de
    degré `degree` sur un segment normalisé (coin-à-coin)."""
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


def classifier_cote(ctrl_points, seuil_plat=SEUIL_PLAT):
    """Classifie un côté à partir de ses points de contrôle normalisés :
    0 = plat (bord de puzzle), 1 = bosse (tenon), 2 = creux."""
    y = ctrl_points[:, 1]
    y_max = np.max(y)
    y_min = np.min(y)

    if abs(y_max) < seuil_plat and abs(y_min) < seuil_plat:
        return 0
    if abs(y_max) > abs(y_min):
        return 1
    return 2


# ========================================================================
# PARTIE 5 — Figure de vérification par pièce
# (reprise telle quelle du pipeline robuste de l'autre script)
# ========================================================================

_COULEURS_SEGMENTS = ["red", "blue", "green", "orange", "purple", "brown"]


def verifier_piece_visuellement(img, masque_final, contour_principal, contour_lisse,
                                 coins_finaux, segments, splines, nom_fichier, dossier_sortie=None):
    """Figure à 4 panneaux (masque, contour+coins, segments+spline, côtés
    normalisés) permettant de vérifier d'un coup d'œil chaque étape du
    pipeline. Enregistrée en PNG si `dossier_sortie` est fourni, sinon
    affichée à l'écran."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    fig.suptitle(nom_fichier, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.imshow(img)
    ax.imshow(masque_final, cmap="Reds", alpha=0.35)
    ax.set_title("1. Image + masque de segmentation")
    ax.axis("off")

    ax = axes[0, 1]
    ax.plot(contour_principal[:, 1], contour_principal[:, 0], '.',
             markersize=1, color="gray", alpha=0.4, label="contour brut")
    ax.plot(contour_lisse[:, 1], contour_lisse[:, 0], '-',
             linewidth=1.2, color="steelblue", label="contour lissé")
    if len(coins_finaux) > 0:
        ax.scatter(coins_finaux[:, 1], coins_finaux[:, 0],
                    color="red", s=90, zorder=5, label="coins retenus")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("2. Contour + coins détectés")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 0]
    for i, (seg, tck) in enumerate(zip(segments, splines)):
        couleur = _COULEURS_SEGMENTS[i % len(_COULEURS_SEGMENTS)]
        ax.plot(seg[:, 1], seg[:, 0], '.', markersize=2, color=couleur, alpha=0.4, label=f"côté {i}")
        u_fin = np.linspace(0, 1, 200)
        x_fit, y_fit = splev(u_fin, tck)
        ax.plot(x_fit, y_fit, '-', linewidth=1.5, color=couleur)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("3. Segments détectés + spline ajustée")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 1]
    for i, seg in enumerate(segments):
        couleur = _COULEURS_SEGMENTS[i % len(_COULEURS_SEGMENTS)]
        try:
            seg_norm = normaliser_segment(seg)
            ax.plot(seg_norm[:, 0], seg_norm[:, 1], color=couleur, label=f"côté {i}")
        except ValueError:
            pass  # segment dégénéré, déjà signalé ailleurs
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_aspect("equal")
    ax.set_title("4. Côtés normalisés (comparaison de forme)")
    ax.legend(fontsize=7, loc="best")

    plt.tight_layout()

    if dossier_sortie:
        os.makedirs(dossier_sortie, exist_ok=True)
        chemin_sortie = os.path.join(
            dossier_sortie, f"verif_{os.path.splitext(nom_fichier)[0]}.png"
        )
        fig.savefig(chemin_sortie, dpi=120)
        plt.close(fig)
        print(f"  → figure de vérification : {chemin_sortie}")
    else:
        plt.show()


# ========================================================================
# PARTIE 6 — Analyse d'une pièce (pipeline robuste de bout en bout)
# (reprise telle quelle du pipeline robuste de l'autre script)
# ========================================================================

def analyser_piece(fichier_image, dossier_verification=None, verifier=True,
                    utiliser_grabcut=UTILISER_GRABCUT,
                    fraction_distance_min_coins=FRACTION_DISTANCE_MIN_COINS,
                    rayon_fraction_coins=RAYON_FRACTION_COINS, coins_manuels=None):
    """Analyse une pièce et renvoie masque, contour, coins, segments (bruts,
    en coordonnées image) et splines. Enregistre/affiche la figure de
    vérification à 4 panneaux si `verifier=True`."""
    nom_fichier = os.path.basename(fichier_image)

    img, img_h, img_s, img_v = charger_image(fichier_image)

    masque_final = obtenir_masque_piece(fichier_image, img_h, img_s, img_v,
                                         utiliser_grabcut=utiliser_grabcut)

    contour_principal = extraire_contour_principal(masque_final)
    contour_lisse = lisser_contour(contour_principal, sigma=7)

    if coins_manuels is not None:
        indices_coins = np.array(sorted(
            trouver_indice_plus_proche(contour_principal, p) for p in coins_manuels
        ))
        coins_finaux = contour_principal[indices_coins]
        print(f"{len(coins_finaux)} coins imposés manuellement")
    else:
        coins_finaux, indices_coins = detecter_coins_par_fraction_piece(
            masque_final, contour_principal, nb_coins=NB_COINS,
            rayon_fraction=rayon_fraction_coins,
            fraction_distance_min=fraction_distance_min_coins,
        )
        print(f"{len(coins_finaux)} coins retenus (fraction de pièce locale)")

    if len(coins_finaux) < NB_COINS:
        print(f"⚠ Seulement {len(coins_finaux)} coin(s) trouvé(s) sur {NB_COINS} attendus pour "
              f"{fichier_image} : les segments risquent d'être incohérents.")

    segments = extraire_segments(contour_principal, indices_coins) if len(coins_finaux) > 0 else []
    for i, seg in enumerate(segments):
        print(f"Segment {i} : {len(seg)} points")

    splines = [ajuster_spline_segment(seg, lissage=len(seg) * 2) for seg in segments]

    if verifier and len(coins_finaux) > 0:
        verifier_piece_visuellement(
            img, masque_final, contour_principal, contour_lisse,
            coins_finaux, segments, splines, nom_fichier, dossier_sortie=dossier_verification,
        )

    return {
        "img": img,
        "masque_final": masque_final,
        "contour_principal": contour_principal,
        "coins": coins_finaux,
        "segments": segments,
        "splines": splines,
    }


# ========================================================================
# PARTIE 6bis — Découpe de la pièce sur fond transparent (pour l'assemblage
# visuel avec les vraies photos, voir PARTIE 9)
# ========================================================================

def decouper_piece(img, masque):
    """Découpe une pièce à partir de son image et de son masque : recadrage
    sur la boîte englobante de la pièce, fond mis à transparent (canal
    alpha = masque). Renvoie (rgba, (r0, c0)) où rgba est un tableau de
    valeurs dans [0, 1] et (r0, c0) le décalage (ligne, colonne) du coin
    supérieur-gauche du recadrage dans l'image d'origine — nécessaire pour
    replacer les coins de la pièce dans le repère local de `rgba`."""
    lignes, colonnes = np.where(masque)
    if len(lignes) == 0:
        raise ValueError("Masque vide : impossible de découper la pièce.")

    r0, r1 = lignes.min(), lignes.max() + 1
    c0, c1 = colonnes.min(), colonnes.max() + 1

    img_zone = img[r0:r1, c0:c1, :3].astype(float)
    if img_zone.max() > 1.0:
        img_zone = img_zone / 255.0

    masque_zone = masque[r0:r1, c0:c1].astype(float)

    rgba = np.dstack([img_zone, masque_zone])
    return rgba, (r0, c0)



# ========================================================================
# PARTIE 7 — Construction de dict_ctrl pour toutes les pièces d'un dossier
# (repris du script "main" d'origine, mais en s'appuyant sur le nouveau
# `analyser_piece` ci-dessus : détection de coins par GrabCut + fraction de
# pièce locale, au lieu de l'ancienne détection par courbure)
# ========================================================================

def construire_dict_ctrl_pour_plusieurs_pieces(dossier, dossier_verification=None,
                                                nb_pieces=NB_PIECES,
                                                afficher_splines_optimisees=False):
    """Analyse les pièces `1_1.jpg`, `1_2.jpg`, ..., `1_{nb_pieces}.jpg` d'un
    dossier et construit `dict_ctrl` : pour chaque pièce, la liste de ses
    côtés, chacun décrit par ses points de contrôle B-spline optimisés
    (`ctrl`) et sa catégorie bosse/creux/plat (`cat`).

    Renvoie le triplet (dict_ctrl, images_pieces, geometrie_pieces) :
      - `images_pieces` : {piece_id: image RGBA} — la photo de chaque pièce
        découpée sur fond transparent ;
      - `geometrie_pieces` : {piece_id: coins (4, 2)} — les 4 coins de la
        pièce en coordonnées (x, y) LOCALES à son image découpée (x = colonne,
        y = ligne, origine = coin supérieur-gauche du recadrage), dans le
        même ordre que les côtés de `dict_ctrl` (le côté k va du coin k au
        coin (k+1) % 4).
    Les deux dictionnaires sont utilisés par `assembler_pieces` /
    `visualiser_assemblage_colle` pour recoller géométriquement les pièces
    entre elles, et par `visualiser_assemblage_images` pour le placement
    approximatif par forces dirigées."""
    dict_ctrl = {}
    images_pieces = {}
    geometrie_pieces = {}

    piece_id = 0
    for i in range(1, nb_pieces + 1):
        nom = f"1_{i}.jpg"
        chemin = os.path.join(dossier, nom)
        if not os.path.exists(chemin):
            print(f"⚠ Fichier introuvable, ignoré : {chemin}")
            continue

        print(f"\n--- Analyse de la pièce {piece_id} : {chemin} ---")

        resultat = analyser_piece(chemin, dossier_verification=dossier_verification, verifier=True)
        segments = resultat["segments"]

        if len(segments) != NB_COINS:
            print(f"⚠ Pièce {piece_id} ({nom}) ignorée : {len(segments)} côté(s) détecté(s) "
                  f"au lieu de {NB_COINS}.")
            piece_id += 1
            continue

        rgba, (r0, c0) = decouper_piece(resultat["img"], resultat["masque_final"])
        images_pieces[piece_id] = rgba

        # coins (row, col) -> repère local de rgba (row-r0, col-c0) -> (x, y)
        # local avec x = colonne, y = ligne : geometrie_pieces[pid][k] est le
        # coin de départ du côté k (cohérent avec `segments`/`dict_ctrl`).
        coins_locaux = resultat["coins"] - np.array([r0, c0])
        geometrie_pieces[piece_id] = coins_locaux[:, ::-1].astype(float)

        dict_ctrl_piece = []
        for cote_id, seg in enumerate(segments):
            seg_norm = normaliser_segment(seg)
            ctrl_opt, knots, t = optimiser_segment(seg_norm)

            if afficher_splines_optimisees:
                plt.figure()
                plt.plot(seg_norm[:, 0], seg_norm[:, 1], 'o', label='Segment normalisé')

                spline_x = BSpline(knots, ctrl_opt[:, 0], DEGRE_SPLINE)
                spline_y = BSpline(knots, ctrl_opt[:, 1], DEGRE_SPLINE)
                C = np.vstack((spline_x(t), spline_y(t))).T

                plt.plot(C[:, 0], C[:, 1], '-', label='Spline optimisée')
                plt.plot(ctrl_opt[:, 0], ctrl_opt[:, 1], 'x', label='Points de contrôle optimisés')

                plt.legend()
                plt.axis("equal")
                plt.title(f"Spline optimisée – Pièce {piece_id}, côté {cote_id}")
                plt.show()

            cat = classifier_cote(ctrl_opt)
            dict_ctrl_piece.append({"ctrl": ctrl_opt, "cat": cat})

        dict_ctrl[piece_id] = dict_ctrl_piece
        piece_id += 1

    return dict_ctrl, images_pieces, geometrie_pieces


# ========================================================================
# PARTIE 8 — Association des pièces + visualisation du schéma
# (reprises telles quelles du script "main" d'origine)
# ========================================================================

def distance_cotes(ctrlA, ctrlB):
    return np.linalg.norm(ctrlA - ctrlB)


def associer_pieces(dict_ctrl):
    """Associe entre eux les côtés « bosse » et « creux » (jamais deux
    côtés plats, jamais deux côtés du même type) en minimisant la distance
    entre leurs points de contrôle normalisés."""
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
                        dict_ctrl[pA][cA]["ctrl"],
                        dict_ctrl[pB][cB]["ctrl"]
                    )

    associations = []
    associes = set()

    for (pA, cA) in cotes:
        if (pA, cA) in associes:
            continue

        meilleur = None
        meilleure_dist = np.inf

        for (pB, cB) in cotes:
            if (pB, cB) in associes:
                continue
            if pA == pB:
                continue

            key = ((pA, cA), (pB, cB))
            if key in distances:
                d = distances[key]
                if d < meilleure_dist:
                    meilleure_dist = d
                    meilleur = (pB, cB)

        if meilleur is not None:
            associations.append(((pA, cA), meilleur))
            associes.add((pA, cA))
            associes.add(meilleur)

    return associations


def visualiser_schema_pieces(dict_ctrl, associations):
    """Visualisation schématique des pièces (carrés) et de leurs côtés
    associés (segments pointillés rouges reliant les côtés appariés)."""
    n_pieces = len(dict_ctrl)

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

    for pid, (x, y) in positions.items():
        ax.add_patch(plt.Rectangle((x - 1, y - 1), 2, 2,
                                   fill=False, linewidth=2))
        ax.text(x, y, f"Pièce {pid}", ha="center", va="center", fontsize=12)

        cotes = [
            ((x - 1, y + 1), (x + 1, y + 1)),  # haut
            ((x + 1, y + 1), (x + 1, y - 1)),  # droite
            ((x - 1, y - 1), (x + 1, y - 1)),  # bas
            ((x - 1, y + 1), (x - 1, y - 1)),  # gauche
        ]

        dict_ctrl[pid].append({"schema_cotes": cotes})

        for i, (p1, p2) in enumerate(cotes):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=1)
            cx = (p1[0] + p2[0]) / 2
            cy = (p1[1] + p2[1]) / 2
            ax.text(cx, cy, f"{i}", fontsize=8, color="gray")

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
    plt.show()


# ========================================================================
# PARTIE 9 — Assemblage visuel à partir des vraies photos des pièces
# ========================================================================

def layout_force_diriges(n_pieces, aretes, iterations=300, seed=0):
    """Positionne `n_pieces` nœuds dans le plan par un algorithme de forces
    dirigées (Fruchterman-Reingold simplifié) : toutes les pièces se
    repoussent, les pièces reliées par une arête (une association) s'attirent.
    Renvoie un tableau (n_pieces, 2) de positions."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, size=(n_pieces, 2))
    k = 1.0 / np.sqrt(n_pieces)

    for it in range(iterations):
        deplacement = np.zeros_like(pos)

        # forces répulsives (toutes les paires de pièces)
        for i in range(n_pieces):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf  # pas d'auto-répulsion
            dist = np.maximum(dist, 1e-6)
            force_rep = (k ** 2) / dist
            deplacement[i] += np.sum((delta.T * (force_rep / dist)).T, axis=0)

        # forces attractives (le long des associations)
        for (a, b) in aretes:
            delta = pos[a] - pos[b]
            dist = max(np.linalg.norm(delta), 1e-6)
            force_att = dist ** 2 / k
            direction = delta / dist
            deplacement[a] -= direction * force_att
            deplacement[b] += direction * force_att

        # limite le déplacement (température décroissante)
        temperature = 0.1 * (1 - it / iterations)
        normes = np.maximum(np.linalg.norm(deplacement, axis=1), 1e-6)
        pos += (deplacement.T * (np.minimum(normes, temperature) / normes)).T

    return pos


def _extent_image(rgba, x, y, taille_cible=1.6):
    """Calcule l'extent (xmin, xmax, ymin, ymax) pour afficher `rgba` centrée
    en (x, y), à l'échelle `taille_cible` sur sa plus grande dimension, en
    conservant son rapport largeur/hauteur d'origine."""
    h, w = rgba.shape[:2]
    if w >= h:
        largeur = taille_cible
        hauteur = taille_cible * h / w
    else:
        hauteur = taille_cible
        largeur = taille_cible * w / h
    return (x - largeur / 2, x + largeur / 2, y - hauteur / 2, y + hauteur / 2)


def visualiser_assemblage_images(dict_ctrl, associations, images_pieces,
                                  iterations=300, taille_cible=1.6, seed=0):
    """Affiche les pièces les unes à côté des autres à l'aide de leurs
    vraies photos (recadrées, fond transparent), positionnées par forces
    dirigées de sorte que les pièces associées (bosse <-> creux) se
    retrouvent proches les unes des autres — une version illustrée, avec
    les photos réelles, du schéma abstrait de `visualiser_schema_pieces`."""
    piece_ids = sorted(pid for pid in dict_ctrl.keys() if pid in images_pieces)
    index = {pid: i for i, pid in enumerate(piece_ids)}
    n = len(piece_ids)

    if n == 0:
        print("⚠ Aucune image de pièce disponible pour l'assemblage visuel.")
        return

    aretes = [
        (index[pA], index[pB]) for (pA, _), (pB, _) in associations
        if pA in index and pB in index
    ]

    pos = layout_force_diriges(n, aretes, iterations=iterations, seed=seed)

    etendue = np.max(np.abs(pos)) if np.max(np.abs(pos)) > 0 else 1.0
    pos = pos / etendue * (0.9 * np.sqrt(n))

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_title("Assemblage des pièces à partir de leurs photos")

    for (a, b) in aretes:
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                "r--", linewidth=1.2, alpha=0.6, zorder=1)

    for pid in piece_ids:
        i = index[pid]
        x, y = pos[i]
        rgba = images_pieces[pid]
        extent = _extent_image(rgba, x, y, taille_cible=taille_cible)
        ax.imshow(rgba, extent=extent, zorder=2)
        ax.text(x, extent[3] + 0.08, f"Pièce {pid}", ha="center", va="bottom", fontsize=9)

    ax.set_aspect("equal")
    ax.axis("off")
    marge = taille_cible
    ax.set_xlim(pos[:, 0].min() - marge, pos[:, 0].max() + marge)
    ax.set_ylim(pos[:, 1].min() - marge, pos[:, 1].max() + marge)
    plt.tight_layout()
    plt.show()


# ========================================================================
# PARTIE 10 — Assemblage géométrique réel (côtés collés bord à bord)
# ========================================================================
#
# Contrairement à `visualiser_assemblage_images` (placement approximatif par
# forces dirigées), on calcule ici la transformation rigide (rotation +
# translation) exacte de chaque pièce de sorte que ses côtés associés
# coïncident réellement avec ceux de ses voisines, en partant d'une pièce
# « racine » et en propageant les placements de proche en proche le long du
# graphe des associations (parcours en largeur).

def _matrice_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def assembler_pieces(dict_ctrl, associations, geometrie_pieces, piece_racine=None):
    """Calcule la transformation rigide (rotation `theta` + translation `t`)
    de chaque pièce atteignable depuis `piece_racine` en suivant le graphe
    des associations, de sorte que les côtés associés coïncident exactement.

    Convention : deux côtés associés sont parcourus en sens opposé l'un de
    l'autre (les contours des différentes pièces étant orientés de façon
    cohérente) — le premier coin du côté de la pièce déjà placée est donc
    apparié au SECOND coin du côté de la pièce à placer, et inversement.

    Renvoie (placements, non_places) :
      - `placements` : {piece_id: (theta, t)} avec t un vecteur (x, y) tel
        que, pour tout coin local p d'une pièce, sa position assemblée est
        `matrice_rotation(theta) @ p + t` ;
      - `non_places` : pièces non atteignables depuis la racine (composante
        du graphe d'association déconnectée)."""
    graphe = {pid: [] for pid in dict_ctrl.keys()}
    for (pA, cA), (pB, cB) in associations:
        graphe[pA].append((cA, pB, cB))
        graphe[pB].append((cB, pA, cA))

    if piece_racine is None:
        piece_racine = min(dict_ctrl.keys())
    if piece_racine not in dict_ctrl:
        raise ValueError(f"Pièce racine inconnue : {piece_racine}")

    placements = {piece_racine: (0.0, np.zeros(2))}
    file_attente = [piece_racine]

    while file_attente:
        p = file_attente.pop(0)
        theta_p, t_p = placements[p]
        R_p = _matrice_rotation(theta_p)
        coins_p = geometrie_pieces[p]

        for (cP, q, cQ) in graphe[p]:
            if q in placements:
                continue

            p1_monde = R_p @ coins_p[cP] + t_p
            p2_monde = R_p @ coins_p[(cP + 1) % 4] + t_p

            coins_q = geometrie_pieces[q]
            q1_local = coins_q[cQ]
            q2_local = coins_q[(cQ + 1) % 4]

            # côtés associés parcourus en sens opposé : q1 <-> p2, q2 <-> p1
            v_monde = p1_monde - p2_monde
            v_local = q2_local - q1_local
            theta_q = (np.arctan2(v_monde[1], v_monde[0])
                       - np.arctan2(v_local[1], v_local[0]))

            R_q = _matrice_rotation(theta_q)
            centre_monde = (p1_monde + p2_monde) / 2
            centre_local = (q1_local + q2_local) / 2
            t_q = centre_monde - R_q @ centre_local

            placements[q] = (theta_q, t_q)
            file_attente.append(q)

    non_places = [pid for pid in dict_ctrl.keys() if pid not in placements]
    return placements, non_places


def visualiser_assemblage_colle(dict_ctrl, associations, images_pieces, geometrie_pieces,
                                 piece_racine=None):
    """Assemble et affiche les pièces en collant réellement leurs côtés
    associés les uns aux autres (rotation + translation exactes calculées
    par `assembler_pieces`), en partant de `piece_racine` (par défaut la
    pièce d'indice le plus bas) et en propageant l'assemblage de proche en
    proche. Renvoie `placements` (voir `assembler_pieces`)."""
    placements, non_places = assembler_pieces(
        dict_ctrl, associations, geometrie_pieces, piece_racine=piece_racine
    )

    if non_places:
        print(f"⚠ Pièces non reliées à l'assemblage principal par une association "
              f"(côtés plats non appariés ou composante séparée) : {non_places}")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Assemblage des pièces (côtés collés bord à bord)")

    tous_les_coins = []

    for pid, (theta, t) in sorted(placements.items()):
        rgba = images_pieces.get(pid)
        if rgba is None:
            continue
        hauteur, largeur = rgba.shape[:2]

        transform = Affine2D().rotate(theta).translate(*t) + ax.transData
        ax.imshow(rgba, extent=(0, largeur, hauteur, 0), transform=transform, zorder=2)

        coins_monde = (geometrie_pieces[pid] @ _matrice_rotation(theta).T) + t
        tous_les_coins.append(coins_monde)

        centre = coins_monde.mean(axis=0)
        ax.text(centre[0], centre[1], f"{pid}", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold", zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")

    if tous_les_coins:
        tous = np.vstack(tous_les_coins)
        marge = 0.06 * max(np.ptp(tous[:, 0]), np.ptp(tous[:, 1]), 1.0)
        ax.set_xlim(tous[:, 0].min() - marge, tous[:, 0].max() + marge)
        # axe y inversé : les coordonnées locales suivent la convention
        # image (y = ligne, vers le bas), donc y croissant doit apparaître
        # vers le bas de la figure.
        ax.set_ylim(tous[:, 1].max() + marge, tous[:, 1].min() - marge)

    plt.tight_layout()
    plt.show()

    return placements


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    # 1. Construction de dict_ctrl (+ images et géométrie des pièces) pour
    #    toutes les pièces du dossier (analyse robuste + figure de
    #    vérification enregistrée pour chacune, puis optimisation B-spline
    #    et classification bosse/creux/plat de chaque côté).
    dict_ctrl, images_pieces, geometrie_pieces = construire_dict_ctrl_pour_plusieurs_pieces(
        DOSSIER_PUZZLE, dossier_verification=DOSSIER_VERIFICATION, nb_pieces=NB_PIECES
    )

    print("\n=== dict_ctrl construit pour toutes les pièces ===")
    for piece_id, cotes_piece in dict_ctrl.items():
        print(f"\nPièce {piece_id}:")
        for cote_id, info in enumerate(cotes_piece):
            print(f"  Côté {cote_id} : catégorie = {info['cat']}")

    # 2. Association des pièces (bosses <-> creux les plus proches en forme)
    associations = associer_pieces(dict_ctrl)

    # 3. Visualisation du schéma abstrait des pièces et de leurs associations
    print("\n=== Schéma des pièces et associations ===")
    visualiser_schema_pieces(dict_ctrl, associations)

    # 4. Assemblage géométrique réel : on part d'une pièce (par défaut la
    #    pièce 0) et on colle les côtés associés les uns aux autres de
    #    proche en proche.
    print("\n=== Assemblage des pièces (côtés collés) ===")
    visualiser_assemblage_colle(dict_ctrl, associations, images_pieces, geometrie_pieces)

    print("\n=== Associations trouvées ===")
    for a in associations:
        print(a)