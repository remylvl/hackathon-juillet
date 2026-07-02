"""
Analyse d'une pièce de puzzle à partir d'une photo.

Étapes :
1. Segmentation de la pièce par couleur (masque HSV calibré automatiquement),
   affiné par GrabCut si dispo (robuste aux ombres).
2. Nettoyage du masque (ouverture/fermeture morphologique, remplissage des trous).
3. Extraction du contour et détection des 4 coins (courbure locale).
4. Découpage du contour en segments (un par côté de la pièce).
5. Ajustement d'une spline sur chaque segment.
6. Normalisation de chaque segment dans un repère coin-à-coin, pour pouvoir
   comparer la forme des côtés entre pièces différentes.

Pour chaque pièce analysée, une figure récapitulative en 4 panneaux
(masque, contour+coins, segments, segments normalisés) est enregistrée
dans un dossier "verification/" — pratique pour vérifier d'un coup d'œil
que le pipeline fonctionne bien sur toutes les pièces d'un lot, sans
ouvrir des dizaines de fenêtres matplotlib.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from skimage import measure
from skimage.color import rgb2hsv
from skimage.feature import corner_harris

try:
    import cv2
    _CV2_DISPONIBLE = True
except ImportError:
    _CV2_DISPONIBLE = False


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

FICHIER_IMAGE = "./resources/piece4.jpeg"
UTILISER_GRABCUT = True     # affine le masque HSV avec GrabCut (robuste aux ombres)
                             # nécessite `pip install opencv-python`

# Paramètres de calibration du masque de couleur
SAT_MIN = 0.20
VAL_MIN = 0.20
LARGEUR_HUE = 0.06

# Paramètres de détection des coins / segments
NB_COINS = 4
# Exprimés en fraction du périmètre du contour plutôt qu'en pixels fixes,
# pour être robustes à des photos de résolutions/tailles de pièces différentes.
FRACTION_SEUIL_COURBURE = 0.02      # taille de la fenêtre de calcul de courbure
FRACTION_DISTANCE_MIN_COINS = 0.15  # distance min entre deux coins retenus


# ----------------------------------------------------------------------
# 1. Chargement de l'image et masque de couleur
# ----------------------------------------------------------------------

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
    """Affine un masque grossier (ex : celui de `creer_masque_couleur`) avec
    l'algorithme GrabCut. Contrairement à un simple seuillage de teinte,
    GrabCut modélise statistiquement les couleurs de la pièce et du fond
    (mélanges de gaussiennes), ce qui le rend beaucoup plus robuste aux
    ombres portées par la pièce sur le fond — la source la plus fréquente
    d'aspérités/faux coins sur le contour détecté.

    `marge_certaine_fond` : au-delà de cette distance (en pixels) de la
    boîte englobante du masque initial, on considère le fond comme certain
    (accélère et stabilise la convergence de GrabCut).
    """
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


def obtenir_masque_piece(fichier_image, img_h, img_s, img_v, utiliser_grabcut=UTILISER_GRABCUT):
    """Construit le masque final de la pièce : seuillage HSV pour une
    estimation grossière, raffinement optionnel par GrabCut (robuste aux
    ombres), puis nettoyage morphologique standard dans tous les cas."""
    masque_brut, h_centre = creer_masque_couleur(img_h, img_s, img_v)
    print(f"Teinte dominante détectée automatiquement : {h_centre:.3f}")

    if utiliser_grabcut:
        if not _CV2_DISPONIBLE:
            print("⚠ opencv-python n'est pas installé (pip install opencv-python) : "
                  "GrabCut désactivé, utilisation du masque HSV brut.")
        else:
            masque_brut = creer_masque_grabcut(fichier_image, masque_brut)

    return nettoyer_masque(masque_brut)


# ----------------------------------------------------------------------
# 2. Nettoyage du masque
# ----------------------------------------------------------------------

def nettoyer_masque(masque):
    """Enlève le bruit, lisse le contour, ne garde que la plus grande
    composante connexe et comble les trous internes."""
    masque_propre = ndimage.binary_opening(masque, structure=np.ones((3, 3)))
    masque_propre = ndimage.binary_closing(masque_propre, structure=np.ones((20, 20)))

    labels, nb = ndimage.label(masque_propre)
    tailles = ndimage.sum(masque_propre, labels, range(1, nb + 1))
    if len(tailles) == 0:
        raise ValueError("Aucune composante détectée : ajuste les seuils HSV du masque.")

    plus_grande = np.argmax(tailles) + 1
    masque_piece = labels == plus_grande

    return ndimage.binary_fill_holes(masque_piece)


# ----------------------------------------------------------------------
# 3. Contour et détection des coins (par courbure)
# ----------------------------------------------------------------------

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


def score_courbure(contour, index, seuil):
    """Score de courbure locale au point `index`, basé sur le produit
    scalaire entre les vecteurs vers les voisins gauche/droite."""
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
    """Sélectionne les `nb_coins` points de plus forte courbure parmi
    `points`, en imposant une distance minimale entre eux."""
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


def detecter_coins_par_courbure(contour, nb_coins=NB_COINS, pas_echantillonnage=3,
                                 fraction_seuil=FRACTION_SEUIL_COURBURE,
                                 fraction_distance_min=FRACTION_DISTANCE_MIN_COINS,
                                 nb_tentatives_max=6):
    """Détecte les `nb_coins` coins d'une pièce par score de courbure locale,
    en échantillonnant TOUT le contour (pas une version simplifiée, qui peut
    perdre trop d'information sur de petites images) et avec des seuils
    exprimés en fraction du périmètre plutôt qu'en pixels fixes, pour rester
    valables quelle que soit la résolution de la photo ou la taille de la pièce.

    Si `distance_min` empêche de trouver `nb_coins` coins distincts, on la
    relâche progressivement (÷ 1.3 à chaque tentative) jusqu'à obtenir
    `nb_coins` coins ou épuiser les tentatives.

    Renvoie (coins, indices_coins_tries) — `coins` en (row, col), et
    `indices_coins_tries` triés dans l'ordre du contour.
    """
    n = len(contour)
    seuil = max(5, int(n * fraction_seuil))
    distance_min = max(10, n * fraction_distance_min)

    candidats = list(range(0, n, max(1, pas_echantillonnage)))

    coins, indices_coins = np.array([]), np.array([])
    for tentative in range(nb_tentatives_max):
        coins, indices_coins = max_courbure(contour, candidats, seuil, distance_min, nb_coins)
        if len(coins) >= nb_coins:
            break
        distance_min /= 1.3
    else:
        print(f"⚠ Impossible d'atteindre {nb_coins} coins même en relâchant distance_min "
              f"(obtenu : {len(coins)}).")

    if tentative > 0 and len(coins) >= nb_coins:
        print(f"(distance_min relâchée à {distance_min:.1f}px après {tentative} tentative(s) "
              f"pour trouver {nb_coins} coins)")

    if len(coins) > 0:
        ordre = np.argsort(indices_coins)
        coins = coins[ordre]
        indices_coins = indices_coins[ordre]

    return coins, indices_coins


def detecter_coins_harris_convexe(masque_final, contour, nb_coins=NB_COINS,
                                   fraction_distance_min=FRACTION_DISTANCE_MIN_COINS,
                                   sigma_harris=3, nb_tentatives_max=6):
    """Détecte les coins d'une pièce en combinant deux critères robustes,
    bien plus fiables qu'un simple score de courbure le long du contour :

    1. Seuls les points de l'ENVELOPPE CONVEXE du contour sont candidats.
       Ça élimine d'office les points concaves (le cou d'un tenon, le fond
       d'une encoche), qui n'ont géométriquement rien à voir avec un coin
       de pièce, mais qui peuvent avoir un score de courbure très élevé et
       tromper une détection purement locale.

    2. Parmi ces candidats, on garde ceux à plus forte réponse de HARRIS
       (`skimage.feature.corner_harris`), qui détecte une vraie structure
       de coin (deux bords ~droits qui se croisent à angle vif) et répond
       beaucoup plus faiblement à une bosse arrondie de tenon — sur nos
       pièces, l'écart observé est net (score ~2.5-3.3 sur un vrai coin,
       ~0.1-0.2 sur le pic d'un tenon).

    IMPORTANT : `contour` doit être le contour BRUT (celui renvoyé par
    `extraire_contour_principal`, PAS `lisser_contour`) : le lissage gaussien
    en mode 'wrap' introduit du bruit numérique qui fait exploser le nombre
    de points jugés "convexes" (observé : 237 points au lieu d'une trentaine
    sur une pièce test), ce qui rend le filtre inutile.

    Renvoie (coins, indices_coins_tries), au même format que
    `detecter_coins_par_courbure` : `coins` en (row, col), et
    `indices_coins_tries` triés dans l'ordre du contour.
    """
    n = len(contour)
    pts_xy = contour[:, ::-1]  # (x, y)
    hull = ConvexHull(pts_xy)

    reponse = corner_harris(masque_final.astype(float), sigma=sigma_harris)
    h_max, l_max = reponse.shape

    candidats = []
    for i in hull.vertices:
        x, y = pts_xy[i]
        xi = int(np.clip(round(x), 0, l_max - 1))
        yi = int(np.clip(round(y), 0, h_max - 1))
        candidats.append((i, reponse[yi, xi]))
    candidats.sort(key=lambda t: -t[1])

    distance_min = max(10, n * fraction_distance_min)
    coins_indices = []

    for tentative in range(nb_tentatives_max):
        coins_indices = []
        for i, score in candidats:
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
        print(f"⚠ Impossible d'atteindre {nb_coins} coins par Harris même en relâchant "
              f"distance_min (obtenu : {len(coins_indices)}).")

    coins_indices = np.array(sorted(coins_indices))
    coins = contour[coins_indices] if len(coins_indices) > 0 else np.array([])

    return coins, coins_indices


# ----------------------------------------------------------------------
# 4. Découpage en segments
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# 5. Ajustement de splines
# ----------------------------------------------------------------------

def nettoyer_doublons_consecutifs(segment):
    """Enlève les points consécutifs identiques (requis par splprep)."""
    garder = [segment[0]]
    for p in segment[1:]:
        if not np.array_equal(p, garder[-1]):
            garder.append(p)
    return np.array(garder)


def ajuster_spline_segment(segment, lissage=0):
    """Ajuste une spline paramétrique (x(u), y(u)) sur un segment."""
    segment = nettoyer_doublons_consecutifs(segment)
    x = segment[:, 1]
    y = segment[:, 0]
    tck, u = splprep([x, y], s=lissage)
    return tck


# ----------------------------------------------------------------------
# 6. Normalisation coin-à-coin
# ----------------------------------------------------------------------

def normaliser_segment(segment):
    """Exprime un segment dans un repère où le premier coin est l'origine
    et l'axe (along) relie les deux coins du segment. `height` mesure
    l'écart perpendiculaire à cet axe (utile pour comparer la forme des
    côtés indépendamment de leur position/orientation)."""
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

    return np.column_stack([along, height])



# ----------------------------------------------------------------------
# Figure de vérification (une par pièce, 4 panneaux = 4 étapes clés)
# ----------------------------------------------------------------------

_COULEURS_SEGMENTS = ["red", "blue", "green", "orange", "purple", "brown"]


def verifier_piece_visuellement(img, masque_final, contour_principal, contour_lisse,
                                 coins_finaux, segments, splines, nom_fichier, dossier_sortie=None):
    """Construit une figure unique à 4 panneaux qui permet de vérifier
    d'un coup d'œil que chaque étape du pipeline s'est bien passée :

      1. image + masque de segmentation en surimpression (la pièce est-elle
         bien isolée du fond ?)
      2. contour lissé + coins retenus (les 4 coins sont-ils aux bons
         endroits ?)
      3. segments/côtés colorés (le découpage en 4 côtés est-il cohérent ?)
      4. côtés normalisés dans le repère coin-à-coin, superposés (permet de
         comparer visuellement la forme des côtés d'une même pièce)

    Si `dossier_sortie` est fourni, la figure est enregistrée en PNG dans ce
    dossier (nom : verif_<nom_fichier>.png) plutôt qu'affichée à l'écran —
    pratique pour vérifier un lot de pièces sans ouvrir une fenêtre par pièce.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    fig.suptitle(nom_fichier, fontsize=14, fontweight="bold")

    # 1. Image + masque
    ax = axes[0, 0]
    ax.imshow(img)
    ax.imshow(masque_final, cmap="Reds", alpha=0.35)
    ax.set_title("1. Image + masque de segmentation")
    ax.axis("off")

    # 2. Contour + coins
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

    # 3. Segments + spline ajustée sur chaque côté
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

    # 4. Segments normalisés
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


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------

def analyser_piece(fichier_image=FICHIER_IMAGE, dossier_verification=None,
                    verifier=True, utiliser_grabcut=UTILISER_GRABCUT,
                    fraction_distance_min_coins=FRACTION_DISTANCE_MIN_COINS,
                    sigma_harris=3):
    """Analyse une pièce et renvoie masque, contour, coins, segments et
    splines. Si `verifier=True`, enregistre (ou affiche, si
    `dossier_verification` est None) la figure récapitulative en 4 panneaux.

    Les coins sont détectés par `detecter_coins_harris_convexe` : on ne
    considère que les points de l'enveloppe convexe du contour (élimine les
    creux d'encoches), puis on garde ceux à plus forte réponse de Harris
    (`sigma_harris` en contrôle l'échelle) — voir la docstring de cette
    fonction pour le détail.

    `fraction_distance_min_coins` : distance minimale entre deux coins
    retenus, en fraction du périmètre du contour. Une valeur plus PETITE
    autorise des coins plus proches les uns des autres (utile si deux vrais
    coins sont proches sur une pièce peu carrée) ; plus GRANDE force les
    coins à être mieux répartis.
    """
    nom_fichier = os.path.basename(fichier_image)

    img, img_h, img_s, img_v = charger_image(fichier_image)

    masque_final = obtenir_masque_piece(fichier_image, img_h, img_s, img_v,
                                         utiliser_grabcut=utiliser_grabcut)

    contour_principal = extraire_contour_principal(masque_final)
    contour_lisse = lisser_contour(contour_principal, sigma=7)

    coins_finaux, indices_coins = detecter_coins_harris_convexe(
        masque_final, contour_principal, nb_coins=NB_COINS,
        fraction_distance_min=fraction_distance_min_coins,
        sigma_harris=sigma_harris,
    )
    print(f"{len(coins_finaux)} coins retenus (enveloppe convexe + réponse de Harris)")

    if len(coins_finaux) < NB_COINS:
        print(f"⚠ Seulement {len(coins_finaux)} coin(s) trouvé(s) sur {NB_COINS} attendus pour "
              f"{fichier_image} : les segments risquent d'être incohérents. Vérifie le masque "
              f"(la pièce est-elle bien isolée du fond ?) ou ajuste FRACTION_DISTANCE_MIN_COINS.")

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
        "masque_final": masque_final,
        "contour_principal": contour_principal,
        "coins": coins_finaux,
        "segments": segments,
        "splines": splines,
    }


if __name__ == "__main__":
    # Charge automatiquement toutes les pièces du dossier elodie/puzzle1/,
    # nommées 1_1.jpg, 1_2.jpg, ..., 1_12.jpg, et génère la figure de
    # vérification de chacune dans puzzle1/verification/.
    DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))
    DOSSIER_PUZZLE = os.path.join(DOSSIER_SCRIPT, "puzzle1")
    DOSSIER_VERIFICATION = os.path.join(DOSSIER_PUZZLE, "verification")
    NB_PIECES = 12

    pieces = {}
    for i in range(1, NB_PIECES + 1):
        nom = f"1_{i}"
        chemin = os.path.join(DOSSIER_PUZZLE, f"{nom}.jpg")
        if not os.path.exists(chemin):
            print(f"⚠ Fichier introuvable, ignoré : {chemin}")
            continue
        print(f"--- Analyse de {nom} ---")
        resultat = analyser_piece(chemin, dossier_verification=DOSSIER_VERIFICATION)
        if len(resultat["segments"]) != NB_COINS:
            print(f"⚠ {nom} ignorée : {len(resultat['segments'])} côté(s) détecté(s) au lieu de {NB_COINS}.")
            continue
        pieces[nom] = resultat

    print(f"\n{len(pieces)}/{NB_PIECES} pièces analysées avec succès : {sorted(pieces.keys())}")