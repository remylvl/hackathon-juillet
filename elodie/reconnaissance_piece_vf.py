"""
Analyse d'une pièce de puzzle à partir d'une photo.

Étapes :
1. Segmentation de la pièce par couleur (masque HSV calibré automatiquement).
2. Nettoyage du masque (ouverture/fermeture morphologique, remplissage des trous).
3. Extraction du contour et détection des coins (courbure locale).
4. Découpage du contour en segments (un par côté de la pièce).
5. Ajustement d'une spline sur chaque segment.
6. Normalisation de chaque segment dans un repère coin-à-coin, pour pouvoir
   comparer la forme des côtés entre pièces différentes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage.feature import corner_harris, corner_peaks
from skimage import measure
from skimage.color import rgb2hsv


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

FICHIER_IMAGE = "./resources/piece4.jpeg"
AFFICHER_GRAPHIQUES = True  # passe à False pour désactiver tous les plt.show()

# Paramètres de calibration du masque bleu
SAT_MIN = 0.20
VAL_MIN = 0.20
LARGEUR_HUE = 0.06

# Paramètres de détection des coins / segments
NB_COINS = 4
SEUIL_COURBURE = 50       # nb de voisins pris en compte de chaque côté d'un point
DISTANCE_MIN_COINS = 200  # distance min entre deux coins retenus


# ----------------------------------------------------------------------
# 1. Chargement de l'image et masque de couleur
# ----------------------------------------------------------------------

def charger_image(chemin):
    """Charge l'image et renvoie les canaux HSV (RGB seulement, sans alpha)."""
    img = image.imread(chemin)
    img_hsv = rgb2hsv(img[:, :, :3])
    return img, img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]


def creer_masque_bleu(img_h, img_s, img_v, sat_min=SAT_MIN, val_min=VAL_MIN,
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

    masque_final = ndimage.binary_fill_holes(masque_piece)
    return masque_final


def extraire_bord(masque_final):
    """Renvoie les coordonnées (X, Y) des pixels de bord du masque."""
    masque_erode = ndimage.binary_erosion(masque_final)
    bord = masque_final & ~masque_erode

    X, Y = np.nonzero(bord)
    return bord, list(X), list(Y)


# ----------------------------------------------------------------------
# 3. Contour et détection des coins
# ----------------------------------------------------------------------

def detecter_coins_harris(masque_final, min_distance=150, threshold_rel=0.1):
    """Détection de coins par la méthode de Harris (donne un premier
    ensemble de candidats, affiné ensuite par courbure)."""
    reponse = corner_harris(masque_final.astype(float))
    coins = corner_peaks(reponse, min_distance=min_distance, threshold_rel=threshold_rel)
    return coins


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


def trouver_indice(contour, point):
    """Indice du point du contour le plus proche d'un point donné."""
    distances = np.sqrt((contour[:, 0] - point[0]) ** 2 + (contour[:, 1] - point[1]) ** 2)
    return np.argmin(distances)


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
    et l'axe (along) relie les deux coins du segment. Les deux coordonnées
    sont normalisées par la longueur du côté, ce qui rend les segments
    comparables entre eux sans amplifier artificiellement les côtés plats."""
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


def _reechantillonner_normalise(segment, nb_points=100, normaliser_echelle=False):
    """Normalise un segment (repère coin-à-coin) puis ré-échantillonne sa
    hauteur (`height`) sur `nb_points` régulièrement espacés le long de
    l'axe coin-à-coin (`along`, ramené à [0, 1]).

    Si `normaliser_echelle=True`, la hauteur est elle aussi divisée par la
    longueur du segment : la comparaison devient alors indépendante de
    l'échelle (utile si deux photos n'ont pas exactement le même zoom).
    """
    seg_norm = normaliser_segment(segment)
    along = seg_norm[:, 0]
    height = seg_norm[:, 1]

    # np.interp exige un x croissant : on trie par along
    ordre = np.argsort(along)
    along = along[ordre]
    height = height[ordre]

    longueur = along[-1] - along[0]
    if longueur <= 0:
        raise ValueError("Segment dégénéré : impossible de le normaliser.")

    along_t = (along - along[0]) / longueur  # dans [0, 1]
    t_commun = np.linspace(0, 1, nb_points)
    height_rééch = np.interp(t_commun, along_t, height)

    if normaliser_echelle:
        height_rééch = height_rééch / longueur

    return height_rééch


def comparer_splines(segment1, segment2, marge=5.0, nb_points=100,
                      normaliser_echelle=False, autoriser_miroir=True):
    """Compare la forme de deux côtés de pièce (segments de contour bruts,
    en (row, col)) à une marge d'erreur près.

    Principe :
      1. Chaque segment est normalisé dans son propre repère coin-à-coin
         (voir `normaliser_segment`), ce qui neutralise translation et
         rotation.
      2. Les deux courbes de hauteur sont ré-échantillonnées sur la même
         grille de `nb_points` le long de l'axe coin-à-coin.
      3. On mesure l'écart RMS et l'écart max entre les deux courbes.
      4. On compare aussi la version miroir de segment2 (utile pour tester
         si deux côtés sont complémentaires plutôt qu'identiques : un
         renflement chez l'un doit correspondre à un creux chez l'autre une
         fois mis en miroir).

    Paramètres
    ----------
    segment1, segment2 : np.ndarray de forme (N, 2), colonnes (row, col)
        Les segments à comparer, tels que renvoyés par `extraire_segments`.
    marge : float
        Tolérance sur l'écart RMS (en pixels, sauf si `normaliser_echelle=True`,
        auquel cas c'est une fraction de la longueur du segment).
    nb_points : int
        Nombre de points de ré-échantillonnage pour la comparaison.
    normaliser_echelle : bool
        Si True, rend la comparaison indépendante de l'échelle/zoom.
    autoriser_miroir : bool
        Si True, teste aussi segment2 mis en miroir et garde le meilleur résultat.

    Renvoie
    -------
    dict avec :
      - "identique" (bool) : True si l'écart RMS <= marge
      - "erreur_rms" (float)
      - "erreur_max" (float)
      - "miroir_utilise" (bool) : True si c'est la version miroir qui a été retenue
    """
    h1 = _reechantillonner_normalise(segment1, nb_points, normaliser_echelle)
    h2 = _reechantillonner_normalise(segment2, nb_points, normaliser_echelle)

    def erreurs(a, b):
        diff = a - b
        rms = np.sqrt(np.mean(diff ** 2))
        maxi = np.max(np.abs(diff))
        return rms, maxi

    rms_direct, max_direct = erreurs(h1, h2)
    miroir_utilise = False

    if autoriser_miroir:
        rms_miroir, max_miroir = erreurs(h1, -h2)
        if rms_miroir < rms_direct:
            rms_direct, max_direct = rms_miroir, max_miroir
            miroir_utilise = True

    return {
        "identique": rms_direct <= marge,
        "erreur_rms": rms_direct,
        "erreur_max": max_direct,
        "miroir_utilise": miroir_utilise,
    }


# ----------------------------------------------------------------------
# 7. Recherche de correspondances entre bords (plusieurs pièces)
# ----------------------------------------------------------------------

def trouver_correspondances_bords(pieces, marge=5.0, nb_points=100,
                                   normaliser_echelle=False, autoriser_miroir=True):
    """Compare tous les bords de toutes les pièces entre eux, deux par deux,
    et renvoie la liste des correspondances jugées « identiques » (à la
    marge d'erreur près), triée par erreur croissante.

    Paramètres
    ----------
    pieces : dict {nom_piece: resultat}
        `resultat` doit être le dict renvoyé par `analyser_piece` (on utilise
        sa clé "segments"). Exemple :
            pieces = {
                "piece4": analyser_piece("./resources/piece4.jpeg", afficher=False),
                "piece3": analyser_piece("./resources/piece3.jpeg", afficher=False),
            }
    marge, nb_points, normaliser_echelle, autoriser_miroir :
        transmis tels quels à `comparer_splines`.

    Renvoie
    -------
    Liste de dicts triée par "erreur_rms" croissante :
        [{"piece1", "bord1", "piece2", "bord2", "erreur_rms", "erreur_max", "miroir"}, ...]
    Un bord n'est jamais comparé à lui-même, et chaque paire n'apparaît qu'une fois
    (y compris pour les deux bords d'une même pièce).
    """
    noms = list(pieces.keys())
    correspondances = []

    for i in range(len(noms)):
        for j in range(i, len(noms)):
            nom1, nom2 = noms[i], noms[j]
            segments1 = pieces[nom1]["segments"]
            segments2 = pieces[nom2]["segments"]
            meme_piece = (i == j)

            for b1, seg1 in enumerate(segments1):
                for b2, seg2 in enumerate(segments2):
                    if meme_piece and b2 <= b1:
                        continue  # évite auto-comparaison et doublons dans la même pièce

                    res = comparer_splines(
                        seg1, seg2,
                        marge=marge,
                        nb_points=nb_points,
                        normaliser_echelle=normaliser_echelle,
                        autoriser_miroir=autoriser_miroir,
                    )
                    if res["identique"]:
                        correspondances.append({
                            "piece1": nom1, "bord1": b1,
                            "piece2": nom2, "bord2": b2,
                            "erreur_rms": res["erreur_rms"],
                            "erreur_max": res["erreur_max"],
                            "miroir": res["miroir_utilise"],
                        })

    correspondances.sort(key=lambda c: c["erreur_rms"])
    return correspondances


def meilleure_correspondance_par_bord(pieces, **kwargs):
    """Comme `trouver_correspondances_bords`, mais ne garde que la MEILLEURE
    correspondance pour chaque bord (utile en pratique : dans un puzzle, un
    bord interne s'emboîte avec exactement un autre bord ; un bord de bordure
    du puzzle, lui, n'a normalement aucun partenaire).

    Renvoie la même structure que `trouver_correspondances_bords`, mais
    filtrée pour qu'aucun bord n'apparaisse plus d'une fois comme "meilleur"
    partenaire d'un autre.
    """
    toutes = trouver_correspondances_bords(pieces, **kwargs)

    deja_matches = set()
    meilleures = []
    for c in toutes:  # déjà trié par erreur croissante
        cle1 = (c["piece1"], c["bord1"])
        cle2 = (c["piece2"], c["bord2"])
        if cle1 in deja_matches or cle2 in deja_matches:
            continue
        meilleures.append(c)
        deja_matches.add(cle1)
        deja_matches.add(cle2)

    return meilleures


def afficher_correspondances(correspondances):
    """Affichage lisible d'une liste de correspondances de bords."""
    if not correspondances:
        print("Aucune correspondance trouvée.")
        return
    for c in correspondances:
        miroir = " (miroir)" if c["miroir"] else ""
        print(
            f"{c['piece1']} [bord {c['bord1']}]  <->  "
            f"{c['piece2']} [bord {c['bord2']}]{miroir}  "
            f"— erreur RMS: {c['erreur_rms']:.2f}, max: {c['erreur_max']:.2f}"
        )


# ----------------------------------------------------------------------
# Affichages (optionnels, activés par AFFICHER_GRAPHIQUES)
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
# Pipeline principal
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


if __name__ == "__main__":
    # Exemple : analyser plusieurs pièces et chercher les correspondances de bords.
    pieces = {
        "piece4": analyser_piece("./resources/piece4.jpeg", afficher=False),
        "piece3": analyser_piece("./resources/piece3.jpeg", afficher=False),
        # ajoute ici les autres pièces à comparer, ex :
        # "piece5": analyser_piece("./resources/piece5.jpeg", afficher=False),
    }

    correspondances = meilleure_correspondance_par_bord(pieces, marge=5.0)
    afficher_correspondances(correspondances)