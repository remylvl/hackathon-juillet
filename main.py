"""
Analyse d'une pièce de puzzle à partir d'une photo + optimisation B-spline
+ construction de dict_ctrl + association des contours.

Ce fichier est autonome.
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

# Configuration

FICHIER_IMAGE = "./resources/n_pieces_ensembles/Test_4_pieces_2.jpg"
AFFICHER_GRAPHIQUES = True  # passe à False pour désactiver tous les plt.show()

# Paramètres de calibration du masque bleu
SAT_MIN = 0.20
VAL_MIN = 0.20
LARGEUR_HUE = 0.06

# Paramètres de détection des coins / segments
NB_COINS = 4
SEUIL_COURBURE = 10       # nb de voisins pris en compte de chaque côté d'un point
DISTANCE_MIN_COINS = 500  # distance min entre deux coins retenus

# 1. Chargement de l'image et masque de couleur

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

# 2. Nettoyage du masque

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

# 3. Contour et détection des coins

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


def trouver_minima_locaux(scores, rayon=5):
    """Renvoie les indices des minima locaux d'une courbe 1D cyclique."""
    n = len(scores)
    minima = []
    for i in range(n):
        indices = [(i + k) % n for k in range(-rayon, rayon + 1)]
        if scores[i] == np.min(scores[indices]):
            minima.append(i)
    return np.array(minima, dtype=int)


def max_courbure(contour, points, seuil, distance_min, nb_coins=NB_COINS,
                 contour_score=None):
    """Sélectionne les `nb_coins` points de plus forte courbure parmi
    `points`, en imposant une distance minimale entre eux.

    La sélection n'est pas strictement greedy : si un meilleur point arrive
    dans la même zone qu'un point déjà retenu, il peut le remplacer.
    """
    if len(points) == 0:
        return np.array([]), np.array([])

    contour_pour_score = contour if contour_score is None else contour_score
    scores = np.array([score_courbure(contour_pour_score, index, seuil) for index in points])
    ordre = np.argsort(scores)

    selection = []

    for pos in ordre:
        index = int(points[pos])
        point = contour[index]
        score = float(scores[pos])

        proches = [k for k, item in enumerate(selection)
                   if np.sqrt((item["point"][0] - point[0]) ** 2 + (item["point"][1] - point[1]) ** 2) < distance_min]

        if proches:
            k_pire = max(proches, key=lambda k: selection[k]["score"])
            if score < selection[k_pire]["score"]:
                selection[k_pire] = {"index": index, "point": point, "score": score}
        elif len(selection) < nb_coins:
            selection.append({"index": index, "point": point, "score": score})
        elif score < max(selection, key=lambda item: item["score"])["score"]:
            k_pire = max(range(len(selection)), key=lambda k: selection[k]["score"])
            selection[k_pire] = {"index": index, "point": point, "score": score}

    selection.sort(key=lambda item: item["index"])
    points_filtres = np.array([item["point"] for item in selection])
    i_filtres = np.array([item["index"] for item in selection], dtype=int)
    return points_filtres, i_filtres

# 4. Découpage en segments

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

# 5. Ajustement de splines

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

# 6. Normalisation coin-à-coin

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

# 7. Recherche de correspondances entre bords (plusieurs pièces)

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


# Affichages (optionnels, activés par AFFICHER_GRAPHIQUES)

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


def afficher_contour_lisse_avec_coins(contour_lisse, coins_finaux):
    plt.figure()
    plt.plot(contour_lisse[:, 1], contour_lisse[:, 0], '.', color='steelblue')
    plt.scatter(coins_finaux[:, 1], coins_finaux[:, 0], color='black', s=80, zorder=5)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.title("Contour lisse avec coins")
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

# Analyse d'un dossier complet de pièces

def analyser_dossier(dossier="./resources/n_pieces_ensemble/", afficher=AFFICHER_GRAPHIQUES):
    """Analyse toutes les images du dossier et renvoie un dict {nom: resultat}."""
    fichiers = sorted(os.listdir(dossier))
    pieces = {}

    for fichier in fichiers:
        if fichier.lower().endswith((".png", ".jpg", ".jpeg")):
            chemin = os.path.join(dossier, fichier)
            print(f"\n=== Analyse de la pièce : {fichier} ===")
            resultat = analyser_piece(chemin, afficher=afficher)
            nom_piece = os.path.splitext(fichier)[0]
            pieces[nom_piece] = resultat

    return pieces

# Pipeline principal

def analyser_piece(fichier_image=FICHIER_IMAGE, afficher=AFFICHER_GRAPHIQUES):
    img, img_h, img_s, img_v = charger_image(fichier_image)

    masque, h_centre = creer_masque_bleu(img_h, img_s, img_v)
    print(f"Teinte bleue détectée automatiquement : {h_centre:.3f}")

    masque_final = nettoyer_masque(masque)
    _, X, Y = extraire_bord(masque_final)

    coins_harris = detecter_coins_harris(masque_final)
    print(f"{len(coins_harris)} coins détectés (Harris)")

    contour_principal = extraire_contour_principal(masque_final)
    contour_lisse = lisser_contour(contour_principal, sigma=10)
    contour_simplifie = measure.approximate_polygon(contour_lisse, tolerance=15)

    indices_coins_harris = sorted(trouver_indice(contour_lisse, c) for c in coins_harris)
    print("Indices des coins (Harris) dans le contour :", indices_coins_harris)

    scores_courbure = np.array([score_courbure(contour_lisse, i, SEUIL_COURBURE) for i in range(len(contour_lisse))])
    indices_points = trouver_minima_locaux(scores_courbure, rayon=5)
    if len(indices_points) < NB_COINS:
        indices_points = np.arange(len(contour_lisse), dtype=int)
    coins_finaux, indices_coins = max_courbure(
        contour_lisse, indices_points, SEUIL_COURBURE, DISTANCE_MIN_COINS,
        contour_score=contour_lisse
    )
    indices_coins = np.sort(indices_coins)
    print(f"{len(coins_finaux)} coins retenus après filtrage par courbure")

    segments = extraire_segments(contour_lisse, indices_coins)
    for i, seg in enumerate(segments):
        print(f"Segment {i} : {len(seg)} points")

    if afficher:
        afficher_segments(segments, coins_finaux)
        afficher_contour_lisse_avec_coins(contour_lisse, coins_finaux)

    splines = [ajuster_spline_segment(seg, lissage=len(seg) * 2) for seg in segments]

    if afficher:
        afficher_splines(segments, splines)
        afficher_segments_normalises(segments)

    segments_norm = [normaliser_segment(seg) for seg in segments]

    return {
        "masque_final": masque_final,
        "contour_principal": contour_lisse,
        "coins": coins_finaux,
        "segments": segments_norm,
        "splines": splines,
    }

# B-spline : ré-échantillonnage + optimisation

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

# Création du dictionnaire de contrôle (dict_ctrl)

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

# Classification bosse / creux / plat

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

# Construction de dict_ctrl à partir d'une seule pièce

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

# Association de contours (reconstruction du puzzle)

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
PENALITE_MEME_PIECE = 1e6


def associer_pieces(dict_ctrl):
    """
    Associe les côtés complémentaires (bosse <-> creux) entre pièces
    différentes, en résolvant un problème d'affectation optimale globale
    via l'algorithme hongrois (`scipy.optimize.linear_sum_assignment`).

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

    lignes, colonnes = linear_sum_assignment(cout)

    associations = []
    for i, j in zip(lignes, colonnes):
        # On écarte les associations "forcées" entre côtés d'une même pièce
        # (n'apparaissent que si le solveur n'avait vraiment aucun autre choix).
        if cout[i, j] >= PENALITE_MEME_PIECE:
            continue
        associations.append((bosses[i], creux[j]))

    return associations

# Visualisation détaillée des résultats d'une pièce

def afficher_resultats_piece(resultats, titre="Résultats pièce"):
    """
    Affiche une visualisation claire et compacte (grille 2x2) des résultats
    de l'analyse d'une pièce 
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


# Main : exécution du pipeline complet

if __name__ == "__main__":

    dossier = "./resources/n_pieces_ensembles/"

    # Analyse de toutes les pièces du dossier
    pieces = analyser_dossier(dossier, afficher=AFFICHER_GRAPHIQUES)

    # Recherche des correspondances
    correspondances = meilleure_correspondance_par_bord(pieces)

    print("\n=== Correspondances trouvées ===")
    afficher_correspondances(correspondances)

    # Construction de dict_ctrl pour toutes les pièces du dossier
    dict_ctrl = construire_dict_ctrl_pour_plusieurs_pieces(dossier)

    print("\n=== dict_ctrl construit pour toutes les pièces ===")
    for piece_id, cotes_piece in dict_ctrl.items():
        print(f"\nPièce {piece_id}:")
        for cote_id, info in enumerate(cotes_piece):
            print(f"  Côté {cote_id} : catégorie = {info['cat']}")

    # Association des pièces (recherche des côtés complémentaires)
    associations = associer_pieces(dict_ctrl)

    print("\n=== Schéma des pièces et associations ===")
    visualiser_schema_pieces(dict_ctrl, associations)

    print("\n=== Associations trouvées ===")
    for a in associations:
        print(a)