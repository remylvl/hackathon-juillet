

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.transforms import Affine2D
from scipy import ndimage
from scipy.interpolate import splprep, splev, BSpline, interp1d
from scipy.optimize import least_squares, linear_sum_assignment
from skimage import measure
from skimage.color import rgb2hsv
from scipy.spatial import ConvexHull
from skimage.feature import corner_harris

# ca permet d'éviter les problèmes pcq cv2 s'importe pas toujours bien 
try: 
    import cv2
    _CV2_DISPONIBLE = True
except ImportError:
    _CV2_DISPONIBLE = False


# configuration

DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PUZZLE = os.path.join(DOSSIER_SCRIPT, "detect_coins_vision_par_ordi", "puzzle1")
DOSSIER_VERIFICATION = os.path.join(DOSSIER_PUZZLE, "verification")

NB_PIECES = 12  # borne haute, le vrai nombre de pièces est détecté automatiquement

UTILISER_GRABCUT = True     # affine le masque avec grabcut (plus robuste aux ombres)
                             # nécessite pip install opencv-python

# seuils pour calibrer le masque de couleur (à changer selon la luminosité des photos)
SAT_MIN = 0.20
VAL_MIN = 0.20
LARGEUR_HUE = 0.06

# détection des coins (enveloppe convexe + harris)
NB_COINS = 4
FRACTION_DISTANCE_MIN_COINS = 0.15  # distance min entre 2 coins, en fraction du périmètre
SIGMA_HARRIS = 2

# optimisation bspline + classification
N_CTRL = 15
DEGRE_SPLINE = 2
SEUIL_PLAT = 0.08


# 1. chargement image / masque couleur / grabcut


def charger_image(chemin):
    # charge l'image et renvoie (rgb, h, s, v)
    img = image.imread(chemin)
    img_rgb = img[:, :, :3]
    img_hsv = rgb2hsv(img_rgb)
    return img_rgb, img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]


def creer_masque_couleur(img_h, img_s, img_v, sat_min=SAT_MIN, val_min=VAL_MIN,
                          largeur_hue=LARGEUR_HUE):
    # construit le masque en calibrant automatiquement la teinte dominante des pixels saturés
    candidat = (img_s > sat_min) & (img_v > val_min)
    if not np.any(candidat):
        raise ValueError("Aucun pixel suffisamment saturé pour calibrer le masque.")  # si mauvaise photo, ça plante pas tout, on s'en rend juste compte

    h_candidats = img_h[candidat]
    hist, bins = np.histogram(h_candidats, bins=60, range=(0.0, 1.0))
    i_pic = np.argmax(hist)
    h_centre = 0.5 * (bins[i_pic] + bins[i_pic + 1])

    masque = candidat & (np.abs(img_h - h_centre) <= largeur_hue)
    return masque, h_centre


def creer_masque_grabcut(fichier_image, masque_initial, iterations=5, marge_certaine_fond=40):
    # grabcut pour affiner le masque, sinon c'est pas robuste aux ombres
    if not _CV2_DISPONIBLE:
        raise ImportError("GrabCut nécessite opencv-python : `pip install opencv-python`.")

    img_bgr = cv2.imread(fichier_image)
    if img_bgr is None:
        raise ValueError(f"Impossible de lire l'image avec OpenCV : {fichier_image}")

    masque_gc = np.full(img_bgr.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    masque_gc[masque_initial] = cv2.GC_PR_FGD

    # ça c'est quasi sûrement la pièce
    noyau = np.ones((15, 15), np.uint8)
    coeur = cv2.erode(masque_initial.astype(np.uint8), noyau, iterations=1).astype(bool)
    masque_gc[coeur] = cv2.GC_FGD

    # et ça c'est quasi sûrement le fond
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
    # enlève le bruit, garde que la plus grande zone connexe, comble les trous
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
    # masque final : seuillage hsv, puis grabcut si dispo, puis nettoyage
    masque_brut, h_centre = creer_masque_couleur(img_h, img_s, img_v)
    print(f"Teinte dominante détectée automatiquement : {h_centre:.3f}")

    if utiliser_grabcut:
        if not _CV2_DISPONIBLE:
            print("⚠ opencv-python n'est pas installé (pip install opencv-python) : "
                  "GrabCut désactivé, utilisation du masque HSV brut.")
        else:
            masque_brut = creer_masque_grabcut(fichier_image, masque_brut)

    return nettoyer_masque(masque_brut)


# 2. contour + détection des coins (enveloppe convexe + harris)


def extraire_contour_principal(masque_final):
    # renvoie le plus long contour du masque
    contours = measure.find_contours(masque_final, level=0.5)
    return max(contours, key=len)


def lisser_contour(contour, sigma=7):
    # lisse le contour fermé avec un filtre gaussien (wrap car c'est une boucle)
    contour_ferme = np.vstack([contour, contour[:1]])
    row_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 0], sigma=sigma, mode="wrap")[:-1]
    col_lisse = ndimage.gaussian_filter1d(contour_ferme[:, 1], sigma=sigma, mode="wrap")[:-1]
    return np.column_stack([row_lisse, col_lisse])


def detecter_coins_harris_convexe(masque_final, contour, nb_coins=NB_COINS,
                                   fraction_distance_min=FRACTION_DISTANCE_MIN_COINS,
                                   sigma_harris=SIGMA_HARRIS, nb_tentatives_max=6):
    # deux critères combinés pour trouver les coins :
    # 1. on prend que les points de l'enveloppe convexe -> ça élimine direct les points concaves (le cou d'un tenon, le fond d'une encoche) qui peuvent avoir une courbure énorme mais qui sont pas des coins
    # 2. parmi ces candidats, on garde ceux avec la plus forte réponse de harris, qui détecte un vrai coin (deux bords ~droits qui se croisent) et répond beaucoup moins à une bosse arrondie de tenon

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
        distance_min /= 1.3  # on relâche si on trouve pas assez de coins
    else:
        print(f"⚠ Impossible d'atteindre {nb_coins} coins par Harris même en relâchant "
              f"distance_min (obtenu : {len(coins_indices)}).")

    coins_indices = np.array(sorted(coins_indices))
    coins = contour[coins_indices] if len(coins_indices) > 0 else np.array([])

    return coins, coins_indices


# 3. découpage en segments + spline (pour l'affichage)


def extraire_segments(contour, indices):
    # découpe le contour en segments entre coins consécutifs (c'est une boucle)
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
    # enlève les points consécutifs identiques (splprep plante sinon)
    garder = [segment[0]]
    for p in segment[1:]:
        if not np.array_equal(p, garder[-1]):
            garder.append(p)
    return np.array(garder)


def ajuster_spline_segment(segment, lissage=0):
    # spline (x(u), y(u)) sur un segment, juste pour l'affichage
    segment = nettoyer_doublons_consecutifs(segment)
    x = segment[:, 1]
    y = segment[:, 0]
    tck, u = splprep([x, y], s=lissage)
    return tck


# 4. normalisation coin à coin, optimisation bspline, classification bosse/creux/plat


def normaliser_segment(segment):
    # met le segment dans un repère où le 1er coin est l'origine et l'axe
    # relie les 2 coins, mis à l'échelle par la longueur du segment
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
    # ré-échantillonne le segment à pas constant (n points)
    d = np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1))
    u = np.concatenate([[0], np.cumsum(d)])
    u = u / u[-1]

    fx = interp1d(u, segment[:, 0])
    fy = interp1d(u, segment[:, 1])

    u_new = np.linspace(0, 1, n)
    return np.column_stack([fx(u_new), fy(u_new)])


def optimiser_segment(segment, n_ctrl=N_CTRL, degree=DEGRE_SPLINE):
    # trouve par moindres carrés les points de contrôle de la bspline qui colle le mieux au segment
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
    # 0 = plat (bord du puzzle), 1 = bosse (tenon), 2 = creux
    y = ctrl_points[:, 1]
    y_max = np.max(y)
    y_min = np.min(y)

    if abs(y_max) < seuil_plat and abs(y_min) < seuil_plat:
        return 0
    if abs(y_max) > abs(y_min):
        return 1
    return 2


# 5. figure de vérification par pièce

_COULEURS_SEGMENTS = ["red", "blue", "green", "orange", "purple", "brown"]


def verifier_piece_visuellement(img, masque_final, contour_principal, contour_lisse,
                                 coins_finaux, segments, splines, nom_fichier, dossier_sortie=None):
    # figure 4 panneaux : masque, contour+coins, segments+spline, côtés normalisés
    # enregistrée en png si dossier_sortie est donné, sinon juste affichée
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
    ax.set_title("2. Contour + coins détectés (convexe + Harris)")
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


# 6. analyse d'une pièce (pipeline complet)


def analyser_piece(fichier_image, dossier_verification=None, verifier=True,
                    utiliser_grabcut=UTILISER_GRABCUT, nb_coins=NB_COINS,
                    fraction_distance_min_coins=FRACTION_DISTANCE_MIN_COINS,
                    sigma_harris=SIGMA_HARRIS):
    # renvoie image, masque, contour, coins, segments (bruts) et splines
    # les coins sont trouvés par detecter_coins_harris_convexe (enveloppe convexe + harris)
    nom_fichier = os.path.basename(fichier_image)

    img, img_h, img_s, img_v = charger_image(fichier_image)

    masque_final = obtenir_masque_piece(fichier_image, img_h, img_s, img_v,
                                         utiliser_grabcut=utiliser_grabcut)

    contour_principal = extraire_contour_principal(masque_final)
    contour_lisse = lisser_contour(contour_principal, sigma=7)

    coins_finaux, indices_coins = detecter_coins_harris_convexe(
        masque_final, contour_principal, nb_coins=nb_coins,
        fraction_distance_min=fraction_distance_min_coins,
        sigma_harris=sigma_harris,
    )
    print(f"{len(coins_finaux)} coins retenus (enveloppe convexe + réponse de Harris)")

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


# 6bis. découpe de la pièce sur fond transparent


def decouper_piece(img, masque):
    # recadre sur la boîte englobante et met le fond transparent (alpha = masque)
    # renvoie (rgba, (r0, c0)) avec (r0, c0) le décalage du recadrage dans l'image d'origine
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


# 7. construction de dict_ctrl pour toutes les pièces d'un dossier


def construire_dict_ctrl_pour_plusieurs_pieces(dossier, dossier_verification=None,
                                                nb_pieces=NB_PIECES,
                                                afficher_splines_optimisees=False):
    # construit dict_ctrl : pour chaque pièce, la liste de ses côtés avec les points de contrôle bspline et la catégorie bosse/creux/plat (cat)
    # renvoie aussi images_pieces (piece_id -> image rgba) et geometrie_pieces (piece_id -> coins (4,2) en coordonnées locales à l'image découpée, dans le même ordre que les côtés de dict_ctrl)

    dict_ctrl = {}
    images_pieces = {}
    geometrie_pieces = {}

    if not os.path.isdir(dossier):
        raise FileNotFoundError(
            f"Dossier introuvable : {dossier}\n"
            f"Vérifie DOSSIER_PUZZLE en haut du script (nom/emplacement du dossier)."
        )

    # on détecte les fichiers vraiment présents (1_1.jpg, 1_2.jpg, ...) au lieu de supposer qu'il y en a exactement nb_pieces, ça évite le spam "fichier introuvable" et ça marche même avec un lot incomplet
    fichiers_trouves = []
    for i in range(1, nb_pieces + 1):
        nom = f"1_{i}.jpg"
        chemin = os.path.join(dossier, nom)
        if os.path.exists(chemin):
            fichiers_trouves.append((i, nom, chemin))

    if not fichiers_trouves:
        print(f"⚠ Aucun fichier '1_1.jpg' à '1_{nb_pieces}.jpg' trouvé dans {dossier}")
        print(f"   Contenu réel du dossier : {sorted(os.listdir(dossier))}")
        return dict_ctrl, images_pieces, geometrie_pieces

    print(f"{len(fichiers_trouves)} photo(s) de pièce trouvée(s) sur {nb_pieces} attendue(s) "
          f"({', '.join(nom for _, nom, _ in fichiers_trouves)})")

    piece_id = 0
    for i, nom, chemin in fichiers_trouves:
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


# 8. association des pièces (algo hongrois) + schéma


def distance_cotes(ctrlA, ctrlB):
    # mesure à quel point 2 côtés s'emboîtent bien (bosse d'un côté, creux de l'autre)
    # on inverse le sens de parcours et le signe de la hauteur de ctrlB avant de comparer
    ctrlB_aligne = ctrlB[::-1].copy()
    ctrlB_aligne[:, 1] = -ctrlB_aligne[:, 1]
    return np.linalg.norm(ctrlA - ctrlB_aligne)


PENALITE_MEME_PIECE = 1e6  # pour empêcher une pièce de s'associer avec elle-même


def associer_pieces(dict_ctrl):
    # associe les côtés complémentaires (bosse <-> creux) entre pièces différentes avec l'algo hongrois, qui minimise la distance totale sur toutes les associations d'un coup (contrairement à un appariement glouton, qui peut gâcher une bonne paire trouvée plus tard) renvoie une liste de paires ((pieceA, coteA), (pieceB, coteB))
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
        if cout[i, j] >= PENALITE_MEME_PIECE:
            continue
        associations.append((bosses[i], creux[j]))

    return associations


def visualiser_schema_pieces(dict_ctrl, associations):
    # dessine les pièces en carrés et relie les côtés associés par des pointillés rouges
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


# 9. assemblage visuel approximatif (forces dirigées)


def layout_force_diriges(n_pieces, aretes, iterations=300, seed=0):
    # place les pièces dans le plan avec des forces dirigées : tout se repousse, sauf les pièces reliées par une association qui s'attirent
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, size=(n_pieces, 2))
    k = 1.0 / np.sqrt(n_pieces)

    for it in range(iterations):
        deplacement = np.zeros_like(pos)

        for i in range(n_pieces):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf
            dist = np.maximum(dist, 1e-6)
            force_rep = (k ** 2) / dist
            deplacement[i] += np.sum((delta.T * (force_rep / dist)).T, axis=0)

        for (a, b) in aretes:
            delta = pos[a] - pos[b]
            dist = max(np.linalg.norm(delta), 1e-6)
            force_att = dist ** 2 / k
            direction = delta / dist
            deplacement[a] -= direction * force_att
            deplacement[b] += direction * force_att

        temperature = 0.1 * (1 - it / iterations)
        normes = np.maximum(np.linalg.norm(deplacement, axis=1), 1e-6)
        pos += (deplacement.T * (np.minimum(normes, temperature) / normes)).T

    return pos


def _extent_image(rgba, x, y, taille_cible=1.6):
    # calcule l'extent pour afficher rgba centrée en (x, y) en gardant son ratio largeur/hauteur
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
    # affiche les vraies photos des pièces, placées par forces dirigées pour que les pièces associées se retrouvent proches les unes des autres
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


# 10. assemblage géométrique réel (côtés collés bord à bord)


def _matrice_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def assembler_pieces(dict_ctrl, associations, geometrie_pieces, piece_racine=None):
    # calcule la rotation + translation de chaque pièce atteignable depuis piece_racine, pour que les côtés associés se superposent exactement renvoie (placements, non_places)
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
    # assemble et affiche les pièces en collant vraiment leurs côtés associés (rotation + translation exactes). renvoie placements
    placements, non_places = assembler_pieces(
        dict_ctrl, associations, geometrie_pieces, piece_racine=piece_racine
    )

    if non_places:
        print(f"⚠ Pièces non reliées à l'assemblage principal "
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
        ax.set_ylim(tous[:, 1].max() + marge, tous[:, 1].min() - marge)

    plt.tight_layout()
    plt.show()

    return placements


# main

if __name__ == "__main__":
    # 1. dict_ctrl (+ images et géométrie) pour toutes les pièces du dossier (détection convexe+harris, figure de vérif enregistrée pour chaque pièce, puis bspline + classification bosse/creux/plat)
    dict_ctrl, images_pieces, geometrie_pieces = construire_dict_ctrl_pour_plusieurs_pieces(
        DOSSIER_PUZZLE, dossier_verification=DOSSIER_VERIFICATION, nb_pieces=NB_PIECES
    )

    if not dict_ctrl:
        print("\n⚠ Aucune pièce n'a été ajoutée à dict_ctrl : soit aucune photo n'a été "
              "trouvée, soit la détection de coins a échoué (4 côtés non trouvés) pour "
              "toutes les photos présentes. Regarde les figures dans le dossier "
              f"'{DOSSIER_VERIFICATION}' et les messages ⚠ ci-dessus pour comprendre "
              "pourquoi, puis relance. Arrêt du script ici.")
        raise SystemExit(1)

    print("\n=== dict_ctrl construit pour toutes les pièces ===")
    for piece_id, cotes_piece in dict_ctrl.items():
        print(f"\nPièce {piece_id}:")
        for cote_id, info in enumerate(cotes_piece):
            print(f"  Côté {cote_id} : catégorie = {info['cat']}")

    # 2. association des pièces avec l'algo hongrois
    associations = associer_pieces(dict_ctrl)

    # 3. schéma abstrait des pièces et de leurs associations
    print("\n=== Schéma des pièces et associations ===")
    visualiser_schema_pieces(dict_ctrl, associations)

    # 4. assemblage géométrique réel : on part d'une pièce (la 0 par défaut) et on colle les côtés associés de proche en proche
    print("\n=== Assemblage des pièces (côtés collés) ===")
    visualiser_assemblage_colle(dict_ctrl, associations, images_pieces, geometrie_pieces)

    print("\n=== Associations trouvées ===")
    for a in associations:
        print(a)