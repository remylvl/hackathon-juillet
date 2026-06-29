import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import ndimage
from skimage.feature import corner_harris, corner_peaks


img = image.imread("piece1.jpg")
#plt.imshow(img)
#plt.show()
img_gris = img[:, :, 2]
seuil = 148  
img_seuil = np.where(img_gris > seuil, 0, 255)
masque = img_seuil > 0

#plt.imshow(img_gris, cmap="grey")
#plt.show()

from scipy import ndimage
import numpy as np

# 1. Enlever les petits points isolés (bruit "poivre" à l'extérieur)
masque_propre = ndimage.binary_opening(masque, structure=np.ones((3, 3)))

# 2. Ne garder que la plus grande composante connexe = la pièce
labels, nb = ndimage.label(masque_propre)
tailles = ndimage.sum(masque_propre, labels, range(1, nb + 1))
plus_grande = np.argmax(tailles) + 1
masque_piece = labels == plus_grande

# 3. Combler les petits trous internes (les points noirs au milieu de la pièce)
masque_final = ndimage.binary_fill_holes(masque_piece)

# 4. Extraire le contour, maintenant propre
masque_erode = ndimage.binary_erosion(masque_final)
bord = masque_final & ~masque_erode

#plt.imshow(bord, cmap='gray')
#plt.show()

X = []
Y = []
for i, ligne in enumerate(bord):       # i = indice de la ligne (row)
    for j, point in enumerate(ligne):   # j = indice de la colonne (col)
        if point == True:
            X.append(i)
            Y.append(j)

plt.figure()
plt.plot( X, Y, '.', markersize=1)
plt.axis("equal")
#plt.show()

# 5. Détection des coins sur la forme pleine (pas juste le contour)
reponse = corner_harris(masque_final.astype(float))
coins = corner_peaks(reponse, min_distance=40, threshold_rel=0.3)
# coins a la forme (nb_coins, 2), colonnes = (row, col)

plt.figure()
plt.plot(X, Y, '.', markersize=1)
plt.scatter(coins[:, 0], coins[:, 1], color='red', s=60)
plt.axis("equal")
#plt.show()

print(f"{len(coins)} coins détectés :")
print(coins)


from skimage import measure

contours = measure.find_contours(masque_final, level=0.5)
contour_principal = max(contours, key=len)

# Simplifie le contour en gardant les sommets significatifs
contour_simplifie = measure.approximate_polygon(contour_principal, tolerance=15)

plt.figure()
plt.plot(X, Y, '.', markersize=1)
plt.scatter(contour_simplifie[:, 0], contour_simplifie[:, 1], color='red', s=60)
plt.axis("equal")
#plt.show()



def fusionner_points_proches(points, distance_min=100):
    points_filtres = [points[0]]
    for p in points[1:]:
        dernier = points_filtres[-1]
        dist = np.sqrt((p[0]-dernier[0])**2 + (p[1]-dernier[1])**2)
        if dist > distance_min:
            points_filtres.append(p)
    return np.array(points_filtres)

contour_simplifie_propre = fusionner_points_proches(contour_simplifie, distance_min=800)
print(f"{len(contour_simplifie_propre)} points après fusion")

plt.figure()
plt.plot(X, Y, '.', markersize=1)
plt.scatter(contour_simplifie_propre[:, 0], contour_simplifie_propre[:, 1], color='red', s=60)
plt.axis("equal")
#plt.show()

print(contour_simplifie_propre)

points = contour_simplifie_propre

vus = set()
points_uniques = []
for p in points:
    cle = tuple(p)          # un tuple est "hashable", utilisable dans un set
    if cle not in vus:
        vus.add(cle)
        points_uniques.append(p)

points_uniques = np.array(points_uniques)
print("points uniques : ", points_uniques)

coins_finaux = points_uniques  

# 1. Retrouver l'indice de chaque coin dans le contour original
def trouver_indice(contour, point):
    distances = np.sqrt((contour[:, 0] - point[0])**2 + (contour[:, 1] - point[1])**2)
    return np.argmin(distances)

indices_coins = sorted([trouver_indice(contour_principal, c) for c in coins_finaux])
print("Indices des coins dans le contour :", indices_coins)

# 2. Découper le contour en segments entre coins consécutifs
def extraire_segments(contour, indices):
    segments = []
    n = len(indices)
    for k in range(n):
        i_debut = indices[k]
        i_fin = indices[(k + 1) % n]
        if i_fin > i_debut:
            segment = contour[i_debut:i_fin + 1]
        else:
            # cas où le segment "boucle" entre la fin et le début du tableau
            segment = np.vstack([contour[i_debut:], contour[:i_fin + 1]])
        segments.append(segment)
    return segments

segments = extraire_segments(contour_principal, indices_coins)

# 3. Affichage : une couleur différente par segment
plt.figure()
couleurs = ['red', 'blue', 'green', 'orange']
for seg, c in zip(segments, couleurs):
    plt.plot(seg[:, 1], seg[:, 0], '.', markersize=2, color=c)

plt.scatter(coins_finaux[:, 1], coins_finaux[:, 0], color='black', s=80, zorder=5)
plt.axis("equal")
plt.gca().invert_yaxis()
plt.show()

for i, seg in enumerate(segments):
    print(f"Segment {i} : {len(seg)} points")

from scipy.interpolate import splprep, splev

def ajuster_spline_segment(segment, lissage=0):
    x = segment[:, 1]  # colonnes
    y = segment[:, 0]  # lignes

    # splprep attend une liste [x, y] et renvoie le paramétrage + les coefficients
    tck, u = splprep([x, y], s=lissage)
    return tck

def nettoyer_doublons_consecutifs(segment):
    garder = [segment[0]]
    for p in segment[1:]:
        if not np.array_equal(p, garder[-1]):
            garder.append(p)
    return np.array(garder)

def ajuster_spline_segment(segment, lissage=0):
    segment = nettoyer_doublons_consecutifs(segment)  # <- nettoyage avant ajustement
    x = segment[:, 1]
    y = segment[:, 0]

    tck, u = splprep([x, y], s=lissage)
    return tck

# Pour chaque segment
splines = []
for seg in segments:
    tck = ajuster_spline_segment(seg, lissage=len(seg) * 2)  # à ajuster
    splines.append(tck)

# Affichage : comparaison points réels vs spline ajustée
plt.figure()
for seg, tck in zip(segments, splines):
    u_fin = np.linspace(0, 1, 200)  # 200 points lissés pour tracer la courbe
    x_fit, y_fit = splev(u_fin, tck)

    plt.plot(seg[:, 1], seg[:, 0], '.', markersize=2, alpha=0.3)  # points réels
    plt.plot(x_fit, y_fit, '-', linewidth=2)                       # spline ajustée

plt.axis("equal")
plt.gca().invert_yaxis()
plt.show()


def normaliser_segment(segment):
    # Conversion explicite en (x, y) pour éviter toute confusion
    x = segment[:, 1].astype(float)
    y = segment[:, 0].astype(float)

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    # Translation : le premier coin devient l'origine
    px = x - x0
    py = y - y0

    # Vecteur coin-à-coin (direction de référence)
    vx, vy = x1 - x0, y1 - y0
    longueur = np.sqrt(vx**2 + vy**2)
    vx, vy = vx / longueur, vy / longueur  # vecteur unitaire

    # Projection de chaque point :
    # "along"  = position le long de l'axe coin-à-coin
    # "height" = déviation perpendiculaire à cet axe
    along = px * vx + py * vy
    height = px * vy - py * vx   # produit vectoriel (z) avec le vecteur unitaire

    return np.column_stack([along, height])

# Affichage
plt.figure()
for i, seg in enumerate(segments):
    seg_norm = normaliser_segment(seg)
    plt.plot(seg_norm[:, 0], seg_norm[:, 1], label=f"Segment {i}")

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend()
plt.axis("equal")
plt.show()