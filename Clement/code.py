
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString 

chemin_image = "/Users/clementpho/cours-info/Puzzle_test/PièceV3.jpg"

with open(chemin_image, "rb") as f:
    chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img_bgr = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise FileNotFoundError("Image introuvable. Vérifie le chemin du fichier.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.title("Image Originale")
plt.imshow(img_rgb)
plt.show()

img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
seuil = 150  
_, binary = cv2.threshold(img_gris, seuil, 255, cv2.THRESH_BINARY_INV)

plt.title(f"Image binarisée (Seuil à {seuil})")
plt.imshow(binary, cmap='gray')
plt.show()

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if not contours:
    raise ValueError("Aucun contour détecté sur l'image.")

contour = max(contours, key=cv2.contourArea)
pts = contour[:, 0, :]  # Formatage en tableau (N, 2)

rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
corners = np.int64(box) # Ce sont les 4 vrais coins extérieurs

vrais_coins = []
for c in corners:
    d = np.linalg.norm(pts - c, axis=1)
    vrais_coins.append(pts[np.argmin(d)])
corners = np.array(vrais_coins)

img_coins = img_rgb.copy()
for c in corners:
    cv2.circle(img_coins, tuple(c), 12, (255, 0, 0), -1) # Gros points rouges
plt.title(f"{len(corners)} Coins principaux détectés")
plt.imshow(img_coins)
plt.show()

indices = []
for c in corners:
    d = np.linalg.norm(pts - c, axis=1)
    indices.append(np.argmin(d))

indices = sorted(indices)
sides = []

for i in range(4):
    a = indices[i]
    b = indices[(i + 1) % 4]

    if a < b:
        side = pts[a:b+1]
    else:
        side = np.vstack((pts[a:], pts[:b+1]))

    sides.append(side)


lines = [LineString(s) for s in sides]
lines = [l.simplify(1.0, preserve_topology=True) for l in lines]

print("\n--- RÉSULTATS DE L'ANALYSE ---")
for i, l in enumerate(lines):
    print(f"Côté {i} :")
    print(f"  Longueur : {l.length:.2f} pixels")
    print(f"  Points   : {len(l.coords)}")