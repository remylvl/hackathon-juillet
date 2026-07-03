import cv2
import numpy as np

def extract_and_normalize_edges(mask: np.ndarray, corners: np.ndarray) -> list:
    """
    Extrait les 4 bords du masque et les normalise mathématiquement 
    pour un algorithme d'ajustement de courbe (Moindres Carrés).
    """
    # 1. Récupère le contour de la pièce (le plus grand, au cas où il y ait du bruit)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Aucun contour détecté dans le masque.")

    contour = max(contours, key=cv2.contourArea).squeeze()  # (N, 2)

    # 2. Pour chaque coin donné, on trouve le point du contour le plus proche
    corner_indices = []
    for corner in corners:
        distances = np.linalg.norm(contour - corner, axis=1)
        closest_idx = np.argmin(distances)
        corner_indices.append(closest_idx)

    # On trie pour parcourir le contour dans l'ordre (sens horaire)
    corner_indices.sort()

    # 3. On découpe le contour en 4 segments, un par côté, entre deux coins consécutifs
    segments = []
    for i in range(4):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % 4]  # modulo pour boucler du dernier coin au 1er

        if start_idx < end_idx:
            segment = contour[start_idx:end_idx + 1]
        else:
            # le segment traverse la fin du tableau : on recolle fin + début
            segment = np.vstack((contour[start_idx:], contour[:end_idx + 1]))
        segments.append(segment)

    # 4. On normalise chaque segment pour pouvoir comparer les côtés entre eux, peu importe leur position, orientation ou taille dans l'image d'origine
    normalized_segments = []
    for segment in segments:
        seg_float = segment.astype(float)

        A = seg_float[0]   # point de départ du côté
        B = seg_float[-1]  # point d'arrivée du côté

        # Translation : A devient l'origine (0, 0)
        seg_translated = seg_float - A

        # Rotation : on aligne le vecteur AB sur l'axe des x
        vector_AB = B - A
        angle = np.arctan2(vector_AB[1], vector_AB[0])

        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        seg_rotated = np.dot(seg_translated, rotation_matrix.T)

        # Mise à l'échelle : la longueur AB devient 1
        length = np.linalg.norm(vector_AB)
        if length > 0:
            seg_normalized = seg_rotated / length
        else:
            seg_normalized = seg_rotated

        normalized_segments.append(seg_normalized)

    return normalized_segments

def extract_four_corners(heatmap: np.ndarray, min_distance_pixels: int = 60) -> np.ndarray:
    """
    Extrait les coordonnées (x, y) des 4 coins à partir de la prédiction du réseau.
    """
    corners = []
    # copie pour ne pas modifier la prédiction originale
    temp_heatmap = heatmap.copy()
    
    for _ in range(4):
        # Le pixel le plus lumineux = le pic de la Gaussienne = un coin prédit
        _, _, _, max_loc = cv2.minMaxLoc(temp_heatmap)
        corners.append(max_loc)
        
        # On efface (met à 0) un disque autour de ce pic pour forcer la
        # prochaine recherche à trouver un coin différent, sinon minMaxLoc
        # retomberait toujours sur le même maximum
        cv2.circle(temp_heatmap, max_loc, min_distance_pixels, 0.0, -1)
        
    return np.array(corners, dtype=np.int32)


def refine_corners_with_math(mask: np.ndarray, nn_corners: np.ndarray, radius: int = 40) -> np.ndarray:
    """
    Raffine les coins prédits par l'IA en cherchant le point de courbure maximale 
    sur le contour dans un rayon donné autour de chaque prédiction.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return nn_corners
    contour = max(contours, key=cv2.contourArea).squeeze()  
    N = len(contour)

    refined_corners = []

    for nn_c in nn_corners:
        # Points du contour situés dans le rayon de recherche autour du coin prédit
        distances = np.linalg.norm(contour - nn_c, axis=1)
        pts_in_radius_indices = np.where(distances <= radius)[0]

        if len(pts_in_radius_indices) == 0:
            # rien à proximité : on garde la prédiction du réseau de neurones telle quelle
            refined_corners.append(nn_c)
            continue

        max_curvature = -1
        best_corner = nn_c

        # step : on regarde des voisins un peu éloignés sur le contour plutôt que les pixels juste adjacents, pour un calcul d'angle plus stable
        step = max(5, N // 100) 

        for idx in pts_in_radius_indices:
            idx_prev = (idx - step) % N
            idx_next = (idx + step) % N

            p = contour[idx]
            p_prev = contour[idx_prev]
            p_next = contour[idx_next]

            v1 = p_prev - p
            v2 = p_next - p

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                # cos_angle proche de 1 = angle très fermé = coin marqué
                # cos_angle proche de -1 = bord plat (angle de 180°)
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                if cos_angle > max_curvature:
                    max_curvature = cos_angle
                    best_corner = p

        refined_corners.append(best_corner)

    return np.array(refined_corners, dtype=np.int32)