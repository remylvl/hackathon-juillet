import cv2
import numpy as np

def extract_and_normalize_edges(mask: np.ndarray, corners: np.ndarray) -> list:
    """
    Extrait les 4 bords du masque et les normalise mathématiquement 
    pour un algorithme d'ajustement de courbe (Moindres Carrés).
    """
    # 1. Extraction du contour complet continu
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Aucun contour détecté dans le masque.")
    
    # On prend le contour le plus grand (la pièce) et on enlève la dimension inutile d'OpenCV
    contour = max(contours, key=cv2.contourArea).squeeze() # Format: (N, 2)

    # 2. Calibrage des coins sur le contour
    corner_indices = []
    for corner in corners:
        # Calcule la distance entre ce coin et TOUS les points du contour
        distances = np.linalg.norm(contour - corner, axis=1)
        closest_idx = np.argmin(distances)
        corner_indices.append(closest_idx)
        
    # On trie les indices pour parcourir le périmètre dans le bon sens (horaire)
    corner_indices.sort()

    # 3. Découpage des 4 segments
    segments = []
    for i in range(4):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % 4] # Le modulo permet de boucler sur le 1er coin
        
        if start_idx < end_idx:
            segment = contour[start_idx:end_idx+1]
        else:
            # Gestion de la boucle : on assemble la fin et le début du tableau
            segment = np.vstack((contour[start_idx:], contour[:end_idx+1]))
        segments.append(segment)

    # 4. Normalisation géométrique
    normalized_segments = []
    for segment in segments:
        seg_float = segment.astype(float)
        
        A = seg_float[0]  # Point de départ
        B = seg_float[-1] # Point d'arrivée
        
        # --- Translation ---
        seg_translated = seg_float - A
        
        # --- Rotation ---
        vector_AB = B - A
        angle = np.arctan2(vector_AB[1], vector_AB[0])
        
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # Application de la matrice de rotation sur tous les points du segment
        seg_rotated = np.dot(seg_translated, rotation_matrix.T)
        
        # --- Mise à l'échelle ---
        length = np.linalg.norm(vector_AB)
        if length > 0:
            seg_normalized = seg_rotated / length
        else:
            seg_normalized = seg_rotated
            
        normalized_segments.append(seg_normalized)

    return normalized_segments