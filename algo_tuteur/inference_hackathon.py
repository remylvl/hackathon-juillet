import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import de l'architecture du modèle
# (Assure-toi que train_unet.py est dans le même dossier)
from train_unet import PuzzleUNet 
import cv2
import numpy as np

def extract_and_normalize_edges(mask: np.ndarray, corners: np.ndarray) -> list:
    """
    Extrait les 4 bords du masque et les normalise mathématiquement 
    pour un algorithme d'ajustement de courbe (Moindres Carrés).
    """
    # 1. Extraction du contour complet continu
    # CHAIN_APPROX_NONE est crucial : on veut tous les pixels, sans simplification
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

def extract_four_corners(heatmap: np.ndarray, min_distance_pixels: int = 15) -> np.ndarray:
    """
    Extrait les coordonnées (x, y) des 4 coins à partir de la prédiction du réseau.
    Utilise une approche d'effacement pour garantir des coins distincts.
    """
    corners = []
    # On travaille sur une copie pour ne pas détruire la prédiction originale
    temp_heatmap = heatmap.copy()
    
    for _ in range(4):
        # 1. Trouve les coordonnées du pixel le plus lumineux (le pic exact de la Gaussienne)
        # cv2.minMaxLoc retourne (min_val, max_val, min_loc, max_loc)
        _, _, _, max_loc = cv2.minMaxLoc(temp_heatmap)
        
        # 2. Sauvegarde la coordonnée (x, y)
        corners.append(max_loc)
        
        # 3. Effacement (Le cœur de l'astuce)
        # On dessine un cercle noir (valeur 0.0) de rayon 'min_distance_pixels' 
        # centré sur le pic qu'on vient de trouver. Cela force OpenCV à 
        # chercher le prochain pic sur un autre coin de la pièce.
        cv2.circle(temp_heatmap, max_loc, min_distance_pixels, 0.0, -1)
        
    return np.array(corners, dtype=np.int32)


def refine_corners_with_math(mask: np.ndarray, nn_corners: np.ndarray) -> np.ndarray:
    """
    Corrige l'imprécision de l'IA en 'aimantant' ses prédictions
    sur les vrais angles géométriques détectés mathématiquement.
    """
    # 1. Détection de tous les angles saillants sur le masque binaire pur (Shi-Tomasi)
    # On autorise jusqu'à 20 coins (les 4 de la pièce + les fausses alertes des encoches)
    math_corners = cv2.goodFeaturesToTrack(
        mask, maxCorners=20, qualityLevel=0.01, minDistance=15
    )
    
    if math_corners is None:
        print("Attention : Aucun coin mathématique trouvé, on garde l'IA brute.")
        return nn_corners
        
    math_corners = math_corners.reshape(-1, 2)
    refined_corners = []
    
    # 2. Association de l'IA avec la Mathématique
    for nn_c in nn_corners:
        # On calcule la distance entre le point prédit par l'IA et tous les vrais coins
        distances = np.linalg.norm(math_corners - nn_c, axis=1)
        closest_idx = np.argmin(distances)
        
        # Si un vrai coin existe dans un rayon de 40 pixels, on s'y accroche
        if distances[closest_idx] < 40: 
            refined_corners.append(math_corners[closest_idx])
        else:
            # Sinon (cas d'erreur rare), on fait confiance à l'IA
            refined_corners.append(nn_c)
            
    return np.array(refined_corners, dtype=np.int32)

def load_trained_model(weights_path="unet_puzzle_weights.pth"):
    """ Charge le modèle en mémoire et le prépare pour la prédiction pure. """
    
    # 1. Détection matérielle optimisée pour ton architecture
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Modèle chargé sur : {device}")

    # 2. Instanciation et chargement des poids
    model = PuzzleUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 3. CRITIQUE : Verrouillage des couches (BatchNorm, Dropout)
    model.eval()
    
    return model, device

def predict_mask(model, device, image_path):
    """ Prend une vraie photo en entrée et retourne le masque binaire et les 4 coins. """
    
    # --- 1. PRÉ-TRAITEMENT (Extraction et formatage) ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erreur : Impossible de lire l'image '{image_path}'. Vérifie le chemin.")
    img = 255 - img
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    # --- 2. PRÉDICTION ---
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # --- 3. POST-TRAITEMENT MULTI-TASK ---
    # Le tenseur en sortie est maintenant (1, 2, 128, 128)
    # On le ramène sur le CPU et on retire la dimension Batch : shape devient (2, 128, 128)
    output_matrix = output_tensor.squeeze().cpu().numpy()
    
    # Séparation des canaux
    probability_mask = output_matrix[0]  # Canal 0 : Segmentation
    heatmap_corners = output_matrix[1]   # Canal 1 : Carte de chaleur
    
    # 1. Traitement du Masque
    binary_mask = (probability_mask > 0.5).astype(np.uint8)
    final_cv2_mask = binary_mask * 255
    
    # 2. Extraction des Coins
    corners = extract_four_corners(heatmap_corners, min_distance_pixels=60)
    
    corners = refine_corners_with_math(final_cv2_mask, corners)
    
    return final_cv2_mask, img_resized, corners



# EXÉCUTION RÉELLE (Liaison IA -> Mathématiques -> Données)
if __name__ == "__main__":
    from inference_hackathon import load_trained_model, predict_mask, refine_corners_with_math
    from geometry_utils import extract_and_normalize_edges
    from optimisation_bspline import fit_spline_to_segment
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import BSpline

    # 1. Initialisation de la structure de données globale
    # Ce dictionnaire contiendra toutes les pièces de ton hackathon
    dict_ctrl = {}

    print("1. Chargement de l'IA...")
    unet_model, device = load_trained_model("unet_puzzle_weights.pth")
    
    # 2. Définition de la pièce en cours de traitement
    image_path = "algo_tuteur/photo_test_5.jpg"
    
    # Astuce : Utiliser le nom du fichier sans l'extension comme ID de la pièce
    piece_id = os.path.splitext(os.path.basename(image_path))[0] 
    
    # On crée l'entrée pour cette pièce spécifique
    dict_ctrl[piece_id] = {}

    print(f"2. Traitement de la pièce : {piece_id}")
    mask, original, corners = predict_mask(unet_model, device, image_path)
    
    print("3. Découpage et Normalisation des 4 bords...")
    segments_normalises = extract_and_normalize_edges(mask, corners)

    print("4. Optimisation, Stockage et Affichage...")
    
    # Préparation de la grille graphique 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # La boucle tourne 4 fois (0: Haut, 1: Droite, 2: Bas, 3: Gauche)
    for cote_idx, segment in enumerate(segments_normalises): 
        print(f" -> Calcul du Bord {cote_idx}...")
        
        # Appel de l'algorithme de ton camarade (qui renvoie un array 9x2)
        ctrl_opt, knots = fit_spline_to_segment(segment)
        
        # ==========================================
        # STRUCTURATION DES DONNÉES
        # ==========================================
        dict_ctrl[piece_id][cote_idx] = {
            "ctrl": ctrl_opt
        }
        
        # ==========================================
        # RECONSTRUCTION GRAPHIQUE
        # ==========================================
        degree = 2
        t_plot = np.linspace(0, 1, 200)
        spline_x = BSpline(knots, ctrl_opt[:, 0], degree)
        spline_y = BSpline(knots, ctrl_opt[:, 1], degree)
        courbe_finale = np.column_stack((spline_x(t_plot), spline_y(t_plot)))
        
        ax = axes[cote_idx]
        ax.plot(segment[:, 0], segment[:, 1], '.', markersize=2, label='Contour IA', color='gray', alpha=0.5)
        ax.plot(courbe_finale[:, 0], courbe_finale[:, 1], '-', label='B-Spline', color='blue', linewidth=2)
        ax.plot(ctrl_opt[:, 0], ctrl_opt[:, 1], 'rx', label='9 Points de Contrôle', markersize=8, markeredgewidth=2)
        
        ax.set_title(f"Bord {cote_idx}")
        ax.axis('equal')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Affichage de vérification dans le terminal
    print("\n=== STRUCTURE DE DONNÉES GÉNÉRÉE ===")
    for cote in dict_ctrl[piece_id]:
        array_shape = dict_ctrl[piece_id][cote]["ctrl"].shape
        print(f"Pièce '{piece_id}', Côté {cote} : Array de dimensions {array_shape}")

    # Lancement de la fenêtre graphique
    plt.tight_layout()
    plt.show()