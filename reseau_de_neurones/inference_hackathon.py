import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import de l'architecture du modèle
from train_unet import PuzzleUNet 

def load_trained_model(weights_path="unet_puzzle_weights.pth"):
    """ Charge le modèle en mémoire et le prépare pour la prédiction pure. """
    
    # Choix du meilleur device disponible (GPU NVIDIA, GPU Apple Silicon, ou CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Modèle chargé sur : {device}")

    model = PuzzleUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # eval() désactive le comportement "entraînement" des couches BatchNorm/Dropout,
    # indispensable pour avoir des prédictions stables en inférence
    model.eval()
    
    return model, device


def predict_mask(model, device, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erreur : Impossible de lire l'image '{image_path}'.")
        
    img = 255 - img  # inversion des couleurs : la pièce doit être claire sur fond sombre
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    output_matrix = output_tensor.squeeze().cpu().numpy()
    probability_mask = output_matrix[0]   # canal 0 : masque de segmentation de la pièce
    heatmap_corners = output_matrix[1]    # canal 1 : carte de chaleur des coins
    
    binary_mask = (probability_mask > 0.5).astype(np.uint8)
    final_cv2_mask = binary_mask * 255

    # ==========================================
    # STRATÉGIE IA : Extraction et Raffinement
    # ==========================================
    # 1. On extrait les 4 zones à partir de la carte de chaleur de l'IA
    corners_bruts = extract_four_corners(heatmap_corners, min_distance_pixels=150)
    
    # 2. On verrouille ces zones sur les vrais pixels pointus du contour
    corners = refine_corners_with_math(final_cv2_mask, corners_bruts)

    return final_cv2_mask, img_resized, corners, heatmap_corners



# EXÉCUTION RÉELLE (Liaison IA -> Mathématiques -> Données)
if __name__ == "__main__":
    from inference_hackathon import load_trained_model, predict_mask, refine_corners_with_math
    from geometry_utils import extract_and_normalize_edges
    from optimisation_bspline import fit_spline_to_segment
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import BSpline

    # Dictionnaire global : contiendra toutes les pièces traitées du hackathon
    dict_ctrl = {}

    print("1. Chargement de l'IA...")
    unet_model, device = load_trained_model("reseau_de_neurones/unet_puzzle_weights.pth")
    
    image_path = "algo_tuteur/photo_test_5.jpg"
    
    # Le nom du fichier (sans extension) sert d'identifiant de pièce
    piece_id = os.path.splitext(os.path.basename(image_path))[0] 
    dict_ctrl[piece_id] = {}

    print(f"2. Traitement de la pièce : {piece_id}")
    mask, original, corners, heatmap = predict_mask(unet_model, device, image_path)
    
    # Affichage de diagnostic : permet de vérifier que l'IA a bien repéré 4 coins nets
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='magma')
    plt.title("Diagnostic IA : Carte de Chaleur des Coins")
    plt.colorbar()
    
    print("3. Découpage et Normalisation des 4 bords...")
    segments_normalises = extract_and_normalize_edges(mask, corners)

    print("4. Optimisation, Stockage et Affichage...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # 4 côtés à traiter : 0 = Haut, 1 = Droite, 2 = Bas, 3 = Gauche
    for cote_idx, segment in enumerate(segments_normalises): 
        print(f" -> Calcul du Bord {cote_idx}...")
        
        # Ajustement B-spline (9 points de contrôle) sur le contour réel du côté
        ctrl_opt, knots = fit_spline_to_segment(segment)
        
        dict_ctrl[piece_id][cote_idx] = {
            "ctrl": ctrl_opt
        }
        
        # Reconstruction de la courbe lissée à partir des points de contrôle,
        # juste pour vérifier visuellement la qualité de l'ajustement
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

    print("\n=== STRUCTURE DE DONNÉES GÉNÉRÉE ===")
    for cote in dict_ctrl[piece_id]:
        array_shape = dict_ctrl[piece_id][cote]["ctrl"].shape
        print(f"Pièce '{piece_id}', Côté {cote} : Array de dimensions {array_shape}")

    plt.tight_layout()
    plt.show()


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

def extract_four_corners(heatmap: np.ndarray, min_distance_pixels: int = 60) -> np.ndarray:
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


def refine_corners_with_math(mask: np.ndarray, nn_corners: np.ndarray, radius: int = 40) -> np.ndarray:
    """
    Raffine les coins prédits par l'IA en cherchant le point de courbure maximale 
    sur le contour dans un rayon donné autour de chaque prédiction.
    """
    # 1. Extraire le contour principal
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return nn_corners
    contour = max(contours, key=cv2.contourArea).squeeze() # (N, 2)
    N = len(contour)

    refined_corners = []

    for nn_c in nn_corners:
        # Trouver les points du contour à l'intérieur du rayon
        distances = np.linalg.norm(contour - nn_c, axis=1)
        pts_in_radius_indices = np.where(distances <= radius)[0]

        if len(pts_in_radius_indices) == 0:
            # Aucun point du contour à proximité, on garde la prédiction IA
            refined_corners.append(nn_c)
            continue

        # Calculer la "courbure" pour chaque point dans le rayon
        max_curvature = -1
        best_corner = nn_c

        # On utilise un pas (step) pour regarder des voisins un peu distants sur le contour,
        # ce qui rend le calcul d'angle plus robuste aux petits défauts du masque.
        step = max(5, N // 100) 

        for idx in pts_in_radius_indices:
            # Indices des voisins
            idx_prev = (idx - step) % N
            idx_next = (idx + step) % N

            p = contour[idx]
            p_prev = contour[idx_prev]
            p_next = contour[idx_next]

            # Vecteurs
            v1 = p_prev - p
            v2 = p_next - p

            # Calcul du cosinus de l'angle entre v1 et v2
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                # Plus l'angle est aigu (proche de 90° ou moins), 
                # plus le cosinus est grand (ou proche de 0, donc on cherche la valeur la plus proche de 1)
                
                # Un angle aigu (coin) aura un cosinus positif (ex: cos(90)=0, cos(45)=0.7)
                # Un bord plat aura un cosinus proche de -1 (ex: cos(180)=-1)
                
                # On cherche la valeur maximale de cos_angle (l'angle le plus fermé)
                if cos_angle > max_curvature:
                    max_curvature = cos_angle
                    best_corner = p

        refined_corners.append(best_corner)

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

def recalculer_coins_geometriques(mask: np.ndarray) -> np.ndarray:
    """
    Utilise la morphologie mathématique pour effacer les bosses et boucher les creux.
    La pièce devient un carré pur, ce qui permet de trouver les vrais angles à 90°.
    """
    # 1. Binarisation stricte
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255 if mask.max() <= 1.0 else mask.astype(np.uint8)
    
    # 2. Astuce Morphologique : Transformer la pièce en carré
    # La taille du noyau s'adapte à la taille de l'image (128x128 -> noyau d'environ 17x17)
    taille_noyau = max(5, mask_uint8.shape[0] // 7)
    kernel = np.ones((taille_noyau, taille_noyau), np.uint8)
    
    # Étape A : Boucher les creux (Fermeture)
    mask_sans_creux = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Étape B : Limer les bosses (Ouverture)
    mask_carre = cv2.morphologyEx(mask_sans_creux, cv2.MORPH_OPEN, kernel)
    
    # 3. Trouver la boîte englobante de ce nouveau masque devenu carré
    contours_carre, _ = cv2.findContours(mask_carre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_carre:
        raise ValueError("Erreur lors de la transformation morphologique.")
        
    contour_carre = max(contours_carre, key=cv2.contourArea).squeeze()
    rect = cv2.minAreaRect(contour_carre)
    coins_carre = cv2.boxPoints(rect)
        
    # 4. Ramener ces 4 coins trouvés sur le VRAI contour de la pièce originale
    contours_vrais, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vrai_contour = max(contours_vrais, key=cv2.contourArea).squeeze()
    
    coins_finaux = []
    for pt in coins_carre:
        distances = np.linalg.norm(vrai_contour - pt, axis=1)
        closest_idx = np.argmin(distances)
        coins_finaux.append(vrai_contour[closest_idx])
        
    return np.array(coins_finaux, dtype=np.int32)

def predict_mask(model, device, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erreur : Impossible de lire l'image '{image_path}'.")
        
    img = 255 - img # N'oublie pas l'inversion des couleurs si besoin !
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    output_matrix = output_tensor.squeeze().cpu().numpy()
    probability_mask = output_matrix[0] 
    heatmap_corners = output_matrix[1]  # LA VOICI
    
    binary_mask = (probability_mask > 0.5).astype(np.uint8)
    final_cv2_mask = binary_mask * 255

    corners = recalculer_coins_geometriques(final_cv2_mask)

    
    # CRITIQUE : on retourne la heatmap en plus
    return final_cv2_mask, img_resized, corners, heatmap_corners


# EXÉCUTION RÉELLE (Liaison IA -> Mathématiques -> Données)
if __name__ == "__main__":
    from inference_hackathon import load_trained_model, predict_mask, refine_corners_with_math
    from geometry_utils import extract_and_normalize_edges, extract_four_corners, refine_corners_with_math
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
    # 1. On récupère les 4 variables
    mask, original, corners, heatmap = predict_mask(unet_model, device, image_path)
    
    # 2. AFFICHAGE DE DIAGNOSTIC CRITIQUE
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='magma')
    plt.title("Diagnostic IA : Carte de Chaleur des Coins")
    plt.colorbar()
    
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