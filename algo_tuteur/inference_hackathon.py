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
# ============================================================
# CLASSIFICATION ET LOGIQUE DE TRANSITION
# ============================================================

def classifier_cote(ctrl_points, seuil_plat=0.08):
    """
    Analyse les 9 points de contrôle d'une B-spline normalisée pour déterminer 
    la forme du côté du puzzle (0: Plat, 1: Bosse, 2: Creux).
    """
    y_coords = ctrl_points[:, 1]
    y_max = np.max(y_coords)
    y_min = np.min(y_coords)
    
    # Test du côté Plat
    if abs(y_max) < seuil_plat and abs(y_min) < seuil_plat:
        return 0
        
    # Différenciation Bosse / Creux
    if abs(y_max) > abs(y_min):
        return 1  # Éloignement vers l'extérieur (Bosse)
    else:
        return 2  # Enfoncement vers l'intérieur (Creux)

# À REMPLACER PAR LA VRAIE FONCTION DE TON CAMARADE
def fit_spline_to_edge_MOCK(segment):
    """ 
    Simulation temporaire qui génère 9 points de contrôle (9, 2) 
    à partir des points du segment pour éviter que le code ne plante.
    """
    # Exemple de courbe factice : 9 points répartis uniformément de X=0 à X=1
    x_fictif = np.linspace(0, 1, 9)
    # On génère un Y sinusoïdal simulé pour l'exemple
    y_fictif = 0.2 * np.sin(x_fictif * np.pi) 
    return np.stack((x_fictif, y_fictif), axis=-1)

# ============================================================
# EXÉCUTION PRINCIPALE
# ============================================================
# ============================================================
# EXÉCUTION PRINCIPALE
# ============================================================
if __name__ == "__main__":
    print("Initialisation du pipeline de segmentation...")
    unet_model, compute_device = load_trained_model("unet_puzzle_weights.pth")
    
    # Remplace par le chemin réel de ta photo de test
    image_test = "algo_tuteur/photo_test_5.jpg" 
    
    try:
        mask, original, corners_extrait = predict_mask(unet_model, compute_device, image_test)
        
        segments_normalises = extract_and_normalize_edges(mask, corners_extrait)
        
        
        for i, segment in enumerate(segments_normalises):
            print(f"Bord {i+1} : {len(segment)} points. Début={segment[0]}, Fin={segment[-1]}")
            # Appel de la fonction des moindres carrés de ton camarade ici :
            # resultat = fit_spline_to_edge(segment)
        print(f"Coordonnées des 4 coins trouvés (X, Y) : \n{corners_extrait}")
        
        # --- Affichage Sécurisé et Augmenté ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Image Originale + Superposition des Coins
        axes[0].imshow(original, cmap='gray')
        # On ajoute des croix rouges sur les coordonnées X (colonne 0) et Y (colonne 1)
        axes[0].scatter(corners_extrait[:, 0], corners_extrait[:, 1], c='red', s=80, marker='x', linewidths=2)
        axes[0].set_title("1. Photo Originale (256x256) + Coins Détectés")
        axes[0].axis('off')
        
        # 2. Masque Binaire
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("2. Masque Extrait par U-Net")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erreur d'exécution : {e}")