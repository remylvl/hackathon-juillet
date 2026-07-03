import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import de l'architecture du modèle
from train_unet import PuzzleUNet 
from geometry_utils import extract_and_normalize_edges, refine_corners_with_math, extract_four_corners
from optimisation_bspline import fit_spline_to_segment
from scipy.interpolate import BSpline
import os

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
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    output_matrix = output_tensor.squeeze().cpu().numpy()
    probability_mask = output_matrix[0]   # canal 0 : masque de segmentation de la pièce
    heatmap_corners = output_matrix[1]    # canal 1 : carte de chaleur des coins
    
    binary_mask = (probability_mask > 0.5).astype(np.uint8)
    final_cv2_mask = binary_mask * 255

    # Extraction et Raffinement
    # 1. On extrait les 4 zones à partir de la carte de chaleur de l'IA
    corners_bruts = extract_four_corners(heatmap_corners, min_distance_pixels=100)
    
    # 2. On verrouille ces zones sur les vrais pixels pointus du contour
    corners = refine_corners_with_math(final_cv2_mask, corners_bruts)

    return final_cv2_mask, img_resized, corners, heatmap_corners



# Éxécution finale
if __name__ == "__main__":
    from reseau_de_neurones.main_reseau_de_neurones import load_trained_model, predict_mask

    # Dictionnaire global : contiendra toutes les pièces traitées 
    dict_ctrl = {}

    print("1. Chargement du réseau de neurones")
    unet_model, device = load_trained_model("reseau_de_neurones/unet_puzzle_weights.pth")
    
    image_path = "reseau_de_neurones/data_photo/photo_test_5.jpg"
    
    # Le nom du fichier sert d'identifiant de pièce
    piece_id = os.path.splitext(os.path.basename(image_path))[0] 
    dict_ctrl[piece_id] = {}

    print(f"2. Traitement de la pièce : {piece_id}")
    mask, original, corners, heatmap = predict_mask(unet_model, device, image_path)
    
    # Affichage de diagnostic : permet de vérifier que le réseau a bien repéré 4 coins nets
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='magma')
    plt.title("Diagnostic IA : Carte de Chaleur des Coins")
    plt.colorbar()
    
    print("3. Découpage et Normalisation des 4 bords")
    segments_normalises = extract_and_normalize_edges(mask, corners)

    print("4. Optimisation, Stockage et Affichage...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # 4 côtés à traiter : 0 = Haut, 1 = Droite, 2 = Bas, 3 = Gauche
    for cote_idx, segment in enumerate(segments_normalises): 
        print(f" -> Calcul du Bord {cote_idx}...")
        
        # Ajustement B-spline (<=> 9 points de contrôle) sur le contour réel du côté
        ctrl_opt, knots = fit_spline_to_segment(segment)
        
        dict_ctrl[piece_id][cote_idx] = {
            "ctrl": ctrl_opt
        }
        
        # Reconstruction de la courbe lissée à partir des points de contrôle
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

    print("Structure de données générées")
    for cote in dict_ctrl[piece_id]:
        array_shape = dict_ctrl[piece_id][cote]["ctrl"].shape
        print(f"Pièce '{piece_id}', Côté {cote} : Array de dimensions {array_shape}")

    plt.tight_layout()
    plt.show()