import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2

# Import de l'architecture et du générateur créés précédemment
from train_unet import PuzzleUNet, PuzzleDataset 

def extract_four_corners(heatmap: np.ndarray, min_distance_pixels: int = 150) -> np.ndarray:
    """
    Extrait les coordonnées (x, y) des 4 coins à partir de la prédiction du réseau.
    Utilise une approche d'effacement pour garantir des coins distincts.
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
    # Binarisation stricte, peu importe le format d'entrée du masque
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255 if mask.max() <= 1.0 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return nn_corners
        
    contour = max(contours, key=cv2.contourArea).squeeze()
    if len(contour.shape) < 2:  # contour trop petit / dégénéré
        return nn_corners
        
    N = len(contour)
    refined_corners = []

    for nn_c in nn_corners:
        # Points du contour situés dans le rayon de recherche autour du coin prédit
        distances = np.linalg.norm(contour - nn_c, axis=1)
        pts_in_radius_indices = np.where(distances <= radius)[0]

        if len(pts_in_radius_indices) == 0:
            # rien à proximité : on garde la prédiction de l'IA telle quelle
            refined_corners.append(nn_c)
            continue

        max_curvature = -1
        best_corner = nn_c
        # step : on compare à des voisins un peu éloignés sur le contour
        # plutôt qu'aux pixels adjacents, pour un calcul d'angle plus stable
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

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """ Calcule l'Intersection over Union pour un batch d'images. """
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    true_bin = true_mask.astype(np.uint8)

    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()

    if union == 0:
        # aucun pixel positif ni dans la prédiction ni dans la vérité terrain :
        # on considère que c'est un accord parfait
        return 1.0 if intersection == 0 else 0.0
    else:
        return intersection / union

def evaluate_model(weights_path="reseau_de_neurones/unet_puzzle_weights.pth", num_test_samples=500):
    # Choix du meilleur device disponible (GPU NVIDIA, GPU Apple Silicon, ou CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Évaluation sur : {device}")

    # Chargement du modèle entraîné, verrouillé en mode évaluation
    # (eval() désactive le comportement "entraînement" des couches BatchNorm/Dropout)
    model = PuzzleUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Nouveau jeu de données généré à la volée, jamais vu pendant l'entraînement,
    # pour mesurer la vraie capacité de généralisation du modèle
    test_dataset = PuzzleDataset(num_samples=num_test_samples, img_size=512, rotation=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    total_iou = 0.0
    all_ious = []

    print(f"Test sur {num_test_samples} images inédites en cours...")

    # no_grad() : on ne fait que de l'inférence, pas d'entraînement, donc pas
    # besoin de calculer les gradients (plus rapide, moins de mémoire utilisée)
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)  # targets contient 2 canaux : masque + heatmap

            preds = model(inputs)

            # canal 0 = masque de segmentation (on ignore le canal 1, la heatmap,
            # pour ce calcul d'IoU)
            preds_masks_np = preds[:, 0, :, :].cpu().numpy()
            true_masks_np = targets[:, 0, :, :].cpu().numpy()

            for i in range(preds_masks_np.shape[0]):
                iou = calculate_iou(preds_masks_np[i], true_masks_np[i])
                all_ious.append(iou)
                total_iou += iou

    mean_iou = total_iou / num_test_samples
    print(f"\n=== RÉSULTATS DE L'ÉVALUATION ===")
    print(f"IoU Moyen : {mean_iou * 100:.2f}%")
    print(f"IoU Médian : {np.median(all_ious) * 100:.2f}%")
    print(f"Pire IoU : {np.min(all_ious) * 100:.2f}%")
    
    return test_dataset, model

def visualize_predictions(dataset, model, num_samples=3):
    """ Affiche l'Image, la Vérité Terrain, et le Masque Prédit avec les Coins superposés. """
    device = next(model.parameters()).device
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    plt.suptitle("Évaluation 512x512 : Masque & Coins", fontsize=14)

    # On tire quelques exemples au hasard dans le jeu de test pour inspection visuelle
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, target_tensor = dataset[idx]
            input_batch = img_tensor.unsqueeze(0).to(device)
            
            pred_tensor = model(input_batch)
            
            img_show = img_tensor.squeeze().cpu().numpy()
            true_mask_show = target_tensor[0].cpu().numpy()
            
            pred_mask_show = pred_tensor[0, 0].cpu().numpy()
            pred_heatmap_show = pred_tensor[0, 1].cpu().numpy()
            
            pred_binary = (pred_mask_show > 0.5).astype(np.float32)

            # rayon plus grand ici car l'image est en 512x512 (donc les
            # taches de la heatmap sont proportionnellement plus larges)
            corners = extract_four_corners(pred_heatmap_show, min_distance_pixels=150)

            axes[i, 0].imshow(img_show, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title("1. Entrée (512x512)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask_show, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title("2. Masque Parfait")
            axes[i, 1].axis('off')

            # masque prédit + coins détectés superposés en rouge
            axes[i, 2].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].scatter(corners[:, 0], corners[:, 1], c='red', s=60, marker='x', linewidths=2)
            axes[i, 2].set_title(f"3. Masque + Coins (IoU: {calculate_iou(pred_mask_show, true_mask_show):.2f})")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Étape 1 : Calcul des métriques statistiques sur 500 nouvelles images
    test_data, trained_model = evaluate_model(num_test_samples=500)
    
    # Étape 2 : Inspection visuelle pour vérifier la netteté des contours
    visualize_predictions(test_data, trained_model, num_samples=4)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2

# Import de l'architecture et du générateur créés précédemment
from train_unet import PuzzleUNet, PuzzleDataset 

def extract_four_corners(heatmap: np.ndarray, min_distance_pixels: int = 150) -> np.ndarray:
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
    # 1. Extraire le contour principal (le mask doit être en uint8, 0 ou 255)
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255 if mask.max() <= 1.0 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return nn_corners
        
    contour = max(contours, key=cv2.contourArea).squeeze()
    if len(contour.shape) < 2: # Sécurité si le contour est trop petit
        return nn_corners
        
    N = len(contour)
    refined_corners = []

    for nn_c in nn_corners:
        distances = np.linalg.norm(contour - nn_c, axis=1)
        pts_in_radius_indices = np.where(distances <= radius)[0]

        if len(pts_in_radius_indices) == 0:
            refined_corners.append(nn_c)
            continue

        max_curvature = -1
        best_corner = nn_c
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
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                if cos_angle > max_curvature:
                    max_curvature = cos_angle
                    best_corner = p

        refined_corners.append(best_corner)

    return np.array(refined_corners, dtype=np.int32)

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """ Calcule l'Intersection over Union pour un batch d'images. """
    # Binarisation de la prédiction
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    true_bin = true_mask.astype(np.uint8)

    # Calcul des intersections et unions matricielles
    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    else:
        return intersection / union

def evaluate_model(weights_path="reseau_de_neurones/unet_puzzle_weights.pth", num_test_samples=500):
        # Utilisation de la carte graphique si disponible, sinon processeur
    if torch.cuda.is_available():
        device = torch.device("cuda") # Pour Nvidia
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Pour ton MacBook Pro (Apple Silicon)
    else:
        device = torch.device("cpu")  # Mode dégradé
    print(f"Évaluation sur : {device}")

    # 1. Chargement du modèle verrouillé en mode évaluation
    model = PuzzleUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Création du Test Set 
    # CRITIQUE : On instancie une nouvelle graine (seed) aléatoire indirectement
    # pour s'assurer que les perturbations générées par 'perturb_params' sont inédites.
    test_dataset = PuzzleDataset(num_samples=num_test_samples, img_size=512, rotation=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    total_iou = 0.0
    all_ious = []

    print(f"Test sur {num_test_samples} images inédites en cours...")

    # 3. Inférence sans calcul de gradient
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device) # 'targets' contient maintenant 2 canaux !

            # Prédiction
            preds = model(inputs)

            # On isole le Canal 0 (La Segmentation) en passant sur le CPU
            preds_masks_np = preds[:, 0, :, :].cpu().numpy()
            true_masks_np = targets[:, 0, :, :].cpu().numpy()

            # Calcul de l'IoU uniquement sur les masques
            for i in range(preds_masks_np.shape[0]):
                iou = calculate_iou(preds_masks_np[i], true_masks_np[i])
                all_ious.append(iou)
                total_iou += iou

    mean_iou = total_iou / num_test_samples
    print(f"\n=== RÉSULTATS DE L'ÉVALUATION ===")
    print(f"IoU Moyen : {mean_iou * 100:.2f}%")
    print(f"IoU Médian : {np.median(all_ious) * 100:.2f}%")
    print(f"Pire IoU : {np.min(all_ious) * 100:.2f}%")
    
    return test_dataset, model

def visualize_predictions(dataset, model, num_samples=3):
    """ Affiche l'Image, la Vérité Terrain, et le Masque Prédit avec les Coins superposés. """
    device = next(model.parameters()).device
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    plt.suptitle("Évaluation 512x512 : Masque & Coins", fontsize=14)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, target_tensor = dataset[idx]
            input_batch = img_tensor.unsqueeze(0).to(device)
            
            pred_tensor = model(input_batch)
            
            # --- Extraction des matrices NumPy ---
            img_show = img_tensor.squeeze().cpu().numpy()
            true_mask_show = target_tensor[0].cpu().numpy()
            
            pred_mask_show = pred_tensor[0, 0].cpu().numpy()
            pred_heatmap_show = pred_tensor[0, 1].cpu().numpy()
            
            pred_binary = (pred_mask_show > 0.5).astype(np.float32)

            # --- Extraction des Coins (Ajusté pour 512x512) ---
            # On utilise un rayon de 60 pixels pour effacer les taches
            corners = extract_four_corners(pred_heatmap_show, min_distance_pixels=150)

            # --- Affichage ---
            axes[i, 0].imshow(img_show, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title("1. Entrée (512x512)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask_show, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title("2. Masque Parfait")
            axes[i, 1].axis('off')

            # 3. Masque Prédit + Superposition des Croix Rouges
            axes[i, 2].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].scatter(corners[:, 0], corners[:, 1], c='red', s=60, marker='x', linewidths=2)
            axes[i, 2].set_title(f"3. Masque + Coins (IoU: {calculate_iou(pred_mask_show, true_mask_show):.2f})")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Étape 1 : Calcul des métriques statistiques sur 500 nouvelles images
    test_data, trained_model = evaluate_model(num_test_samples=500)
    
    # Étape 2 : Inspection visuelle pour vérifier la netteté des contours
    visualize_predictions(test_data, trained_model, num_samples=4)