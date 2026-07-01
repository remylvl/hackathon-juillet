import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2

# Import de l'architecture et du générateur créés précédemment
# (Assure-toi que ces classes sont bien accessibles dans ton projet)
from train_unet import PuzzleUNet, PuzzleDataset 

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

def evaluate_model(weights_path="unet_puzzle_weights.pth", num_test_samples=500):
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
    test_dataset = PuzzleDataset(num_samples=num_test_samples, img_size=512)
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
            corners = extract_four_corners(pred_heatmap_show, min_distance_pixels=60)

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