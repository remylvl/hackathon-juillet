import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import time
import random

# Import de ta logique métier géométrique
from jigsaw5 import perturb_params, default_params, make_piece_from_params #[cite: 1]
# ============================================================
# 1) L'Architecture du Modèle (U-Net)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def generate_gaussian_heatmap(img_size: int, center_x: float, center_y: float, sigma: float = 3.0) -> np.ndarray:
    """
    Génère une matrice 2D contenant une tache gaussienne centrée sur (center_x, center_y).
    'sigma' contrôle l'étalement (le rayon) de la tache.
    """
    x = np.arange(0, img_size, 1, np.float32)
    y = np.arange(0, img_size, 1, np.float32)
    xx, yy = np.meshgrid(x, y)
    
    # Équation de la distribution Gaussienne 2D
    heatmap = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
    
    return heatmap


class PuzzleUNet(nn.Module):
    def __init__(self):
        super(PuzzleUNet, self).__init__()
        self.down1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)
        self.up_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up2 = DoubleConv(64, 32)
        self.up_trans3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up3 = DoubleConv(32, 16)

        self.out_conv = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        bn = self.bottleneck(p3)

        up_val1 = self.up_trans1(bn)
        d1 = self.up1(torch.cat([up_val1, x3], dim=1))
        up_val2 = self.up_trans2(d1)
        d2 = self.up2(torch.cat([up_val2, x2], dim=1))
        up_val3 = self.up_trans3(d2)
        d3 = self.up3(torch.cat([up_val3, x1], dim=1))

        out = self.out_conv(d3)
        return torch.sigmoid(out)

# ============================================================
# 2) Le Générateur de Données (PyTorch Dataset)
# ============================================================
class PuzzleDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=512, rotation=True, max_rotation=5): 
        self.rotation = rotation
        self.num_samples = num_samples
        self.img_size = img_size
        self.max_rotation = max_rotation
        self.rng = np.random.default_rng()
        self.coord_min = -0.3
        self.coord_max = 1.3

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
            # 1. Génération de la géométrie de base
            top_p = perturb_params(default_params(), self.rng, noise=1.5)
            right_p = perturb_params(default_params(), self.rng, noise=1.5)
            bottom_p = perturb_params(default_params(), self.rng, noise=1.5)
            left_p = perturb_params(default_params(), self.rng, noise=1.5)

            piece_data = make_piece_from_params(top_p, right_p, bottom_p, left_p, n=150)
            piece_curve = piece_data["piece"]

            # 2. Conversion du masque de segmentation (Canal 0)
            norm_coords = (piece_curve - self.coord_min) / (self.coord_max - self.coord_min)
            pixel_coords = np.round(norm_coords * (self.img_size - 1)).astype(np.int32)
            
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            cv2.fillPoly(mask, [pixel_coords], color=1.0)
            
            # 3. Création de la Carte de Chaleur des Coins (Canal 1)
            # Les coins théoriques définis dans ton jigsaw5.py
            math_corners = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
            
            # On projette ces coins dans le repère de l'image (pixels)
            norm_corners = (math_corners - self.coord_min) / (self.coord_max - self.coord_min)
            pixel_corners = norm_corners * (self.img_size - 1)
            
            heatmap_total = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            for cx, cy in pixel_corners:
                # On additionne les 4 taches gaussiennes (une pour chaque coin)
                heatmap_total += generate_gaussian_heatmap(self.img_size, cx, cy, sigma=2.5)
                
            # On s'assure que les valeurs ne dépassent pas 1.0 au cas où deux taches se chevauchent
            heatmap_total = np.clip(heatmap_total, 0.0, 1.0)

            # 4. Création de l'image d'entrée bruitée (X)
            bg_color = self.rng.uniform(0.1, 0.4)
            piece_color = self.rng.uniform(0.6, 0.9)
            noisy_img = np.where(mask == 1.0, piece_color, bg_color).astype(np.float32)
            noise = self.rng.normal(0, 0.05, (self.img_size, self.img_size)).astype(np.float32)
            noisy_img = np.clip(noisy_img + noise, 0.0, 1.0)
            
            if self.rng.random() > 0.5:
                noisy_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)
            
            # On choisit un angle de rotation totalement aléatoire entre -180° et +180°
            centre = (self.img_size // 2, self.img_size // 2)
            
            # On utilise uniquement ta nouvelle logique 'max_rotation'
            if hasattr(self, 'max_rotation') and self.max_rotation > 0:
                angle = self.rng.uniform(-self.max_rotation, self.max_rotation)
            else:
                angle = 0.0
    
            # Matrice de rotation
            zoom_aleatoire = self.rng.uniform(0.6, 0.85)

            M = cv2.getRotationMatrix2D(centre, angle, scale=zoom_aleatoire)
            
            # On fait pivoter les 3 matrices EXACTEMENT du même angle
            # 1. L'image bruitée (le fond est rempli avec la couleur de fond bg_color)
            noisy_img = cv2.warpAffine(noisy_img, M, (self.img_size, self.img_size), 
                                    flags=cv2.INTER_LINEAR, borderValue=float(bg_color))
            
            # 2. Le masque (Interpolation 'Nearest' obligatoire pour garder du binaire pur)
            mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size), 
                                flags=cv2.INTER_NEAREST, borderValue=0.0)
            
            # 3. La Heatmap des coins (Les taches lumineuses tournent avec la pièce)
            heatmap_total = cv2.warpAffine(heatmap_total, M, (self.img_size, self.img_size), 
                                        flags=cv2.INTER_LINEAR, borderValue=0.0)
            # ==========================================

            # 5. Formatage final des Tenseurs (Ton code existant)
            X_tensor = torch.from_numpy(np.expand_dims(noisy_img, axis=0))
            Y_stacked = np.stack([mask, heatmap_total], axis=0)
            Y_tensor = torch.from_numpy(Y_stacked)

            return X_tensor, Y_tensor


    
# ============================================================
# 3) La Boucle d'Entraînement
# ============================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_weight=10.0):
        super(MultiTaskLoss, self).__init__()
        self.bce = nn.BCELoss()      # Pour le canal 0 (Segmentation Binaire)
        self.mse = nn.MSELoss()      # Pour le canal 1 (Carte de chaleur continue)
        self.lambda_weight = lambda_weight # Équilibreur de gradient

    def forward(self, predictions, targets):
        # predictions[:, 0, :, :] -> Récupère toutes les images du batch, canal 0
        
        # Calcul de l'erreur sur le Masque (Canal 0)
        loss_mask = self.bce(predictions[:, 0:1, :, :], targets[:, 0:1, :, :])
        
        # Calcul de l'erreur sur les Coins (Canal 1)
        loss_corners = self.mse(predictions[:, 1:2, :, :], targets[:, 1:2, :, :])
        
        # Addition pondérée
        total_loss = loss_mask + (self.lambda_weight * loss_corners)
        
        return total_loss

def train_model():
    # Paramètres d'entraînement
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    # Utilisation de la carte graphique si disponible, sinon processeur
    if torch.cuda.is_available():
        device = torch.device("cuda") # Pour Nvidia
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Pour ton MacBook Pro (Apple Silicon)
    else:
        device = torch.device("cpu")  # Mode dégradé
    print(f"Début de l'entraînement sur : {device}")

    # Initialisation du modèle, de la perte (BCELoss) et de l'optimiseur (Adam)
    model = PuzzleUNet().to(device)
    criterion = MultiTaskLoss(lambda_weight=40.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Création du Dataset (10 000 images générées à la volée) et du DataLoader
    dataset = PuzzleDataset(num_samples=10000, img_size=512, max_rotation = 5)
    dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=8,        # Nombre de cœurs CPU dédiés à la génération des images
    pin_memory=False,      # Accélère le transfert des données vers la carte graphique
    prefetch_factor=2     # Prépare 2 batchs d'avance par cœur CPU
)

    model.train() # Passe le modèle en mode entraînement
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        running_loss = 0.0

        for batch_idx, (inputs, masks) in enumerate(dataloader):
            # Envoi des données sur le GPU/CPU
            inputs, masks = inputs.to(device), masks.to(device)

            # 1. Remise à zéro des gradients
            optimizer.zero_grad()

            # 2. Passe avant (Forward) : prédiction du modèle
            predictions = model(inputs)

            # 3. Calcul de l'erreur (Loss)
            loss = criterion(predictions, masks)

            # 4. Passe arrière (Backward) : calcul des gradients
            loss.backward()

            # 5. Mise à jour des poids du modèle
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Erreur BCE : {epoch_loss:.4f} | Temps : {elapsed:.2f}s")

    # Sauvegarde du modèle entraîné
    torch.save(model.state_dict(), "unet_puzzle_weights.pth")
    print("Entraînement terminé. Poids sauvegardés dans 'unet_puzzle_weights.pth'")

if __name__ == "__main__":
    train_model()