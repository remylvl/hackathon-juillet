import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import time
import random

# Import de la logique métier géométrique (code de Damien Corral)
from jigsaw5 import perturb_params, default_params, make_piece_from_params #[cite: 1]


# 1) L'Architecture du modèle : U-Net

class DoubleConv(nn.Module):
    """ Bloc de base du U-Net : deux convolutions 3x3 successives. """
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
    'sigma' contrôle l'étalement (le rayon) de la tache, pour créer du flou sur nos entrées.
    """
    x = np.arange(0, img_size, 1, np.float32)
    y = np.arange(0, img_size, 1, np.float32)
    xx, yy = np.meshgrid(x, y)
    
    heatmap = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
    
    return heatmap


class PuzzleUNet(nn.Module):
    """
    U-Net : le réseau a la forme d'un "U".

    Descente (gauche) : à chaque étage, l'image est réduite de moitié en
    résolution, mais avec plus de canaux, pour capter des infos de plus en
    plus abstraites (contours -> formes -> "ceci est une pièce de puzzle").

    Montée (droite) : on regrandit l'image étage par étage pour revenir à
    la résolution de départ. Pour ne pas perdre les détails fins écrasés
    pendant la descente, chaque étage de la montée recolle les
    features du même étage de la descente : c'est la "skip connection".

    Sortie : 2 canaux, un pour le masque de segmentation, un pour la heatmap des coins.
    """
    def __init__(self):
        super(PuzzleUNet, self).__init__()
        # Descente  : chaque étage réduit la résolution par 2 et augmente le nombre de canaux
        self.down1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        # Montée : on ré-augmente la résolution en concaténant à chaque étage les features correspondantes de la descente
        self.up_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)
        self.up_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up2 = DoubleConv(64, 32)
        self.up_trans3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up3 = DoubleConv(32, 16)

        # Conv 1x1 finale : ramène à 2 canaux de sortie (masque + heatmap)
        self.out_conv = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        # Descente, en gardant x1/x2/x3 pour les skip connections
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        bn = self.bottleneck(p3)

        # Montée : à chaque étage, on concatène avec la sortie du même niveau de la descente pour récupérer le détail spatial perdu
        up_val1 = self.up_trans1(bn)
        d1 = self.up1(torch.cat([up_val1, x3], dim=1))
        up_val2 = self.up_trans2(d1)
        d2 = self.up2(torch.cat([up_val2, x2], dim=1))
        up_val3 = self.up_trans3(d2)
        d3 = self.up3(torch.cat([up_val3, x1], dim=1))

        out = self.out_conv(d3)
        # les deux canaux de sortie (masque + heatmap) sont bornés entre 0 et 1
        return torch.sigmoid(out)


# 2) Le générateur de données 

class PuzzleDataset(Dataset):
    """
    Génère des images de pièces de puzzle synthétiques à la volée (avec bruit,
    rotation, zoom aléatoire), accompagnées de leur masque de segmentation et
    de leur heatmap de coins : c'est cette paire (image, vérité terrain) que
    le U-Net apprend à reproduire.
    """
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
            # 1. Génère une géométrie de pièce aléatoire (4 côtés indépendants, chacun perturbé aléatoirement autour d'une forme par défaut)
            top_p = perturb_params(default_params(), self.rng, noise=1.5)
            right_p = perturb_params(default_params(), self.rng, noise=1.5)
            bottom_p = perturb_params(default_params(), self.rng, noise=1.5)
            left_p = perturb_params(default_params(), self.rng, noise=1.5)

            piece_data = make_piece_from_params(top_p, right_p, bottom_p, left_p, n=150)
            piece_curve = piece_data["piece"]

            # 2. Masque de segmentation (canal 0) : on projette le contour mathématique de la pièce vers des coordonnées pixels, puis on remplit l'intérieur du polygone
            norm_coords = (piece_curve - self.coord_min) / (self.coord_max - self.coord_min)
            pixel_coords = np.round(norm_coords * (self.img_size - 1)).astype(np.int32)
            
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            cv2.fillPoly(mask, [pixel_coords], color=1.0)
            
            # 3. Heatmap des coins (canal 1) : les 4 coins théoriques de la pièce (dans le repère mathématique) sont projetés en pixels, puis on dépose une tache gaussienne sur chacun
            math_corners = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
            
            norm_corners = (math_corners - self.coord_min) / (self.coord_max - self.coord_min)
            pixel_corners = norm_corners * (self.img_size - 1)
            
            heatmap_total = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            for cx, cy in pixel_corners:
                heatmap_total += generate_gaussian_heatmap(self.img_size, cx, cy, sigma=2.5)
                
            # évite de dépasser 1.0 si deux taches gaussiennes se chevauchent
            heatmap_total = np.clip(heatmap_total, 0.0, 1.0)

            # 4. Image d'entrée bruitée (X) : on colore la pièce et le fond avec des teintes aléatoires, puis on ajoute du bruit gaussien pour simuler une vraie photo 
            bg_color = self.rng.uniform(0.1, 0.4)
            piece_color = self.rng.uniform(0.6, 0.9)
            noisy_img = np.where(mask == 1.0, piece_color, bg_color).astype(np.float32)
            noise = self.rng.normal(0, 0.05, (self.img_size, self.img_size)).astype(np.float32)
            noisy_img = np.clip(noisy_img + noise, 0.0, 1.0)
            
            if self.rng.random() > 0.5:
                noisy_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)
            
            centre = (self.img_size // 2, self.img_size // 2)
            
            # rotation légère aléatoire (utile pour rendre le modèle robuste à un petit désalignement de la pièce sur la photo réelle)
            if hasattr(self, 'max_rotation') and self.max_rotation > 0:
                angle = self.rng.uniform(-self.max_rotation, self.max_rotation)
            else:
                angle = 0.0
    
            zoom_aleatoire = self.rng.uniform(0.6, 0.85)

            M = cv2.getRotationMatrix2D(centre, angle, scale=zoom_aleatoire)
            
            # On applique EXACTEMENT la même transformation aux 3 matrices, pour qu'image, masque et heatmap restent alignés
            noisy_img = cv2.warpAffine(noisy_img, M, (self.img_size, self.img_size), 
                                    flags=cv2.INTER_LINEAR, borderValue=float(bg_color))
            
            # interpolation "nearest" pour le masque : on veut rester en valeurs binaires pures (0 ou 1), pas de flou
            mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size), 
                                flags=cv2.INTER_NEAREST, borderValue=0.0)
            
            heatmap_total = cv2.warpAffine(heatmap_total, M, (self.img_size, self.img_size), 
                                        flags=cv2.INTER_LINEAR, borderValue=0.0)

            # 5. Mise en forme finale en tenseurs PyTorch
            X_tensor = torch.from_numpy(np.expand_dims(noisy_img, axis=0))
            Y_stacked = np.stack([mask, heatmap_total], axis=0)
            Y_tensor = torch.from_numpy(Y_stacked)

            return X_tensor, Y_tensor


    

# 3) La boucle d'entraînement


class MultiTaskLoss(nn.Module):
    """
    Combine deux erreurs différentes puisque le modèle a deux objectifs :
    - BCE pour le masque (une tâche de classification binaire pixel par pixel)
    - MSE pour la heatmap (une tâche de régression continue)
    lambda_weight sert à équilibrer les deux, car sans ça la loss du masque
    dominerait largement l'entraînement.
    """
    def __init__(self, lambda_weight=10.0):
        super(MultiTaskLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.lambda_weight = lambda_weight

    def forward(self, predictions, targets):
        loss_mask = self.bce(predictions[:, 0:1, :, :], targets[:, 0:1, :, :])
        loss_corners = self.mse(predictions[:, 1:2, :, :], targets[:, 1:2, :, :])
        
        total_loss = loss_mask + (self.lambda_weight * loss_corners)
        
        return total_loss

def train_model():
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    # Choix du meilleur device disponible (GPU NVIDIA, GPU Apple Silicon, ou CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Début de l'entraînement sur : {device}")

    model = PuzzleUNet().to(device)
    criterion = MultiTaskLoss(lambda_weight=40.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dataset : 10 000 pièces différentes, aucune n'est stockée sur le disque, tout est recalculé à chaque appel de __getitem__
    dataset = PuzzleDataset(num_samples=10000, img_size=512, max_rotation = 5)
    dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=8,        # nombre de coeurs CPU dédiés à la génération des images
    pin_memory=False,
    prefetch_factor=2     # prépare des batchs à l'avance pendant que le GPU travaille
)

    model.train()  # active le comportement "entraînement" 
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        running_loss = 0.0

        for batch_idx, (inputs, masks) in enumerate(dataloader):
            inputs, masks = inputs.to(device), masks.to(device)

            # cycle standard d'entraînement PyTorch :
            optimizer.zero_grad()          # 1. reset des gradients
            predictions = model(inputs)    # 2. forward
            loss = criterion(predictions, masks)  # 3. calcul de l'erreur
            loss.backward()                # 4. backward (calcul des gradients)
            optimizer.step()               # 5. mise à jour des poids

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Erreur BCE : {epoch_loss:.4f} | Temps : {elapsed:.2f}s")

    torch.save(model.state_dict(), "unet_puzzle_weights.pth")
    print("Entraînement terminé. Poids sauvegardés dans 'unet_puzzle_weights.pth'")

if __name__ == "__main__":
    train_model()
