import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

def fit_spline_to_segment(Q, degree=2, n_ctrl=9):
    """
    Ajuste une B-spline avec paramétrisation par longueur d'arc (Chordal)
    et verrouillage strict des extrémités à (0,0) et (1,0).
    """
    N = len(Q)

    # ==========================================
    # CORRECTION CRITIQUE 1 : Paramétrisation Chordal
    # On calcule la distance réelle entre chaque pixel du contour IA.
    # ==========================================
    diffs = np.diff(Q, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    
    t = np.zeros(N)
    t[1:] = np.cumsum(dists) # Distance cumulée
    t /= t[-1]               # On normalise pour que t aille de 0.0 à 1.0

    # Création du vecteur de nœuds
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ))

    # ==========================================
    # CORRECTION CRITIQUE 2 : Verrouillage des coins
    # ==========================================
    def cost(ctrl_flat_reduced):
        # L'optimiseur ne manipule que les 7 points du milieu
        ctrl_interior = ctrl_flat_reduced.reshape((n_ctrl - 2, 2))
        
        # On force les points 0 et 8 aux extrémités géométriques
        ctrl = np.vstack(([0, 0], ctrl_interior, [1, 0]))

        spline_x = BSpline(knots, ctrl[:, 0], degree)
        spline_y = BSpline(knots, ctrl[:, 1], degree)
        
        # Évaluation aux vrais instants géométriques 't'
        C = np.column_stack((spline_x(t), spline_y(t)))
        
        return (C - Q).ravel()

    # ==========================================
    # CORRECTION CRITIQUE 3 : Initialisation intelligente
    # On place les points initiaux en se basant sur les distances réelles.
    # ==========================================
    target_t = np.linspace(0, 1, n_ctrl)
    indices_initiaux = [np.argmin(np.abs(t - val)) for val in target_t]
    
    init_ctrl_full = Q[indices_initiaux]
    # On retire les extrémités de l'initialisation car elles sont verrouillées
    init_ctrl_reduced = init_ctrl_full[1:-1] 

    # Optimisation (uniquement sur les 14 variables du milieu)
    result = least_squares(cost, init_ctrl_reduced.ravel(), method='lm')

    # Reconstruction de la matrice finale des 9 points
    ctrl_opt_interior = result.x.reshape((n_ctrl - 2, 2))
    ctrl_opt_final = np.vstack(([0, 0], ctrl_opt_interior, [1, 0]))
    
    return ctrl_opt_final, knots


# ============================================================
# EXÉCUTION (Simulation avec notre Pipeline)
# ============================================================
# ============================================================
# EXÉCUTION RÉELLE (Liaison IA -> Mathématiques)
# ============================================================
if __name__ == "__main__":
    # 1. Imports de tes propres scripts (Ajuste les noms de fichiers si besoin)
    from inference_hackathon import load_trained_model, predict_mask
    from geometry_utils import extract_and_normalize_edges

    print("1. Chargement de l'IA...")
    unet_model, device = load_trained_model("unet_puzzle_weights.pth")
    
    print("2. Prédiction sur la photo...")
    # Remplace par le nom de ta vraie image scannée/photographiée
    mask, original, corners = predict_mask(unet_model, device, "algo_tuteur/photo_test_5.jpg")
    
    print("3. Découpage et Normalisation des 4 bords...")
    # C'est cette fonction qui va te créer une liste de 4 vrais segments !
    segments_normalises = extract_and_normalize_edges(mask, corners)

    print("4. Optimisation des Moindres Carrés...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # La boucle va maintenant tourner 4 fois, une pour chaque bord de la pièce
    for i, segment in enumerate(segments_normalises): 
        print(f" -> Calcul du Bord {i+1}...")
        
        # Appel de l'algorithme de ton camarade
        ctrl_opt, knots = fit_spline_to_segment(segment)
        degree = 2
        
        # Reconstruction mathématique pour l'affichage
        t_plot = np.linspace(0, 1, 200)
        spline_x = BSpline(knots, ctrl_opt[:, 0], degree)
        spline_y = BSpline(knots, ctrl_opt[:, 1], degree)
        courbe_finale = np.column_stack((spline_x(t_plot), spline_y(t_plot)))
        
        # Affichage
        ax = axes[i]
        ax.plot(segment[:, 0], segment[:, 1], '.', markersize=2, label='Contour IA', color='gray', alpha=0.5)
        ax.plot(courbe_finale[:, 0], courbe_finale[:, 1], '-', label='B-Spline', color='blue', linewidth=2)
        ax.plot(ctrl_opt[:, 0], ctrl_opt[:, 1], 'rx', label='9 Points de Contrôle', markersize=8, markeredgewidth=2)
        
        ax.set_title(f"Bord {i+1}")
        ax.axis('equal')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()