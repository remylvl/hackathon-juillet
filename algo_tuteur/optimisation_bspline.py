import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

def fit_spline_to_segment(Q, degree=3, n_ctrl=9):
    """
    Ajuste une courbe B-spline sur un segment de points Q.
    Q : numpy array de shape (N, 2) contenant les pixels normalisés du bord.
    """
    N = len(Q)
    
    # Paramétrisation temporelle des points du contour (de 0 à 1)
    t = np.linspace(0, 1, N)

    # Création du vecteur de nœuds uniforme (knots)
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ))

    def cost(ctrl_flat):
        """ Fonction de perte (résidus) pour les moindres carrés """
        ctrl = ctrl_flat.reshape((n_ctrl, 2))

        # Construction des splines X et Y
        spline_x = BSpline(knots, ctrl[:, 0], degree)
        spline_y = BSpline(knots, ctrl[:, 1], degree)

        # Évaluation de la spline aux temps t
        C = np.column_stack((spline_x(t), spline_y(t)))

        # Retourne le vecteur des distances (erreurs) aplati
        return (C - Q).ravel()

    # Initialisation intelligente : on prend 9 points répartis uniformément sur le contour OpenCV
    indices_initiaux = np.linspace(0, N - 1, n_ctrl).astype(int)
    init_ctrl = Q[indices_initiaux]

    # Optimisation Levenberg-Marquardt
    result = least_squares(cost, init_ctrl.ravel())

    # Formatage du résultat
    ctrl_opt = result.x.reshape((n_ctrl, 2))
    
    return ctrl_opt, knots


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
    mask, original, corners = predict_mask(unet_model, device, "algo_tuteur/photo_test_1.jpg")
    
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
        degree = 3
        
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