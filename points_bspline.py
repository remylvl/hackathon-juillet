import numpy as np
from scipy.interpolate import BSpline, splev
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import sys
import os

# ----------------------------------------------------------------------
# Import du module de reconnaissance (segments pivotés)
# ----------------------------------------------------------------------

parent = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(parent)

from elodie.reconnaissance_piece_vf import analyser_piece, normaliser_segment


# ----------------------------------------------------------------------
# 1. Ré-échantillonnage uniforme en longueur d’arc
# ----------------------------------------------------------------------

def echantillonner_segment(segment, n=200):
    """Ré-échantillonne un segment uniformément en longueur d’arc."""
    d = np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1))
    u = np.concatenate([[0], np.cumsum(d)])
    u = u / u[-1]

    from scipy.interpolate import interp1d
    fx = interp1d(u, segment[:, 0])
    fy = interp1d(u, segment[:, 1])

    u_new = np.linspace(0, 1, n)
    return np.column_stack([fx(u_new), fy(u_new)])


# ----------------------------------------------------------------------
# 2. Optimisation des 9 points de contrôle
# ----------------------------------------------------------------------

def optimiser_segment(segment, n_ctrl=15, degree=2):
    """Optimise 9 points de contrôle pour approximer un segment pivoté."""
    
    # Segment déjà pivoté par reconnaissance_piece_vf
    Q = echantillonner_segment(segment, n=200)
    N = len(Q)
    t = np.linspace(0, 1, N)

    # Noeuds uniformes
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ))

    # Initialisation des points de contrôle
    init_ctrl = Q[np.linspace(0, N-1, n_ctrl).astype(int)]

    # Fonction coût
    def cost(ctrl_flat):
        ctrl = ctrl_flat.reshape((n_ctrl, 2))
        spline_x = BSpline(knots, ctrl[:, 0], degree)
        spline_y = BSpline(knots, ctrl[:, 1], degree)
        C = np.vstack((spline_x(t), spline_y(t))).T
        return (C - Q).ravel()

    result = least_squares(cost, init_ctrl.ravel())
    return result.x.reshape((n_ctrl, 2)), knots, t


# ----------------------------------------------------------------------
# 3. Pipeline d’optimisation pour une pièce
# ----------------------------------------------------------------------

def optimiser_piece(fichier="./resources/piece4.jpeg", segment_id=3):
    """Optimise un segment d’une pièce analysée."""
    
    data = analyser_piece(fichier, afficher=False)
    segments = data["segments"]  # segments pivotés coin-à-coin

    segment = segments[segment_id]
    ctrl_opt, knots, t = optimiser_segment(segment)

    # Affichage
    plt.figure()
    plt.plot(segment[:, 0], segment[:, 1], 'o', label='Segment pivoté')
    
    spline_x = BSpline(knots, ctrl_opt[:, 0], 3)
    spline_y = BSpline(knots, ctrl_opt[:, 1], 3)
    C = np.vstack((spline_x(t), spline_y(t))).T

    plt.plot(C[:, 0], C[:, 1], '-', label='Spline optimisée')
    plt.plot(ctrl_opt[:, 0], ctrl_opt[:, 1], 'x', label='Points de contrôle optimisés')

    plt.legend()
    plt.axis("equal")
    plt.title(f"Spline optimisée – Segment {segment_id}")
    plt.show()

    return ctrl_opt


# ----------------------------------------------------------------------
# Exécution directe
# ----------------------------------------------------------------------

if __name__ == "__main__":
    optimiser_piece()
