import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

def fit_spline_to_segment(Q, degree=2, n_ctrl=12):
    """
    Ajuste une B-spline avec paramétrisation par longueur d'arc 
    et verrouille strictement des extrémités à (0,0) et (1,0).
    """
    N = len(Q)

    # Paramétrisation : on associe à chaque point Q un paramètre t proportionnel à la distance parcourue le long du contour (et pas juste à son indice), pour que la spline suive fidèlement les zones où les points sont plus ou moins espacés
    diffs = np.diff(Q, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    t = np.zeros(N)
    t[1:] = np.cumsum(dists)  # distance cumulée depuis le premier point
    t /= t[-1]                # normalisation pour que t aille de 0 à 1

    # Vecteur de nœuds pour une B-spline de degré `degree` avec `n_ctrl` points de contrôle
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ))

    def cost(ctrl_flat_reduced):
        # Les points de contrôle 0 et n_ctrl-1 sont fixés aux coins du côté (0,0) et (1,0) : on ne touche donc qu'aux points du milieu
        ctrl_interior = ctrl_flat_reduced.reshape((n_ctrl - 2, 2))
        ctrl = np.vstack(([0, 0], ctrl_interior, [1, 0]))

        spline_x = BSpline(knots, ctrl[:, 0], degree)
        spline_y = BSpline(knots, ctrl[:, 1], degree)
        
        # On évalue la spline aux paramètres t pour pouvoir comparer directement les deux courbes point à point
        C = np.column_stack((spline_x(t), spline_y(t)))
        
        return (C - Q).ravel()  # résidus : ce que least_squares va minimiser

    # Initialisation : on choisit, pour chaque point, le point de Q dont le paramètre t est le plus proche. Ça donne un point de départ déjà proche de la solution, donc une convergence plus rapide
    target_t = np.linspace(0, 1, n_ctrl)
    indices_initiaux = [np.argmin(np.abs(t - val)) for val in target_t]
    
    init_ctrl_full = Q[indices_initiaux]
    init_ctrl_reduced = init_ctrl_full[1:-1]  # on retire les extrémités, déjà fixées

    # Optimisation par moindres carrés, uniquement sur les points de contrôle du milieu
    result = least_squares(cost, init_ctrl_reduced.ravel(), method='lm')

    # On rajoute les deux extrémités fixes pour reconstituer les 9 points finaux
    ctrl_opt_interior = result.x.reshape((n_ctrl - 2, 2))
    ctrl_opt_final = np.vstack(([0, 0], ctrl_opt_interior, [1, 0]))
   
    return ctrl_opt_final, knots