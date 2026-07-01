import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
import sys
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splev

# Ajouter le dossier parent au path
parent = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(parent)

# Maintenant tu peux importer le module frère
from elodie.reconnaissance_piece2 import splines


# Tes points du contour
t = np.linspace(0, 1, 2000)
x, y = splev(t, splines[1], der=0)  # Utilisation de la première spline pour générer les points du contour
Q = np.column_stack([x, y])

N = len(Q)

# Paramétrisation des points du contour
t = np.linspace(0, 1, N)

# Degré de la spline
degree = 2

# Nombre de points de contrôle
n_ctrl = 9

# Vecteur de noeuds uniforme
knots = np.concatenate((
    np.zeros(degree),
    np.linspace(0, 1, n_ctrl - degree + 1),
    np.ones(degree)
))

# Fonction coût
def cost(ctrl_flat):
    # ctrl_flat = [x0, y0, x1, y1, ..., x8, y8]
    ctrl = ctrl_flat.reshape((n_ctrl, 2))

    # On construit une spline pour x et une pour y
    spline_x = BSpline(knots, ctrl[:,0], degree)
    spline_y = BSpline(knots, ctrl[:,1], degree)

    # Évaluation
    C = np.vstack((spline_x(t), spline_y(t))).T

    # Distances aux points du contour
    return (C - Q).ravel()

# Initialisation des points de contrôle (par ex. échantillonnage du contour)
init_ctrl = Q[np.linspace(0, N-1, n_ctrl).astype(int)]

# Optimisation
result = least_squares(cost, init_ctrl.ravel())

# Points de contrôle optimisés
ctrl_opt = result.x.reshape((n_ctrl, 2))

plt.figure()
plt.plot(Q[:,0], Q[:,1], 'o', label='Contour original')
plt.plot(ctrl_opt[:,0], ctrl_opt[:,1], 'x', label='Points de contrôle optimisés')
plt.plot(splev(t, BSpline(knots, ctrl_opt[:,0], degree)), splev(t, BSpline(knots, ctrl_opt[:,1], degree)), label='Spline ajustée')
plt.legend()
plt.show()

