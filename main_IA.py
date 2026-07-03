import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# =====================================================================
# IMPORTS DE VOS MODULES LOCAUX
# =====================================================================
try:
    from inference_hackton_affichage import load_trained_model, predict_mask
    from geometry_utils import extract_and_normalize_edges
    from optimisation_bspline import fit_spline_to_segment
except ImportError as e:
    print(f"⚠️ Erreur d'importation : {e}")
    print("Assure-toi de lancer ce script depuis le dossier contenant tes autres fichiers Python.")


# =====================================================================
# ÉTAPE 1 : CLASSIFICATION ET LOGIQUE MÉTIER
# =====================================================================

def classifier_cote(ctrl_points, seuil_plat=0.08):
    y = ctrl_points[:, 1]
    y_max = np.max(y)
    y_min = np.min(y)

    if abs(y_max) < seuil_plat and abs(y_min) < seuil_plat:
        return 0 # Plat
    if abs(y_max) > abs(y_min):
        return 1 # Bosse (Vers l'extérieur)
    return 2 # Creux (Vers l'intérieur)

def afficher_piece_individuelle(piece_id, dict_ctrl_piece):
    """ Affiche les 4 bords d'une pièce normalisés de 0 à 1 pour vérifier la détection """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle(f"Diagnostic Détection : Pièce {piece_id}", fontsize=14, fontweight='bold')
    axes = axes.ravel()
    
    t = np.linspace(0, 1, 100)
    knots = np.concatenate((np.zeros(2), np.linspace(0, 1, 9 - 2 + 1), np.ones(2)))

    for i in range(4):
        ctrl = dict_ctrl_piece[i]["ctrl"]
        spline_x = BSpline(knots, ctrl[:, 0], 2)(t)
        spline_y = BSpline(knots, ctrl[:, 1], 2)(t)

        axes[i].plot(spline_x, spline_y, 'b-', linewidth=2)
        axes[i].plot(ctrl[:, 0], ctrl[:, 1], 'rx', markersize=8)
        
        cat_nom = ["Plat", "Bosse", "Creux"][dict_ctrl_piece[i]["cat"]]
        axes[i].set_title(f"Bord {i} (Classé: {cat_nom})")
        axes[i].axis('equal')
        axes[i].grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()

# =====================================================================
# ÉTAPE 2 : RÉSOLUTION DU PUZZLE
# =====================================================================

def distance_cotes(ctrlA, ctrlB):
    ctrlB_inverse = np.zeros_like(ctrlB)
    for i in range(9):
        ctrlB_inverse[i, 0] = 1.0 - ctrlB[8-i, 0]
        ctrlB_inverse[i, 1] = -ctrlB[8-i, 1]
    return np.linalg.norm(ctrlA - ctrlB_inverse)

def associer_pieces(dict_ctrl):
    cotes = []
    for pid, cotes_piece in dict_ctrl.items():
        for cid, info in enumerate(cotes_piece):
            if info["cat"] != 0:
                cotes.append((pid, cid))

    distances = {}
    for (pA, cA) in cotes:
        for (pB, cB) in cotes:
            if pA != pB:
                catA = dict_ctrl[pA][cA]["cat"]
                catB = dict_ctrl[pB][cB]["cat"]
                if (catA, catB) in [(1, 2), (2, 1)]:
                    distances[((pA, cA), (pB, cB))] = distance_cotes(
                        dict_ctrl[pA][cA]["ctrl"], dict_ctrl[pB][cB]["ctrl"]
                    )

    associations = []
    associes = set()

    for (pA, cA) in cotes:
        if (pA, cA) in associes: continue
        meilleur, meilleure_dist = None, np.inf

        for (pB, cB) in cotes:
            if (pB, cB) in associes or pA == pB: continue
            key = ((pA, cA), (pB, cB))
            if key in distances and distances[key] < meilleure_dist:
                meilleure_dist = distances[key]
                meilleur = (pB, cB)

        if meilleur is not None:
            associations.append(((pA, cA), meilleur))
            associes.add((pA, cA))
            associes.add(meilleur)

    return associations


# =====================================================================
# ÉTAPE 3 : RECONSTRUCTION GÉOMÉTRIQUE AVEC GESTION DE LA ROTATION
# =====================================================================

def reconstruire_coordonnees(dict_ctrl, associations):
    adj = {pid: {} for pid in dict_ctrl}
    for (pA, cA), (pB, cB) in associations:
        adj[pA][cA] = (pB, cB)
        adj[pB][cB] = (pA, cA)
        
    # 1. Trouver la racine (Idéalement un coin du puzzle)
    root = None
    root_rot = 0
    for pid, cotes in dict_ctrl.items():
        plats = [i for i, c in enumerate(cotes) if c["cat"] == 0]
        if len(plats) >= 2:
            root = pid
            # Orientons la pièce de coin pour que ses bords plats pointent en Haut(0) et Gauche(3)
            # 'root_rot' est le nombre de quarts de tour horaires (90°) à appliquer à la photo
            if 0 in plats and 3 in plats: root_rot = 0
            elif 0 in plats and 1 in plats: root_rot = 3 # Tourner de 270° pour amener (0,1) vers (Haut, Gauche)
            elif 1 in plats and 2 in plats: root_rot = 2 # Tourner de 180°
            elif 2 in plats and 3 in plats: root_rot = 1 # Tourner de 90°
            break
            
    if root is None:
        print("⚠️ Aucun coin pur détecté, on commence par une pièce au hasard sans rotation.")
        root = list(dict_ctrl.keys())[0]

    # Le dictionnaire stocke désormais (Ligne, Colonne, Rotation en Quarts de Tour)
    positions = {root: (0, 0, root_rot)}
    queue = [root]
    
    while queue:
        curr = queue.pop(0)
        r, c, rot_curr = positions[curr]
        
        for cote_curr, (voisin, cote_voisin) in adj[curr].items():
            if voisin not in positions:
                # Direction absolue du bord de la pièce courante dans le monde global (0:Haut, 1:Droite...)
                dir_abs_curr = (cote_curr + rot_curr) % 4
                
                # Le bord cible de la pièce voisine DOIT faire face à la direction opposée
                dir_abs_voisin_cible = (dir_abs_curr + 2) % 4
                
                # On calcule la rotation que doit subir le voisin pour que son bord s'aligne
                rot_voisin = (dir_abs_voisin_cible - cote_voisin) % 4
                
                if dir_abs_curr == 0: n_r, n_c = r - 1, c
                elif dir_abs_curr == 1: n_r, n_c = r, c + 1
                elif dir_abs_curr == 2: n_r, n_c = r + 1, c
                elif dir_abs_curr == 3: n_r, n_c = r, c - 1
                
                positions[voisin] = (n_r, n_c, rot_voisin)
                queue.append(voisin)
                
    # Gestion des orphelines
    orphelines = set(dict_ctrl.keys()) - set(positions.keys())
    offset_c = max([c for r, c, rot in positions.values()] + [0]) + 2
    for pid in list(orphelines):
        positions[pid] = (0, offset_c, 0)
        offset_c += 1
        
    return positions

def afficher_puzzle_resolu(dict_ctrl, positions):
    fig, ax = plt.subplots(figsize=(12, 10))
    t = np.linspace(0, 1, 100)
    knots = np.concatenate((np.zeros(2), np.linspace(0, 1, 9 - 2 + 1), np.ones(2)))

    for piece_id, (r, c, rot) in positions.items():
        cotes = dict_ctrl[piece_id]
        
        for cote_idx, data in enumerate(cotes):
            ctrl = data["ctrl"]
            
            spline_x = BSpline(knots, ctrl[:, 0], 2)(t)
            spline_y = BSpline(knots, ctrl[:, 1], 2)(t)
            
            # Application de la rotation globale pour le dessin
            dir_abs = (cote_idx + rot) % 4
            
            if dir_abs == 0:   # Haut
                plot_x = c + spline_x ; plot_y = -r + spline_y
            elif dir_abs == 1: # Droite
                plot_x = c + 1 + spline_y ; plot_y = -r - spline_x
            elif dir_abs == 2: # Bas
                plot_x = c + 1 - spline_x ; plot_y = -r - 1 - spline_y
            elif dir_abs == 3: # Gauche
                plot_x = c - spline_y ; plot_y = -r - 1 + spline_x

            couleur = 'black' if data["cat"] == 0 else 'blue'
            ax.plot(plot_x, plot_y, color=couleur, linewidth=2.5)
            
        ax.text(c + 0.5, -r - 0.5, str(piece_id), color='red', fontsize=14, 
                ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    ax.axis('equal')
    ax.axis('off')
    ax.set_title("Résolution du Puzzle avec Correction des Rotations", fontsize=16)
    plt.tight_layout()
    plt.show()


# =====================================================================
# ÉTAPE 4 : ORCHESTRATION GLOBALE
# =====================================================================

def analyser_dossier_photos(dossier_photos, model_path="unet_puzzle_weights.pth"):
    print(f"--- 1. Chargement du modèle IA depuis '{model_path}' ---")
    unet_model, device = load_trained_model(model_path)
    
    dict_ctrl = {}
    fichiers = [f for f in sorted(os.listdir(dossier_photos)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not fichiers:
        raise ValueError(f"Aucune image trouvée dans le dossier {dossier_photos}")

    print(f"\n--- 2. Traitement IA et Extraction Mathématique ({len(fichiers)} images) ---")
    
    for fichier in fichiers:
        chemin = os.path.join(dossier_photos, fichier)
        piece_id = os.path.splitext(fichier)[0]
        print(f" -> Analyse de la pièce : {piece_id}")
        
        mask, _, corners, _ = predict_mask(unet_model, device, chemin)
        segments_normalises = extract_and_normalize_edges(mask, corners)
        
        dict_ctrl_piece = []
        for cote_idx, segment in enumerate(segments_normalises):
            ctrl_opt, knots = fit_spline_to_segment(segment)
            
            # ==========================================
            # CORRECTION DE SÉCURITÉ ("ANT-SPAGHETTI")
            # On verrouille mathématiquement le 1er et le dernier point 
            # pour être sûr que les bords se connectent parfaitement aux angles
            # ==========================================
            ctrl_opt[0] = [0.0, 0.0]
            ctrl_opt[-1] = [1.0, 0.0]
            
            cat = classifier_cote(ctrl_opt)
            
            dict_ctrl_piece.append({
                "ctrl": ctrl_opt,
                "cat": cat
            })
            
        dict_ctrl[piece_id] = dict_ctrl_piece
        afficher_piece_individuelle(piece_id, dict_ctrl_piece)
    return dict_ctrl


if __name__ == "__main__":
    DOSSIER_TEST = "resources/4_pieces_ensembles" 
    
    if not os.path.exists(DOSSIER_TEST):
        print(f"Création du dossier '{DOSSIER_TEST}'. Mets tes 4 photos dedans et relance le script !")
        os.makedirs(DOSSIER_TEST)
    else:
        try:
            dict_ctrl_reelles = analyser_dossier_photos(DOSSIER_TEST)
            
            print("\n--- 3. Algorithme de Résolution ---")
            associations = associer_pieces(dict_ctrl_reelles)
            print(f"{len(associations)} associations trouvées.")
            for a in associations:
                print(f"  [{a[0][0]}] côté {a[0][1]}  <--->  [{a[1][0]}] côté {a[1][1]}")
            
            print("\n--- 4. Affichage du Puzzle Reconstruit ---")
            pos_reconstruites = reconstruire_coordonnees(dict_ctrl_reelles, associations)
            afficher_puzzle_resolu(dict_ctrl_reelles, pos_reconstruites)
            
        except Exception as e:
            print(f"\n❌ Une erreur critique est survenue : {e}")
            import traceback
            traceback.print_exc()