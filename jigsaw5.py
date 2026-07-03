import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from scipy.interpolate import splprep, splev


# ============================================================
# 1) Paramètres métier sémantiques
# ============================================================

@dataclass
class SideParams:
    # --------------------------------------------------------
    # Intervalles horizontaux positifs
    # Ordre imposé :
    # x0 < x1 < x3 < x2 < x4 < x6 < x5 < x7 < x8
    # --------------------------------------------------------
    left_approach: float = 0.18
    left_lobe_entry: float = 0.10
    left_lobe_return: float = 0.08
    rise_to_peak: float = 0.14
    drop_from_peak: float = 0.14
    right_lobe_exit: float = 0.08
    right_approach_entry: float = 0.10
    right_approach: float = 0.18

    # --------------------------------------------------------
    # Paramètres verticaux
    # --------------------------------------------------------
    shoulder_lift_base: float = 0.02
    shoulder_skew: float = 0.0

    neck_rise_base: float = 0.14
    neck_skew: float = 0.0

    peak_rise: float = 0.09

    # --------------------------------------------------------
    # Offsets signés sur les points d'approche
    # y0 = 0 et y8 = 0 restent fixes
    # --------------------------------------------------------
    left_entry_offset: float = 0.0
    right_entry_offset: float = 0.0

    # +1 = téton, -1 = creux
    direction: int = 1


def default_params() -> SideParams:
    return SideParams()


# ============================================================
# 2) Outils
# ============================================================

def positive_part(values: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.maximum(v, eps)


def normalize_positive(values: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    v = positive_part(values, eps=eps)
    return v / v.sum()


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([
        [c, -s],
        [s,  c]
    ])
    return points @ R.T


def translate(points: np.ndarray, offset) -> np.ndarray:
    return points + np.asarray(offset)


# ============================================================
# 3) Extraction sémantique des intervalles
# ============================================================

def horizontal_intervals(p: SideParams) -> np.ndarray:
    return np.array([
        p.left_approach,
        p.left_lobe_entry,
        p.left_lobe_return,
        p.rise_to_peak,
        p.drop_from_peak,
        p.right_lobe_exit,
        p.right_approach_entry,
        p.right_approach,
    ], dtype=float)


def vertical_intervals(p: SideParams) -> np.ndarray:
    return np.array([
        p.shoulder_lift_base,
        p.neck_rise_base,
        p.peak_rise,
    ], dtype=float)


# ============================================================
# 4) Canonicalisation
# ============================================================

def canonicalize_params(p: SideParams) -> SideParams:
    q = replace(p)

    # horizontaux : strictement positifs
    q.left_approach = max(q.left_approach, 1e-4)
    q.left_lobe_entry = max(q.left_lobe_entry, 1e-4)
    q.left_lobe_return = max(q.left_lobe_return, 1e-4)
    q.rise_to_peak = max(q.rise_to_peak, 1e-4)
    q.drop_from_peak = max(q.drop_from_peak, 1e-4)
    q.right_lobe_exit = max(q.right_lobe_exit, 1e-4)
    q.right_approach_entry = max(q.right_approach_entry, 1e-4)
    q.right_approach = max(q.right_approach, 1e-4)

    # verticaux centraux : strictement positifs
    q.shoulder_lift_base = max(q.shoulder_lift_base, 1e-4)
    q.neck_rise_base = max(q.neck_rise_base, 1e-4)
    q.peak_rise = max(q.peak_rise, 1e-4)

    # asymétries bornées
    q.shoulder_skew = float(np.clip(q.shoulder_skew, -0.03, 0.03))
    q.neck_skew = float(np.clip(q.neck_skew, -0.04, 0.04))

    # offsets d'approche signés mais bornés
    q.left_entry_offset = float(np.clip(q.left_entry_offset, -0.05, 0.05))
    q.right_entry_offset = float(np.clip(q.right_entry_offset, -0.05, 0.05))

    q.direction = 1 if q.direction >= 0 else -1
    return q


# ============================================================
# 5) Reconstruction des coordonnées
# ============================================================

def build_x_coords(p: SideParams):
    gaps = normalize_positive(horizontal_intervals(p))

    d01, d13, d32, d24, d46, d65, d57, d78 = gaps
    xs = np.concatenate([[0.0], np.cumsum([d01, d13, d32, d24, d46, d65, d57, d78])])

    x0 = xs[0]
    x1 = xs[1]
    x3 = xs[2]
    x2 = xs[3]
    x4 = xs[4]
    x6 = xs[5]
    x5 = xs[6]
    x7 = xs[7]
    x8 = xs[8]

    return x0, x1, x2, x3, x4, x5, x6, x7, x8


def build_y_levels(p: SideParams):
    shoulder_base, neck_base, peak_rise = positive_part(vertical_intervals(p))

    shoulder_left = shoulder_base + p.shoulder_skew
    shoulder_right = shoulder_base - p.shoulder_skew

    neck_left = shoulder_base + neck_base + p.neck_skew
    neck_right = shoulder_base + neck_base - p.neck_skew

    peak = shoulder_base + neck_base + peak_rise

    # sécurité : positifs
    shoulder_left = max(shoulder_left, 1e-4)
    shoulder_right = max(shoulder_right, 1e-4)

    # sécurité : le cou doit rester au-dessus de l'épaule
    neck_left = max(neck_left, shoulder_left + 1e-4)
    neck_right = max(neck_right, shoulder_right + 1e-4)

    # sécurité : le sommet doit rester au-dessus des deux côtés du cou
    peak = max(peak, max(neck_left, neck_right) + 1e-4)

    sgn = p.direction
    return (
        sgn * shoulder_left,   # y2
        sgn * neck_left,       # y3
        sgn * peak,            # y4
        sgn * neck_right,      # y5
        sgn * shoulder_right,  # y6
    )


# ============================================================
# 6) Construction des points de contrôle
# ============================================================

def build_control_points(p: SideParams) -> np.ndarray:
    p = canonicalize_params(p)

    x0, x1, x2, x3, x4, x5, x6, x7, x8 = build_x_coords(p)
    y2, y3, y4, y5, y6 = build_y_levels(p)

    y0 = 0.0
    y1 = p.left_entry_offset
    y7 = p.right_entry_offset
    y8 = 0.0

    ctrl = np.array([
        [x0, y0],  # 0
        [x1, y1],  # 1
        [x2, y2],  # 2
        [x3, y3],  # 3
        [x4, y4],  # 4
        [x5, y5],  # 5
        [x6, y6],  # 6
        [x7, y7],  # 7
        [x8, y8],  # 8
    ], dtype=float)

    return ctrl


# ============================================================
# 7) Spline quadratique
# ============================================================

def spline_side(ctrl: np.ndarray, n: int = 250) -> np.ndarray:
    tck, _ = splprep(ctrl.T, s=0, k=2)
    u = np.linspace(0.0, 1.0, n)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def make_side(params: SideParams, n: int = 250):
    ctrl = build_control_points(params)
    curve = spline_side(ctrl, n=n)
    return ctrl, curve


# ============================================================
# 8) Perturbation
# ============================================================

def perturb_params(base: SideParams, rng: np.random.Generator, noise: float = 1.0) -> SideParams:
    q = replace(base)

    # bruit horizontal
    q.left_approach += rng.normal(0.0, 0.020 * noise)
    q.left_lobe_entry += rng.normal(0.0, 0.018 * noise)
    q.left_lobe_return += rng.normal(0.0, 0.018 * noise)
    q.rise_to_peak += rng.normal(0.0, 0.020 * noise)
    q.drop_from_peak += rng.normal(0.0, 0.020 * noise)
    q.right_lobe_exit += rng.normal(0.0, 0.018 * noise)
    q.right_approach_entry += rng.normal(0.0, 0.018 * noise)
    q.right_approach += rng.normal(0.0, 0.020 * noise)

    # bruit vertical de base
    q.shoulder_lift_base += rng.normal(0.0, 0.008 * noise)
    q.neck_rise_base += rng.normal(0.0, 0.012 * noise)
    q.peak_rise += rng.normal(0.0, 0.012 * noise)

    # asymétries
    q.shoulder_skew += rng.normal(0.0, 0.006 * noise)
    q.neck_skew += rng.normal(0.0, 0.008 * noise)

    # offsets signés
    q.left_entry_offset += rng.normal(0.0, 0.010 * noise)
    q.right_entry_offset += rng.normal(0.0, 0.010 * noise)

    q.direction = rng.choice([-1, 1])

    return canonicalize_params(q)


# ============================================================
# 9) Pièce complète à partir d'une face
# ============================================================

def transform_side_to_edge(side: np.ndarray, start, end) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    edge = end - start
    length = np.linalg.norm(edge)
    if length <= 1e-12:
        raise ValueError("Le segment cible est de longueur nulle.")

    angle = np.arctan2(edge[1], edge[0])

    transformed = rotate(side, angle)
    transformed = transformed * length
    transformed = transformed + start
    return transformed


# ============================================================
# 10) Pièce complète à partir de quatre faces différentes
# ============================================================

def make_piece(top_side: np.ndarray,
               right_side: np.ndarray,
               bottom_side: np.ndarray,
               left_side: np.ndarray) -> tuple:

    top_t = transform_side_to_edge(top_side, [0, 1], [1, 1])
    right_t = transform_side_to_edge(right_side, [1, 1], [1, 0])
    bottom_t = transform_side_to_edge(bottom_side, [1, 0], [0, 0])
    left_t = transform_side_to_edge(left_side, [0, 0], [0, 1])

    piece = np.vstack([
        top_t[:-1],
        right_t[:-1],
        bottom_t[:-1],
        left_t[:-1],
        top_t[:1]
    ])
    return top_t, right_t, bottom_t, left_t, piece

# ============================================================
# 10) Debug / validation
# ============================================================

def print_debug_example():
    p = canonicalize_params(default_params())
    ctrl = build_control_points(p)

    print("Points de contrôle:")
    for i, (x, y) in enumerate(ctrl):
        print(f"{i}: x={x:.4f}, y={y:.4f}")

    xs = ctrl[:, 0]
    ys = ctrl[:, 1]

    horizontal_ok = xs[0] < xs[1] < xs[3] < xs[2] < xs[4] < xs[6] < xs[5] < xs[7] < xs[8]

    central_vertical_left_ok = abs(ys[2]) < abs(ys[3]) < abs(ys[4])
    central_vertical_right_ok = abs(ys[6]) < abs(ys[5]) < abs(ys[4])

    print("\nOrdre horizontal valide :", horizontal_ok)
    print("Ordre vertical gauche   :", central_vertical_left_ok)
    print("Ordre vertical droit    :", central_vertical_right_ok)
    print("Offsets entrée          :", ys[1], ys[7])
    print("Asymétrie épaules       :", ys[2], ys[6])
    print("Asymétrie cou           :", ys[3], ys[5])


# ============================================================
# 12) Utilitaires puzzle
# ============================================================

def complementary_params(p: SideParams) -> SideParams:
    q = replace(p)
    q.direction = -p.direction
    return q


def transform_control_points_to_edge(ctrl: np.ndarray, start, end) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    edge = end - start
    length = np.linalg.norm(edge)
    if length <= 1e-12:
        raise ValueError("Le segment cible est de longueur nulle.")

    angle = np.arctan2(edge[1], edge[0])

    transformed = rotate(ctrl, angle)
    transformed = transformed * length
    transformed = transformed + start
    return transformed


def make_piece_from_params(top_p, right_p, bottom_p, left_p, n=250):
    top_ctrl, top_side = make_side(top_p, n=n)
    right_ctrl, right_side = make_side(right_p, n=n)
    bottom_ctrl, bottom_side = make_side(bottom_p, n=n)
    left_ctrl, left_side = make_side(left_p, n=n)

    top_t, right_t, bottom_t, left_t, piece = make_piece(
        top_side, right_side, bottom_side, left_side
    )

    top_ctrl_t = transform_control_points_to_edge(top_ctrl, [0, 0], [1, 0])
    right_ctrl_t = transform_control_points_to_edge(right_ctrl, [1, 0], [1, 1])
    bottom_ctrl_t = transform_control_points_to_edge(bottom_ctrl, [1, 1], [0, 1])
    left_ctrl_t = transform_control_points_to_edge(left_ctrl, [0, 1], [0, 0])

    return {
        "top_side": top_t,
        "right_side": right_t,
        "bottom_side": bottom_t,
        "left_side": left_t,
        "piece": piece,
        "top_ctrl": top_ctrl_t,
        "right_ctrl": right_ctrl_t,
        "bottom_ctrl": bottom_ctrl_t,
        "left_ctrl": left_ctrl_t,
        "params": {
            "top": top_p,
            "right": right_p,
            "bottom": bottom_p,
            "left": left_p,
        }
    }


# ============================================================
# 13) Génération des arêtes partagées
# ============================================================

def random_inner_edge(rng: np.random.Generator, noise: float = 1.0) -> SideParams:
    p = perturb_params(default_params(), rng, noise=noise)
    p.direction = int(rng.choice([-1, 1]))
    return canonicalize_params(p)


def generate_puzzle_edge_tables(rows: int, cols: int, rng: np.random.Generator, noise: float = 1.0):
    vertical_edges = [[None for _ in range(cols - 1)] for _ in range(rows)]
    horizontal_edges = [[None for _ in range(cols)] for _ in range(rows - 1)]

    for r in range(rows):
        for c in range(cols - 1):
            vertical_edges[r][c] = random_inner_edge(rng, noise=noise)

    for r in range(rows - 1):
        for c in range(cols):
            horizontal_edges[r][c] = random_inner_edge(rng, noise=noise)

    return horizontal_edges, vertical_edges

# ============================================================
# 14) Faces plates pour le bord du puzzle
# ============================================================

def flat_side_params() -> SideParams:
    p = default_params()
    p.shoulder_lift_base = 1e-4
    p.neck_rise_base = 1e-4
    p.peak_rise = 1e-4
    p.shoulder_skew = 0.0
    p.neck_skew = 0.0
    p.left_entry_offset = 0.0
    p.right_entry_offset = 0.0
    p.direction = 1
    return canonicalize_params(p)

# ============================================================
# 15) Construction de toutes les pièces du puzzle
# ============================================================

def generate_puzzle_pieces(rows: int, cols: int, rng: np.random.Generator, noise: float = 1.0, n=250):
    horizontal_edges, vertical_edges = generate_puzzle_edge_tables(rows, cols, rng, noise=noise)

    pieces = []

    for r in range(rows):
        row_pieces = []
        for c in range(cols):
            # top
            if r == 0:
                top_p = flat_side_params()
            else:
                top_p = complementary_params(horizontal_edges[r - 1][c])

            # right
            if c == cols - 1:
                right_p = flat_side_params()
            else:
                right_p = vertical_edges[r][c]

            # bottom
            if r == rows - 1:
                bottom_p = flat_side_params()
            else:
                bottom_p = horizontal_edges[r][c]

            # left
            if c == 0:
                left_p = flat_side_params()
            else:
                left_p = complementary_params(vertical_edges[r][c - 1])

            piece_data = make_piece_from_params(top_p, right_p, bottom_p, left_p, n=n)
            row_pieces.append(piece_data)
        pieces.append(row_pieces)

    return pieces

# ============================================================
# 16) Démonstration puzzle complet
# ============================================================

def demo_puzzle(rows: int = 3, cols: int = 4, noise: float = 0.9, n: int = 250):
    rng = np.random.default_rng(2026)
    pieces = generate_puzzle_pieces(rows, cols, rng, noise=noise, n=n)

    plt.figure(figsize=(3 * cols, 3 * rows))

    for r in range(rows):
        for c in range(cols):
            data = pieces[r][c]
            piece = data["piece"] + np.array([c, r])

            plt.plot(piece[:, 0], piece[:, 1], color="black", lw=1.8)

            # points de contrôle des 4 faces
            for key, color in [
                ("top_ctrl", "red"),
                ("right_ctrl", "green"),
                ("bottom_ctrl", "blue"),
                ("left_ctrl", "turquoise"),
            ]:
                ctrl = data[key] + np.array([c, r])
                plt.scatter(ctrl[:, 0], ctrl[:, 1], s=10, color=color, alpha=0.6)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(True, alpha=0.2)
    plt.title(f"Puzzle généré : {rows} x {cols}")
    plt.show()


# ============================================================
# 11) Démonstrations
# ============================================================

def demo_faces():
    rng = np.random.default_rng(2026)
    base = default_params()

    base.left_entry_offset = -0.015
    base.right_entry_offset = -0.012
    base.shoulder_skew = 0.004
    base.neck_skew = -0.006

    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        params = base if i == 0 else perturb_params(base, rng, noise=1.2)
        ctrl, curve = make_side(params)

        ax.plot(curve[:, 0], curve[:, 1], lw=2, color="tab:red")
        ax.scatter(ctrl[:, 0], ctrl[:, 1], s=28, color="black")
        ax.plot(ctrl[:, 0], ctrl[:, 1], "--", color="gray", alpha=0.25)

        ax.axhline(0.0, color="black", lw=0.8, alpha=0.25)

        ax.set_title(f"Face {i}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.40, 0.40)

    plt.tight_layout()
    plt.show()


def demo_piece():
    rng = np.random.default_rng(2026)

    top_params = perturb_params(default_params(), rng, noise=2)
    right_params = perturb_params(default_params(), rng, noise=2)
    bottom_params = perturb_params(default_params(), rng, noise=2)
    left_params = perturb_params(default_params(), rng, noise=2)

    _, top_side = make_side(top_params)
    _, right_side = make_side(right_params)
    _, bottom_side = make_side(bottom_params)
    _, left_side = make_side(left_params)

    top_t, right_t, bottom_t, left_t, piece = make_piece(
        top_side, right_side, bottom_side, left_side
    )

    plt.figure(figsize=(10, 10))
    plt.plot(top_t[:, 0], top_t[:, 1], color="red", label="top")
    plt.plot(right_t[:, 0], right_t[:, 1], color="green", label="right")
    plt.plot(bottom_t[:, 0], bottom_t[:, 1], color="blue", label="bottom")
    plt.plot(left_t[:, 0], left_t[:, 1], color="turquoise", label="left")
    plt.plot(piece[:, 0], piece[:, 1], color="black", lw=2, ls="dotted", alpha=0.7, label="piece")

    plt.scatter([0, 1, 1, 0], [0, 0, 1, 1], color="black", s=35, zorder=5)

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Pièce générée avec quatre faces indépendantes")
    plt.show()


if __name__ == "__main__":
    print_debug_example()
    demo_faces()
    demo_piece()
    #demo_puzzle()