# Copyright © MERL 2026
# Description: Filters UAV directions so no UAV drops below a minimum safety height after one step.
import numpy as np


def height_filter(X, Q0, V, dt, min_height, tol=1e-9):
    """Closed-form projection of directions onto {q : z + v dt q_z >= min_height, |q| = 1}.

    For a violating UAV the vertical component is raised to the smallest value that keeps it at
    min_height after the step (clipped to [-1, 1]); the horizontal part keeps its heading and is
    rescaled so the direction stays unit-norm. A UAV that cannot reach min_height in one step
    climbs straight up. Zero directions (a stationary head) are left unchanged unless they violate.
    """
    X = np.asarray(X, dtype=float)
    Q = np.array(Q0, dtype=float)
    V = np.asarray(V, dtype=float)

    for i in range(X.shape[0]):
        step = V[i] * dt
        if step <= tol:
            continue
        qz_min = (min_height - X[i, 2]) / step
        if Q[i, 2] >= qz_min - tol:
            continue
        qz = float(np.clip(qz_min, -1.0, 1.0))
        horiz = np.linalg.norm(Q[i, :2])
        radial = np.sqrt(max(0.0, 1.0 - qz**2))
        if horiz > tol:
            Q[i, :2] *= radial / horiz
        elif np.linalg.norm(Q[i]) > tol:
            Q[i, :2] = [radial, 0.0]
        Q[i, 2] = qz
    return Q
