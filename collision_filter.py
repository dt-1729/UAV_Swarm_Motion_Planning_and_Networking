# Copyright © MERL 2026
# Description: Filters multi-agent UAV directions to keep a minimum pairwise distance (collision avoidance).
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult


def multi_agent_collision_filter(
    X,
    Q0,
    V,
    dt,
    u2u_safe_dist,
    alpha=0.5,
    tol=1e-8,
    u2u_range=None,
    uav_connectivity_matrix=None,
):
    """Minimally modify nominal directions Q0 so no UAV pair gets closer than u2u_safe_dist.

    Same setup as the connectivity filter: agent 0 (head) is held fixed, followers keep unit-norm
    directions, and each pair (i, j) gets a discrete-time barrier constraint
        |x_ij + dt (v_j q_j - v_i q_i)|^2 - d_safe^2 >= (1 - alpha) (|x_ij|^2 - d_safe^2),
    which keeps the pair outside d_safe if it already is, and pushes it out if it is not.
    Only pairs that could come within d_safe during one step are constrained.

    If u2u_range and uav_connectivity_matrix are given, the connectivity constraint of the connectivity
    filter is kept for the connected pairs, so running this filter after the connectivity filter does not
    undo link preservation.
    """
    X = np.asarray(X, dtype=float)
    Q0 = np.asarray(Q0, dtype=float)
    V = np.asarray(V, dtype=float)

    if X.ndim != 2 or Q0.ndim != 2 or V.ndim != 1:
        raise ValueError("X and Q0 must be 2-D arrays, V must be a 1-D array")
    if X.shape[1] != 3 or Q0.shape != X.shape:
        raise ValueError("X and Q0 must have shape (N, 3) for N agents")
    if V.shape[0] != X.shape[0]:
        raise ValueError("V must have the same length as the number of agents in X")
    if X.shape[0] == 0:
        raise ValueError("At least one agent is required")
    if X.shape[0] > 1 and np.linalg.norm(Q0[1:], axis=1).min() <= tol:
        raise ValueError("All follower nominal directions in Q0 must be nonzero")

    N = X.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            reach = (V[i] + V[j]) * dt
            if np.linalg.norm(X[j] - X[i]) - reach < u2u_safe_dist and reach > 0.0:
                pairs.append((i, j))

    link_pairs = []
    if uav_connectivity_matrix is not None and u2u_range is not None:
        link_matrix = np.asarray(uav_connectivity_matrix, dtype=bool)
        link_pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if link_matrix[i, j]]

    if not pairs:
        return OptimizeResult(x=Q0.flatten(), success=True, status=0, message="no pair near the safe distance", nit=0)

    def objective(Q):
        difference = Q - Q0.flatten()
        return 0.5 / N * np.dot(difference, difference)

    def objective_jac(Q):
        return (Q - Q0.flatten()) / N

    def future_relative(Q, i, j):
        return (X[j] - X[i]) + V[j] * dt * Q[j] - V[i] * dt * Q[i]

    def collision_mat(Q):
        Q = Q.reshape(-1, 3)
        out = np.zeros(len(pairs))
        for idx, (i, j) in enumerate(pairs):
            rel = X[j] - X[i]
            fut = future_relative(Q, i, j)
            out[idx] = (
                np.dot(fut, fut) - u2u_safe_dist**2
                - (1.0 - alpha) * (np.dot(rel, rel) - u2u_safe_dist**2)
            )
        return out

    def collision_mat_jac(Q):
        Q = Q.reshape(-1, 3)
        jac = np.zeros((len(pairs), N * 3))
        for idx, (i, j) in enumerate(pairs):
            fut = future_relative(Q, i, j)
            jac[idx, i * 3:(i + 1) * 3] = -2.0 * V[i] * dt * fut
            jac[idx, j * 3:(j + 1) * 3] = 2.0 * V[j] * dt * fut
        return jac

    def link_mat(Q):
        Q = Q.reshape(-1, 3)
        out = np.zeros(len(link_pairs))
        for idx, (i, j) in enumerate(link_pairs):
            rel = X[j] - X[i]
            fut = future_relative(Q, i, j)
            out[idx] = u2u_range**2 - np.dot(fut, fut) - (1.0 - alpha) * (u2u_range**2 - np.dot(rel, rel))
        return out

    def link_mat_jac(Q):
        return -collision_pair_jac(Q, link_pairs)

    def collision_pair_jac(Q, pair_list):
        Q = Q.reshape(-1, 3)
        jac = np.zeros((len(pair_list), N * 3))
        for idx, (i, j) in enumerate(pair_list):
            fut = future_relative(Q, i, j)
            jac[idx, i * 3:(i + 1) * 3] = -2.0 * V[i] * dt * fut
            jac[idx, j * 3:(j + 1) * 3] = 2.0 * V[j] * dt * fut
        return jac

    def unit_norm(Q):
        return np.sum(Q.reshape(-1, 3)[1:] ** 2, axis=1) - 1.0

    def unit_norm_jac(Q):
        Q = Q.reshape(-1, 3)
        jac = np.zeros((N - 1, N * 3))
        for i in range(1, N):
            jac[i - 1, i * 3:(i + 1) * 3] = 2.0 * Q[i]
        return jac

    constraints = [
        {"type": "ineq", "fun": collision_mat, "jac": collision_mat_jac},
    ]
    if link_pairs:
        constraints.append({"type": "ineq", "fun": link_mat, "jac": link_mat_jac})
    constraints += [
        {"type": "eq", "fun": unit_norm, "jac": unit_norm_jac},
        {
            "type": "eq",
            "fun": lambda Q: Q[0:3] - Q0[0],
            "jac": lambda Q: np.concatenate([np.eye(3), np.zeros((3, Q.size - 3))], axis=1),
        },
    ]

    def nudged_start(axis, size):
        # Symmetric (e.g. head-on collinear) configurations are stationary points of the linearised
        # problem; a small sideways nudge of the followers breaks the symmetry.
        Q = Q0.copy()
        for i in range(1, N):
            side = np.cross(Q0[i], axis)
            if np.linalg.norm(side) < 1e-6:
                side = np.cross(Q0[i], np.roll(axis, 1))
            Q[i] = Q0[i] + size * side / np.linalg.norm(side)
            Q[i] /= np.linalg.norm(Q[i])
        return Q.flatten()

    starts = [Q0.flatten(), nudged_start(np.array([0.0, 0.0, 1.0]), 0.3), nudged_start(np.array([0.0, 1.0, 0.0]), 0.3)]
    result = None
    for x0 in starts:
        result = minimize(
            objective,
            x0=x0,
            jac=objective_jac,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": tol, "maxiter": 500, "disp": False},
        )
        if result.success:
            break
    return result
