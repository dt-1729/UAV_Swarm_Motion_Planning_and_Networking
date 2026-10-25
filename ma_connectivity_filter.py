# Copyright © MERL 2026
# Description: Filters multi-agent UAV actions to preserve connectivity and visualize corrections.
import numpy as np
from scipy.optimize import minimize

def multi_agent_connectivity_filter(
    X,
    Q0,
    V,
    dt,
    u2u_range,
    uav_connectivity_matrix,
    alpha=0.5,
    tol=1e-8,
):
    X = np.asarray(X, dtype=float) # agent current positions
    Q0 = np.asarray(Q0, dtype=float) # agent nominal directions
    V = np.asarray(V, dtype=float) # agent speeds

    if X.ndim != 2 or Q0.ndim != 2 or V.ndim != 1:
        raise ValueError("X and Q0 must be 2-D arrays, V must be a 1-D array")

    if X.shape[1] != 3 or Q0.shape[1] != 3 or Q0.shape[0] != X.shape[0]:
        raise ValueError("X and Q0 must have shape (N, 3) for N agents")

    if V.shape[0] != X.shape[0]:
        raise ValueError("V must have the same length as the number of agents in X")

    if uav_connectivity_matrix.shape != (X.shape[0], X.shape[0]):
        raise ValueError("uav_connectivity_matrix must have shape (N, N)")

    if X.shape[0] == 0:
        raise ValueError("At least one agent is required")

    # Agent 0 (head) is held fixed and may have a zero direction.
    if X.shape[0] > 1 and np.linalg.norm(Q0[1:], axis=1).min() <= tol:
        raise ValueError("All follower nominal directions in Q0 must be nonzero")

    def objective(Q):
        difference = Q - Q0.flatten()
        return 0.5 / Q0.shape[0] * np.dot(difference, difference)

    def objective_jac(Q):
        return (Q - Q0.flatten()) / Q0.shape[0]

    def get_connectivity_pairs():
        pairs = []
        N = uav_connectivity_matrix.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                if uav_connectivity_matrix[i, j]:
                    pairs.append((i, j))
        return pairs
    
    def connectivity_ij(qi, qj, xi, xj, vi, vj):
        relative_position = xj - xi
        step_i = vi * dt * qi
        step_j = vj * dt * qj
        future_relative_position = relative_position + step_j - step_i
        return (
            u2u_range**2
            - np.dot(future_relative_position, future_relative_position)
            - (1.0 - alpha) * (u2u_range**2 - np.dot(relative_position, relative_position))
        )

    def connectivity_ij_jac(qi, qj, xi, xj, vi, vj):
        step_i = vi * dt * qi
        step_j = vj * dt * qj
        future_relative_position = (xj - xi) + step_j - step_i
        jac_qi = 2.0 * vi * dt * future_relative_position
        jac_qj = -2.0 * vj * dt * future_relative_position
        return jac_qi, jac_qj
    
    def connectivity_mat(Q, X, V, connectivity_pairs):
        mat = np.zeros(len(connectivity_pairs))
        for idx, (i, j) in enumerate(connectivity_pairs):
            mat[idx] = connectivity_ij(Q[i], Q[j], X[i], X[j], V[i], V[j])
        return mat

    def connectivity_mat_jac(Q, X, V, connectivity_pairs):
        N = X.shape[0]
        jac = np.zeros((len(connectivity_pairs), N*3))
        for idx, (i, j) in enumerate(connectivity_pairs):
            jac_qi, jac_qj = connectivity_ij_jac(Q[i], Q[j], X[i], X[j], V[i], V[j])
            jac[idx, i*3:(i+1)*3] = jac_qi
            jac[idx, j*3:(j+1)*3] = jac_qj
        return jac

    def unit_norm(Q):
        return np.sum(Q[1:]**2, axis=1) - 1.0

    def unit_norm_jac(Q):
        N = Q.shape[0]

        # Unit-norm constraints only for agents 1, ..., N-1.
        jac = np.zeros((N - 1, N * 3))

        for i in range(1, N):
            jac[i - 1, i * 3:(i + 1) * 3] = 2.0 * Q[i]

        return jac
    
    result = minimize(
        objective,
        x0=Q0.flatten(),
        jac=objective_jac,
        method="SLSQP",
        constraints=[
            {
                "type": "ineq",
                "fun": lambda Q: connectivity_mat(Q.reshape(-1, 3), X, V, get_connectivity_pairs()),
                "jac": lambda Q: connectivity_mat_jac(Q.reshape(-1, 3), X, V, get_connectivity_pairs()),
            },
            {
                "type": "eq",
                "fun": lambda Q: unit_norm(Q.reshape(-1, 3)),
                "jac": lambda Q: unit_norm_jac(Q.reshape(-1, 3)),
            },
            {
                "type": "eq",
                "fun": lambda Q: Q[0:3] - Q0[0],
                "jac": lambda Q: np.concatenate([np.eye(3), np.zeros((3, Q.size - 3))], axis=1),
            }
        ],
        options={
            "ftol": tol,
            "maxiter": 500,
            "disp": False,
        },
    )
    return result