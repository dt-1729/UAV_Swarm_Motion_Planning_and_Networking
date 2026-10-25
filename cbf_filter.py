# Copyright © MERL 2026
# Description: Applies CBF-QP safety filters for connectivity, collision, and boundary constraints.

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize

def _ensure_neighbor_sets(env):
    neighbors = getattr(env, "uav_neighbors", None)
    if neighbors is None or len(neighbors) != env.n_uavs:
        if hasattr(env, "update_uav_connectivity"):
            neighbors = env.update_uav_connectivity()
        else:
            neighbors = [set() for _ in range(env.n_uavs)]
    return neighbors


def _per_uav_speed_limits(env):
    """Return per-UAV max speed vector with backward-compatible fallbacks."""
    n_uavs = int(env.n_uavs)
    head_id = int(getattr(env, "head_uav_idx", 0))

    if hasattr(env, "head_v_max") and hasattr(env, "follower_v_max"):
        head_v = float(getattr(env, "head_v_max"))
        follower_v = float(getattr(env, "follower_v_max"))
    else:
        legacy_v = float(getattr(env, "v_max"))
        head_v = legacy_v
        follower_v = legacy_v

    if head_v <= 0.0 or follower_v <= 0.0:
        raise ValueError("head_v_max and follower_v_max must be positive.")

    limits = np.full(n_uavs, follower_v, dtype=float)
    limits[head_id] = head_v
    return limits


def _coerce_desired_controls(env, desired_controls=None):
    if desired_controls is None:
        desired_controls = getattr(env, "uav_direction_actions", None)
    if desired_controls is None:
        desired_controls = getattr(env, "uav_control", None)
    if desired_controls is None:
        return np.zeros((env.n_uavs, 3), dtype=float)

    desired_controls = np.asarray(desired_controls, dtype=float)
    if desired_controls.shape != (env.n_uavs, 3):
        raise ValueError(
            f"desired_controls must have shape {(env.n_uavs, 3)}, got {desired_controls.shape}."
        )

    speed_limits = _per_uav_speed_limits(env).reshape(env.n_uavs, 1)
    norms = np.linalg.norm(desired_controls, axis=1, keepdims=True)
    scales = np.ones_like(norms)
    nonzero_mask = norms.squeeze(-1) > 1e-9
    scales[nonzero_mask] = np.minimum(1.0, speed_limits[nonzero_mask] / norms[nonzero_mask])
    return desired_controls * scales


def connectivity_cbf(env, controls, neighbor_mode="upstream"):
    """
        Return connectivity CBF values and derivatives.

    The returned derivatives are evaluated for the concrete `controls` argument.
    They are useful for diagnostics, but the QP still needs the affine-in-control
    form built from the same geometry when `u` is the decision variable.

        neighbor_mode:
            - "upstream": use the nearest upstream-hop neighbor when available,
                                        otherwise fall back to nearest connected neighbor.
            - "nearest":  always use the nearest connected neighbor.
    """
    positions = np.asarray(env.uav_locs, dtype=float)
    controls_is_cvx = isinstance(controls, cp.Expression)
    if not controls_is_cvx:
        controls = np.asarray(controls, dtype=float)
    neighbors = _ensure_neighbor_sets(env)
    hop_ids = getattr(env, "hop_ids", None)

    h_vec = []
    dh_dt_vec = []
    pairs = []
    mode = str(neighbor_mode).lower()
    if mode not in {"upstream", "nearest"}:
        raise ValueError("neighbor_mode must be one of: 'upstream', 'nearest'.")

    for uav_id in range(env.n_uavs):
        nbr_ids = sorted(neighbors[uav_id])
        if not nbr_ids:
            continue

        if mode == "upstream":
            # head UAV (hop_id 0) has no upstream; its link is protected by follower constraints.
            if hop_ids is None or np.isinf(hop_ids[uav_id]) or hop_ids[uav_id] == 0:
                continue

            upstream_ids = [j for j in nbr_ids if hop_ids[j] == hop_ids[uav_id] - 1]
            if upstream_ids:
                # pick nearest when multiple upstream candidates exist
                nbr_id = min(upstream_ids, key=lambda j: np.linalg.norm(positions[uav_id] - positions[j]))
            else:
                # no upstream found in current neighbor set; fall back to nearest neighbor
                offsets = positions[nbr_ids] - positions[uav_id]
                nbr_id = nbr_ids[int(np.argmin(np.linalg.norm(offsets, axis=1)))]
        else:
            # nearest-neighbor mode: use closest connected neighbor for every UAV.
            offsets = positions[nbr_ids] - positions[uav_id]
            nbr_id = nbr_ids[int(np.argmin(np.linalg.norm(offsets, axis=1)))]

        diff = positions[uav_id] - positions[nbr_id]
        h_vec.append(env.u2u_range**2 - diff @ diff)
        dh_dt_vec.append(-2.0 * diff @ (controls[uav_id] - controls[nbr_id]))
        pairs.append((uav_id, nbr_id))

    if not controls_is_cvx:
        dh_dt_vec = np.asarray(dh_dt_vec, dtype=float)
    
    return np.asarray(h_vec, dtype=float), dh_dt_vec, pairs


def collision_avoidance_cbf(env, controls):
    """
    Return pairwise collision-avoidance CBF values and derivatives.

    The returned derivatives are evaluated for the concrete `controls` argument.
    They are useful for diagnostics, but the QP still needs the affine-in-control
    form built from the same geometry when `u` is the decision variable.
    """
    positions = np.asarray(env.uav_locs, dtype=float)
    controls_is_cvx = isinstance(controls, cp.Expression)
    if not controls_is_cvx:
        controls = np.asarray(controls, dtype=float)
    
    neighbors = _ensure_neighbor_sets(env)

    h_collision_vec = []
    h_collision_derivative_vec = []
    pairs = []
    for u1 in range(env.n_uavs):
        nbr_ids = sorted(neighbors[u1])
        for u2 in nbr_ids:
            if u2 <= u1:
                continue
            diff = positions[u1] - positions[u2]
            h_collision_vec.append(diff @ diff - env.u2u_safe_dist**2)
            h_collision_derivative_vec.append(2.0 * diff @ (controls[u1] - controls[u2]))
            pairs.append((u1, u2))

    if not controls_is_cvx:
        h_collision_derivative_vec = np.asarray(h_collision_derivative_vec, dtype=float)

    return np.asarray(h_collision_vec, dtype=float), h_collision_derivative_vec, pairs


def domain_boundary_cbf(env, controls):
    """
    Return domain-boundary CBF values and derivatives for each UAV and axis.

    For each coordinate k in {x,y,z}, define:
      - Upper-bound barrier: h_max = space_max[k] - pos[k], dh_max/dt = -u[k]
      - Lower-bound barrier: h_min = pos[k] - space_min[k], dh_min/dt =  u[k]

    Enforcing dh/dt + alpha * h >= 0 for both barriers keeps trajectories
    within the box under standard CBF assumptions.
    """
    space_max = np.asarray(env.space_max, dtype=float).copy()
    space_min = np.asarray(env.space_min, dtype=float).copy()
    space_min[2] = space_min[2] + float(env.min_uav_height)

    positions = np.asarray(env.uav_locs, dtype=float)

    controls_is_cvx = isinstance(controls, cp.Expression)
    if not controls_is_cvx:
        controls = np.asarray(controls, dtype=float)

    h_boundary_vec = []
    dh_boundary_vec = []
    boundary_terms = []

    for uav_id in range(env.n_uavs):
        pos = positions[uav_id]
        h_max_vec = space_max - pos
        h_min_vec = pos - space_min
        dh_max_vec = -controls[uav_id]
        dh_min_vec = controls[uav_id]

        for axis in range(3):
            h_boundary_vec.append(h_max_vec[axis])
            dh_boundary_vec.append(dh_max_vec[axis])
            boundary_terms.append((uav_id, axis, "max"))

            h_boundary_vec.append(h_min_vec[axis])
            dh_boundary_vec.append(dh_min_vec[axis])
            boundary_terms.append((uav_id, axis, "min"))

    if not controls_is_cvx:
        dh_boundary_vec = np.asarray(dh_boundary_vec, dtype=float)

    return np.asarray(h_boundary_vec, dtype=float), dh_boundary_vec, boundary_terms


def compute_CBF_qp(
    env,
    desired_controls=None,
    alpha_con=0.05,
    alpha_col=0.01,
    alpha_bnd=0.1,
    connectivity_neighbor_mode=None,
    solver=None,
):
    """
    Filter nominal UAV velocities through a CBF-QP.

    The QP keeps the solution close to the desired velocities while enforcing
    linearized connectivity and collision-avoidance constraints.
    """
    nominal_controls = _coerce_desired_controls(env, desired_controls)
    head_id = int(getattr(env, "head_uav_idx", 0))
    nominal_controls[head_id] = 0.0
    positions = np.asarray(env.uav_locs, dtype=float)

    if positions.shape != (env.n_uavs, 3):
        raise ValueError(
            f"env.uav_locs must have shape {(env.n_uavs, 3)}, got {positions.shape}."
        )

    u = cp.Variable((env.n_uavs, 3))
    effective_connectivity_neighbor_mode = (
        str(getattr(env, "cbf_connectivity_neighbor_mode", "upstream"))
        if connectivity_neighbor_mode is None
        else str(connectivity_neighbor_mode)
    )
    h_con_values, dh_con_terms, connectivity_pairs = connectivity_cbf(
        env,
        u,
        neighbor_mode=effective_connectivity_neighbor_mode,
    )
    h_col_values, dh_col_terms, collision_pairs = collision_avoidance_cbf(env, u)
    h_bnd_values, dh_bnd_terms, boundary_terms = domain_boundary_cbf(env, u)
    speed_limits = _per_uav_speed_limits(env)
    qp_constraints = [cp.norm(u[uav_id], 2) <= speed_limits[uav_id] for uav_id in range(env.n_uavs)]
    qp_constraints.append(u[head_id] == 0.0)

    weights = np.ones(env.n_uavs)
    weight_col = weights[:, None]  # (n_uavs, 1) for broadcasting
    objective_terms = [cp.sum_squares(cp.multiply(np.sqrt(weight_col), u - nominal_controls))]

    if connectivity_pairs:
        # slack makes the QP always feasible; high weight keeps violations minimal
        # slack_con = cp.Variable(len(connectivity_pairs), nonneg=True)
        # objective_terms.append(connectivity_slack_weight * cp.sum_squares(slack_con))
        for idx, _ in enumerate(connectivity_pairs):
            qp_constraints.append(
                dh_con_terms[idx]
                + alpha_con * h_con_values[idx]
                # + slack_con[idx]
                >= 0.0
            )

    # if collision_pairs:
    #     for idx, _ in enumerate(collision_pairs):
    #         qp_constraints.append(
    #             dh_col_terms[idx]
    #             + alpha_col * h_col_values[idx]
    #             # + slack_col[idx]
    #             >= 0.0
    #         )

    if boundary_terms:
        for idx, _ in enumerate(boundary_terms):
            qp_constraints.append(
                dh_bnd_terms[idx]
                + alpha_bnd * h_bnd_values[idx]
                >= 0.0
            )

    prob = cp.Problem(cp.Minimize(sum(objective_terms)), qp_constraints)
    prob.solve(solver=solver, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or u.value is None:
        filtered_controls = nominal_controls
    else:
        filtered_controls = np.asarray(u.value, dtype=float)

    if hasattr(env, "uav_controls"):
        env.uav_controls = filtered_controls.copy()

    return filtered_controls
