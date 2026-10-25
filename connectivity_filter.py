# Copyright © MERL 2026
# Description: Computes connectivity-preserving action filters and diagnostic 3D plots.
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def multi_agent_connectivity_filter(
    X,
    Q0,
    V,
    dt,
    u2u_range,
    connectivity_pairs,
    alpha=0.5,
    tol=1e-8,
):
    X = np.asarray(X, dtype=float) # agent current positions
    Q0 = np.asarray(Q0, dtype=float) # agent nominal directions
    V = np.asarray(V, dtype=float) # agent speeds

    if X.ndim != 2 or Q0.ndim != 2 or V.ndim != 1:
        raise ValueError("X and Q0 must be 2-D arrays, V must be a 1-D array")

    if X.shape[1] != 3 or Q0.shape[1] != 3:
        raise ValueError("X and Q0 must have shape (N, 3) for N agents")

    if V.shape[0] != X.shape[0]:
        raise ValueError("V must have the same length as the number of agents in X")

    if np.linalg.norm(Q0, axis=1).min() <= tol:
        raise ValueError("All nominal directions in Q0 must be nonzero")

    def objective(Q):
        return 1/Q0.shape[0] * np.linalg.norm(Q - Q0, axis=1).sum()

    def objective_jac(Q):
        return (Q - Q0).flatten() / Q0.shape[0]
    
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
        N = X.shape[0]
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
        return np.sum(Q**2, axis=1) - 1.0

    def unit_norm_jac(Q):
        jac = np.zeros((Q.shape[0], Q.shape[0]*3))
        for i in range(Q.shape[0]):
            jac[i, i*3:(i+1)*3] = 2.0 * Q[i]
        return jac
    
    result = minimize(
        objective,
        x0=Q0.flatten(),
        jac=objective_jac,
        method="SLSQP",
        constraints=[
            {
                "type": "ineq",
                "fun": lambda Q: connectivity_mat(Q.reshape(-1, 3), X, V, connectivity_pairs),
                "jac": lambda Q: connectivity_mat_jac(Q.reshape(-1, 3), X, V, connectivity_pairs),
            },
            {
                "type": "eq",
                "fun": lambda Q: unit_norm(Q.reshape(-1, 3)),
                "jac": lambda Q: unit_norm_jac(Q.reshape(-1, 3)),
            },
        ],
        options={
            "ftol": tol,
            "maxiter": 500,
            "disp": False,
        },
    )
    return result


    
def filter_connectivity_direction(
    z0,
    v,
    xt,
    dt,
    u2u_range,
    q0,
    alpha=0.5,
    tol=1e-8,
    return_result=False,
):
    z0 = np.asarray(z0, dtype=float)
    xt = np.asarray(xt, dtype=float)
    q0 = np.asarray(q0, dtype=float)

    if z0.ndim != 1 or xt.shape != z0.shape or q0.shape != z0.shape:
        raise ValueError("z0, xt, and q0 must be same-sized 1-D arrays")

    if not np.isscalar(v):
        raise ValueError("v must be a scalar speed")

    v = float(v)
    dt = float(dt)
    u2u_range = float(u2u_range)

    if dt < 0:
        raise ValueError("dt must be nonnegative")
    if u2u_range <= 0:
        raise ValueError("u2u_range must be positive")

    q0_norm = np.linalg.norm(q0)

    if not np.isfinite(q0_norm) or q0_norm <= tol:
        raise ValueError("q0 must be finite and nonzero. Its norm is {:.3e}".format(q0_norm))

    q0 = q0 / q0_norm

    relative_position = xt - z0
    current_margin = (
        u2u_range**2 - np.dot(relative_position, relative_position)
    )
    step = v * dt

    # Minimize squared distance to the nominal unit direction.
    # On ||q|| = 1, this is equivalent to maximizing q @ q0.
    def objective(q):
        return 0.5 * np.dot(q - q0, q - q0)

    def objective_jac(q):
        return q - q0

    def connectivity(q):
        future_relative_position = relative_position + step * q
        return (
            u2u_range**2
            - np.dot(future_relative_position, future_relative_position)
            - (1.0 - alpha) * current_margin
        )

    def connectivity_jac(q):
        return -2.0 * step * (relative_position + step * q)

    def unit_norm(q):
        return np.dot(q, q) - 1.0

    def unit_norm_jac(q):
        return 2.0 * q

    # Avoid running an optimizer when the nominal direction is feasible.
    if connectivity(q0) >= 0.0:
        if return_result:
            return q0, None
        future_range_margin = u2u_range**2 - np.dot(relative_position + step * q0, relative_position + step * q0)
        print(f"Nominal direction is feasible with future_range_margin={future_range_margin:.3e} and connectivity={connectivity(q0):.3e}; returning q0.")
        return q0

    q_init = z0 - xt
    q_init_norm = np.linalg.norm(q_init)
    if q_init_norm <= tol:
        q_init = q0 + np.random.uniform(-1.0, 1.0, size=q0.shape)
        q_init /= np.linalg.norm(q_init)
    else:
        q_init /= q_init_norm

    result = minimize(
        objective,
        x0=q_init,
        jac=objective_jac,
        method="SLSQP",
        constraints=[
            {
                "type": "ineq",
                "fun": connectivity,
                "jac": connectivity_jac,
            },
            {
                "type": "eq",
                "fun": unit_norm,
                "jac": unit_norm_jac,
            },
        ],
        options={
            "ftol": tol,
            "maxiter": 500,
            "disp": False,
        },
    )

    q = np.asarray(result.x)
    q_norm = np.linalg.norm(q)

    connectivity_error = min(0.0, connectivity(q))
    norm_error = abs(q_norm - 1.0)

    if (
        not result.success
        or q_norm <= tol
        or connectivity_error < -tol
        or norm_error > 10.0 * tol
    ):
        raise RuntimeError(
            "Unable to find a feasible direction: "
            f"{result.message}; "
            f"connectivity={connectivity(q):.3e}, "
            f"norm_error={norm_error:.3e}"
        )

    q = q / q_norm

    if return_result:
        return q, result
    return q

import matplotlib.pyplot as plt


def set_axes_equal(ax, center, radius):
    """Set equal scaling for a Matplotlib 3-D axis."""
    center = np.asarray(center)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_box_aspect((1, 1, 1))


def visualize_connectivity_direction(
    z0,
    xt,
    q0,
    q,
    v,
    dt,
    u2u_range,
    sphere_alpha=0.12,
):
    z0 = np.asarray(z0, dtype=float)
    xt = np.asarray(xt, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q = np.asarray(q, dtype=float)

    # Actual displacement over one time step.
    arrow_length = abs(v * dt)

    # Generate the communication-range sphere.
    azimuth = np.linspace(0.0, 2.0 * np.pi, 80)
    polar = np.linspace(0.0, np.pi, 40)
    azimuth, polar = np.meshgrid(azimuth, polar)

    sphere_x = (
        z0[0]
        + u2u_range * np.sin(polar) * np.cos(azimuth)
    )
    sphere_y = (
        z0[1]
        + u2u_range * np.sin(polar) * np.sin(azimuth)
    )
    sphere_z = z0[2] + u2u_range * np.cos(polar)

    nominal_endpoint = xt + arrow_length * q0
    filtered_endpoint = xt + arrow_length * q

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Connectivity sphere.
    ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color="royalblue",
        alpha=sphere_alpha,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    # Optional wireframe to make the sphere easier to perceive.
    ax.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=4,
        cstride=6,
        color="royalblue",
        alpha=0.15,
        linewidth=0.5,
    )

    # Center location z0.
    ax.scatter(
        *z0,
        color="black",
        marker="o",
        s=100,
        label=r"$z_0$",
        depthshade=False,
    )

    # Initial location xt.
    ax.scatter(
        *xt,
        color="darkorange",
        marker="o",
        s=100,
        label=r"$x_t$",
        depthshade=False,
    )

    # Nominal direction q0.
    ax.quiver(
        *xt,
        *(arrow_length * q0),
        color="crimson",
        linewidth=2.5,
        arrow_length_ratio=0.15,
        label=r"Nominal direction $q_0$",
    )

    # Filtered direction q.
    ax.quiver(
        *xt,
        *(arrow_length * q),
        color="green",
        linewidth=3.0,
        arrow_length_ratio=0.15,
        label=r"Filtered direction $q$",
    )

    # Show endpoints after one time step.
    ax.scatter(
        *nominal_endpoint,
        color="crimson",
        marker="x",
        s=70,
        depthshade=False,
    )
    ax.scatter(
        *filtered_endpoint,
        color="green",
        marker="x",
        s=70,
        depthshade=False,
    )

    # Line connecting the center to the current location.
    ax.plot(
        [z0[0], xt[0]],
        [z0[1], xt[1]],
        [z0[2], xt[2]],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        alpha=0.8,
    )

    distance = np.linalg.norm(xt - z0)
    angle = np.degrees(
        np.arccos(np.clip(np.dot(q0, q), -1.0, 1.0))
    )

    ax.set_title(
        "Connectivity Direction Filter\n"
        f"Current distance = {distance:.2f}, "
        f"range = {u2u_range:.2f}, "
        f"direction change = {angle:.2f}°"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_axes_equal(
        ax,
        center=z0,
        radius=1.1 * u2u_range,
    )

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

np.random.seed(4000)

alphas = np.linspace(0.99, 0.99, 1)

# Generate the shared parameters once.
z0 = np.random.uniform(-10.0, 10.0, size=3)
v = 10.0
u2u_range = 100.0
dt = 5.0

rel_xt = np.random.uniform(-1.0, 1.0, size=3)
rel_xt /= np.linalg.norm(rel_xt)

xt = z0 + rel_xt * np.random.uniform(
    0.5 * u2u_range,
    1.0 * u2u_range,
)

q0 = np.random.uniform(-1.0, 1.0, size=3)
q0 /= np.linalg.norm(q0)

# Each alpha is stored as a separate test case.
testcases = {}

for i, alpha in enumerate(alphas):
    testcases[i] = {
        "z0": z0.copy(),
        "v": v,
        "xt": xt.copy(),
        "dt": dt,
        "u2u_range": u2u_range,
        "q0": q0.copy(),
        "alpha": float(alpha),
    }

# visualize_connectivity_direction(
#     z0=testcases[0]["z0"],
#     xt=testcases[0]["xt"],
#     q0=testcases[0]["q0"],
#     q=q,
#     v=testcases[0]["v"],
#     dt=testcases[0]["dt"],
#     u2u_range=testcases[0]["u2u_range"],
# )

for i, testcase in testcases.items():
    try:
        testcase["q"] = filter_connectivity_direction(
            z0=testcase["z0"],
            v=testcase["v"],
            xt=testcase["xt"],
            dt=testcase["dt"],
            u2u_range=testcase["u2u_range"],
            q0=testcase["q0"],
            alpha=testcase["alpha"],
        )
        testcase["success"] = True

    except RuntimeError as error:
        testcase["q"] = None
        testcase["success"] = False
        testcase["error"] = str(error)

for testcase in testcases.values():
    if not testcase["success"]:
        testcase["angle_change"] = np.nan
        testcase["h_next"] = np.nan
        testcase["required_h"] = np.nan
        continue

    relative_position = testcase["xt"] - testcase["z0"]

    h_current = (
        testcase["u2u_range"]**2
        - np.dot(relative_position, relative_position)
    )

    future_relative_position = (
        relative_position
        + testcase["v"] * testcase["dt"] * testcase["q"]
    )

    h_next = (
        testcase["u2u_range"]**2
        - np.dot(
            future_relative_position,
            future_relative_position,
        )
    )

    angle_change = np.degrees(
        np.arccos(
            np.clip(
                np.dot(testcase["q0"], testcase["q"]),
                -1.0,
                1.0,
            )
        )
    )

    testcase["h_current"] = h_current
    testcase["h_next"] = h_next
    testcase["required_h"] = (
        (1.0 - testcase["alpha"]) * h_current
    )
    testcase["angle_change"] = angle_change

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")

first_case = testcases[0]
z0 = first_case["z0"]
xt = first_case["xt"]
radius = first_case["u2u_range"]
arrow_length = first_case["v"] * first_case["dt"]

# Connectivity sphere.
azimuth = np.linspace(0.0, 2.0 * np.pi, 60)
polar = np.linspace(0.0, np.pi, 30)
azimuth, polar = np.meshgrid(azimuth, polar)

sphere_x = z0[0] + radius * np.sin(polar) * np.cos(azimuth)
sphere_y = z0[1] + radius * np.sin(polar) * np.sin(azimuth)
sphere_z = z0[2] + radius * np.cos(polar)

ax.plot_surface(
    sphere_x,
    sphere_y,
    sphere_z,
    color="royalblue",
    alpha=0.08,
    linewidth=0,
)

ax.scatter(*z0, color="black", s=90, label=r"$z_0$")
ax.scatter(*xt, color="orange", s=90, label=r"$x_t$")

# Shared nominal direction.
ax.quiver(
    *xt,
    *(arrow_length * first_case["q0"]),
    color="red",
    linewidth=3,
    arrow_length_ratio=0.1,
    label=r"Nominal $q_0$",
)

cmap = cm.viridis
normalizer = colors.Normalize(
    vmin=alphas.min(),
    vmax=alphas.max(),
)

for testcase in testcases.values():
    if not testcase["success"]:
        continue

    ax.quiver(
        *testcase["xt"],
        *(arrow_length * testcase["q"]),
        color=cmap(normalizer(testcase["alpha"])),
        linewidth=2,
        arrow_length_ratio=0.1,
    )

colorbar = fig.colorbar(
    cm.ScalarMappable(norm=normalizer, cmap=cmap),
    ax=ax,
    shrink=0.65,
    pad=0.1,
)
colorbar.set_label(r"$\alpha$")

ax.set_title("Independent test cases for different alpha values")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

set_axes_equal(
    ax,
    center=z0,
    radius=1.1 * radius,
)

plt.tight_layout()
# plt.show()

plot_alphas = np.array([
    testcase["alpha"] for testcase in testcases.values()
])

angle_changes = np.array([
    testcase["angle_change"] for testcase in testcases.values()
])

h_next_values = np.array([
    testcase["h_next"] for testcase in testcases.values()
])

required_h_values = np.array([
    testcase["required_h"] for testcase in testcases.values()
])


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(
    plot_alphas,
    angle_changes,
    "o-",
    color="darkgreen",
)
axes[0].set_xlabel(r"$\alpha$")
axes[0].set_ylabel("Angle between q and q0 (degrees)")
axes[0].set_title("Direction modification")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    plot_alphas,
    h_next_values,
    "o-",
    label=r"$h(x_{k+1})$",
)
axes[1].plot(
    plot_alphas,
    required_h_values,
    "s--",
    label=r"$(1-\alpha)h(x_k)$",
)
axes[1].set_xlabel(r"$\alpha$")
axes[1].set_ylabel("Safety margin")
axes[1].set_title("Discrete CBF constraint")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
