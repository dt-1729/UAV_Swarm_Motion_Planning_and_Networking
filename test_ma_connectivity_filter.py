
import random
import numpy as np
import matplotlib.pyplot as plt
import swarm_env as se
import ma_connectivity_filter as mcf
from pathlib import Path

# Build the UAV swarm environment with specified parameters.
env = se.UAVSwarmEnv3D(
    n_uavs=5,
    n_sensors=10,
    n_obstacles=0,
    box_size=1000.0,
    min_uav_height=2.0,
    com_params={
        "u2u_range": 10.0,
        "u2u_safe_dist": 2.0,
        "u2s_range": 5.0,
        "u2b_range": 20.0,
        "u2s_visibility_range": 100.0,
    },
    simulation_params={"init_time": 0.0, "dt": 5.0, "max_time": 100.0, "max_energy": 1.0e5},
    reward_params={
        "sensor_collect_coeff": 500.0*np.ones(5, dtype=np.float32),
        "potential_diff_coeff": 10.0*np.ones(5, dtype=np.float32),
        "min_uav_height_penalty_coeff": 0.0,
        "collision_penalty_coeff": 0.0,
        "connectivity_penalty_coeff": 0.0,
        "time_penalty_coeff": 1000.0*0.0,
        "cbf_control_penalty_coeff": 0.0,
    },
    power_params={
        "P0": 79.86,
        "P1": 88.63,
        "v_tip": 120.0,
        "v0": 4.03,
        "d0": 0.6,
        "rho": 1.225,
        "s": 0.05,
        "A": 0.503,
    },
    head_v_max=4.0,
    follower_v_max=7.0,
    seed=random.randint(0, 2**32 - 1),
)

def visualize_multi_agent_filter(
    X,
    Q0,
    Q,
    V,
    dt,
    u2u_range,
    connectivity_pairs,
    save_dir="ma_connectivity_filter_plots",
    filename="multi_agent_connectivity_filter.png",
):
    X = np.asarray(X)
    Q0 = np.asarray(Q0)
    Q = np.asarray(Q)
    V = np.asarray(V)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Unit sphere coordinates.
    azimuth = np.linspace(0.0, 2.0 * np.pi, 40)
    polar = np.linspace(0.0, np.pi, 20)
    azimuth, polar = np.meshgrid(azimuth, polar)

    unit_x = np.sin(polar) * np.cos(azimuth)
    unit_y = np.sin(polar) * np.sin(azimuth)
    unit_z = np.cos(polar)

    for i in range(X.shape[0]):
        xi = X[i]
        arrow_length = V[i] * dt

        # Communication sphere centered at agent i.
        sphere_x = xi[0] + u2u_range * unit_x
        sphere_y = xi[1] + u2u_range * unit_y
        sphere_z = xi[2] + u2u_range * unit_z

        ax.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            color="royalblue",
            alpha=0.06,
            linewidth=0,
            shade=False,
        )

        ax.plot_wireframe(
            sphere_x,
            sphere_y,
            sphere_z,
            rstride=4,
            cstride=4,
            color="royalblue",
            alpha=0.10,
            linewidth=0.5,
        )

        # Agent position.
        ax.scatter(
            *xi,
            color="black",
            s=70,
            depthshade=False,
        )

        ax.text(
            xi[0],
            xi[1],
            xi[2],
            f"  UAV {i}",
        )

        # Nominal control.
        ax.quiver(
            *xi,
            *(arrow_length * Q0[i]),
            color="red",
            linewidth=2.5,
            arrow_length_ratio=0.15,
        )

        # Filtered control.
        ax.quiver(
            *xi,
            *(arrow_length * Q[i]),
            color="green",
            linewidth=2.5,
            arrow_length_ratio=0.15,
        )

    # Plot existing connectivity links.
    for i, j in connectivity_pairs:
        ax.plot(
            [X[i, 0], X[j, 0]],
            [X[i, 1], X[j, 1]],
            [X[i, 2], X[j, 2]],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

    # Dummy lines for a clean legend.
    ax.plot(
        [],
        [],
        [],
        color="red",
        linewidth=2.5,
        label=r"Nominal control $Q_0$",
    )
    ax.plot(
        [],
        [],
        [],
        color="green",
        linewidth=2.5,
        label=r"Filtered control $Q$",
    )
    ax.plot(
        [],
        [],
        [],
        color="gray",
        linestyle="--",
        label="Connectivity link",
    )
    ax.scatter(
        [],
        [],
        [],
        color="black",
        s=70,
        label="UAV position",
    )

    # Equal scaling that includes all communication spheres.
    lower = np.min(X, axis=0) - u2u_range
    upper = np.max(X, axis=0) + u2u_range
    center = 0.5 * (lower + upper)
    half_width = 0.5 * np.max(upper - lower)

    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_zlim(center[2] - half_width, center[2] + half_width)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Z position")
    ax.set_title("Multi-Agent Connectivity Filter")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create the output folder if it does not exist.
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Plot saved to: {output_path.resolve()}")

    plt.show()
    plt.close(fig)
    return output_path.resolve()

X = env.uav_locs  # Get current UAV positions
Q0 = np.random.uniform(-1, 1, size=(env.n_uavs, 3))  # Random nominal directions
Q0 /= np.linalg.norm(Q0, axis=1, keepdims=True)  # Normalize to unit vectors
V = env.follower_v_max * np.ones(env.n_uavs)  # Set speeds to max follower speed
dt = env.dt
u2u_range = env.u2u_range
uav_connectivity = env.uav_connectivity
connectivity_pairs = [
    (i, j)
    for i in range(env.n_uavs)
    for j in range(i + 1, env.n_uavs)
    if uav_connectivity[i, j]
]
alpha = 0.5

result = mcf.multi_agent_connectivity_filter(
    X,
    Q0,
    V,
    dt,
    u2u_range,
    uav_connectivity,
    alpha=alpha,
)

if not result.success:
    raise RuntimeError(
        "Connectivity filter failed: "
        f"{result.message}; "
        f"objective={result.fun:.3e}"
    )

Q_filtered = result.x.reshape(env.n_uavs, 3)
assert Q_filtered.shape == X.shape
assert np.allclose(np.linalg.norm(Q_filtered, axis=1), 1.0, atol=1e-5)

future_positions = X + (V * dt)[:, np.newaxis] * Q_filtered
for i, j in connectivity_pairs:
    current_distance_squared = np.sum((X[j] - X[i]) ** 2)
    future_distance_squared = np.sum(
        (future_positions[j] - future_positions[i]) ** 2
    )
    maximum_distance_squared = (
        u2u_range**2
        - (1.0 - alpha) * (u2u_range**2 - current_distance_squared)
    )
    assert future_distance_squared <= maximum_distance_squared + 1e-5

plot_path = visualize_multi_agent_filter(
    X=X,
    Q0=Q0,
    Q=Q_filtered,
    V=V,
    dt=dt,
    u2u_range=u2u_range,
    connectivity_pairs=connectivity_pairs,
    save_dir="ma_connectivity_filter_plots",
    filename="multi_agent_connectivity_filter.png",
)
assert plot_path.is_file()
