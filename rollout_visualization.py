# Copyright © MERL 2026
# Description: Visualizes multi-agent PPO rollouts, metrics, directions, and connectivity.
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def _sphere_mesh(center: np.ndarray, radius: float, n_u: int = 20, n_v: int = 20):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _segment_alpha_ramp(
    n_segments: int,
    alpha_min: float = 0.08,
    alpha_max: float = 0.95,
) -> np.ndarray:
    """Return per-segment alpha ramp so older segments are faint and newer are strong."""
    if n_segments <= 0:
        return np.zeros((0,), dtype=float)
    if n_segments == 1:
        return np.asarray([float(alpha_max)], dtype=float)
    return np.linspace(float(alpha_min), float(alpha_max), int(n_segments))


def _draw_ground_plane(
    ax,
    space_min: np.ndarray,
    space_max: np.ndarray,
    n: int = 24,
) -> None:
    """Draw a subtle textured ground plane at z=space_min[2] for depth cues."""
    x_min, y_min, z_min = float(space_min[0]), float(space_min[1]), float(space_min[2])
    x_max, y_max = float(space_max[0]), float(space_max[1])

    gx = np.linspace(x_min, x_max, max(4, int(n)))
    gy = np.linspace(y_min, y_max, max(4, int(n)))
    xx, yy = np.meshgrid(gx, gy)
    zz = np.full_like(xx, z_min)

    span_x = max(1e-6, x_max - x_min)
    span_y = max(1e-6, y_max - y_min)
    cell_x = span_x / 8.0
    cell_y = span_y / 8.0

    # Blend a gentle Y-gradient with a faint checker pattern for realistic texture.
    y_grad = (yy - y_min) / span_y
    checker = (
        (np.floor((xx - x_min) / max(cell_x, 1e-6)) + np.floor((yy - y_min) / max(cell_y, 1e-6))) % 2
    )
    tone = 0.90 - 0.08 * y_grad + 0.02 * checker
    tone = np.clip(tone, 0.0, 1.0)

    facecolors = np.zeros(xx.shape + (4,), dtype=float)
    facecolors[..., 0] = tone
    facecolors[..., 1] = tone
    facecolors[..., 2] = tone
    facecolors[..., 3] = 0.18

    ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def _compute_uav_speeds(uav_locs: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Estimate per-step per-UAV speed magnitudes from trajectory positions."""
    t_steps = int(uav_locs.shape[0])
    n_uavs = int(uav_locs.shape[1])
    speeds = np.zeros((t_steps, n_uavs), dtype=float)
    if t_steps <= 1:
        return speeds

    dt = np.diff(times)
    dt = np.where(np.abs(dt) < 1e-9, 1.0, dt)
    disp = np.linalg.norm(np.diff(uav_locs, axis=0), axis=-1)
    step_speeds = disp / dt[:, None]
    speeds[1:, :] = step_speeds
    speeds[0, :] = step_speeds[0, :]
    return speeds


def _rolling_mean_percentile(
    values: np.ndarray,
    window: int,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute trailing rolling mean and percentile band for 1D or 2D values."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("values must be 1D or 2D.")

    n_time = arr.shape[0]
    window = max(1, int(window))
    mean = np.full(n_time, np.nan, dtype=float)
    low = np.full(n_time, np.nan, dtype=float)
    high = np.full(n_time, np.nan, dtype=float)

    for idx in range(n_time):
        start = max(0, idx - window + 1)
        chunk = arr[start : idx + 1].reshape(-1)
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == 0:
            continue
        mean[idx] = float(np.mean(chunk))
        low[idx] = float(np.percentile(chunk, lower_percentile))
        high[idx] = float(np.percentile(chunk, upper_percentile))

    return mean, low, high


def _pairwise_distance_matrix(uav_locs: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Return all upper-triangle UAV pairs and their distances over time."""
    n_time = int(uav_locs.shape[0])
    n_uavs = int(uav_locs.shape[1])
    pairs: List[Tuple[int, int]] = [
        (i, j)
        for i in range(n_uavs)
        for j in range(i + 1, n_uavs)
    ]
    distances = np.full((n_time, len(pairs)), np.nan, dtype=float)
    for pair_idx, (i, j) in enumerate(pairs):
        distances[:, pair_idx] = np.linalg.norm(uav_locs[:, i, :] - uav_locs[:, j, :], axis=-1)
    return pairs, distances


def _active_link_count(connectivity: np.ndarray) -> np.ndarray:
    """Count undirected active links in each connectivity matrix."""
    conn = np.asarray(connectivity, dtype=bool)
    if conn.ndim != 3 or conn.shape[1] != conn.shape[2]:
        return np.zeros((0,), dtype=float)
    upper = np.triu(np.ones(conn.shape[1:], dtype=bool), k=1)
    return np.sum(conn[:, upper], axis=1).astype(float)


def _connectivity_violation_mask(connectivity: np.ndarray, head_uav_idx: int = 0) -> np.ndarray:
    """Mark timesteps where the undirected UAV graph is not fully connected."""
    conn = np.asarray(connectivity, dtype=bool)
    if conn.ndim != 3 or conn.shape[1] != conn.shape[2]:
        return np.zeros((0,), dtype=bool)

    n_time, n_uavs, _ = conn.shape
    if n_uavs <= 1:
        return np.zeros(n_time, dtype=bool)

    head_uav_idx = max(0, min(int(head_uav_idx), n_uavs - 1))
    violation = np.zeros(n_time, dtype=bool)
    for t_idx in range(n_time):
        visited = np.zeros(n_uavs, dtype=bool)
        stack = [head_uav_idx]
        visited[head_uav_idx] = True
        while stack:
            current = stack.pop()
            for nbr in np.where(conn[t_idx, current])[0]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(int(nbr))
        violation[t_idx] = not bool(np.all(visited))
    return violation


def _time_edges(times: np.ndarray) -> np.ndarray:
    """Build plotting edges around time samples for shaded event windows."""
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return np.zeros((0,), dtype=float)
    if t.size == 1:
        return np.asarray([t[0] - 0.5, t[0] + 0.5], dtype=float)

    mids = 0.5 * (t[:-1] + t[1:])
    first = t[0] - 0.5 * (t[1] - t[0])
    last = t[-1] + 0.5 * (t[-1] - t[-2])
    return np.concatenate([[first], mids, [last]])


def _add_event_ribbons(
    ax,
    times: np.ndarray,
    collision_violation: np.ndarray,
    connectivity_violation: np.ndarray,
) -> None:
    """Shade collision/connectivity violation windows behind a metric plot."""
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return

    edges = _time_edges(times)
    events = [
        (np.asarray(collision_violation, dtype=bool), "#6A1B9A", "Collision window"),
        (np.asarray(connectivity_violation, dtype=bool), "#EF6C00", "Connectivity window"),
    ]
    used_labels = {
        text.get_text()
        for text in ax.get_legend().texts
    } if ax.get_legend() is not None else set()

    for mask, color, label in events:
        if mask.shape[0] != times.shape[0]:
            continue
        idx = 0
        first_label = label not in used_labels
        while idx < mask.shape[0]:
            if not bool(mask[idx]):
                idx += 1
                continue
            start = idx
            while idx + 1 < mask.shape[0] and bool(mask[idx + 1]):
                idx += 1
            end = idx
            ax.axvspan(
                edges[start],
                edges[end + 1],
                color=color,
                alpha=0.11,
                linewidth=0.0,
                label=label if first_label else None,
                zorder=0,
            )
            first_label = False
            idx += 1


def _legend_if_labeled(ax, **kwargs) -> None:
    """Show a legend only when labeled artists are present."""
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def _ellipse_shadow_xy(
    center_x: float,
    center_y: float,
    major_radius: float,
    minor_radius: float,
    angle_rad: float,
    n_pts: int = 28,
) -> np.ndarray:
    """Build XY vertices for a rotated ellipse centered at (center_x, center_y)."""
    theta = np.linspace(0.0, 2.0 * np.pi, max(12, int(n_pts)), endpoint=False)
    ex = major_radius * np.cos(theta)
    ey = minor_radius * np.sin(theta)

    ca = float(np.cos(angle_rad))
    sa = float(np.sin(angle_rad))
    rx = ca * ex - sa * ey
    ry = sa * ex + ca * ey

    return np.stack([rx + center_x, ry + center_y], axis=-1)


def visualize_rollout_trajectory_3d(
    rollout_trajectory: Dict[str, Any],
    frame_idx: int = -1,
    show_spheres: bool = False,
    figsize: Sequence[float] = (11, 9),
    recent_trace_steps: int = 40,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize MAPPO rollout trajectories in 3D.

    Required keys in rollout_trajectory:
      - uav_locs:        (T, U, 3)
      - times:           (T,)
      - connectivity:    (T, U, U)
      - sensor_locs:     (S, 3)
      - base_station:    (3,)
      - u2u_range:       float
      - u2u_safe_dist:   float
      - space_min:       (3,)
      - space_max:       (3,)
            - visited_fraction:(T,)  # optional for title annotations
            - visited_mask:    (T, S) # optional for red/green sensor coloring
    """
    uav_locs = np.asarray(rollout_trajectory["uav_locs"], dtype=float)
    times = np.asarray(rollout_trajectory["times"], dtype=float)
    connectivity = np.asarray(rollout_trajectory["connectivity"], dtype=bool)
    sensor_locs = np.asarray(rollout_trajectory["sensor_locs"], dtype=float)
    base_station = np.asarray(rollout_trajectory["base_station"], dtype=float)
    u2u_range = float(rollout_trajectory["u2u_range"])
    u2u_safe_dist = float(rollout_trajectory["u2u_safe_dist"])
    space_min = np.asarray(rollout_trajectory["space_min"], dtype=float)
    space_max = np.asarray(rollout_trajectory["space_max"], dtype=float)
    visited_fraction = rollout_trajectory.get("visited_fraction")
    if visited_fraction is not None:
        visited_fraction = np.asarray(visited_fraction, dtype=float)
    visited_mask = rollout_trajectory.get("visited_mask")
    if visited_mask is not None:
        visited_mask = np.asarray(visited_mask, dtype=bool)

    if uav_locs.ndim != 3 or uav_locs.shape[-1] != 3:
        raise ValueError("uav_locs must have shape (T, U, 3).")
    if times.ndim != 1 or times.shape[0] != uav_locs.shape[0]:
        raise ValueError("times must have shape (T,) and match uav_locs length.")

    t_steps, n_uavs, _ = uav_locs.shape
    speeds = _compute_uav_speeds(uav_locs, times)
    speed_max = max(float(np.max(speeds)), 1e-6)
    resolved_frame_idx = frame_idx if frame_idx >= 0 else (t_steps + frame_idx)
    resolved_frame_idx = max(0, min(int(resolved_frame_idx), t_steps - 1))
    head_uav_idx = int(rollout_trajectory.get("head_uav_idx", 0))
    head_uav_idx = max(0, min(head_uav_idx, n_uavs - 1))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    recent_trace_steps = max(2, int(recent_trace_steps))
    _draw_ground_plane(ax, space_min=space_min, space_max=space_max)

    if visited_mask is not None and visited_mask.ndim == 2 and visited_mask.shape[0] == t_steps and visited_mask.shape[1] == sensor_locs.shape[0]:
        sensor_visited = visited_mask[resolved_frame_idx]
    else:
        sensor_visited = np.zeros(sensor_locs.shape[0], dtype=bool)

    unvisited = ~sensor_visited
    if np.any(unvisited):
        ax.scatter(
            sensor_locs[unvisited, 0], sensor_locs[unvisited, 1], sensor_locs[unvisited, 2],
            marker="^", s=55, c="#C62828", alpha=0.9, label="Unvisited Sensors"
        )
    if np.any(sensor_visited):
        ax.scatter(
            sensor_locs[sensor_visited, 0], sensor_locs[sensor_visited, 1], sensor_locs[sensor_visited, 2],
            marker="^", s=60, c="#2E7D32", alpha=0.95, label="Visited Sensors"
        )
    ax.scatter(
        [base_station[0]], [base_station[1]], [base_station[2]],
        marker="*", s=220, c="#B71C1C", edgecolors="black", linewidths=0.7, label="Base Station"
    )

    norm = plt.Normalize(vmin=float(times.min()), vmax=float(times.max()) if float(times.max()) > float(times.min()) else float(times.min()) + 1.0)
    cmaps = ["Blues", "Oranges", "Greens", "Purples", "Reds", "Greys"]

    for uav_id in range(n_uavs):
        path = uav_locs[:, uav_id, :]
        is_head = (uav_id == head_uav_idx)
        speed_start = float(speeds[0, uav_id])
        speed_end = float(speeds[-1, uav_id])
        speed_start_norm = float(np.clip(speed_start / speed_max, 0.0, 1.0))
        speed_end_norm = float(np.clip(speed_end / speed_max, 0.0, 1.0))
        if path.shape[0] >= 2:
            points = path.reshape(-1, 1, 3)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = plt.get_cmap("Reds") if is_head else plt.get_cmap(cmaps[uav_id % len(cmaps)])
            seg_colors = cmap(norm(times[:-1]))
            seg_colors[:, 3] = _segment_alpha_ramp(
                segs.shape[0],
                alpha_min=0.12 if is_head else 0.06,
                alpha_max=0.98 if is_head else 0.90,
            )
            line = Line3DCollection(segs, colors=seg_colors, linewidths=3.4 if is_head else 2.4)
            ax.add_collection3d(line)

            # Overlay a brighter recent trace ending at resolved_frame_idx.
            trace_end = min(resolved_frame_idx + 1, path.shape[0])
            trace_start = max(0, trace_end - recent_trace_steps)
            recent_path = path[trace_start:trace_end, :]
            if recent_path.shape[0] >= 2:
                # Wide low-alpha underlay creates a motion-glow effect near the latest positions.
                ax.plot(
                    recent_path[:, 0],
                    recent_path[:, 1],
                    recent_path[:, 2],
                    color=cmap(0.92),
                    linewidth=8.0 if is_head else 5.8,
                    alpha=0.13 if is_head else 0.10,
                    linestyle="-",
                )
                ax.plot(
                    recent_path[:, 0],
                    recent_path[:, 1],
                    recent_path[:, 2],
                    color=cmap(0.95),
                    linewidth=4.2 if is_head else 3.0,
                    alpha=0.98 if is_head else 0.94,
                    linestyle="-",
                )

            legend_label = "Head UAV" if is_head else f"Follower UAV {uav_id}"
            ax.plot([], [], [], color=cmap(0.8), linewidth=3.4 if is_head else 2.4, label=legend_label)

        ax.scatter(
            [path[0, 0]], [path[0, 1]], [path[0, 2]],
            s=(60 + 18 * speed_start_norm) if is_head else (42 + 14 * speed_start_norm),
            c="#FCE4EC" if is_head else "white",
            edgecolors="#C62828" if is_head else "black",
            linewidths=1.0 if is_head else 0.7,
        )
        ax.scatter(
            [path[-1, 0]], [path[-1, 1]], [path[-1, 2]],
            s=(76 + 26 * speed_end_norm) if is_head else (52 + 18 * speed_end_norm),
            c="#B71C1C" if is_head else "black",
            edgecolors="white", linewidths=0.8 if is_head else 0.6,
        )

    # Realism: shadow footprints and altitude stems at the selected frame.
    z_ground = float(space_min[2])
    frame_pos = uav_locs[resolved_frame_idx]
    frame_speeds = speeds[resolved_frame_idx]
    frame_speed_norm = np.clip(frame_speeds / speed_max, 0.0, 1.0)
    span_xy = np.maximum(space_max[:2] - space_min[:2], 1e-6)
    base_shadow_scale = 0.022 * float(max(span_xy[0], span_xy[1]))
    for uav_id in range(n_uavs):
        pos = frame_pos[uav_id]
        alt = max(0.0, float(pos[2] - z_ground))

        if t_steps > 1:
            prev_idx = max(0, resolved_frame_idx - 1)
            vel_xy = frame_pos[uav_id, :2] - uav_locs[prev_idx, uav_id, :2]
            if np.linalg.norm(vel_xy) < 1e-9 and resolved_frame_idx < t_steps - 1:
                vel_xy = uav_locs[resolved_frame_idx + 1, uav_id, :2] - frame_pos[uav_id, :2]
            heading = float(np.arctan2(vel_xy[1], vel_xy[0])) if np.linalg.norm(vel_xy) > 1e-9 else 0.0
        else:
            heading = 0.0

        major_r = base_shadow_scale * (0.85 + 0.75 * float(frame_speed_norm[uav_id]))
        minor_r = 0.60 * major_r
        shadow_xy = _ellipse_shadow_xy(
            center_x=float(pos[0]),
            center_y=float(pos[1]),
            major_radius=float(major_r),
            minor_radius=float(minor_r),
            angle_rad=heading,
        )
        shadow_xyz = np.column_stack([shadow_xy, np.full((shadow_xy.shape[0],), z_ground, dtype=float)])
        shadow_poly = Poly3DCollection(
            [shadow_xyz],
            facecolors="black",
            edgecolors="none",
            linewidths=0.0,
            alpha=0.15,
        )
        ax.add_collection3d(shadow_poly)

        ax.plot(
            [pos[0], pos[0]],
            [pos[1], pos[1]],
            [z_ground, pos[2]],
            color="black",
            alpha=min(0.22, 0.08 + 0.0015 * alt),
            linewidth=0.8,
            linestyle=":",
        )

    # Mark exact trajectory points with violation crosses instead of drawing spheres.
    collision_violation = np.zeros((t_steps, n_uavs), dtype=bool)
    connectivity_violation = np.zeros((t_steps, n_uavs), dtype=bool)

    for t_idx in range(t_steps):
        pos_t = uav_locs[t_idx]
        conn_t = connectivity[t_idx] if t_idx < connectivity.shape[0] else np.zeros((n_uavs, n_uavs), dtype=bool)

        # Connectivity violation: UAV disconnected from all others.
        for uav_id in range(n_uavs):
            if n_uavs > 1 and np.sum(conn_t[uav_id]) == 0:
                connectivity_violation[t_idx, uav_id] = True

        # Collision violation: UAV too close to any other UAV.
        for i in range(n_uavs):
            for j in range(i + 1, n_uavs):
                dist_ij = float(np.linalg.norm(pos_t[i] - pos_t[j]))
                if dist_ij < u2u_safe_dist:
                    collision_violation[t_idx, i] = True
                    collision_violation[t_idx, j] = True

    connectivity_plotted = False
    collision_plotted = False
    for uav_id in range(n_uavs):
        conn_idx = np.where(connectivity_violation[:, uav_id])[0]
        if conn_idx.size > 0:
            ax.scatter(
                uav_locs[conn_idx, uav_id, 0],
                uav_locs[conn_idx, uav_id, 1],
                uav_locs[conn_idx, uav_id, 2],
                marker="x",
                s=70,
                c="#FF8F00",
                linewidths=1.8,
                alpha=0.95,
                label="Connectivity Violation" if not connectivity_plotted else None,
            )
            connectivity_plotted = True

        coll_idx = np.where(collision_violation[:, uav_id])[0]
        if coll_idx.size > 0:
            ax.scatter(
                uav_locs[coll_idx, uav_id, 0],
                uav_locs[coll_idx, uav_id, 1],
                uav_locs[coll_idx, uav_id, 2],
                marker="x",
                s=70,
                c="#4A148C",
                linewidths=1.8,
                alpha=0.95,
                label="Collision Violation" if not collision_plotted else None,
            )
            collision_plotted = True

    # Draw current connectivity links and color them by link quality (distance ratio).
    conn_t = connectivity[resolved_frame_idx] if resolved_frame_idx < connectivity.shape[0] else np.zeros((n_uavs, n_uavs), dtype=bool)
    quality_cmap = plt.get_cmap("RdYlGn")
    for i in range(n_uavs):
        for j in range(i + 1, n_uavs):
            if not bool(conn_t[i, j]):
                continue
            pi = uav_locs[resolved_frame_idx, i, :]
            pj = uav_locs[resolved_frame_idx, j, :]
            dist_ij = float(np.linalg.norm(pi - pj))
            ratio = dist_ij / max(float(u2u_range), 1e-6)
            quality = float(np.clip(1.0 - ratio, 0.0, 1.0))
            link_color = quality_cmap(quality)
            ax.plot(
                [pi[0], pj[0]],
                [pi[1], pj[1]],
                [pi[2], pj[2]],
                color=link_color,
                linewidth=1.8,
                alpha=0.82,
            )

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("Greys")), ax=ax, pad=0.08)
    cbar.set_label("Time")

    ax.set_xlim(space_min[0], space_max[0])
    ax.set_ylim(space_min[1], space_max[1])
    ax.set_zlim(space_min[2], space_max[2])
    # Preserve true geometric proportions across axes so spherical shells do not
    # appear as ellipsoids due to unequal 3D axis scaling.
    span = np.maximum(space_max - space_min, 1e-6)
    ax.set_box_aspect((span[0], span[1], span[2]))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = f"UAV Rollout Trajectory (frame={resolved_frame_idx}/{t_steps - 1})"
    if visited_fraction is not None and visited_fraction.shape[0] == t_steps:
        title += f" | visited fraction={visited_fraction[resolved_frame_idx]:.3f}"
    ax.set_title(title)

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    if save_path:
        save_path_obj = Path(save_path)
        if save_path_obj.parent and not save_path_obj.parent.exists():
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_rollout_metrics(
    rollout_trajectory: Dict[str, Any],
    figsize: Sequence[float] = (16, 9),
    save_path: Optional[str] = None,
    show: bool = True,
    rolling_window: int = 20,
    percentile_band: Tuple[float, float] = (10.0, 90.0),
):
    """
    Plot rollout metrics:
      1) cumulative reward vs time
      2) reward per step vs time with rolling mean and percentile band
      3) total sensors visited vs time
      4) UAV pair distances vs time with rolling mean and percentile band
      5) active UAV-UAV links vs time
      6) minimum pairwise distance vs time with safety threshold
      7) UAV-UAV connectivity vs time

    Collision and connectivity violation windows are shaded as event ribbons.
    """
    times = np.asarray(rollout_trajectory["times"], dtype=float)
    reward_per_step = np.asarray(rollout_trajectory.get("reward_per_step", []), dtype=float)
    visited_mask = rollout_trajectory.get("visited_mask")
    if visited_mask is not None:
        visited_mask = np.asarray(visited_mask, dtype=bool)
    visited_fraction = rollout_trajectory.get("visited_fraction")
    if visited_fraction is not None:
        visited_fraction = np.asarray(visited_fraction, dtype=float)
    sensor_locs  = np.asarray(rollout_trajectory["sensor_locs"], dtype=float)
    uav_locs     = np.asarray(rollout_trajectory["uav_locs"], dtype=float)
    connectivity = np.asarray(rollout_trajectory["connectivity"], dtype=bool)
    u2u_range    = float(rollout_trajectory["u2u_range"])
    u2u_safe_dist = float(rollout_trajectory["u2u_safe_dist"])

    if times.ndim != 1:
        raise ValueError("times must be a 1D array.")
    if uav_locs.ndim != 3 or uav_locs.shape[-1] != 3:
        raise ValueError("uav_locs must have shape (T, U, 3).")

    n_time = times.shape[0]
    n_uavs = uav_locs.shape[1]
    n_sensors = sensor_locs.shape[0]
    rolling_window = max(1, int(rolling_window))
    pct_low, pct_high = float(percentile_band[0]), float(percentile_band[1])
    if pct_low > pct_high:
        pct_low, pct_high = pct_high, pct_low

    reward_times = times[1:1 + reward_per_step.shape[0]] if reward_per_step.shape[0] > 0 else np.array([], dtype=float)
    cumulative_reward = np.cumsum(reward_per_step) if reward_per_step.shape[0] > 0 else np.array([], dtype=float)
    reward_mean, reward_low, reward_high = (
        _rolling_mean_percentile(reward_per_step, rolling_window, pct_low, pct_high)
        if reward_per_step.shape[0] > 0
        else (np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float))
    )

    if visited_mask is not None and visited_mask.ndim == 2 and visited_mask.shape[0] == n_time:
        visited_count = visited_mask.sum(axis=1).astype(float)
    elif visited_fraction is not None and visited_fraction.shape[0] == n_time:
        visited_count = visited_fraction * float(n_sensors)
    else:
        visited_count = np.zeros(n_time, dtype=float)

    pairs, pair_distances = _pairwise_distance_matrix(uav_locs)
    min_pair_distance = (
        np.nanmin(pair_distances, axis=1)
        if pair_distances.shape[1] > 0
        else np.full(n_time, np.nan, dtype=float)
    )
    collision_violation = np.isfinite(min_pair_distance) & (min_pair_distance < u2u_safe_dist)
    connectivity_violation = _connectivity_violation_mask(
        connectivity,
        head_uav_idx=int(rollout_trajectory.get("head_uav_idx", 0)),
    )
    if connectivity_violation.shape[0] != n_time:
        connectivity_violation = np.zeros(n_time, dtype=bool)
    active_links = _active_link_count(connectivity)
    if active_links.shape[0] != n_time:
        active_links = np.zeros(n_time, dtype=float)

    reward_collision_violation = collision_violation[1:1 + reward_per_step.shape[0]]
    reward_connectivity_violation = connectivity_violation[1:1 + reward_per_step.shape[0]]

    # Ever-connected pairs keep pair traces readable; all pair distances feed the summary band.
    ever_connected = np.any(connectivity, axis=0)
    chain_pairs: List[Tuple[int, int]] = [
        (i, j)
        for i in range(n_uavs)
        for j in range(i + 1, n_uavs)
        if ever_connected[i, j]
    ]
    chain_pair_indices = [
        pair_idx
        for pair_idx, pair in enumerate(pairs)
        if pair in chain_pairs
    ]
    distance_summary_source = (
        pair_distances[:, chain_pair_indices]
        if len(chain_pair_indices) > 0
        else pair_distances
    )
    distance_mean, distance_low, distance_high = _rolling_mean_percentile(
        distance_summary_source,
        rolling_window,
        pct_low,
        pct_high,
    )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])

    _add_event_ribbons(ax1, times, collision_violation, connectivity_violation)
    ax1.plot(reward_times, cumulative_reward, color="#283593", linewidth=2.0)
    ax1.set_title("Cumulative Reward vs Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cumulative Reward")
    ax1.grid(True, alpha=0.3)

    _add_event_ribbons(ax2, reward_times, reward_collision_violation, reward_connectivity_violation)
    if reward_per_step.shape[0] > 0:
        ax2.plot(reward_times, reward_per_step, color="#80CBC4", linewidth=0.9, alpha=0.55, label="raw reward")
        ax2.fill_between(
            reward_times,
            reward_low,
            reward_high,
            color="#00897B",
            alpha=0.18,
            linewidth=0.0,
            label=f"rolling p{pct_low:.0f}-p{pct_high:.0f}",
        )
        ax2.plot(
            reward_times,
            reward_mean,
            color="#00695C",
            linewidth=2.2,
            label=f"rolling mean (w={rolling_window})",
        )
    ax2.set_title("Reward Per Step vs Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Step Reward")
    ax2.grid(True, alpha=0.3)
    _legend_if_labeled(ax2, loc="best", fontsize=8)

    _add_event_ribbons(ax3, times, collision_violation, connectivity_violation)
    ax3.plot(times, visited_count, color="#2E7D32", linewidth=2.0)
    ax3.set_title("Total Sensors Visited vs Time")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Visited Sensors")
    ax3.set_ylim(0.0, max(float(n_sensors), 1.0))
    ax3.grid(True, alpha=0.3)

    pair_colors = plt.cm.tab10(np.linspace(0, 1, max(len(chain_pairs), 1)))
    for color, (i, j) in zip(pair_colors, chain_pairs):
        dist_ij = np.linalg.norm(uav_locs[:, i, :] - uav_locs[:, j, :], axis=-1)
        # NaN where disconnected so matplotlib draws gaps instead of misleading dips
        dist_masked = np.where(connectivity[:, i, j], dist_ij, np.nan)
        # faint dotted line reveals the actual distance during disconnected gaps
        ax4.plot(times, dist_ij, linewidth=0.7, color=color, alpha=0.25, linestyle=":")
        ax4.plot(times, dist_masked, linewidth=1.8, color=color, label=f"UAV {i}\u2013{j}")

    _add_event_ribbons(ax4, times, collision_violation, connectivity_violation)
    if distance_summary_source.shape[1] > 0:
        ax4.fill_between(
            times,
            distance_low,
            distance_high,
            color="#455A64",
            alpha=0.16,
            linewidth=0.0,
            label=f"rolling p{pct_low:.0f}-p{pct_high:.0f}",
        )
        ax4.plot(
            times,
            distance_mean,
            color="#263238",
            linewidth=2.3,
            label=f"rolling mean (w={rolling_window})",
        )
    ax4.axhline(y=u2u_range,     color="#00695C", linestyle="--", linewidth=1.8, label="u2u_range")
    ax4.axhline(y=u2u_safe_dist, color="#BF360C", linestyle="--", linewidth=1.8, label="u2u_safe_dist")
    ax4.set_title("UAV Pair Distances vs Time")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Distance")
    ax4.grid(True, alpha=0.3)
    _legend_if_labeled(ax4, loc="best", fontsize=8)

    _add_event_ribbons(ax5, times, collision_violation, connectivity_violation)
    ax5.step(times, active_links, where="post", color="#1565C0", linewidth=2.1)
    ax5.set_title("Active Links Count vs Time")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Active Links")
    max_links = max(float(n_uavs * (n_uavs - 1) / 2), 1.0)
    ax5.set_ylim(-0.1, max_links + 0.5)
    ax5.grid(True, alpha=0.3)

    _add_event_ribbons(ax6, times, collision_violation, connectivity_violation)
    ax6.plot(times, min_pair_distance, color="#5D4037", linewidth=2.0, label="minimum pair distance")
    ax6.axhline(
        y=u2u_safe_dist,
        color="#BF360C",
        linestyle="--",
        linewidth=1.8,
        label="safety threshold",
    )
    ax6.set_title("Minimum Pairwise Distance vs Time")
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Distance")
    ax6.grid(True, alpha=0.3)
    _legend_if_labeled(ax6, loc="best", fontsize=8)

    # One binary step line per chain pair, using the same pairs and colors as ax4.
    for color, (i, j) in zip(pair_colors, chain_pairs):
        conn_ij = connectivity[:, i, j].astype(float)
        ax7.step(times, conn_ij, where="post", linewidth=1.8, color=color, label=f"UAV {i}\u2013{j}")

    _add_event_ribbons(ax7, times, collision_violation, connectivity_violation)
    ax7.set_yticks([0, 1])
    ax7.set_yticklabels(["Disconnected", "Connected"], fontsize=8)
    ax7.set_ylim(-0.1, 1.1)
    ax7.set_xlabel("Time")
    ax7.set_title("UAV-UAV Connectivity vs Time")
    _legend_if_labeled(ax7, loc="best", fontsize=8)
    ax7.grid(True, alpha=0.3)

    if save_path:
        save_path_obj = Path(save_path)
        if save_path_obj.parent and not save_path_obj.parent.exists():
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    return fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7)


def plot_direction_weights(
    rollout_trajectory: Dict[str, Any],
    uav_id: int,
    window: int = 20,
    figsize: Sequence[float] = (10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot windowed-mean direction weights (sensor vs upstream) for a follower UAV.
    """
    weights = rollout_trajectory.get("follower_direction_weights_raw")
    if weights is None:
        weights = rollout_trajectory.get("follower_direction_weights")
    follower_uav_ids = rollout_trajectory.get("follower_uav_ids", [])
    times = np.asarray(rollout_trajectory["times"], dtype=float)

    if weights is None or len(weights) == 0:
        print(f"[plot_direction_weights] no direction_weights in trajectory; skipping.")
        return None, None

    weights = np.asarray(weights, dtype=float)  # (T, F, 2)

    # map global uav_id to follower index
    if uav_id in follower_uav_ids:
        f_idx = list(follower_uav_ids).index(uav_id)
    else:
        f_idx = uav_id - 1  # fallback: assume followers are 1..n
        f_idx = max(0, min(f_idx, weights.shape[1] - 1))
        print(f"[plot_direction_weights] UAV {uav_id} not in follower_uav_ids {follower_uav_ids}; using follower index {f_idx}.")

    w = weights[:, f_idx, :]  # (T, 2)
    sensor_w = w[:, 0]
    upstream_w = w[:, 1]

    # aligned time axis: weights are recorded after env.step so start from index 1
    t = times[1 : 1 + len(sensor_w)] if len(times) > len(sensor_w) else times[: len(sensor_w)]

    def _moving_average(x, k):
        if k <= 1:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="valid")

    sensor_smooth = _moving_average(sensor_w, window)
    upstream_smooth = _moving_average(upstream_w, window)
    t_smooth = t[window - 1 :] if window > 1 else t

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, sensor_w,   color="#B0BEC5", linewidth=0.8, alpha=0.5, label="_raw sensor")
    ax.plot(t, upstream_w, color="#FFCC80", linewidth=0.8, alpha=0.5, label="_raw upstream")
    ax.plot(t_smooth, sensor_smooth,   color="#1565C0", linewidth=2.0, label=f"sensor weight (w={window})")
    ax.plot(t_smooth, upstream_smooth, color="#E65100", linewidth=2.0, label=f"upstream weight (w={window})")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Weight")
    ax.set_title(f"Direction Weights — UAV {uav_id} (follower index {f_idx})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_connected_neighbor_distances(
    rollout_trajectory: Dict[str, Any],
    figsize: Sequence[float] = (10, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    uav_locs     = np.asarray(rollout_trajectory["uav_locs"], dtype=float)       # (T, U, 3)
    connectivity = np.asarray(rollout_trajectory["connectivity"], dtype=bool)    # (T, U, U)
    times        = np.asarray(rollout_trajectory["times"], dtype=float)          # (T,)
    u2u_range    = float(rollout_trajectory["u2u_range"])
    u2u_safe_dist = float(rollout_trajectory["u2u_safe_dist"])

    n_uavs = uav_locs.shape[1]

    # all upper-triangle pairs so cross-chain distances are visible alongside chain pairs
    pairs: List[Tuple[int, int]] = [
        (i, j)
        for i in range(n_uavs)
        for j in range(i + 1, n_uavs)
    ]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 1)))
    for color, (i, j) in zip(colors, pairs):
        dist = np.linalg.norm(uav_locs[:, i, :] - uav_locs[:, j, :], axis=-1)
        # NaN where disconnected so matplotlib draws gaps instead of misleading dips
        dist_masked = np.where(connectivity[:, i, j], dist, np.nan)
        # faint dotted line reveals the actual distance during disconnected gaps
        ax.plot(times, dist, linewidth=0.7, color=color, alpha=0.25, linestyle=":")
        ax.plot(times, dist_masked, linewidth=1.8, color=color, label=f"UAV {i}\u2013{j}")

    ax.axhline(y=u2u_range,     color="#00695C", linestyle="--", linewidth=1.6, label="u2u_range")
    ax.axhline(y=u2u_safe_dist, color="#BF360C", linestyle="--", linewidth=1.6, label="u2u_safe_dist")

    ax.set_title("Connected UAV Pair Distances vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path_obj = Path(save_path)
        if save_path_obj.parent and not save_path_obj.parent.exists():
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def animate_rollout_trajectory_3d(
    rollout_trajectory: Dict[str, Any],
    frame_stride: int = 1,
    interval_ms: int = 120,
    figsize: Sequence[float] = (11, 9),
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 160,
):
    """
    Animate MAPPO rollout trajectories in 3D and optionally save as GIF.

    Notes:
    - Uses PillowWriter to avoid ffmpeg dependency.
    - If save_path suffix is not .gif, the function saves with .gif suffix.
    """
    uav_locs = np.asarray(rollout_trajectory["uav_locs"], dtype=float)
    times = np.asarray(rollout_trajectory["times"], dtype=float)
    connectivity = np.asarray(rollout_trajectory["connectivity"], dtype=bool)
    sensor_locs = np.asarray(rollout_trajectory["sensor_locs"], dtype=float)
    base_station = np.asarray(rollout_trajectory["base_station"], dtype=float)
    space_min = np.asarray(rollout_trajectory["space_min"], dtype=float)
    space_max = np.asarray(rollout_trajectory["space_max"], dtype=float)
    visited_fraction = rollout_trajectory.get("visited_fraction")
    if visited_fraction is not None:
        visited_fraction = np.asarray(visited_fraction, dtype=float)
    visited_mask = rollout_trajectory.get("visited_mask")
    if visited_mask is not None:
        visited_mask = np.asarray(visited_mask, dtype=bool)

    if uav_locs.ndim != 3 or uav_locs.shape[-1] != 3:
        raise ValueError("uav_locs must have shape (T, U, 3).")
    if times.ndim != 1 or times.shape[0] != uav_locs.shape[0]:
        raise ValueError("times must have shape (T,) and match uav_locs length.")

    t_steps, n_uavs, _ = uav_locs.shape
    speeds = _compute_uav_speeds(uav_locs, times)
    speed_max = max(float(np.max(speeds)), 1e-6)
    head_uav_idx = int(rollout_trajectory.get("head_uav_idx", 0))
    head_uav_idx = max(0, min(head_uav_idx, n_uavs - 1))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _draw_ground_plane(ax, space_min=space_min, space_max=space_max)
    norm = plt.Normalize(
        vmin=float(times.min()),
        vmax=float(times.max()) if float(times.max()) > float(times.min()) else float(times.min()) + 1.0,
    )

    ax.scatter(
        [base_station[0]], [base_station[1]], [base_station[2]],
        marker="*", s=220, c="#B71C1C", edgecolors="black", linewidths=0.7, label="Base Station"
    )
    sensor_scatter = ax.scatter(
        sensor_locs[:, 0], sensor_locs[:, 1], sensor_locs[:, 2],
        marker="^", s=45, c="#C62828", alpha=0.9, label="Sensors"
    )

    trail_collections = []
    marker_points = []
    glow_lines = []
    shadow_polys = []
    altitude_stems = []
    cmaps = ["Blues", "Oranges", "Greens", "Purples", "Reds", "Greys"]
    for uav_id in range(n_uavs):
        is_head = (uav_id == head_uav_idx)
        cmap = plt.get_cmap("Reds") if is_head else plt.get_cmap(cmaps[uav_id % len(cmaps)])
        color = cmap(0.8)
        label = "Head UAV" if is_head else f"Follower UAV {uav_id}"

        # Add a legend handle while rendering trajectory as per-segment collection.
        ax.plot([], [], [], color=color, linewidth=3.2 if is_head else 2.2, alpha=0.9, label=label)

        p0 = uav_locs[0, uav_id, :]
        seed_seg = np.asarray([[p0, p0]], dtype=float)
        trail_collection = Line3DCollection(seed_seg, linewidths=3.2 if is_head else 2.2)
        trail_collection.set_color([(0.0, 0.0, 0.0, 0.0)])
        ax.add_collection3d(trail_collection)

        glow_line, = ax.plot(
            [],
            [],
            [],
            color=cmap(0.92),
            linewidth=7.2 if is_head else 5.2,
            alpha=0.0,
        )

        marker, = ax.plot([], [], [], marker="o", linestyle="None", markersize=7 if is_head else 5, color=color)
        seed_poly = np.array([[0.0, 0.0, float(space_min[2])]] * 3, dtype=float)
        shadow_poly = Poly3DCollection(
            [seed_poly],
            facecolors="black",
            edgecolors="none",
            linewidths=0.0,
            alpha=0.0,
        )
        ax.add_collection3d(shadow_poly)
        stem, = ax.plot([], [], [], color="black", linewidth=0.8, alpha=0.12, linestyle=":")
        trail_collections.append((trail_collection, cmap, is_head))
        marker_points.append(marker)
        glow_lines.append(glow_line)
        shadow_polys.append(shadow_poly)
        altitude_stems.append(stem)

    pair_lines = {}
    quality_cmap = plt.get_cmap("RdYlGn")
    for i in range(n_uavs):
        for j in range(i + 1, n_uavs):
            conn_line, = ax.plot([], [], [], color="#FFB300", linewidth=1.3, alpha=0.75)
            conn_line.set_visible(False)
            pair_lines[(i, j)] = conn_line

    ax.set_xlim(space_min[0], space_max[0])
    ax.set_ylim(space_min[1], space_max[1])
    ax.set_zlim(space_min[2], space_max[2])
    span = np.maximum(space_max - space_min, 1e-6)
    ax.set_box_aspect((span[0], span[1], span[2]))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    frame_stride = max(1, int(frame_stride))
    frame_indices = list(range(0, t_steps, frame_stride))
    if frame_indices[-1] != (t_steps - 1):
        frame_indices.append(t_steps - 1)
    recent_window = 14
    span_xy = np.maximum(space_max[:2] - space_min[:2], 1e-6)
    base_shadow_scale = 0.022 * float(max(span_xy[0], span_xy[1]))

    def _update(frame_id: int):
        t_idx = frame_indices[frame_id]

        # Slow cinematic camera orbit with gentle elevation breathing.
        if t_steps > 1:
            phase = float(t_idx) / float(t_steps - 1)
        else:
            phase = 0.0
        phase_slow = 0.5 * phase
        ax.view_init(elev=22.0 + 4.0 * np.sin(2.0 * np.pi * phase_slow), azim=36.0 + 72.0 * phase_slow)

        if visited_mask is not None and visited_mask.ndim == 2 and visited_mask.shape[0] == t_steps and visited_mask.shape[1] == sensor_locs.shape[0]:
            is_visited = visited_mask[t_idx]
            sensor_colors = np.where(is_visited, "#2E7D32", "#C62828")
            sensor_scatter.set_color(sensor_colors)

        for uav_id in range(n_uavs):
            path = uav_locs[: t_idx + 1, uav_id, :]
            trail_collection, cmap, is_head = trail_collections[uav_id]
            if path.shape[0] >= 2:
                points = path.reshape(-1, 1, 3)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_colors = cmap(norm(times[:t_idx]))
                seg_colors[:, 3] = _segment_alpha_ramp(
                    segs.shape[0],
                    alpha_min=0.12 if is_head else 0.06,
                    alpha_max=0.98 if is_head else 0.90,
                )
                trail_collection.set_segments(segs)
                trail_collection.set_color(seg_colors)
            else:
                trail_collection.set_segments([])

            cur = uav_locs[t_idx, uav_id, :]
            marker_points[uav_id].set_data([cur[0]], [cur[1]])
            marker_points[uav_id].set_3d_properties([cur[2]])
            speed_norm = float(np.clip(speeds[t_idx, uav_id] / speed_max, 0.0, 1.0))
            marker_points[uav_id].set_markersize((7.0 + 2.0 * speed_norm) if is_head else (5.0 + 1.8 * speed_norm))

            # Motion glow around most recent segment window.
            glow_start = max(0, t_idx - recent_window)
            glow_path = uav_locs[glow_start : t_idx + 1, uav_id, :]
            if glow_path.shape[0] >= 2:
                glow_lines[uav_id].set_data(glow_path[:, 0], glow_path[:, 1])
                glow_lines[uav_id].set_3d_properties(glow_path[:, 2])
                glow_lines[uav_id].set_alpha(0.16 if is_head else 0.12)
            else:
                glow_lines[uav_id].set_data([], [])
                glow_lines[uav_id].set_3d_properties([])
                glow_lines[uav_id].set_alpha(0.0)

            # Shadow footprint and altitude stem.
            z_ground = float(space_min[2])
            if t_steps > 1:
                prev_idx = max(0, t_idx - 1)
                vel_xy = uav_locs[t_idx, uav_id, :2] - uav_locs[prev_idx, uav_id, :2]
                if np.linalg.norm(vel_xy) < 1e-9 and t_idx < t_steps - 1:
                    vel_xy = uav_locs[t_idx + 1, uav_id, :2] - uav_locs[t_idx, uav_id, :2]
                heading = float(np.arctan2(vel_xy[1], vel_xy[0])) if np.linalg.norm(vel_xy) > 1e-9 else 0.0
            else:
                heading = 0.0

            major_r = base_shadow_scale * (0.85 + 0.75 * speed_norm)
            minor_r = 0.60 * major_r
            shadow_xy = _ellipse_shadow_xy(
                center_x=float(cur[0]),
                center_y=float(cur[1]),
                major_radius=float(major_r),
                minor_radius=float(minor_r),
                angle_rad=heading,
            )
            shadow_xyz = np.column_stack([shadow_xy, np.full((shadow_xy.shape[0],), z_ground, dtype=float)])
            shadow_polys[uav_id].set_verts([shadow_xyz])
            shadow_polys[uav_id].set_alpha(0.14)

            altitude_stems[uav_id].set_data([cur[0], cur[0]], [cur[1], cur[1]])
            altitude_stems[uav_id].set_3d_properties([z_ground, cur[2]])
            altitude_stems[uav_id].set_alpha(min(0.22, 0.08 + 0.0015 * max(0.0, cur[2] - z_ground)))

        conn_t = connectivity[t_idx] if t_idx < connectivity.shape[0] else np.zeros((n_uavs, n_uavs), dtype=bool)
        for (i, j), line in pair_lines.items():
            if conn_t[i, j]:
                pi = uav_locs[t_idx, i, :]
                pj = uav_locs[t_idx, j, :]
                dist_ij = float(np.linalg.norm(pi - pj))
                ratio = dist_ij / max(float(rollout_trajectory["u2u_range"]), 1e-6)
                quality = float(np.clip(1.0 - ratio, 0.0, 1.0))
                line.set_data([pi[0], pj[0]], [pi[1], pj[1]])
                line.set_3d_properties([pi[2], pj[2]])
                line.set_color(quality_cmap(quality))
                line.set_linewidth(1.1 + 1.7 * quality)
                line.set_alpha(0.35 + 0.55 * quality)
                line.set_visible(True)
            else:
                line.set_visible(False)

        title = f"UAV Rollout Animation (frame={t_idx}/{t_steps - 1}, time={times[t_idx]:.2f})"
        if visited_fraction is not None and visited_fraction.shape[0] == t_steps:
            title += f" | visited fraction={visited_fraction[t_idx]:.3f}"
        ax.set_title(title)

        return [c[0] for c in trail_collections] + glow_lines + marker_points + shadow_polys + altitude_stems + list(pair_lines.values()) + [sensor_scatter]

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        interval=max(10, int(interval_ms)),
        blit=False,
        repeat=False,
    )

    saved_path = None
    if save_path:
        save_path_obj = Path(save_path)
        if save_path_obj.suffix.lower() != ".gif":
            save_path_obj = save_path_obj.with_suffix(".gif")
        if save_path_obj.parent and not save_path_obj.parent.exists():
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        fps = max(1, int(round(1000.0 / max(1, int(interval_ms)))))
        anim.save(str(save_path_obj), writer=PillowWriter(fps=fps), dpi=dpi)
        saved_path = str(save_path_obj)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, anim, saved_path
