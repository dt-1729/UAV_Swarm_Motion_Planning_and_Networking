# Copyright © MERL 2026
# Description: Entry point for multi-agent PPO training, evaluation, rollout, and plotting.
import swarm_env as se
import numpy as np
import torch
import follower_actor as fa
import random
from mappo_env_interface import build_trainer
from datetime import datetime
from pathlib import Path
import pandas as pd
from rollout_visualization import animate_rollout_trajectory_3d, plot_rollout_metrics, visualize_rollout_trajectory_3d
from mappo_env_interface import obs_adapter, action_adapter, reset_fn
from cli_args import parse_main_args
from training_utils import (
    build_results_run_dir,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    find_latest_training_metrics,
    plot_training_results,
    build_tensorboard_logger,
)

n_obstacles = 0
args = parse_main_args()
n_uavs = args.n_uavs
n_sensors = args.n_sensors
n_chains = args.n_chains
run_mode = args.run_mode
eval_checkpoint_path = args.eval_checkpoint_path
finetune_checkpoint_path = args.finetune_checkpoint_path
checkpoint_every_updates = args.checkpoint_every_updates
plot_csv_path = args.plot_csv_path
plot_output_path = args.plot_output_path
smoothing_window = args.smoothing_window
show_variance = bool(args.show_variance)
anim_frame_stride = args.anim_frame_stride
anim_interval_ms = args.anim_interval_ms
direction_head_temperature = args.direction_head_temperature
direction_head_temperature_final = args.direction_head_temperature_final
temperature_schedule = args.temperature_schedule
rollout_steps = args.rollout_steps
max_time = args.max_time
epochs = args.epochs
minibatch_size = args.minibatch_size
use_mean_direction = args.use_mean_direction
cbf_connectivity_neighbor_mode = args.cbf_connectivity_neighbor_mode
sensor_init_mode = args.sensor_init_mode
n_sensor_clusters = args.n_sensor_clusters

print(f"CBF CONNECTIVITY NEIGHBOR MODE: {cbf_connectivity_neighbor_mode}")
print(f"SENSOR INIT MODE: {sensor_init_mode} (clusters={n_sensor_clusters})")
print(f"ENV SIZE: n_uavs={n_uavs}, n_sensors={n_sensors}")
print(f"CHAIN CONFIG: n_chains={n_chains}")

# Build the UAV swarm environment with specified parameters.
env = se.UAVSwarmEnv3D(
    n_uavs=n_uavs,
    n_sensors=n_sensors,
    n_obstacles=n_obstacles,
    box_size=1000.0,
    min_uav_height=2.0,
    com_params={
        "u2u_range": 10.0,
        "u2u_safe_dist": 2.0,
        "u2s_range": 5.0,
        "u2b_range": 20.0,
        "u2s_visibility_range": 100.0,
    },
    simulation_params={"init_time": 0.0, "dt": 0.1, "max_time": max_time, "max_energy": 1.0e5},
    reward_params={
        "sensor_collect_coeff": 1000.0*np.ones(n_uavs, dtype=np.float32),
        "potential_diff_coeff": 1.0*np.ones(n_uavs, dtype=np.float32),
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
    sensor_sequence_strategy="tsp",
    sensor_init_mode=sensor_init_mode,
    n_sensor_clusters=n_sensor_clusters,
    n_chains=n_chains,
    connectivity_update_freq=10000,
)
env.cbf_connectivity_neighbor_mode = cbf_connectivity_neighbor_mode
env.set_head_motion(args.head_motion, n_head_clusters=args.head_n_clusters)

# Build the MAPPO trainer and initialize the environment.
trainer, initial_obs = build_trainer(
    env,
    rollout_length=rollout_steps,
    minibatch_size=minibatch_size,
    ppo_epochs=8,
    direction_head_temperature=direction_head_temperature,
    use_mean_direction=use_mean_direction,
)
if run_mode == "sim":
    print("Simulating the swarm...")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rollout_dir = Path("rollout_visualizations") / (
        f"sim_n_sensors_{env.n_sensors}_n_uavs_{env.n_uavs}_head_{args.head_motion}_{run_stamp}"
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)

    if eval_checkpoint_path:
        model_path = Path(eval_checkpoint_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        print(f"Simulating the swarm with trained follower model: {model_path}")
    else:
        model_path = rollout_dir / "random_follower_model.pt"
        random_actor = fa.Actor(sensor_feature_dim=7, upstream_feature_dim=14)
        torch.save({"follower_actor_state_dict": random_actor.state_dict()}, model_path)
        print("No checkpoint given (-ec); using a randomly initialized follower model.")

    rollout_trajectory = se.simulate(
        env,
        str(model_path),
        use_mean_direction=use_mean_direction,
        direction_temperature=direction_head_temperature,
    )
    print(
        f"Simulated {len(rollout_trajectory['reward_per_step'])} steps "
        f"(terminated={rollout_trajectory['terminated']}, truncated={rollout_trajectory['truncated']})"
    )

    if rollout_trajectory is not None:
        trajectory_plot_path = rollout_dir / "rollout_trajectory_3d.png"
        trajectory_anim_path = rollout_dir / "rollout_trajectory_3d.gif"
        metrics_plot_path = rollout_dir / "rollout_metrics.png"

        visualize_rollout_trajectory_3d(
            rollout_trajectory=rollout_trajectory,
            frame_idx=-1,
            show_spheres=False,
            save_path=str(trajectory_plot_path),
            show=True,
        )
        plot_rollout_metrics(
            rollout_trajectory=rollout_trajectory,
            save_path=str(metrics_plot_path),
            show=True,
        )
        _, _, _, saved_anim_path = animate_rollout_trajectory_3d(
            rollout_trajectory=rollout_trajectory,
            frame_stride=anim_frame_stride,
            interval_ms=anim_interval_ms,
            save_path=str(trajectory_anim_path),
            show=False,
        )

        if saved_anim_path is not None:
            print(f"Saved rollout animation to: {saved_anim_path}")

        print(f"Saved rollout plots to: {rollout_dir}")

elif run_mode == "train":
    print(
        f"Starting follower training with head motion '{args.head_motion}', "
        f"rollout_steps={rollout_steps}, epochs={epochs}, minibatch_size={minibatch_size}, "
        f"max_time={max_time}..."
    )
    if finetune_checkpoint_path is not None:
        finetune_path = Path(finetune_checkpoint_path)
        if not finetune_path.exists():
            raise FileNotFoundError(f"Finetune checkpoint not found: {finetune_path}")
        load_checkpoint(trainer, finetune_path)
        print(f"Loaded weights from checkpoint: {finetune_path}")
    run_dir = build_results_run_dir(env, head_motion=args.head_motion)
    tb_dir = run_dir / "tensorboard"
    live_training_csv_path = run_dir / "training.csv"
    if live_training_csv_path.exists():
        live_training_csv_path.unlink()
    # Create CSV immediately so it exists before rollout collection starts.
    live_training_csv_path.touch()
    writer, tb_log_callback = build_tensorboard_logger(tb_dir)

    def _train_log_callback(logs: dict) -> None:
        tb_log_callback(logs)

        # Persist each update as it arrives so progress is available before training ends.
        write_header = (not live_training_csv_path.exists()) or (live_training_csv_path.stat().st_size == 0)
        pd.DataFrame([logs]).to_csv(
            live_training_csv_path,
            mode="a",
            header=write_header,
            index=False,
        )

        if checkpoint_every_updates <= 0:
            return

        update_idx = int(logs.get("update", 0))
        if update_idx > 0 and (update_idx % checkpoint_every_updates == 0):
            periodic_ckpt_path = run_dir / f"checkpoints/model_update_{update_idx:06d}.pt"
            periodic_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(trainer, periodic_ckpt_path)
            print(f"[checkpoint] saved update={update_idx} path={periodic_ckpt_path}")

    training_logs = trainer.train_ppo(
        env=env,
        obs_adapter=obs_adapter,
        action_adapter=action_adapter,
        num_updates=epochs,
        reset_fn=reset_fn,
        initial_obs=initial_obs,
        stop_on_termination=False,
        rollout_progress=False,
        rollout_progress_every=1,
        log_interval=1,
        log_callback=_train_log_callback,
        direction_temperature_initial=direction_head_temperature,
        direction_temperature_final=direction_head_temperature_final,
        temperature_schedule=temperature_schedule,
    )

    writer.flush()
    writer.close()

    training_df = pd.DataFrame(training_logs)
    training_csv_path = run_dir / "training_metrics.csv"
    training_df.to_csv(training_csv_path, index=False)

    final_plot_path = run_dir / "training_summary.png"
    plot_training_results(
        training_df,
        final_plot_path,
        smoothing_window=smoothing_window,
        show_variance=show_variance,
    )

    checkpoint_path = run_dir / "model_final.pt"
    save_checkpoint(trainer, checkpoint_path)

    print(f"Saved training artifacts to: {run_dir}")
    print(f"Live per-update CSV: {live_training_csv_path}")
    print(f"TensorBoard log dir: {tb_dir}")
    print(f"Saved model checkpoint: {checkpoint_path}")

elif run_mode == "eval":
    checkpoint_path = Path(eval_checkpoint_path) if eval_checkpoint_path else find_latest_checkpoint(Path("training_results"))
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(
            "No checkpoint found. Set eval_checkpoint_path explicitly or train first to create model_final.pt."
        )

    load_checkpoint(trainer, checkpoint_path)

    checkpoint_name = checkpoint_path.stem
    model_folder_name = checkpoint_path.parent.name
    if model_folder_name == "checkpoints" and checkpoint_path.parent.parent is not None:
        model_folder_name = checkpoint_path.parent.parent.name

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path("rollout_visualizations") / (
        f"eval_{model_folder_name}_{checkpoint_name}_"
        f"n_sensors_{env.n_sensors}_n_uavs_{env.n_uavs}_head_{args.head_motion}_{run_stamp}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running follower evaluation with head motion '{args.head_motion}'")
    metrics = trainer.collect_rollout(
        env=env,
        obs_adapter=obs_adapter,
        action_adapter=action_adapter,
        reset_fn=reset_fn,
        initial_obs=initial_obs,
        stop_on_termination=True,
        rollout_progress=True,
        rollout_progress_every=1,
        record_rollout_trajectory=True,
    )

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(eval_dir / "eval_metrics.csv", index=False)

    rollout_trajectory = metrics.get("rollout_trajectory")
    if rollout_trajectory is not None:
        trajectory_plot_path = eval_dir / "rollout_trajectory_3d.png"
        trajectory_anim_path = eval_dir / "rollout_trajectory_3d.gif"
        metrics_plot_path = eval_dir / "rollout_metrics.png"

        visualize_rollout_trajectory_3d(
            rollout_trajectory=rollout_trajectory,
            frame_idx=-1,
            show_spheres=False,
            save_path=str(trajectory_plot_path),
            show=True,
        )
        plot_rollout_metrics(
            rollout_trajectory=rollout_trajectory,
            save_path=str(metrics_plot_path),
            show=True,
        )
        _, _, _, saved_anim_path = animate_rollout_trajectory_3d(
            rollout_trajectory=rollout_trajectory,
            frame_stride=anim_frame_stride,
            interval_ms=anim_interval_ms,
            save_path=str(trajectory_anim_path),
            show=False,
        )

        if saved_anim_path is not None:
            print(f"Saved evaluation animation to: {saved_anim_path}")

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved evaluation artifacts to: {eval_dir}")

elif run_mode == "plot":
    csv_path = Path(plot_csv_path) if plot_csv_path else find_latest_training_metrics(Path("training_results"))
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "No training metrics CSV found. Provide --plot-csv-path or run training first."
        )

    training_df = pd.read_csv(csv_path)
    if "update" not in training_df.columns:
        raise ValueError(f"CSV at {csv_path} does not contain required 'update' column.")

    if plot_output_path is not None:
        output_path = Path(plot_output_path)
    else:
        suffix = f"_ma{smoothing_window}" if smoothing_window > 1 else ""
        output_path = csv_path.parent / f"training_summary_replot{suffix}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training_results(
        training_df,
        output_path,
        smoothing_window=smoothing_window,
        show_variance=show_variance,
    )

    print(f"Loaded training metrics: {csv_path}")
    print(f"Saved post-training plot: {output_path}")
