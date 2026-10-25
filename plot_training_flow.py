# Copyright © MERL 2026
# Description: Draws a schematic of the multi-agent PPO training workflow.
"""
Generates a figure illustrating the MAPPO training flow for the UAV swarm system.
Run:  python plot_training_flow.py
Output: training_flow.png  (in the same directory)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
C_ENV   = "#4A90D9"   # blue   – environment
C_OBS   = "#7B68EE"   # medium-slate – adapter
C_ACT   = "#E8A838"   # orange – actors
C_CRIT  = "#E05C5C"   # red    – critic
C_BUF   = "#4CAF7D"   # green  – buffer / GAE
C_PPO   = "#9C59B6"   # purple – PPO update
C_SCHED = "#78909C"   # grey   – scheduler
C_LOOP  = "#37474F"   # dark   – loop annotations
BG      = "#F8F9FA"

fig = plt.figure(figsize=(20, 13), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.set_facecolor(BG)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, color, text, fontsize=9, text_color="white",
         alpha=1.0, style="round,pad=0.1", bold=False):
    """Draw a rounded box with centred text."""
    fc = (*matplotlib.colors.to_rgb(color), alpha)
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=style,
                         linewidth=1.2,
                         edgecolor="#333333",
                         facecolor=fc,
                         zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight=weight, zorder=4,
            wrap=True)

def arrow(ax, x1, y1, x2, y2, color="#555555", label="", lw=1.5,
          arrowstyle="-|>", label_side="top"):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=arrowstyle,
                                color=color,
                                lw=lw,
                                mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dy = 0.18 if label_side == "top" else -0.22
        ax.text(mx, my + dy, label,
                ha="center", va="center",
                fontsize=7.5, color=color, style="italic", zorder=5)

def section_bg(ax, x, y, w, h, color, title, title_fs=9):
    """Draw a lightly shaded section background with a title."""
    bg = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.05",
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor=(*matplotlib.colors.to_rgb(color), 0.07),
                        zorder=1, linestyle="--")
    ax.add_patch(bg)
    ax.text(x + w / 2, y + h + 0.05, title,
            ha="center", va="bottom",
            fontsize=title_fs, color=color, fontweight="bold", zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(10, 12.55, "MAPPO Training Flow — UAV Swarm Coverage",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=C_LOOP)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════════════════
# Left column: rollout phase
section_bg(ax, 0.25, 1.3, 9.5, 10.7, C_ENV, "① Rollout Collection  (T = 512 timesteps / update)", 9)
# Right column: PPO update phase
section_bg(ax, 10.25, 1.3, 9.5, 10.7, C_PPO, "② PPO Update  (8 epochs × minibatches of 256)", 9)

# ═══════════════════════════════════════════════════════════════════════════════
# ── LEFT COLUMN: ROLLOUT ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# [A] Temperature scheduler
rbox(ax, 0.55, 10.8, 3.6, 0.72, C_SCHED,
     "Temperature Scheduler\n(sensor & direction sampling τ)", fontsize=8)

# [B] Environment
rbox(ax, 0.55, 9.4, 3.6, 0.92, C_ENV,
     "UAV Swarm Environment\n(3-D, sensors, energy, time)",
     fontsize=8.5, bold=True)

# [C] Observation Adapter
rbox(ax, 0.55, 7.85, 3.6, 0.92, C_OBS,
     "obs_adapter()\nraw state → follower obs, critic obs",
     fontsize=7.8)

# [D] Follower Actor  (shared weights, F followers)
rbox(ax, 0.55, 5.75, 3.6, 1.5, C_ACT,
     "Follower Actors  (×F, shared)\n──────────────\n"
     "Transformer + attention pooling\n"
     "over nearby & upstream sensors\n"
     "→ Direction (VonMises3D)",
     fontsize=7.8)

# [E] Central Critic
rbox(ax, 0.55, 3.8, 3.6, 1.1, C_CRIT,
     "Central Critic\n──────────────\n"
     "Follower state → V(s)",
     fontsize=7.8)

# [G] action_adapter  (small inline)
rbox(ax, 4.6, 6.6, 1.9, 0.55, C_OBS,
     "action_adapter()\n→ env actions", fontsize=7.5)

# ── arrows left column ──────────────────────────────────────────────────────
arrow(ax, 2.35, 10.8, 2.35, 10.32, C_SCHED, "schedule τ")           # sched → env
arrow(ax, 2.35, 9.4,  2.35, 8.77,  C_ENV,   "raw state")            # env → obs adapter
arrow(ax, 2.35, 7.85, 2.35, 7.25,  C_OBS,   "follower obs",
      label_side="top")
arrow(ax, 2.35, 7.85, 2.35, 5.30,  C_OBS,   "critic obs",
      label_side="top")

# follower → action_adapter
arrow(ax, 4.15, 6.5, 5.2,  6.6,   C_ACT,
      "dir actions", label_side="top")

# action_adapter → env (feedback)
arrow(ax, 5.5, 6.6, 5.5, 9.86, C_OBS, "env_actions")
arrow(ax, 5.5, 9.86, 4.15, 9.86, C_OBS, "")

# critic → buffer (via a label)
arrow(ax, 2.35, 3.8, 2.35, 2.2, C_CRIT, "V(s_t)")

# ── Rollout Buffer ─────────────────────────────────────────────────────────
rbox(ax, 0.55, 1.35, 3.6, 0.85, C_BUF,
     "Rollout Buffer\n(obs, actions, log_probs, rewards, dones, V(s))",
     fontsize=7.5, bold=True)

# reward back from env to buffer (bypass)
ax.annotate("", xy=(0.55, 1.77), xytext=(0.4, 9.86),
            arrowprops=dict(arrowstyle="-|>", color=C_ENV, lw=1.4,
                            connectionstyle="arc3,rad=0.0",
                            mutation_scale=14), zorder=2)
ax.plot([0.4, 0.4], [1.77, 9.86], color=C_ENV, lw=1.4, zorder=2)
ax.text(0.15, 5.8, "reward\n  done", ha="center", va="center",
        fontsize=7, color=C_ENV, rotation=90)

# ── GAE (connects buffer to right column) ──────────────────────────────────
rbox(ax, 3.75, 1.35, 6.0, 0.85, C_BUF,
     "GAE  (γ=0.99, λ=0.95) → returns & advantages   [backward in time]",
     fontsize=8, bold=True)
arrow(ax, 4.15, 1.77, 3.75, 1.77, C_BUF, "")

# ═══════════════════════════════════════════════════════════════════════════════
# ── RIGHT COLUMN: PPO UPDATE ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# [H] Minibatch sampler
rbox(ax, 10.55, 9.9, 8.9, 0.72, C_PPO,
     "Shuffle & iterate minibatches  (size 256)  ×  8 PPO epochs",
     fontsize=8.5, bold=True)

# [I] Re-evaluate follower
rbox(ax, 10.55, 8.4, 4.1, 1.1, C_ACT,
     "Re-evaluate Follower Actions\n──────────────\n"
     "→ new log_prob  (aggregate F followers)\n"
     "→ entropy",
     fontsize=7.8)

# [K] PPO Clipping Loss
rbox(ax, 10.55, 6.5, 4.1, 1.5, C_PPO,
     "PPO Clipping Loss\n──────────────\n"
     "ratio = exp(Δ log_prob)\n"
     "surr = min(ratio·A, clip(ratio,1±ε)·A)\n"
     "L_policy = –surr – α_H · entropy",
     fontsize=7.8)

# [L] Value Loss
rbox(ax, 15.05, 6.5, 4.1, 1.1, C_CRIT,
     "Value Loss\n──────────────\n"
     "Re-predict V(s)  via Central Critic\n"
     "L_value = MSE( V(s), returns )",
     fontsize=7.8)

# [M] Backward / optimiser
rbox(ax, 10.55, 4.8, 4.1, 1.25, C_PPO,
     "Backward Pass\n──────────────\n"
     "Actor optimiser  (Adam)\n"
     "+ Critic optimiser  (Adam)\n"
     "+ gradient clip (max norm 0.5)",
     fontsize=7.8)

# [N] KL divergence monitor
rbox(ax, 15.05, 4.8, 4.1, 1.1, C_SCHED,
     "Approx KL Divergence Monitor\n──────────────\n"
     "KL ≈ old_log_prob – new_log_prob\n"
     "(logged, no early stopping)",
     fontsize=7.8)

# [O] Network update
rbox(ax, 10.55, 3.35, 4.1, 1.05, C_PPO,
     "Network Weights Updated\n──────────────\n"
     "Follower Actor · Critic",
     fontsize=7.8, bold=True)

# [P] Logging / checkpoint
rbox(ax, 15.05, 3.35, 4.1, 1.05, C_BUF,
     "Logging & Checkpointing\n──────────────\n"
     "TensorBoard  ·  CSV  ·  model.pt\n"
     "(every log_interval updates)",
     fontsize=7.8)

# ── arrows right column ─────────────────────────────────────────────────────
# GAE → minibatch sampler
arrow(ax, 9.75, 1.77, 14.95, 1.77, C_BUF, "returns & advantages", label_side="top")
arrow(ax, 14.95, 1.77, 14.95, 9.9,  C_BUF, "")
arrow(ax, 14.95, 10.62, 19.45, 10.62, C_BUF, "")
ax.annotate("", xy=(19.45, 9.9), xytext=(19.45, 10.62),
            arrowprops=dict(arrowstyle="-|>", color=C_BUF, lw=1.4,
                            mutation_scale=14), zorder=2)
arrow(ax, 19.45, 9.9, 19.45, 9.62, C_BUF, "")
arrow(ax, 19.45, 9.62, 19.15, 9.62, C_BUF, "")

# minibatch → follower re-evaluation
arrow(ax, 12.6,  9.9,  12.6,  9.5,  C_PPO, "")

# follower re-evaluation → clipping loss
arrow(ax, 12.6, 8.4, 12.6, 8.0, C_ACT,  "")

# clipping loss → backward
arrow(ax, 12.6, 6.5, 12.6, 6.05, C_PPO, "")

# critic re-eval → value loss → backward
rbox(ax, 10.55, 5.65, 4.1, 0.7, C_CRIT,
     "Central Critic  (re-forward on minibatch)", fontsize=7.8)
arrow(ax, 12.6, 5.65, 12.6, 5.28, C_PPO, "")

# value loss → backward
arrow(ax, 17.1, 6.5, 17.1, 6.05, C_CRIT, "")
arrow(ax, 15.05, 5.35, 14.65, 5.35, C_CRIT, "")
arrow(ax, 14.65, 5.35, 14.65, 5.15, C_CRIT, "")
arrow(ax, 14.65, 5.15, 14.65, 4.80, C_CRIT, "value loss")

# KL monitor from follower re-evaluation
arrow(ax, 12.6, 8.4, 17.1, 5.9, C_SCHED, "")

# backward → weights updated
arrow(ax, 12.6, 4.8, 12.6, 4.4, C_PPO, "")

# weights updated → logging
arrow(ax, 15.05, 3.875, 19.45, 3.875, C_BUF, "")
# weights updated → loop back arrow (continue to next minibatch)
ax.annotate("", xy=(10.55, 10.26), xytext=(10.2, 10.26),
            arrowprops=dict(arrowstyle="-|>", color=C_PPO, lw=1.4,
                            mutation_scale=14), zorder=2)
ax.plot([10.2, 10.2], [3.875, 10.26], color=C_PPO, lw=1.4, zorder=2)
ax.plot([10.2, 10.55], [3.875, 3.875], color=C_PPO, lw=1.4, zorder=2)

# ── outer training loop arrow (right side back to left) ────────────────────
# After all updates → collect new rollout
ax.annotate("", xy=(0.55, 9.86), xytext=(6.5, 9.86),
            arrowprops=dict(arrowstyle="-|>", color=C_LOOP, lw=2.0,
                            mutation_scale=16,
                            connectionstyle="arc3,rad=0.0"), zorder=2)
ax.plot([6.5, 6.5], [3.35, 9.86], color=C_LOOP, lw=2.0, zorder=2)
ax.plot([6.5, 12.6], [3.35, 3.35], color=C_LOOP, lw=2.0, zorder=2)
ax.text(9.6, 3.1, "next update iteration  (repeat N updates)",
        ha="center", va="top", fontsize=8, color=C_LOOP,
        fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_ENV,   "Environment / data"),
    (C_OBS,   "Observation / action adapters"),
    (C_ACT,   "Actor networks"),
    (C_CRIT,  "Critic network"),
    (C_BUF,   "Buffer / GAE / logging"),
    (C_PPO,   "PPO algorithm"),
    (C_SCHED, "Schedulers / monitoring"),
]
lx, ly = 0.55, 0.85
for i, (color, label) in enumerate(legend_items):
    rect = FancyBboxPatch((lx + i * 2.7, ly - 0.25), 0.35, 0.35,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor="#333", linewidth=0.8,
                          zorder=5)
    ax.add_patch(rect)
    ax.text(lx + i * 2.7 + 0.5, ly - 0.07, label,
            va="center", fontsize=7.5, color="#222222", zorder=6)

plt.tight_layout(pad=0)
out_path = "training_flow.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out_path}")
