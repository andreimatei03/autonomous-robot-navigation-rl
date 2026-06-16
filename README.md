# Autonomous Navigation using Reinforcement Learning

Bachelor's thesis project. A **Deep Q-Network (DQN)** agent drives a **Pioneer 3-AT**
robot autonomously through an **unstructured environment** in the **Webots R2025a**
simulator. Navigation is **pure reinforcement learning on sensors** — the learned
policy decides everything from LiDAR + IMU + the relative goal position. There is
**no hand-crafted geometric path planner**.

| Evaluation (100 episodes) | Success | Collision | Stuck | SPL |
|---|---:|---:|---:|---:|
| DQN — random start | **64%** | 22% | 13% | 0.83 |
| DQN — centre start | **80%** | 12% | 7% | 0.86 |
| DQN — **unseen map** | **62%** | 22% | 13% | 0.67 |
| Reactive baseline | 31% | 4% | 64% | 0.30 |

The DQN roughly **doubles** the reactive baseline, and performance on a map it has
**never seen** (62%) matches the training map (64%) — so the policy learned to
navigate, not to memorise the layout.

---

## Environment

A **150 × 150 m irregular-terrain map**, built procedurally from a Python terrain
model:

- an **organic navigable clearing** (radius ~45–54 m, not a circle) of gentle
  rolling hills with real pitch/roll;
- a **pine forest whose density increases toward the map edge**, ending in an
  **impassable belt** at the perimeter — the robot physically cannot fall off the map;
- **2 utility buildings** (warehouse + barn) on flat graded pads, plus several
  **inaccessible obstacle zones** (boulder fields, a thicket, a container pile, a
  rubble heap) and scattered clutter (rocks, logs, barrels, crates, vehicles).

Goals are always sampled inside the clearing; the forest is a graded obstacle field
the policy learns to avoid. The terrain heightfield is **baked from the same Python
functions the RL queries**, so the simulated surface equals what the agent perceives.

## Approach

- **State (23 dims):** 12 LiDAR sector minima covering the full 180° FOV, a frontal
  vertical profile (3), goal distance & angle, linear/angular velocity, and IMU
  (z, roll, pitch, yaw-rate). LiDAR range = 10 m.
- **Actions (5, discrete):** forward, forward-left, forward-right, rotate-left,
  rotate-right.
- **Reward:** potential-based progress-to-goal shaping (the only positive term, so it
  cannot be farmed by looping), plus proximity/instability/idle penalties and terminal
  rewards (+200 goal, −20 collision/rollover/out-of-bounds/stuck).
- **Agent:** Double DQN + Dueling network (128→128, value/advantage streams), 200k
  replay buffer, target network, ε-greedy with per-episode decay, LR schedule with a
  floor, best-avg50 checkpointing. Runs on CPU (the network is tiny; Webots is the
  bottleneck).

## How to Run

**Prerequisites:** Webots R2025a; Python 3.10 with the dependencies in
`requirements.txt`.

```powershell
python -m venv rl_env
.\rl_env\Scripts\activate
pip install -r requirements.txt

# (Re)build the world from the terrain model + object scatter (offline):
cd diploma_webots_project\controllers\rl_scout
python generate_world.py            # writes ../../worlds/world.wbt
```

Then open `diploma_webots_project/worlds/world.wbt` in Webots; it launches the
`rl_scout` controller, which shows a menu (train from scratch / resume / batch
evaluation / centre-start comparison / unseen-map evaluation / manual run / reactive
baseline / fixed demo scenarios).

Generate the thesis figures after a run (offline, not inside Webots):

```powershell
cd diploma_webots_project\controllers\rl_scout
python plot_metrics.py              # reads logs/*.csv -> writes figures/*.png
```

> The Webots binding `controller` is **not** a pip package — it ships with Webots and
> is added to the path automatically when a world is opened from the simulator.
> On first load Webots fetches and caches the object PROTOs from GitHub (internet
> required once).

## Project Structure

```
diploma_webots_project/
├── controllers/rl_scout/
│   ├── scout_env.py        # RL environment: terrain model, state, reward, dynamic obstacle scan
│   ├── dqn_agent.py        # Double + Dueling DQN, replay buffer
│   ├── rl_scout.py         # Webots Supervisor, training/eval/baseline loops, Tkinter menu
│   ├── generate_world.py   # offline procedural world generator (terrain + objects + forest)
│   ├── plot_metrics.py     # offline figures + SPL + consolidated results table
│   ├── tests/              # offline unit/smoke tests (reward, state, agent) — no Webots needed
│   ├── analysis/           # offline log-analysis helpers (per-run breakdowns, stuck diagnosis)
│   ├── logs/               # CSV metrics + evaluation summaries/trajectories
│   ├── figures/            # generated PNG figures
│   └── runs/               # archived runs (run2…run7): models, logs, figures
├── protos/                 # local Pioneer3at PROTO
└── worlds/                 # world.wbt (training) + world_eval.wbt (unseen map)
```

## Results & Figures

`plot_metrics.py` produces, in `figures/`: the learning curve, success / collision
rates, episode length, TD-loss, ε-decay, evaluation trajectories over the obstacle
map, steps-to-goal histograms, a 4-series comparison bar chart (with 95% confidence
intervals), success-by-distance, and **SPL** (Success weighted by Path Length). The
consolidated numbers are written to `logs/results_summary.csv`.

The development history (`runs/run2…run7/`) documents the full convergence and
reward-design process, including ablations and **negative results** (an action-space
regression and a value-function divergence), which informed the final configuration.

## Tests

```powershell
cd diploma_webots_project\controllers\rl_scout
python tests\test_reward.py         # 11 reward-shaping scenarios
python tests\test_state.py          # state-extraction / LiDAR sectors
python tests\test_agent_smoke.py    # network dims + checkpoint loading
```

## Notes

- Results are reported from the headline run (run 5). They come from a single training
  seed; evaluation uses 100 episodes per setting with binomial confidence intervals.
- SPL is computed from logged trajectories (`save_traj`); the archived run-5 data logged
  the first 20 trajectories per evaluation, so `spl_summary.csv` lists both the
  subset and the full-population success rate for transparency.

Developed as part of a Bachelor's thesis on autonomous systems and reinforcement
learning applied to robotics.
