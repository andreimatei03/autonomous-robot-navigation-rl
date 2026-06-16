"""Smoke-test al agentului (config run 5 revenita): retea 128, stare 23, 5
actiuni + CONFIRMA ca modelul arhivat run 5 se incarca in reteaua revenita."""

import os
import sys

# Bootstrap cale: ruleaza din orice director. Adauga folderul controllerului
# (parintele) la sys.path (pentru `import scout_env` / `dqn_agent`) si fixeaza
# cwd acolo (pentru path-urile relative logs/ si runs/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import os

import numpy as np
import torch

import scout_env as S
from dqn_agent import DQNAgent

sdim = S.FRAME_STACK * (S.N_LIDAR_SECTORS + 11)
adim = 5
agent = DQNAgent(sdim, adim)
print(f"OK  state_dim={sdim} (asteptat 23) action_dim={adim}")
h = agent.q_network.shared[0].out_features
print(f"OK  trunchi hidden={h} (asteptat 128)" if h == 128 else f"FAIL hidden={h}")

s = np.random.rand(sdim).astype(np.float32)
a = agent.select_action(s)
print(f"OK  select_action -> {a}" if 0 <= a < adim else f"FAIL action {a}")

# Incarca modelul arhivat run 5 in reteaua revenita (verifica dimensiunile).
m = os.path.join("runs", "run5_20260612", "dqn_model_best.pth")
sd = torch.load(m, map_location="cpu")
agent.q_network.load_state_dict(sd)
print(f"OK  modelul run 5 ({m}) se incarca in reteaua revenita")

print("AGENT SMOKE OK")
