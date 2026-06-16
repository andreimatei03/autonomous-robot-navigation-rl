import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==============================
# Q-Network (Dueling Architecture)
# ==============================

class QNetwork(nn.Module):
    # Trunchi 128→128, streamuri →64 (configurația run 5, headline-ul). Run 7 a
    # încercat 256/128 pentru starea stivuită de 69 dims: ABLAȚIA a arătat că
    # rețeaua mai mare + bugetul lung au provocat divergența funcției de valoare
    # (loss 6→1246, avg50 +622→−23) — rețeaua mică e robustă la deadly triad.
    HIDDEN = 128
    STREAM = 64

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        h, s = self.HIDDEN, self.STREAM

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(h, s),
            nn.ReLU(),
            nn.Linear(s, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(h, s),
            nn.ReLU(),
            nn.Linear(s, action_dim)
        )

    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ==============================
# Replay Buffer
# ==============================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # float32 compact (o listă Python de float-uri ocupă ~5× mai mult)
        self.buffer[self.position] = (
            np.asarray(state, dtype=np.float32),
            action,
            reward,
            np.asarray(next_state, dtype=np.float32),
            done
        )

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ==============================
# DQN Agent
# ==============================

class DQNAgent:
    def __init__(self, state_dim, action_dim):

        self.device = torch.device("cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        # LR cu decădere lentă și PRAG MINIM. Vechiul StepLR(5000, 0.9) era
        # apelat la FIECARE pas de antrenare (~1300 pași/episod → ×0.9 la
        # fiecare ~4 episoade), deci lr ≈ 0 pe la ep. 300 — rețeaua îngheța
        # exact când curriculum-ul ajungea la ținte îndepărtate (run-ul
        # 2026-06-09: vârf avg50 la ep. 226, apoi stagnare 700 de episoade).
        # Acum: 5e-4 → podea 1e-4 (factor 0.2), atinsă pe la ep. ~450.
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: max(0.2, 0.95 ** (step // 20000)))

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # Decădere PER EPISOD (apelată o dată/episod din bucla de training prin
        # decay_epsilon()). Per-pas, epsilon se prăbușea la min pe la ep. 55:
        # episoadele aleatoare ating 5000 pași, deci bugetul de explorare se
        # consuma înainte ca curriculum-ul să introducă ținte îndepărtate.
        # 0.993/episod → epsilon ~0.24 la ep.200, ~0.06 la ep.400 (final
        # curriculum), atinge 0.01 pe la ep.650 → explorare vie tot curriculum-ul.
        self.epsilon_decay = 0.993

        self.batch_size = 64

        # 50k ≈ doar ~36 de episoade de istorie (~1400 pași/episod): episoadele
        # rare cu succes la ținte îndepărtate erau suprascrise înainte să fie
        # exploatate. 200k ≈ ~150 de episoade; stocat float32 → ~100 MB RAM.
        self.replay_buffer = ReplayBuffer(200000)

        self.update_target_every = 500  # Update mai des pentru mai multă stabilitate
        self.step_counter = 0
        
        self.clip_grad_norm = 10.0  # Gradient clipping pentru stabilitate

    # -----------------------------

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Inferență sub no_grad. Rețeaua e doar Linear + ReLU (fără BatchNorm
        # sau Dropout), deci modul train/eval nu schimbă ieșirea → nu comutăm
        # aici; respectăm modul setat din afară (eval() la evaluare).
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax().item()

    # -----------------------------

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # -----------------------------

    def decay_epsilon(self):
        """Decădere epsilon o dată pe episod (vezi epsilon_decay)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -----------------------------

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a)
        current_q = self.q_network(states).gather(1, actions)

        # Double DQN: Folosim q_network pentru selectare, target_network pentru evaluare
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping pentru stabilitate
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clip_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()

        self.step_counter += 1

        # Update target network
        if self.step_counter % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()