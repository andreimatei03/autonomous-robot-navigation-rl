import torch
import os
from dqn_agent import DQNAgent
from scout_env import ScoutEnv
from controller import Supervisor

# ===============================
# INIT WEBOTS
# ===============================

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Robot node
robot_node = robot.getFromDef("Pioneer3AT")
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# Motoare
fl = robot.getDevice("front left wheel")
bl = robot.getDevice("back left wheel")
fr = robot.getDevice("front right wheel")
br = robot.getDevice("back right wheel")

motors = [fl, bl, fr, br]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

# Lidar
lidar = robot.getDevice("lidar")
lidar.enable(timestep)

# Conul Obiectiv (preluat din modificarile anterioare)
goal_node = robot.getFromDef("GOAL_CONE")
goal_translation_field = goal_node.getField("translation")

# ===============================
# ENVIRONMENT & AGENT (MOD EVALUARE)
# ===============================

env = ScoutEnv(
    robot,
    motors,
    lidar,
    translation_field,
    rotation_field,
    goal_translation_field,
    timestep
)

state_dim = env.state_dim
action_dim = env.action_dim

# Initializam agentul
agent = DQNAgent(state_dim, action_dim)

# Incarcam cel mai bun model obtinut in antrenament
model_path = "dqn_model_best.pth"
if os.path.exists(model_path):
    # Incarcam "greutatile" retelei
    agent.q_network.load_state_dict(torch.load(model_path))
    # Trecem rețeaua explicit in modul de evaluare (opreste eventuale dropouts)
    agent.q_network.eval() 
    print(f"✓ Model de succes incarcat din {model_path}")
else:
    print(f"❌ EROARE: Fisierul {model_path} nu a fost gasit!")
    # Oprim rularea daca nu avem model
    exit()

# ---------------------------------------------------------
# CEA MAI IMPORTANTA LINIE PENTRU EVALUARE:
agent.epsilon = 0.0  # Robotul nu mai exploreaza NIMIC aleatoriu.
# ---------------------------------------------------------

# ===============================
# TESTING LOOP
# ===============================

num_episodes = 5  # Vom rula doar 5 episoade demonstrative
max_steps = 2000  

print("\n🚀 Incepere testare! Robotul foloseste 100% inteligenta artificiala.")

for episode in range(num_episodes):
    
    state = env.reset()
    total_reward = 0
    success = False

    for step in range(max_steps):
        
        robot.step(timestep)

        # Agentul alege actiunea calculata de retea, fara Epsilon
        action = agent.select_action(state)

        # Executam actiunea
        next_state, reward, done = env.step(action)

        state = next_state
        total_reward += reward

        # Daca primeste bonusul maxim, inseamna ca a atins conul
        if reward >= 190.0:  
            success = True

        if done:
            break
            
    # Mesaj vizual clar in consola pentru evaluare
    status = "✅ SUCCES (Destinatie atinsa)" if success else "❌ ESEC (Coliziune / Timp expirat)"
    print(f"Test Episod {episode+1}/{num_episodes} | Status: {status} | Reward: {total_reward:8.2f} | Pasi folositi: {step}")

# ===============================
# FINALIZARE
# ===============================

# Oprim motoarele ferm la sfarsit ca sa nu se invarta in cerc!
for m in motors:
    m.setVelocity(0.0)
    
print("\n🎉 Prezentare demonstrativa incheiata!")