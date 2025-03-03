import isaacgym
import isaacgymenvs
import torch

num_envs = 200
envs = isaacgymenvs.make(
    seed=0,
    task="FrankaToy",
    num_envs=num_envs,
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()

# Admittance Control Parameters (Only used when controlType is "admittance")
inertia = 1
stiffness = 1000.0

while True:
    l_dp = []
    l_quat = []

    actions = torch.zeros((num_envs,) + envs.action_space.shape, device = 'cuda:0')
    actions[:, :2] = envs.states["box_pos"][:, :2] - envs.states['eef_pos'][:, :2] # Desired position = box position (2D)
    actions[:, 2] = inertia
    actions[:, 3] = stiffness
    envs.step(actions)

