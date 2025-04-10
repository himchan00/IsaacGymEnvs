import isaacgym
import isaacgymenvs
import torch

num_envs = 4
envs = isaacgymenvs.make(
    seed=0,
    task="PushToy",
    num_envs=num_envs,
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()

# for i in range(100):
while True:
    l_dp = []
    l_quat = []

    actions = torch.zeros((num_envs,) + envs.action_space.shape, device = 'cuda:0')
    actions[:, :2] = 10*(envs.states["obj_pos"][:, :2] - envs.states['eef_pos'][:, :2]) # Desired position = object position (2D)
    actions[:, 2] = 0.0
    obs_dict, _ ,_, _ = envs.step(actions)