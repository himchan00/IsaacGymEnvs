import isaacgym
import isaacgymenvs
import torch
import math
from isaacgymenvs.utils.torch_jit_utils import * 

def circle_trajectory(i, t):
    x = 0.1 + 0.2 * math.cos(1.5 * t - math.pi * float(i) / num_envs)
    y = 0.2 * math.sin(1.5 * t - math.pi * float(i) / num_envs)
    z = 1.2
    return torch.tensor([x, y, z], dtype=torch.float32)

def push_trajectory(i, t):
    x = 0.1 + 0.2 * math.cos(- math.pi * float(i) / num_envs)
    y = 0.2 * math.sin(- math.pi * float(i) / num_envs)
    if t < 1.0:
        z = 1.103- 0.1 * t
    else:
        z = 1.003
    return torch.tensor([x, y, z], dtype=torch.float32)

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

t = 0
goal_quat = quat_from_euler_xyz(torch.tensor([math.pi]), torch.tensor([0]), torch.tensor([0])).repeat(num_envs, 1).to(envs.device)

while True:
    n_control_loop = envs.n_control_loop if hasattr(envs, 'n_control_loop') else 1
    t += envs.dt * n_control_loop
    l_dp = []
    l_quat = []

    for i in range(num_envs):
        # Desired position - current position
        l_dp.append(push_trajectory(i,t).to(envs.device) - envs.states['eef_pos'][i]) 
        # l_dp.append(circle_trajectory(i,t).to(envs.device) - envs.states['eef_pos'][i]) 
        l_quat.append(envs.states['eef_quat'][i])   
    dp = torch.stack(l_dp)

    quat = torch.stack(l_quat)
    # Desired orientation - current orientation
    cc = quat_conjugate(quat)
    angle, axis = quat_to_angle_axis(quat_mul(goal_quat, cc))
    dor = angle.unsqueeze(-1) * axis

    # Apply actions
    actions = torch.zeros((num_envs,) + envs.action_space.shape, device=envs.device)
    actions[:, :3] = dp
    actions[:, 3:6] = dor
    actions[:, 6] = inertia
    actions[:, 7] = stiffness
    envs.step(actions)

    # Debug
    force = envs._contact_forces[:, envs.handles['franka_fingertip']]
    print("Force", force[103]) # Force on the 103rd environment (This environment is directly in front of the viewer.)