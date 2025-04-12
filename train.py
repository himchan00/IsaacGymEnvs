
################# training procedure ################
import isaacgym
import isaacgymenvs
from project.policy.actor import Contextual_A2C
import torch
import numpy as np
import random

obs_dim = 7
action_dim = 3
context_dim = 64

device = 'cuda:0'
torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Initialize environment
num_envs = 256
n_steps = 4
envs = isaacgymenvs.make(
    seed=0,
    task="PushToy",
    num_envs=num_envs,
    sim_device=device,
    rl_device=device,
    graphics_device_id=0,
    headless=True,
    force_render=False
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

# initialize a PPO agent
ppo_agent = Contextual_A2C(obs_dim=obs_dim, action_dim = action_dim, context_dim = context_dim,  device = device)

max_training_timesteps = 1e6
# printing and logging variables


time_step = 1

obs_dict = envs.reset()
context = ppo_agent.sample_initial_context(num_envs) # (num_envs, context_dim)
while time_step <= max_training_timesteps:
    obs = obs_dict['obs'] # s_t
    image = obs_dict['images'] # I_t
    sampled_action = obs_dict['sampled_actions'] 
    sampled_transition = obs_dict['sampled_transitions']
    # Select action with policy
    action = ppo_agent.select_action(obs, image, context) # (s_t, a_t, I_t, c_t, log a_t, v_t) is added to buffer
    obs_dict, reward, done, info = envs.step(action)
    # Update Context (c_t -> c_t+1)
    is_initial = info['initials'] # (num_envs,)
    context = ppo_agent.update_context(context, image, obs_dict['images'], action)
    context[is_initial] = ppo_agent.sample_initial_context(is_initial.sum()) # (num_envs, context_dim)

    # Add to buffer
    ppo_agent.buffer.rewards.append(reward)
    ppo_agent.buffer.is_terminals.append(done)
    ppo_agent.buffer.is_timeouts.append(info['time_outs'])
    ppo_agent.buffer.sampled_actions.append(sampled_action)
    ppo_agent.buffer.sampled_transitions.append(sampled_transition)
    ppo_agent.buffer.is_initials.append(is_initial)

    # update PPO agent
    if time_step % n_steps == 0:
        print("Training A2C agent")
        d_train = ppo_agent.update()
        for key, value in d_train.items():
            print(f"{key}: {value}")
        
        print("Policy Rollout")

    time_step += 1