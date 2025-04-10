
################# training procedure ################
import isaacgym
import isaacgymenvs
from project.policy.actor import Contextual_PPO

obs_dim = 7
action_dim = 3
context_dim = 64

device = 'cuda:0'
# Initialize environment
num_envs = 256
n_steps = 32
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
ppo_agent = Contextual_PPO(obs_dim=obs_dim, action_dim = action_dim, context_dim = context_dim, batch_size=128, lr = 1e-5, lr_e = 1e-5, gamma = 0.99, device = device, entropy_coef=0.001, clip_grad=1.0)

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
    # Update Context
    is_initial = info['initials'] # (num_envs,)
    context = ppo_agent.update_context(context, image, obs_dict['images'], action, is_initial)

    # Add to buffer
    ppo_agent.buffer.rewards.append(reward)
    ppo_agent.buffer.is_terminals.append(done)
    ppo_agent.buffer.is_timeouts.append(info['time_outs'])
    ppo_agent.buffer.sampled_actions.append(sampled_action)
    ppo_agent.buffer.sampled_transitions.append(sampled_transition)
    ppo_agent.buffer.is_initials.append(info['initials'])

    # update PPO agent
    if time_step % n_steps == 0:
        ppo_agent.update()
    time_step += 1