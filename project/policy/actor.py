"""
This code was written with reference to https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import torch
from torch.distributions import transformed_distribution, Normal, transforms
from ..wideresnet_cond.wideresnet_cond import Wide_ResNet_Cond



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.images = []
        self.observations = []
        self.contexts = []
        self.rewards = []
        self.values = []
        self.sampled_actions = []
        self.sampled_transitions = []
        self.is_terminals = []
        self.is_timeouts = []
        self.is_initials = []
    
    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.images[:]
        del self.contexts[:]
        del self.rewards[:]
        del self.values[:]
        del self.sampled_actions[:]
        del self.sampled_transitions[:]
        del self.is_terminals[:]
        del self.is_timeouts[:]
        del self.is_initials[:]

class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, context_dim):
        super(ActorCritic, self).__init__()
        self.network = Wide_ResNet_Cond(
                        depth = 10, 
                        widen_factor = 4, 
                        cond_dim = obs_dim + 2 *context_dim,
                        num_classes= 2*action_dim + 1,
                        input_channels = 1,
                        norm = 'layer',
                        leak = 0.0,
                        dropout_rate = 0.0
                        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.logstd_clip = [-20, 2]

    def forward(self):
        raise NotImplementedError

    def act(self, obs, image, context):
        """
        (Input)
        Obs: (batch_size, obs_dim)
        Image: (batch_size, 1, height, width)
        Context: (batch_size, 2 * context_dim)
        (Output)
        Action: (batch_size, action_dim)
        Action log probability: (batch_size, )
        Obs value: (batch_size, )
        """
        cond = torch.cat([obs, context], dim = -1) # (batch_size, obs_dim + 2 * context_dim)
        out = self.network(image, cond)
        action_mean = out[:, :self.action_dim]
        logstd = out[:, self.action_dim:2*self.action_dim]
        logstd = torch.clamp(logstd, self.logstd_clip[0], self.logstd_clip[1])
        value = out[:, -1] # (batch_size, )
        action_std = torch.exp(logstd)
        dist = transformed_distribution.TransformedDistribution(
            Normal(action_mean, action_std),
            [transforms.TanhTransform(cache_size=1)]
        )
        action = dist.sample() # (batch_size, action_dim)

        return action.detach(), value.detach()

    def evaluate(self, obs, image, context, action):
        """
        (Input)
        obs: (batch_size, obs_dim)
        Image: (batch_size, 1, height, width)
        Context: (batch_size, 2 * context_dim)
        Action: (batch_size, action_dim)
        (Output)
        Action log probability: (batch_size, )
        obs value: (batch_size, )
        Entropy: (batch_size, )
        """
        cond = torch.cat([obs, context], dim = -1) # (batch_size, obs_dim + 2 * context_dim)
        out = self.network(image, cond)
        action_mean = out[:, :self.action_dim]
        logstd = out[:, self.action_dim:2*self.action_dim]
        logstd = torch.clamp(logstd, self.logstd_clip[0], self.logstd_clip[1])
        value = out[:, -1] # (batch_size, )
        action_std = torch.exp(logstd)
        dist = transformed_distribution.TransformedDistribution(
            Normal(action_mean, action_std),
            [transforms.TanhTransform(cache_size=1)]
        )
        action_logprob = dist.log_prob(action).sum(dim=-1) # (batch_size, )
        # entropy = dist.entropy().sum(dim = -1) # This is not implemented for tanh normal
    
        return action_logprob, value



class Contextual_A2C:
            
    def __init__(self, obs_dim, action_dim, context_dim, device, lr = 1e-4, lr_e = 3e-5, gamma = 0.99, critic_coef = 0.5, entropy_coef = 0.0, clip_grad = 0.5, normalize_advantage = True):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.gamma = gamma
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        self.buffer = RolloutBuffer()

        self.actor_critic = ActorCritic(obs_dim, action_dim, context_dim).to(device)

        self.encoder = Wide_ResNet_Cond(
                        depth = 10, 
                        widen_factor = 4, 
                        cond_dim = action_dim,
                        num_classes= 2*context_dim,
                        input_channels = 2,
                        norm = 'layer',
                        leak = 0.0,
                        dropout_rate = 0.0
                        ).to(device)

        self.optimizer = torch.optim.Adam([{'params': self.actor_critic.parameters(), 'lr': lr}, 
                                            {'params': self.encoder.parameters(), 'lr': lr_e}])
                                          
        self.normalize_advantage = normalize_advantage

        self.device = device
        self.clip_grad = clip_grad


    def sample_initial_context(self, batch_size):
        initial_context = torch.zeros(batch_size, 2 * self.context_dim).to(self.device)
        initial_context[:, self.context_dim:] = 1.0
        return initial_context

    def select_action(self, obs, image, context):
        """
        obs: (batch_size, obs_dim)
        Image: (batch_size, 1, height, width)
        Context: (batch_size, 2 * context_dim)
        """
        self.actor_critic.eval()
        with torch.no_grad():
            action, value = self.actor_critic.act(obs, image, context)
            self.buffer.observations.append(obs)
            self.buffer.images.append(image)
            self.buffer.contexts.append(context)
            self.buffer.actions.append(action)
            self.buffer.values.append(value)
        
        return action
    
    def update_context(self, context, image, next_image, action):
        """
        Context: (batch_size, 2 * context_dim)
        Image: (batch_size, 1, height, width)
        Next image: (batch_size, 1, height, width)
        Action: (batch_size, action_dim)
        """
        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(torch.cat([image, next_image], dim = 1), action)
            new_context_mean = out[:, :self.context_dim]
            log_var = out[:, self.context_dim:]
            new_contex_var = torch.exp(log_var)
            new_contex_prec = 1 / new_contex_var

        prev_context_mean = context[:, :self.context_dim]
        prev_context_var = context[:, self.context_dim:]
        prev_context_prec = 1 / prev_context_var
        context_var = 1/(prev_context_prec + new_contex_prec)
        context_mean = (prev_context_mean * prev_context_prec + new_context_mean * new_contex_prec) * context_var
        context = torch.cat([context_mean, context_var], dim = -1)
        return context

    def update(self):
        # Monte Carlo estimate of returns
        is_initials = torch.cat(self.buffer.is_initials, dim = 0).detach().to(self.device) # (n_envs * n_steps, )
        not_initials = ~ is_initials

        returns = []
        n_envs = self.buffer.actions[0].shape[0]
        discounted_reward = torch.zeros(n_envs, device=self.device)
        for (i, (reward, is_terminal, is_timeout, value)) in enumerate(zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), reversed(self.buffer.is_timeouts), reversed(self.buffer.values))):
            # reward (n_envs,), is_terminal (n_envs,)
            discounted_reward = (1 - is_terminal.float()) * discounted_reward
            if i == 0:
                reward = value
            else:
                reward = (1 - is_timeout.float()) * reward + is_timeout.float() * value
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.cat(returns, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, )
        return_mean = returns.mean().item()
        return_std = returns.std().item()

        # Convert to tensor
        old_observations = torch.cat(self.buffer.observations, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, obs_dim)
        old_images = torch.cat(self.buffer.images, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, 1, height, width)
        old_contexts = torch.cat(self.buffer.contexts, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, 2 * context_dim)
        old_actions = torch.cat(self.buffer.actions, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, action_dim)
        old_values = torch.cat(self.buffer.values, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps,)
        # For encoder training
        old_sampled_actions = torch.cat(self.buffer.sampled_actions, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, action_dim)
        old_sampled_transitions = torch.cat(self.buffer.sampled_transitions, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, 2, height, width)


        # Calculate advantages
        advantages = returns - old_values
        # Normalize advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy
        self.encoder.train()
        self.actor_critic.train()
        self.optimizer.zero_grad()

        out = self.encoder(old_sampled_transitions, old_sampled_actions)
        new_context_mean = out[:, :self.context_dim]
        new_context_prec = 1 / torch.exp(out[:, self.context_dim:])
        prev_context_prec = (1 / old_contexts[:, self.context_dim:] - new_context_prec).detach()
        # context_mean_rest * context_prec_rest + context_mean * context_prec = sampled_contexts[:, :self.context_dim] * sampled_contexts[:, self.context_dim:]
        prev_context_mean = (old_contexts[:, :self.context_dim] /old_contexts[:, self.context_dim:] - new_context_mean * new_context_prec).detach() / prev_context_prec
        context_var = 1/(prev_context_prec + new_context_prec)
        context_mean = (prev_context_mean * prev_context_prec + new_context_mean * new_context_prec) * context_var
        contexts = torch.cat([context_mean, context_var], dim = -1)

        # Evaluating old actions and values
        logprobs, values = self.actor_critic.evaluate(old_observations, old_images, contexts, old_actions)

        # Policy gradient loss
        actor_loss = -(advantages * logprobs).mean()

        # Critic loss
        critic_loss = (returns - values).pow(2).mean()

        # Entropy loss (Not sure this is correct)
        entropy_loss = logprobs.mean() # (batch_size, ) 

        # Total loss
        loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

        # Take gradient step
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad)
        self.optimizer.step()

        # clear buffer
        self.buffer.clear()

        # Calculate mean loss
        actor_loss_mean = actor_loss.item()
        critic_loss_mean = critic_loss.item()
        entropy_loss_mean = entropy_loss.item()
        d_train = {
            "return": return_mean,
            "return_std": return_std,
            "actor_loss": actor_loss_mean,
            "critic_loss": critic_loss_mean,
            "entropy_loss": entropy_loss_mean
        }
        return d_train
