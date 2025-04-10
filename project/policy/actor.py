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
        self.logprobs = []
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
        del self.logprobs[:]
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
                        widen_factor = 8, 
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
        value = out[:, -1] # (batch_size, )
        action_std = torch.exp(logstd)
        dist = transformed_distribution.TransformedDistribution(
            Normal(action_mean, action_std),
            [transforms.TanhTransform(cache_size=1)]
        )
        action = dist.sample() # (batch_size, action_dim)
        action_logprob = dist.log_prob(action).sum(dim=-1) # (batch_size, )

        return action.detach(), action_logprob.detach(), value.detach()

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
        value = out[:, -1] # (batch_size, )
        action_std = torch.exp(logstd)
        dist = transformed_distribution.TransformedDistribution(
            Normal(action_mean, action_std),
            [transforms.TanhTransform(cache_size=1)]
        )
        action_logprob = dist.log_prob(action).sum(dim=-1) # (batch_size, )
        # entropy = dist.entropy().sum(dim = -1) # This is not implemented for tanh normal
    
        return action_logprob, value


class Contextual_PPO:
            
    def __init__(self, obs_dim, action_dim, context_dim, batch_size, lr, lr_e, gamma, device, K_epochs = 10, eps_clip = 0.2 , critic_coef = 8, entropy_coef = 0.0, context_prior = "unit_normal", clip_grad = None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        self.buffer = RolloutBuffer()

        self.actor_critic = ActorCritic(obs_dim, action_dim, context_dim).to(device)
        self.actor_critic_old = ActorCritic(obs_dim, action_dim, context_dim).to(device)
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        self.actor_critic_old.eval()

        self.encoder = Wide_ResNet_Cond(
                        depth = 10, 
                        widen_factor = 8, 
                        cond_dim = action_dim,
                        num_classes= 2*context_dim,
                        input_channels = 2,
                        norm = 'layer',
                        leak = 0.0,
                        dropout_rate = 0.0
                        ).to(device)
        
        self.optimizer = torch.optim.Adam([{"params": self.actor_critic.parameters(), "lr": lr},
                                          {"params": self.encoder.parameters(), "lr": lr_e}])

        assert context_prior in ["unit_normal", "uniform"]
        self.context_prior = context_prior
        self.device = device
        self.clip_grad = clip_grad


    def sample_initial_context(self, batch_size):
        initial_context = torch.zeros(batch_size, 2 * self.context_dim).to(self.device)
        if self.context_prior == "unit_normal":
            initial_context[:, self.context_dim:] = torch.ones(batch_size, self.context_dim).to(self.device)
        return initial_context

    def select_action(self, obs, image, context):
        """
        obs: (batch_size, obs_dim)
        Image: (batch_size, 1, height, width)
        Context: (batch_size, 2 * context_dim)
        """
        with torch.no_grad():
            action, action_logprob, value = self.actor_critic_old.act(obs, image, context)
            self.buffer.observations.append(obs)
            self.buffer.images.append(image)
            self.buffer.contexts.append(context)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.values.append(value)
        
        return action
    
    def update_context(self, context, image, next_image, action, is_initial):
        """
        Context: (batch_size, 2 * context_dim)
        Image: (batch_size, 1, height, width)
        Next image: (batch_size, 1, height, width)
        Action: (batch_size, action_dim)
        is_initial: (batch_size,)
        """
        self.encoder.eval()
        is_initial_mask = is_initial.float().unsqueeze(-1) # (batch_size, 1)
        with torch.no_grad():
            out = self.encoder(torch.cat([image, next_image], dim = 1), action)
            new_context_mean = out[:, :self.context_dim] * (1 - is_initial_mask)
            log_prec = out[:, self.context_dim:]
            new_contex_prec = torch.exp(log_prec) * (1 - is_initial_mask)

        prev_context_mean = context[:, :self.context_dim]
        prev_context_prec = context[:, self.context_dim:]
        context_mean = (prev_context_mean * prev_context_prec + new_context_mean * new_contex_prec) / (prev_context_prec + new_contex_prec)
        context_prec = prev_context_prec + new_contex_prec
        context = torch.cat([context_mean, context_prec], dim = -1)
        return context

    def update(self):
        # Monte Carlo estimate of returns
        returns = []
        n_envs = self.buffer.actions[0].shape[0]
        discounted_reward = torch.zeros(n_envs, device=self.device)
        for reward, is_terminal, is_timeout, value in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), reversed(self.buffer.is_timeouts), reversed(self.buffer.values)):
            # reward (n_envs,), is_terminal (n_envs,)
            discounted_reward = (1 - is_terminal.float()) * discounted_reward
            reward = (1 - is_timeout.float()) * reward + is_timeout.float() * value
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.cat(returns, dim = 0).detach().to(self.device) # (n_envs * n_steps, )

        # Convert to tensor
        old_observations = torch.cat(self.buffer.observations, dim = 0).detach().to(self.device) # (n_envs * n_steps, obs_dim)
        old_images = torch.cat(self.buffer.images, dim = 0).detach().to(self.device) # (n_envs * n_steps, 1, height, width)
        old_contexts = torch.cat(self.buffer.contexts, dim = 0).detach().to(self.device) # (n_envs * n_steps, 2 * context_dim)
        old_actions = torch.cat(self.buffer.actions, dim = 0).detach().to(self.device) # (n_envs * n_steps, action_dim)
        old_logprobs = torch.cat(self.buffer.logprobs, dim = 0).detach().to(self.device) # (n_envs * n_steps,)
        old_values = torch.cat(self.buffer.values, dim = 0).detach().to(self.device) # (n_envs * n_steps,)
        # For encoder training
        old_sampled_actions = torch.cat(self.buffer.sampled_actions, dim = 0).detach().to(self.device) # (n_envs * n_steps, action_dim)
        old_sampled_transitions = torch.cat(self.buffer.sampled_transitions, dim = 0).detach().to(self.device) # (n_envs * n_steps, 2, height, width)
        is_initial_masks = torch.cat(self.buffer.is_initials, dim = 0).float().unsqueeze(-1).detach().to(self.device) # (n_envs * n_steps, 1)

        # Calculate advantages
        advantages = returns - old_values

        assert len(old_observations) % self.batch_size == 0, "n_envs * n_steps must be divisible by batch_size"
        n_updates = int(len(old_observations) / self.batch_size)
        # Optimize policy for K epochs
        self.encoder.train()
        self.actor_critic.train()
        for _ in range(self.K_epochs):
            # Randomize the order of the data
            indices = torch.randperm(len(old_observations))
            for i in range(n_updates):
                self.optimizer.zero_grad()
                # Sample a batch of data
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                sampled_observations = old_observations[batch_indices]
                sampled_images = old_images[batch_indices]
                sampled_contexts = old_contexts[batch_indices]
                sampled_actions = old_actions[batch_indices]
                sampled_logprobs = old_logprobs[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_sampled_actions = old_sampled_actions[batch_indices]
                sampled_sampled_transitions = old_sampled_transitions[batch_indices]
                sampled_is_initial_masks = is_initial_masks[batch_indices]

                initial_context = self.sample_initial_context(sampled_observations.shape[0])
                out = self.encoder(sampled_sampled_transitions, sampled_sampled_actions) * (1 - sampled_is_initial_masks) + sampled_is_initial_masks * initial_context
                new_context_mean = out[:, :self.context_dim]
                log_prec = out[:, self.context_dim:]
                new_context_prec = torch.exp(log_prec)
                prev_context_prec = sampled_contexts[:, self.context_dim:] - new_context_prec.detach()
                # context_mean_rest * context_prec_rest + context_mean * context_prec = sampled_contexts[:, :self.context_dim] * sampled_contexts[:, self.context_dim:]
                prev_context_mean = (sampled_contexts[:, :self.context_dim] * sampled_contexts[:, self.context_dim:] - (new_context_mean * new_context_prec).detach()) / prev_context_prec
                context_mean = (prev_context_mean * prev_context_prec + new_context_mean * new_context_prec) / (prev_context_prec + new_context_prec)
                context_prec = prev_context_prec + new_context_prec
                contexts = torch.cat([context_mean, context_prec], dim = -1)

                # Evaluating old actions and values
                logprobs, values = self.actor_critic.evaluate(sampled_observations, sampled_images, contexts, sampled_actions)

                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - sampled_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * sampled_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * sampled_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic loss
                critic_loss = (sampled_returns - values).pow(2).mean()

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

        # Copy new weights into old policy
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())

        # clear buffer
        self.buffer.clear()