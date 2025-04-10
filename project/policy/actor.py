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
        self.logstd_clip = [-10, 2]

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


class Contextual_PPO:
            
    def __init__(self, obs_dim, action_dim, context_dim, batch_size, device, lr = 1e-4, lr_e = 3e-5, gamma = 0.99, K_epochs = 10, eps_clip = 0.2 , critic_coef = 0.5, entropy_coef = 0.0, clip_grad = 0.5):
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
                        widen_factor = 4, 
                        cond_dim = action_dim,
                        num_classes= 2*context_dim,
                        input_channels = 2,
                        norm = 'layer',
                        leak = 0.0,
                        dropout_rate = 0.0
                        ).to(device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=lr_e)

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
        with torch.no_grad():
            action, action_logprob, value = self.actor_critic_old.act(obs, image, context)
            self.buffer.observations.append(obs)
            self.buffer.images.append(image)
            self.buffer.contexts.append(context)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
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
            log_var = torch.clamp(log_var, -10, 2)
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
        old_logprobs = torch.cat(self.buffer.logprobs, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps,)
        old_values = torch.cat(self.buffer.values, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps,)
        # For encoder training
        old_sampled_actions = torch.cat(self.buffer.sampled_actions, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, action_dim)
        old_sampled_transitions = torch.cat(self.buffer.sampled_transitions, dim = 0)[not_initials].detach().to(self.device) # (n_envs * n_steps, 2, height, width)


        # Calculate advantages
        advantages = returns - old_values
        # Normalize advantages

        n_updates = int(len(old_observations) // self.batch_size)
        # Optimize policy for K epochs
        self.encoder.train()
        self.actor_critic.train()
        actor_loss_sum = 0
        critic_loss_sum = 0
        entropy_loss_sum = 0
        cnt = 0
        accumumated_grads = [torch.zeros_like(p) for p in self.encoder.parameters()]
        self.optimizer_e.zero_grad()
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

                out = self.encoder(sampled_sampled_transitions, sampled_sampled_actions)
                new_context_mean = out[:, :self.context_dim]
                new_context_prec = 1 / torch.exp(torch.clamp(out[:, self.context_dim:], -10, 2))
                prev_context_prec = (1 / sampled_contexts[:, self.context_dim:] - new_context_prec).detach()
                # context_mean_rest * context_prec_rest + context_mean * context_prec = sampled_contexts[:, :self.context_dim] * sampled_contexts[:, self.context_dim:]
                prev_context_mean = (sampled_contexts[:, :self.context_dim] /sampled_contexts[:, self.context_dim:] - new_context_mean * new_context_prec).detach() / prev_context_prec
                context_var = 1/(prev_context_prec + new_context_prec)
                context_mean = (prev_context_mean * prev_context_prec + new_context_mean * new_context_prec) * context_var
                contexts = torch.cat([context_mean, context_var], dim = -1)

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
                for acc, p in zip(accumumated_grads, self.encoder.parameters()):
                    acc += p.grad.data
                self.optimizer_e.zero_grad()
                self.optimizer.step()
                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                entropy_loss_sum += entropy_loss.item()
                cnt += 1
        
        for acc, p in zip(accumumated_grads, self.encoder.parameters()):
            p.grad = acc.clone()
        self.optimizer_e.step()

        # Copy new weights into old policy
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())

        # clear buffer
        self.buffer.clear()

        # Calculate mean loss
        actor_loss_mean = actor_loss_sum / cnt
        critic_loss_mean = critic_loss_sum / cnt
        entropy_loss_mean = entropy_loss_sum / cnt
        d_train = {
            "return": return_mean,
            "return_std": return_std,
            "actor_loss": actor_loss_mean,
            "critic_loss": critic_loss_mean,
            "entropy_loss": entropy_loss_mean
        }
        return d_train

