import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3_BC_ensemble(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		num_nets=1,
		device=None
	):

		self.device = device
		self.num_nets = num_nets
		self.L_actor, self.L_actor_target, self.L_critic, self.L_critic_target = [], [], [], []
		for _ in range(self.num_nets):
			self.L_actor.append(Actor(state_dim, action_dim, max_action).to(self.device))
			self.L_actor_target.append(Actor(state_dim, action_dim, max_action).to(self.device))
			self.L_critic.append(Critic(state_dim, action_dim).to(self.device))
			self.L_critic_target.append(Critic(state_dim, action_dim).to(self.device))
		self.L_actor_optimizer, self.L_critic_optimizer = [], []
		for en_index in range(self.num_nets):
			self.L_actor_optimizer.append(torch.optim.Adam(self.L_actor[en_index].parameters(), lr=3e-4))
			self.L_critic_optimizer.append(torch.optim.Adam(self.L_critic[en_index].parameters(), lr=3e-4))

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0


	def ensemble_eval_select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		a = None
		for en_index in range(self.num_nets):
			_a = self.L_actor[en_index](state).cpu().data.numpy().flatten()
			if en_index == 0:
				a = _a
			else:
				a += _a
		a = a / self.num_nets
		return a


	def ensemble_expl_select_action(self, state, trans_parameter):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		current_Qs = []
		actions = []
		with torch.no_grad():
			for en_index in range(self.num_nets):
				action = self.L_actor[en_index](state)
				current_Q1, current_Q2 = self.L_critic[en_index](state, action)
				current_Q = torch.min(current_Q1, current_Q2)
				current_Qs.append(current_Q.squeeze(-1))
				actions.append(action.cpu().data.numpy().flatten())
		current_Qs = torch.stack(current_Qs, dim=-1)
		logits = current_Qs
		logits = logits * trans_parameter
		w_dist = torch.distributions.Categorical(logits=logits)
		w = w_dist.sample()
		w = w.squeeze(-1).detach().cpu().numpy()
		action = actions[w]

		return action


	def train(self, replay_buffer, offline_replay_buffer, batch_size=256, t=None, Utd=None):
		self.total_it += 1
		
		# Sample replay buffer 
		online_batch_size = batch_size * Utd
		offline_batch_size = batch_size * Utd

		online_state, online_action, online_next_state, online_reward, online_not_done = replay_buffer.sample(online_batch_size)
		offline_state, offline_action, offline_next_state, offline_reward, offline_not_done = offline_replay_buffer.sample(offline_batch_size)
		
		for i in range(Utd):
			state = torch.concat([online_state[batch_size*i:batch_size*(i+1)], offline_state[batch_size*i:batch_size*(i+1)]])
			action = torch.concat([online_action[batch_size*i:batch_size*(i+1)], offline_action[batch_size*i:batch_size*(i+1)]])
			next_state = torch.concat([online_next_state[batch_size*i:batch_size*(i+1)], offline_next_state[batch_size*i:batch_size*(i+1)]])
			reward = torch.concat([online_reward[batch_size*i:batch_size*(i+1)], offline_reward[batch_size*i:batch_size*(i+1)]])
			not_done = torch.concat([online_not_done[batch_size*i:batch_size*(i+1)], offline_not_done[batch_size*i:batch_size*(i+1)]])
			
			for en_index in range(self.num_nets):
				with torch.no_grad():
					# Select action according to policy and add clipped noise
					noise = (
						torch.randn_like(action) * self.policy_noise
					).clamp(-self.noise_clip, self.noise_clip)
					
					next_action = (
						self.L_actor_target[en_index](next_state) + noise
					).clamp(-self.max_action, self.max_action)

					# Compute the target Q value
					target_Q1, target_Q2 = self.L_critic_target[en_index](next_state, next_action)
					target_Q = torch.min(target_Q1, target_Q2)
					target_Q = reward + not_done * self.discount * target_Q

				# Get current Q estimates
				current_Q1, current_Q2 = self.L_critic[en_index](state, action)

				# Compute critic loss
				critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

				# Optimize the critic
				self.L_critic_optimizer[en_index].zero_grad()
				critic_loss.backward()
				self.L_critic_optimizer[en_index].step()

				# Update the frozen target models
				for param, target_param in zip(self.L_critic[en_index].parameters(), self.L_critic_target[en_index].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for en_index in range(self.num_nets):
			# Compute TD3 actor losse
			actor_loss = -self.L_critic[en_index].Q1(state, self.L_actor[en_index](state)).mean()

			# Optimize the actor 
			self.L_actor_optimizer[en_index].zero_grad()
			actor_loss.backward()
			self.L_actor_optimizer[en_index].step()

			for param, target_param in zip(self.L_actor[en_index].parameters(), self.L_actor_target[en_index].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		return current_Q1


	def load(self, policy_file, file_name):
		for en_index in range(self.num_nets):
			self.L_critic[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_critic", map_location=self.device))
			self.L_critic_optimizer[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_critic_optimizer", map_location=self.device))
			self.L_critic_target[en_index] = copy.deepcopy(self.L_critic[en_index])

			self.L_actor[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_actor", map_location=self.device))
			self.L_actor_optimizer[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_actor_optimizer", map_location=self.device))
			self.L_actor_target[en_index] = copy.deepcopy(self.L_actor[en_index])
			print('model ', en_index, ' load done...')