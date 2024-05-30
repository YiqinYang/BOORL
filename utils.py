import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, mask_prob, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		print('capacity: ', max_size, 'state capacity: ', self.state.shape)
		self.device = device

		self.mask_prob = mask_prob

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		# print(self.size, self.ptr)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		bootstrap_mask = np.array(np.random.binomial(1, self.mask_prob, len(dataset['observations'])),  dtype = bool)
		self.state = dataset['observations'][bootstrap_mask]
		self.action = dataset['actions'][bootstrap_mask]
		self.reward = dataset['rewards'][bootstrap_mask].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'][bootstrap_mask].reshape(-1,1)
		self.next_state = dataset['next_observations'][bootstrap_mask]
		
		self.size = self.state.shape[0]
		self.ptr = self.state.shape[0]
		print('convert buffer size: ', self.size)


	def initialize_with_dataset(self, dataset):
		dataset_size = len(dataset['observations'])
		num_samples = dataset_size
		indices = np.arange(num_samples)
		self.state[:num_samples] = dataset['observations'][indices]
		self.action[:num_samples] = dataset['actions'][indices]
		self.next_state[:num_samples] = dataset['next_observations'][indices]
		self.reward[:num_samples] = dataset['rewards'].reshape(-1,1)[indices]
		self.not_done[:num_samples] = 1. - dataset['terminals'].reshape(-1,1)[indices]
		self.size = num_samples
		self.ptr = num_samples
		print('convert buffer size: ', self.size, ' ptr: ', self.ptr)


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std


class Online_ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		print('capacity: ', max_size, 'state capacity: ', self.state.shape)
		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]
		self.ptr = self.state.shape[0]
		print('convert buffer size: ', self.size)


	def initialize_with_dataset(self, dataset):
		dataset_size = len(dataset['observations'])
		num_samples = dataset_size
		indices = np.arange(num_samples)
		self.state[:num_samples] = dataset['observations'][indices]
		self.action[:num_samples] = dataset['actions'][indices]
		self.next_state[:num_samples] = dataset['next_observations'][indices]
		self.reward[:num_samples] = dataset['rewards'].reshape(-1,1)[indices]
		self.not_done[:num_samples] = 1. - dataset['terminals'].reshape(-1,1)[indices]
		self.size = num_samples
		self.ptr = num_samples
		print('convert buffer size: ', self.size, ' ptr: ', self.ptr)


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std