import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
from torch.utils.tensorboard import SummaryWriter


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, agent_i, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Agent {agent_i} Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=False)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_seed_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	mask_prob = 0.9
	num_nets = 5
	device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
	writer = SummaryWriter(f"offline_runs/{'offline_TD3_BC' + '_mask_' + str(mask_prob) + '_ensemble_' + str(num_nets) + '_' + str(args.env) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed)}/")
	
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha,
		"device": device
	}

	datasets = []
	for _ in range(num_nets):
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim, mask_prob, device)
		replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
		if args.normalize:
			mean,std = replay_buffer.normalize_states() 
		else:
			mean,std = 0,1
		datasets.append(replay_buffer)
	print('='*30, '\n', 'normalize: ', args.normalize, ' mean: ', mean, ' std: ', std, '\n', '='*30)
	
	agents = []
	for _ in range(num_nets):
		policy = TD3_BC.TD3_BC(**kwargs)
		agents.append(policy)
	
	evaluations = []
	training_iters = 0
	for t in range(int(args.max_timesteps)):
		for agent_i in range(num_nets):
			Q1 = agents[agent_i].train(datasets[agent_i], args.batch_size)

			# Evaluate episode
			if (t + 1) % args.eval_freq == 0:
				training_iters += args.eval_freq
				print(f"Time steps: {t+1}")
				
				d4rl_score = eval_policy(agents[agent_i], args.env, args.seed, mean, std, agent_i)
				
				print('save model: ', args.save_model)
				if args.save_model:
					os.makedirs(f'./models/non_normalize_nets_{str(num_nets)}_mask_{str(mask_prob)}', exist_ok=True)
					agents[agent_i].save(f"./models/non_normalize_nets_{str(num_nets)}_mask_{str(mask_prob)}/{file_name}_agent_{str(agent_i)}")

				writer.add_scalar(f'd4rl score {str(agent_i)}', d4rl_score, t)
				writer.add_scalar(f'Q value {str(agent_i)}', Q1.mean().detach().cpu().numpy(), t)
				print('writing score done...')