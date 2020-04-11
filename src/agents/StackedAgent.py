from stable_baselines import SAC
import sys
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
from src.agents.EmbeddedAgent import EmbeddedAgent


class StackedAgent(SAC):

	def __init__(self, policy_1, policy_2, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
				 learning_starts=100, train_freq=1, batch_size=64,
				 tau=0.005, ent_coef='auto', target_update_interval=1,
				 gradient_steps=1, target_entropy='auto', action_noise=None,
				 random_exploration=0.0, verbose=0, tensorboard_log=None,
				 _init_setup_model=True, policy_1_kwargs=None, policy_2_kwargs=None, full_tensorboard_log=False):

		super(StackedAgent, self).__init__(policy_1, env, gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size,
				 learning_starts=learning_starts, train_freq=train_freq, batch_size=batch_size,
				 tau=tau, ent_coef=ent_coef, target_update_interval=target_update_interval,
				 gradient_steps=gradient_steps, target_entropy=target_entropy, action_noise=action_noise,
				 random_exploration=random_exploration, verbose=verbose, tensorboard_log=tensorboard_log,
				 _init_setup_model=_init_setup_model, policy_kwargs=policy_1_kwargs, full_tensorboard_log=full_tensorboard_log)

		self.embedded_agent = EmbeddedAgent(policy_2, env, gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size,
				 learning_starts=learning_starts, train_freq=train_freq, batch_size=batch_size,
				 tau=tau, ent_coef=ent_coef, target_update_interval=target_update_interval,
				 gradient_steps=gradient_steps, target_entropy=target_entropy, action_noise=action_noise,
				 random_exploration=random_exploration, verbose=verbose, tensorboard_log=tensorboard_log,
				 _init_setup_model=_init_setup_model, policy_kwargs=policy_2_kwargs, full_tensorboard_log=full_tensorboard_log)

	def learn(self, total_timesteps, callback=None, seed=None,
			  log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

		new_tb_log = self._init_num_timesteps(reset_num_timesteps)

		if replay_wrapper is not None:
			self.replay_buffer = replay_wrapper(self.replay_buffer)

		with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
				as writer:

			self._setup_learn(seed)

			self.embedded_agent.start_learning(seed=seed,reset_num_timesteps=reset_num_timesteps)

			# Transform to callable if needed
			self.learning_rate = get_schedule_fn(self.learning_rate)
			# Initial learning rate
			current_lr = self.learning_rate(1)

			start_time = time.time()
			episode_rewards = [0.0]
			episode_successes = []
			if self.action_noise is not None:
				self.action_noise.reset()
			obs = self.env.reset()
			self.episode_reward = np.zeros((1,))
			ep_info_buf = deque(maxlen=100)
			n_updates = 0
			infos_values = []

			for step in range(total_timesteps):
				if callback is not None:
					# Only stop training if return value is False, not when it is None. This is for backwards
					# compatibility with callbacks that have no return statement.
					if callback(locals(), globals()) is False:
						break

				# Before training starts, randomly sample actions
				# from a uniform distribution for better exploration.
				# Afterwards, use the learned policy
				# if random_exploration is set to 0 (normal setting)
				reshaped_obs = obs.reshape((len(self.env.tickers),-1))
				first_layer_obs = reshaped_obs[:,:self.env.first_layer_feature_set_size]
				second_layer_obs = reshaped_obs[:,self.env.first_layer_feature_set_size:]
				second_layer_obs_final = np.zeros((0,1))

				observations = []
				actions = []
				for group in range(self.env.group_count):

					selected_obs = first_layer_obs[group*self.env.group_size:(group+1)*self.env.group_size,:].flatten()
					if (self.num_timesteps < self.learning_starts
						or np.random.rand() < self.random_exploration):
						# No need to rescale when sampling random action
						rescaled_action = action = self.env.action_space.sample()
					else:
						action = self.policy_tf.step(selected_obs[None], deterministic=False).flatten()
						# Add noise to the action (improve exploration,
						# not needed in general)
						if self.action_noise is not None:
							action = np.clip(action + self.action_noise(), -1, 1)
						# Rescale from [-1, 1] to the correct bounds
						rescaled_action = action * np.abs(self.action_space.low)

					rescaled_action = np.exp(rescaled_action)/sum(np.exp(rescaled_action))
					self.env.set_first_layer_portfolio(group,rescaled_action.flatten()[int(self.env.include_cash):])

					assert action.shape == self.env.action_space.shape

					observations.append(selected_obs)
					actions.append(action)

					scaled_second_layer_features = (second_layer_obs[group*self.env.group_size:(group+1)*self.env.group_size,:-1]*(rescaled_action.flatten()[int(self.env.include_cash):][:, np.newaxis])).sum(axis=0).flatten()
					scaled_second_layer_features = np.array(list(scaled_second_layer_features)+[self.env.second_layer_portfolio[group,0],]).reshape((-1,1))
					second_layer_obs_final = np.concatenate((second_layer_obs_final,scaled_second_layer_features))

				second_layer_obs_final = second_layer_obs_final.flatten()

				self.env, new_obs, reward, done, info = self.embedded_agent.learn_single_step(self.env,second_layer_obs_final,step,total_timesteps,callback=callback, seed=seed,
					log_interval=log_interval, tb_log_name=tb_log_name, replay_wrapper=replay_wrapper)

				obs = new_obs

				new_reshaped_obs = obs.reshape((len(self.env.tickers),-1))
				new_first_layer_obs = reshaped_obs[:,:self.env.first_layer_feature_set_size]
				for group in range(self.env.group_count):
					selected_new_obs = new_first_layer_obs[group*self.env.group_size:(group+1)*self.env.group_size,:].flatten()
					selected_obs = observations[group]
					selected_action = actions[group]
					selected_done = done if group == self.env.group_count-1 else False

					# Store transition in the replay buffer.
					self.replay_buffer.add(selected_obs, selected_action, reward, selected_new_obs, float(selected_done))
					selected_obs = selected_new_obs

					# Retrieve reward and episode length if using Monitor wrapper
					maybe_ep_info = info.get('episode')
					if maybe_ep_info is not None:
						ep_info_buf.extend([maybe_ep_info])

					if writer is not None:
						# Write reward per episode to tensorboard
						ep_reward = np.array([reward]).reshape((1, -1))
						ep_done = np.array([selected_done]).reshape((1, -1))
						self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
																		  ep_done, writer, self.num_timesteps)

					if step % self.train_freq == 0:
						mb_infos_vals = []
						# Update policy, critics and target networks
						for grad_step in range(self.gradient_steps):
							# Break if the warmup phase is not over
							# or if there are not enough samples in the replay buffer
							if not self.replay_buffer.can_sample(self.batch_size) \
							   or self.num_timesteps < self.learning_starts:
								break
							n_updates += 1
							# Compute current learning_rate
							frac = 1.0 - step / total_timesteps
							current_lr = self.learning_rate(frac)
							# Update policy and critics (q functions)
							mb_infos_vals.append(self._train_step(step, writer, current_lr))
							# Update target network
							if (step + grad_step) % self.target_update_interval == 0:
								# Update target network
								self.sess.run(self.target_update_op)
						# Log losses and entropy, useful for monitor training
						if len(mb_infos_vals) > 0:
							infos_values = np.mean(mb_infos_vals, axis=0)

				episode_rewards[-1] += reward
				if done:
					if self.action_noise is not None:
						self.action_noise.reset()
					if not isinstance(self.env, VecEnv):
						obs = self.env.reset()
					episode_rewards.append(0.0)

					maybe_is_success = info.get('is_success')
					if maybe_is_success is not None:
						episode_successes.append(float(maybe_is_success))

				if len(episode_rewards[-101:-1]) == 0:
					mean_reward = -np.inf
				else:
					mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

				num_episodes = len(episode_rewards)
				self.num_timesteps += 1
				# Display training infos
				if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
					fps = int(step / (time.time() - start_time))
					logger.logkv("episodes", num_episodes)
					logger.logkv("mean 100 episode reward", mean_reward)
					if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
						logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
						logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
					logger.logkv("n_updates", n_updates)
					logger.logkv("current_lr", current_lr)
					logger.logkv("fps", fps)
					logger.logkv('time_elapsed', int(time.time() - start_time))
					if len(episode_successes) > 0:
						logger.logkv("success rate", np.mean(episode_successes[-100:]))
					if len(infos_values) > 0:
						for (name, val) in zip(self.infos_names, infos_values):
							logger.logkv(name, val)
					logger.logkv("total timesteps", self.num_timesteps)
					logger.dumpkvs()
					# Reset infos:
					infos_values = []

			return self

	def predict(self, observation, state=None, mask=None, deterministic=True):
		observation = np.array(observation)
		reshaped_obs = observation.reshape((len(self.env.tickers),-1))
		first_layer_obs = reshaped_obs[:,:self.env.first_layer_feature_set_size]
		second_layer_obs = reshaped_obs[:,self.env.first_layer_feature_set_size:]

		second_layer_obs_final = np.zeros((0,1))
		for group in range(self.env.group_count):
			selected_obs = first_layer_obs[group*self.env.group_size:(group+1)*self.env.group_size,:].flatten()
			vectorized_env = self._is_vectorized_observation(selected_obs, self.observation_space)

			selected_obs = selected_obs.reshape((-1,) + self.observation_space.shape)
			actions = self.policy_tf.step(selected_obs, deterministic=deterministic)
			actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
			actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

			if not vectorized_env:
				actions = actions[0]

			actions = np.exp(actions)/sum(np.exp(actions))
			self.env.set_first_layer_portfolio(group,actions.flatten()[int(self.env.include_cash):])

			scaled_second_layer_features = (second_layer_obs[group*self.env.group_size:(group+1)*self.env.group_size,:-1]*(actions.flatten()[int(self.env.include_cash):][:, np.newaxis])).sum(axis=0).flatten()
			scaled_second_layer_features = np.array(list(scaled_second_layer_features)+[self.env.second_layer_portfolio[group,0],]).reshape((-1,1))
			second_layer_obs_final = np.concatenate((second_layer_obs_final,scaled_second_layer_features))

		second_layer_obs_final = second_layer_obs_final.flatten()
		actions, _ = self.embedded_agent.predict(second_layer_obs_final,state=state,mask=mask,deterministic=deterministic)

		actions = np.exp(actions)/sum(np.exp(actions))
		self.env.set_second_layer_portfolio(actions.flatten()[int(self.env.include_cash):])
		portfolio_vector = np.concatenate([actions[int(self.env.include_cash):].reshape((-1,1))]*self.env.group_size,axis=1).flatten()
		portfolio_vector = portfolio_vector*self.env.first_layer_portfolio.flatten()
		portfolio_vector = np.array([1-portfolio_vector.sum(),]+list(portfolio_vector))

		return portfolio_vector, None

