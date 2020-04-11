from stable_baselines import SAC
import sys
import time
import multiprocessing
from collections import deque
import warnings
from abc import ABC

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv
import gym
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.sac.sac import get_vars
from stable_baselines.common.policies import get_policy_from_name

class EmbeddedAgent(SAC):

	def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
				 learning_starts=100, train_freq=1, batch_size=64,
				 tau=0.005, ent_coef='auto', target_update_interval=1,
				 gradient_steps=1, target_entropy='auto', action_noise=None,
				 random_exploration=0.0, verbose=0, tensorboard_log=None,
				 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):

		super(ABC,self).__init__()

		requires_vec_env = False
		policy_base = SACPolicy
		if isinstance(policy, str) and policy_base is not None:
			self.policy = get_policy_from_name(policy_base, policy)
		else:
			self.policy = policy
		self.env = env
		self.verbose = verbose
		self._requires_vec_env = requires_vec_env
		self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
		self.observation_space = None
		self.action_space = None
		self.n_envs = None
		self._vectorize_action = False
		self.num_timesteps = 0
		self.graph = None
		self.sess = None
		self.params = None
		self._param_load_ops = None

		if env is not None:
			if isinstance(env, str):
				if self.verbose >= 1:
					print("Creating environment from the given name, wrapped in a DummyVecEnv.")
				self.env = env = DummyVecEnv([lambda: gym.make(env)])

			self.observation_space = env.second_layer_observation_space
			self.action_space = env.second_layer_action_space
			if requires_vec_env:
				if isinstance(env, VecEnv):
					self.n_envs = env.num_envs
				else:
					raise ValueError("Error: the model requires a vectorized environment, please use a VecEnv wrapper.")
			else:
				if isinstance(env, VecEnv):
					if env.num_envs == 1:
						self.env = _UnvecWrapper(env)
						self._vectorize_action = True
					else:
						raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
										 " environment.")
				self.n_envs = 1

		self.replay_buffer = None

		self.buffer_size = buffer_size
		self.learning_rate = learning_rate
		self.learning_starts = learning_starts
		self.train_freq = train_freq
		self.batch_size = batch_size
		self.tau = tau
		# In the original paper, same learning rate is used for all networks
		# self.policy_lr = learning_rate
		# self.qf_lr = learning_rate
		# self.vf_lr = learning_rate
		# Entropy coefficient / Entropy temperature
		# Inverse of the reward scale
		self.ent_coef = ent_coef
		self.target_update_interval = target_update_interval
		self.gradient_steps = gradient_steps
		self.gamma = gamma
		self.action_noise = action_noise
		self.random_exploration = random_exploration

		self.value_fn = None
		self.graph = None
		self.replay_buffer = None
		self.episode_reward = None
		self.sess = None
		self.tensorboard_log = tensorboard_log
		self.verbose = verbose
		self.params = None
		self.summary = None
		self.policy_tf = None
		self.target_entropy = target_entropy
		self.full_tensorboard_log = full_tensorboard_log

		self.obs_target = None
		self.target_policy = None
		self.actions_ph = None
		self.rewards_ph = None
		self.terminals_ph = None
		self.observations_ph = None
		self.action_target = None
		self.next_observations_ph = None
		self.value_target = None
		self.step_ops = None
		self.target_update_op = None
		self.infos_names = None
		self.entropy = None
		self.target_params = None
		self.learning_rate_ph = None
		self.processed_obs_ph = None
		self.processed_next_obs_ph = None
		self.log_ent_coef = None

		if _init_setup_model:
			self.setup_model()

	def start_learning(self, seed=None, reset_num_timesteps=True,):
		
		self._setup_learn(seed)

		# Transform to callable if needed
		self.learning_rate = get_schedule_fn(self.learning_rate)
		# Initial learning rate
		self.current_lr = self.learning_rate(1)

		self.start_time = time.time()
		self.episode_rewards = [0.0]
		self.episode_successes = []
		if self.action_noise is not None:
			self.action_noise.reset()
		self.episode_reward = np.zeros((1,))
		self.ep_info_buf = deque(maxlen=100)
		self.n_updates = 0
		self.infos_values = []

		self.new_tb_log = self._init_num_timesteps(reset_num_timesteps)

	def learn_single_step(self, env, obs, step, total_timesteps, callback=None, seed=None,
			  log_interval=4, tb_log_name="SAC", replay_wrapper=None):

		self.env = env

		if replay_wrapper is not None:
			self.replay_buffer = replay_wrapper(self.replay_buffer)

		with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, self.new_tb_log) \
				as writer:


			if callback is not None:
				# Only stop training if return value is False, not when it is None. This is for backwards
				# compatibility with callbacks that have no return statement.
				if callback(locals(), globals()) is False:
					return

			# Before training starts, randomly sample actions
			# from a uniform distribution for better exploration.
			# Afterwards, use the learned policy
			# if random_exploration is set to 0 (normal setting)
			if (self.num_timesteps < self.learning_starts
				or np.random.rand() < self.random_exploration):
				# No need to rescale when sampling random action
				rescaled_action = action = self.env.second_layer_action_space.sample()
			else:
				action = self.policy_tf.step(obs[None], deterministic=False).flatten()
				# Add noise to the action (improve exploration,
				# not needed in general)
				if self.action_noise is not None:
					action = np.clip(action + self.action_noise(), -1, 1)
				# Rescale from [-1, 1] to the correct bounds
				rescaled_action = action * np.abs(self.action_space.low)

			assert action.shape == self.env.second_layer_action_space.shape

			rescaled_action = np.exp(rescaled_action)/sum(np.exp(rescaled_action))
			self.env.set_second_layer_portfolio(rescaled_action[int(self.env.include_cash):])
			portfolio_vector = np.concatenate([rescaled_action[int(self.env.include_cash):].reshape((-1,1))]*self.env.group_size,axis=1).flatten()
			portfolio_vector = portfolio_vector*self.env.first_layer_portfolio.flatten()
			portfolio_vector = np.array([1-portfolio_vector.sum(),]+list(portfolio_vector))
			new_obs, reward, done, info = self.env.step(portfolio_vector)

			selected_obs = obs
			# TODO: is this correct? should we not use the first layer weights of the NEXT step??
			reshaped_obs = new_obs.reshape((len(self.env.tickers),-1))
			second_layer_obs = reshaped_obs[:,self.env.first_layer_feature_set_size:]
			selected_new_obs = np.zeros((0,1))
			for group in range(self.env.group_count):
				weights = self.env.first_layer_portfolio[group*self.env.group_size:(group+1)*self.env.group_size,:]
				scaled_second_layer_features = (second_layer_obs[group*self.env.group_size:(group+1)*self.env.group_size,:-1]*(weights.flatten()[:, np.newaxis])).sum(axis=0).flatten()
				scaled_second_layer_features = np.array(list(scaled_second_layer_features)+[self.env.second_layer_portfolio[group,0],]).reshape((-1,1))
				selected_new_obs = np.concatenate((selected_new_obs,scaled_second_layer_features))

			selected_new_obs = selected_new_obs.flatten()

			# Store transition in the replay buffer.
			self.replay_buffer.add(selected_obs, action, reward, selected_new_obs, float(done))
			obs = new_obs

			# Retrieve reward and episode length if using Monitor wrapper
			maybe_ep_info = info.get('episode')
			if maybe_ep_info is not None:
				self.ep_info_buf.extend([maybe_ep_info])

			if writer is not None:
				# Write reward per episode to tensorboard
				ep_reward = np.array([reward]).reshape((1, -1))
				ep_done = np.array([done]).reshape((1, -1))
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
					self.n_updates += 1
					# Compute current learning_rate
					frac = 1.0 - step / total_timesteps
					self.current_lr = self.learning_rate(frac)
					# Update policy and critics (q functions)
					mb_infos_vals.append(self._train_step(step, writer, self.current_lr))
					# Update target network
					if (step + grad_step) % self.target_update_interval == 0:
						# Update target network
						self.sess.run(self.target_update_op)
				# Log losses and entropy, useful for monitor training
				if len(mb_infos_vals) > 0:
					self.infos_values = np.mean(mb_infos_vals, axis=0)

			self.episode_rewards[-1] += reward
			if done:
				if self.action_noise is not None:
					self.action_noise.reset()
				if not isinstance(self.env, VecEnv):
					obs = self.env.reset()
				self.episode_rewards.append(0.0)

				maybe_is_success = info.get('is_success')
				if maybe_is_success is not None:
					self.episode_successes.append(float(maybe_is_success))

			if len(self.episode_rewards[-101:-1]) == 0:
				mean_reward = -np.inf
			else:
				mean_reward = round(float(np.mean(self.episode_rewards[-101:-1])), 1)

			num_episodes = len(self.episode_rewards)
			self.num_timesteps += 1
			# Display training infos
			if self.verbose >= 1 and done and log_interval is not None and len(self.episode_rewards) % log_interval == 0:
				fps = int(step / (time.time() - self.start_time))
				logger.logkv("episodes", num_episodes)
				logger.logkv("mean 100 episode reward", mean_reward)
				if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
					logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
					logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
				logger.logkv("self.n_updates", self.n_updates)
				logger.logkv("self.current_lr", self.current_lr)
				logger.logkv("fps", fps)
				logger.logkv('time_elapsed', int(time.time() - self.start_time))
				if len(self.episode_successes) > 0:
					logger.logkv("success rate", np.mean(self.episode_successes[-100:]))
				if len(self.infos_values) > 0:
					for (name, val) in zip(self.infos_names, self.infos_values):
						logger.logkv(name, val)
				logger.logkv("total timesteps", self.num_timesteps)
				logger.dumpkvs()
				# Reset infos:
				self.infos_values = []
			return self.env, new_obs, reward, done, info

	def setup_model(self):
		with SetVerbosity(self.verbose):
			self.graph = tf.Graph()
			with self.graph.as_default():
				n_cpu = multiprocessing.cpu_count()
				if sys.platform == 'darwin':
					n_cpu //= 2
				self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

				self.replay_buffer = ReplayBuffer(self.buffer_size)

				with tf.variable_scope("input", reuse=False):
					# Create policy and target TF objects
					self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
												 **self.policy_kwargs)
					self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
													 **self.policy_kwargs)

					# Initialize Placeholders
					self.observations_ph = self.policy_tf.obs_ph
					# Normalized observation for pixels
					self.processed_obs_ph = self.policy_tf.processed_obs
					self.next_observations_ph = self.target_policy.obs_ph
					self.processed_next_obs_ph = self.target_policy.processed_obs
					self.action_target = self.target_policy.action_ph
					self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
					self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
					self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
													 name='actions')
					self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

				with tf.variable_scope("model", reuse=False):
					# Create the policy
					# first return value corresponds to deterministic actions
					# policy_out corresponds to stochastic actions, used for training
					# logp_pi is the log probabilty of actions taken by the policy
					self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
					# Monitor the entropy of the policy,
					# this is not used for training
					self.entropy = tf.reduce_mean(self.policy_tf.entropy)
					#  Use two Q-functions to improve performance by reducing overestimation bias.
					qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
																	 create_qf=True, create_vf=True)
					qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
																	policy_out, create_qf=True, create_vf=False,
																	reuse=True)

					# Target entropy is used when learning the entropy coefficient
					if self.target_entropy == 'auto':
						# automatically set target entropy if needed
						self.target_entropy = -np.prod(self.env.second_layer_action_space.shape).astype(np.float32)
					else:
						# Force conversion
						# this will also throw an error for unexpected string
						self.target_entropy = float(self.target_entropy)

					# The entropy coefficient or entropy can be learned automatically
					# see Automating Entropy Adjustment for Maximum Entropy RL section
					# of https://arxiv.org/abs/1812.05905
					if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
						# Default initial value of ent_coef when learned
						init_value = 1.0
						if '_' in self.ent_coef:
							init_value = float(self.ent_coef.split('_')[1])
							assert init_value > 0., "The initial value of ent_coef must be greater than 0"

						self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
															initializer=np.log(init_value).astype(np.float32))
						self.ent_coef = tf.exp(self.log_ent_coef)
					else:
						# Force conversion to float
						# this will throw an error if a malformed string (different from 'auto')
						# is passed
						self.ent_coef = float(self.ent_coef)

				with tf.variable_scope("target", reuse=False):
					# Create the value network
					_, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
																		 create_qf=False, create_vf=True)
					self.value_target = value_target

				with tf.variable_scope("loss", reuse=False):
					# Take the min of the two Q-Values (Double-Q Learning)
					min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

					# Target for Q value regression
					q_backup = tf.stop_gradient(
						self.rewards_ph +
						(1 - self.terminals_ph) * self.gamma * self.value_target
					)

					# Compute Q-Function loss
					# TODO: test with huber loss (it would avoid too high values)
					qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
					qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

					# Compute the entropy temperature loss
					# it is used when the entropy coefficient is learned
					ent_coef_loss, entropy_optimizer = None, None
					if not isinstance(self.ent_coef, float):
						ent_coef_loss = -tf.reduce_mean(
							self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
						entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

					# Compute the policy loss
					# Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
					policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

					# NOTE: in the original implementation, they have an additional
					# regularization loss for the gaussian parameters
					# this is not used for now
					# policy_loss = (policy_kl_loss + policy_regularization_loss)
					policy_loss = policy_kl_loss


					# Target for value fn regression
					# We update the vf towards the min of two Q-functions in order to
					# reduce overestimation bias from function approximation error.
					v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
					value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

					values_losses = qf1_loss + qf2_loss + value_loss

					# Policy train op
					# (has to be separate from value train op, because min_qf_pi appears in policy_loss)
					policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
					policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

					# Value train op
					value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
					values_params = get_vars('model/values_fn')

					source_params = get_vars("model/values_fn/vf")
					target_params = get_vars("target/values_fn/vf")

					# Polyak averaging for target variables
					self.target_update_op = [
						tf.assign(target, (1 - self.tau) * target + self.tau * source)
						for target, source in zip(target_params, source_params)
					]
					# Initializing target to match source variables
					target_init_op = [
						tf.assign(target, source)
						for target, source in zip(target_params, source_params)
					]

					# Control flow is used because sess.run otherwise evaluates in nondeterministic order
					# and we first need to compute the policy action before computing q values losses
					with tf.control_dependencies([policy_train_op]):
						train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

						self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
						# All ops to call during one training step
						self.step_ops = [policy_loss, qf1_loss, qf2_loss,
										 value_loss, qf1, qf2, value_fn, logp_pi,
										 self.entropy, policy_train_op, train_values_op]

						# Add entropy coefficient optimization operation if needed
						if ent_coef_loss is not None:
							with tf.control_dependencies([train_values_op]):
								ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
								self.infos_names += ['ent_coef_loss', 'ent_coef']
								self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

					# Monitor losses and entropy in tensorboard
					tf.summary.scalar('policy_loss', policy_loss)
					tf.summary.scalar('qf1_loss', qf1_loss)
					tf.summary.scalar('qf2_loss', qf2_loss)
					tf.summary.scalar('value_loss', value_loss)
					tf.summary.scalar('entropy', self.entropy)
					if ent_coef_loss is not None:
						tf.summary.scalar('ent_coef_loss', ent_coef_loss)
						tf.summary.scalar('ent_coef', self.ent_coef)

					tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

				# Retrieve parameters that must be saved
				self.params = get_vars("model")
				self.target_params = get_vars("target/values_fn/vf")

				# Initialize Variables and target network
				with self.sess.as_default():
					self.sess.run(tf.global_variables_initializer())
					self.sess.run(target_init_op)

				self.summary = tf.summary.merge_all()
