from stable_baselines import SAC, PPO2, A2C
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
from src.agents.EmbeddedAgentPPO import EmbeddedAgentPPO
from stable_baselines.ppo2.ppo2 import *
from os.path import dirname as dirname


class StackedAgentPPO(PPO2):

	def __init__(self, policy_1, policy_2, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
				 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
				 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_1_kwargs=None, policy_2_kwargs=None,
				 full_tensorboard_log=False):

		super(StackedAgentPPO, self).__init__(policy_1, env, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef, learning_rate=learning_rate, vf_coef=vf_coef,
				 max_grad_norm=max_grad_norm, lam=lam, nminibatches=nminibatches, noptepochs=noptepochs, cliprange=cliprange, cliprange_vf=cliprange_vf,
				 verbose=verbose, tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model, policy_kwargs=policy_1_kwargs,
				 full_tensorboard_log=full_tensorboard_log)


		if policy_2 is not None: self.embedded_agent = EmbeddedAgentPPO(policy_2, env, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef, learning_rate=learning_rate, vf_coef=vf_coef,
				 max_grad_norm=max_grad_norm, lam=lam, nminibatches=nminibatches, noptepochs=noptepochs, cliprange=cliprange, cliprange_vf=cliprange_vf,
				 verbose=verbose, tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model, policy_kwargs=policy_2_kwargs,
				 full_tensorboard_log=full_tensorboard_log)

	def setup_model(self):
		with SetVerbosity(self.verbose):

			assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
															   "an instance of common.policies.ActorCriticPolicy."

			self.n_batch = self.n_envs * self.n_steps * self.env.group_count
			#TODO: we zaten hier met converteren

			n_cpu = multiprocessing.cpu_count()
			if sys.platform == 'darwin':
				n_cpu //= 2

			self.graph = tf.Graph()
			with self.graph.as_default():
				self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

				n_batch_step = None
				n_batch_train = None
				if issubclass(self.policy, RecurrentActorCriticPolicy):
					assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
						"the number of environments run in parallel should be a multiple of nminibatches."
					n_batch_step = self.n_envs
					n_batch_train = self.n_batch // self.nminibatches

				act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
										n_batch_step, reuse=False, **self.policy_kwargs)
				with tf.variable_scope("train_model", reuse=True,
									   custom_getter=tf_util.outer_scope_getter("train_model")):
					train_model = self.policy(self.sess, self.observation_space, self.action_space,
											  self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
											  reuse=True, **self.policy_kwargs)

				with tf.variable_scope("loss", reuse=False):
					self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
					self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
					self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
					self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
					self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
					self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
					self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

					neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
					self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

					vpred = train_model.value_flat

					# Value function clipping: not present in the original PPO
					if self.cliprange_vf is None:
						# Default behavior (legacy from OpenAI baselines):
						# use the same clipping as for the policy
						self.clip_range_vf_ph = self.clip_range_ph
						self.cliprange_vf = self.cliprange
					elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
						# Original PPO implementation: no value function clipping
						self.clip_range_vf_ph = None
					else:
						# Last possible behavior: clipping range
						# specific to the value function
						self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

					if self.clip_range_vf_ph is None:
						# No clipping
						vpred_clipped = train_model.value_flat
					else:
						# Clip the different between old and new value
						# NOTE: this depends on the reward scaling
						vpred_clipped = self.old_vpred_ph + \
							tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
											 - self.clip_range_vf_ph, self.clip_range_vf_ph)


					vf_losses1 = tf.square(vpred - self.rewards_ph)
					vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
					self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

					ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
					pg_losses = -self.advs_ph * ratio
					pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
																  self.clip_range_ph)
					self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
					self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
					self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
																	  self.clip_range_ph), tf.float32))
					loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

					tf.summary.scalar('entropy_loss', self.entropy)
					tf.summary.scalar('policy_gradient_loss', self.pg_loss)
					tf.summary.scalar('value_function_loss', self.vf_loss)
					tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
					tf.summary.scalar('clip_factor', self.clipfrac)
					tf.summary.scalar('loss', loss)

					with tf.variable_scope('model'):
						self.params = tf.trainable_variables()
						if self.full_tensorboard_log:
							for var in self.params:
								tf.summary.histogram(var.name, var)
					grads = tf.gradients(loss, self.params)
					if self.max_grad_norm is not None:
						grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
					grads = list(zip(grads, self.params))
				trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
				self._train = trainer.apply_gradients(grads)

				self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

				with tf.variable_scope("input_info", reuse=False):
					tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
					tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
					tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
					tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
					if self.clip_range_vf_ph is not None:
						tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

					tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
					tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

					if self.full_tensorboard_log:
						tf.summary.histogram('discounted_rewards', self.rewards_ph)
						tf.summary.histogram('learning_rate', self.learning_rate_ph)
						tf.summary.histogram('advantage', self.advs_ph)
						tf.summary.histogram('clip_range', self.clip_range_ph)
						tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
						tf.summary.histogram('old_value_pred', self.old_vpred_ph)
						if tf_util.is_image(self.observation_space):
							tf.summary.image('observation', train_model.obs_ph)
						else:
							tf.summary.histogram('observation', train_model.obs_ph)

				self.train_model = train_model
				self.act_model = act_model
				self.step = act_model.step
				self.proba_step = act_model.proba_step
				self.value = act_model.value
				self.initial_state = act_model.initial_state
				tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

				self.summary = tf.summary.merge_all()

	def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
			  reset_num_timesteps=True):
		# Transform to callable if needed
		self.learning_rate = get_schedule_fn(self.learning_rate)
		self.cliprange = get_schedule_fn(self.cliprange)
		cliprange_vf = get_schedule_fn(self.cliprange_vf)

		new_tb_log = self._init_num_timesteps(reset_num_timesteps)

		with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
				as writer:
			self._setup_learn(seed)

			runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
			self.episode_reward = np.zeros((self.n_envs,))

			ep_info_buf = deque(maxlen=100)
			t_first_start = time.time()

			n_updates = total_timesteps // self.n_batch
			for update in range(1, n_updates + 1):
				assert self.n_batch % self.nminibatches == 0
				batch_size = self.n_batch // self.nminibatches
				t_start = time.time()
				frac = 1.0 - (update - 1.0) / n_updates
				lr_now = self.learning_rate(frac)
				cliprange_now = self.cliprange(frac)
				cliprange_vf_now = cliprange_vf(frac)
				# true_reward is the reward without discount
				obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
				self.num_timesteps += self.n_batch
				ep_info_buf.extend(ep_infos)
				mb_loss_vals = []
				if states is None:  # nonrecurrent version
					update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
					inds = np.arange(self.n_batch)
					for epoch_num in range(self.noptepochs):
						np.random.shuffle(inds)
						for start in range(0, self.n_batch, batch_size):
							timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
																			self.n_batch + start) // batch_size)
							end = start + batch_size
							mbinds = inds[start:end]
							slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
							mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
																 update=timestep, cliprange_vf=cliprange_vf_now))
				else:  # recurrent version
					update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
					assert self.n_envs % self.nminibatches == 0
					env_indices = np.arange(self.n_envs)
					flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
					envs_per_batch = batch_size // self.n_steps
					for epoch_num in range(self.noptepochs):
						np.random.shuffle(env_indices)
						for start in range(0, self.n_envs, envs_per_batch):
							timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
																			self.n_envs + start) // envs_per_batch)
							end = start + envs_per_batch
							mb_env_inds = env_indices[start:end]
							mb_flat_inds = flat_indices[mb_env_inds].ravel()
							slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
							mb_states = states[mb_env_inds]
							mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
																 writer=writer, states=mb_states,
																 cliprange_vf=cliprange_vf_now))

				loss_vals = np.mean(mb_loss_vals, axis=0)
				t_now = time.time()
				fps = int(self.n_batch / (t_now - t_start))

				if writer is not None:
					self.episode_reward = total_episode_reward_logger(self.episode_reward,
																	  true_reward.reshape((self.n_envs, self.n_steps)),
																	  masks.reshape((self.n_envs, self.n_steps)),
																	  writer, self.num_timesteps)

				if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
					explained_var = explained_variance(values, returns)
					logger.logkv("serial_timesteps", update * self.n_steps)
					logger.logkv("n_updates", update)
					logger.logkv("total_timesteps", self.num_timesteps)
					logger.logkv("fps", fps)
					logger.logkv("explained_variance", float(explained_var))
					if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
						logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
						logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
					logger.logkv('time_elapsed', t_start - t_first_start)
					for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
						logger.logkv(loss_name, loss_val)
					logger.dumpkvs()

				if callback is not None:
					# Only stop training if return value is False, not when it is None. This is for backwards
					# compatibility with callbacks that have no return statement.
					if callback(locals(), globals()) is False:
						break

			return self

	def learn_old(self, total_timesteps, callback=None, seed=None,
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

	def save(self, save_name, cloudpickle=False):
		first_layer_path = "{}/saved_models/0_{}.model".format(dirname(__file__),save_name)
		second_layer_path = "{}/saved_models/1_{}.model".format(dirname(__file__),save_name)
		super(StackedAgentPPO,self).save(first_layer_path,cloudpickle=cloudpickle)

		self.embedded_agent.save(second_layer_path,cloudpickle=cloudpickle)

	@classmethod
	def load(cls, load_name, env=None, custom_objects=None, **kwargs):
		first_layer_path = "{}/saved_models/0_{}.model".format(dirname(__file__),load_name)
		second_layer_path = "{}/saved_models/1_{}.model".format(dirname(__file__),load_name)



		data, params = cls._load_from_file(first_layer_path, custom_objects=custom_objects)

		if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
			raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
							 "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
																			  kwargs['policy_kwargs']))

		model = cls(policy_1=data["policy"],policy_2=None, env=None, _init_setup_model=False)
		model.__dict__.update(data)
		model.__dict__.update(kwargs)
		model.set_env(env)
		model.setup_model()

		model.load_parameters(params)


		model.embedded_agent = EmbeddedAgent.load(second_layer_path,env=env,custom_objects=custom_objects,**kwargs)

		return model


