from stable_baselines.common.policies import LstmPolicy, nature_cnn
import warnings
import tensorflow as tf
import numpy as np

from stable_baselines.a2c.utils import linear, batch_to_seq, seq_to_batch, lstm

class SharedStackedRecurrentPolicy(LstmPolicy):
	"""
	Policy object that implements actor critic, using LSTMs.

	:param sess: (TensorFlow session) The current TensorFlow session
	:param ob_space: (Gym Space) The observation space of the environment
	:param ac_space: (Gym Space) The action space of the environment
	:param n_env: (int) The number of environments to run
	:param n_steps: (int) The number of steps to run for each environment
	:param n_batch: (int) The number of batch to run (n_envs * n_steps)
	:param n_lstm: (int) The number of LSTM cells (for recurrent policies)
	:param reuse: (bool) If the policy is reusable or not
	:param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
	:param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
		format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
	:param act_fun: (tf.func) the activation function to use in the neural network.
	:param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
	:param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
	:param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
	:param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
	"""

	recurrent = True

	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_groups=2, group_size=4, n_first_layer_input_features=9, n_lstm=256, reuse=False, layers=None,
				 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
				 **kwargs):
		# state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
		super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
										 state_shape=(2 * n_lstm, ), reuse=reuse,
										 scale=(feature_extraction == "cnn"))

		self._kwargs_check(feature_extraction, kwargs)

		if net_arch is None:  # Legacy mode
			if layers is None:
				layers = [64, 64]
			else:
				warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

			with tf.variable_scope("model", reuse=reuse):
				if feature_extraction == "cnn":
					extracted_features = cnn_extractor(self.processed_obs, **kwargs)
				else:
					extracted_features = tf.layers.flatten(self.processed_obs)
					for i, layer_size in enumerate(layers):
						extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
															init_scale=np.sqrt(2)))
				input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
				masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
				rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
											 layer_norm=layer_norm)
				rnn_output = seq_to_batch(rnn_output)
				value_fn = linear(rnn_output, 'vf', 1)

				self._proba_distribution, self._policy, self.q_value = \
					self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

			self._value_fn = value_fn
		else:  # Use the new net_arch parameter
			if layers is not None:
				warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
			if feature_extraction == "cnn":
				raise NotImplementedError()

			with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
				observations_1 = tf.slice(self.processed_obs,[0,0,0,0],[-1,-1,-1,n_first_layer_input_features])
				observations_2 = tf.slice(self.processed_obs,[0,0,0,n_first_layer_input_features],[-1,-1,-1,-1])

				latents = [tf.layers.flatten(tf.slice(observations_1,[0,0,i*group_size,0],[-1,-1,group_size,-1])) for i in range(n_groups)]

				flattened_observations_2 = tf.layers.flatten(observations_2)


				latent = None
				policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
				value_only_layers = []  # Layer sizes of the network that only belongs to the value network
				parallel = True	# Are we still building multiple parallel networks?

				# Iterate through the shared layers and build the shared parts of the network
				lstm_layer_constructed = False
				for idx, layer in enumerate(net_arch):
					if isinstance(layer, int):  # Check that this is a shared layer
						layer_size = layer
						if parallel:
							for i in range(len(latents)):
								latents[i] = act_fun(linear(latents[i], "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
						else:
							latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
					elif layer == "lstm":
						if lstm_layer_constructed:
							raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
						if parallel:
							for i in range(len(latents)):
								input_sequence = batch_to_seq(latents[i], self.n_env, n_steps)
								masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
								rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
															 layer_norm=layer_norm)
								latents[i] = seq_to_batch(rnn_output)
						else:
							input_sequence = batch_to_seq(latent, self.n_env, n_steps)
							masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
							rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
														 layer_norm=layer_norm)
							latent = seq_to_batch(rnn_output)
						lstm_layer_constructed = True
					elif layer == 'merge':
						assert parallel, "Error: cannot merge net_arch twice!"
						# Merge the parallel lanes
						latent = tf.concat(latents+[flattened_observations_2,],1)
						parallel = False
					elif layer == 'softmax':
						if parallel:
							for i in range(len(latents)):
								latents[i] = tf.nn.softmax(latents[i])
						else:
							latent = tf.nn.softmax(latent)
					else:
						assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
						if 'pi' in layer:
							assert isinstance(layer['pi'],
											  list), "Error: net_arch[-1]['pi'] must contain a list of integers."
							policy_only_layers = layer['pi']

						if 'vf' in layer:
							assert isinstance(layer['vf'],
											  list), "Error: net_arch[-1]['vf'] must contain a list of integers."
							value_only_layers = layer['vf']
						break  # From here on the network splits up in policy and value network

				# Build the non-shared part of the policy-network
				latents_policy = latents
				latent_policy = latent
				latents_value = latents
				latent_value = latent
				policy_parallel = parallel
				value_parallel = parallel
				for idx, pi_layer_size in enumerate(policy_only_layers):
					if isinstance(pi_layer_size, str):
						if pi_layer_size == "lstm":
							raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
						elif pi_layer_size == 'merge':
							assert policy_parallel, "Error: cannot merge policy net_arch twice!"
							# Merge the parallel lanes
							latent_policy = tf.concat(latents_policy+[flattened_observations_2,],1)
							policy_parallel = False
						elif pi_layer_size == 'softmax':
							if policy_parallel:
								for i in range(len(latents_policy)):
									latents_policy[i] = tf.nn.softmax(latents_policy[i])
							else:
								latent_policy = tf.nn.softmax(latent_policy)
						else:
							raise Exception("Error: unknown keyword {}".format(pi_layer_size))
					else:
						assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
						if policy_parallel:
							for i in range(len(latents_policy)):
								latents_policy[i] = act_fun(linear(latents_policy[i], "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))
						else:
							latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

				# Build the non-shared part of the value-network
				for idx, vf_layer_size in enumerate(value_only_layers):
					if isinstance(vf_layer_size, str):
						if vf_layer_size == "lstm":
							raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
													  "network.")
						elif vf_layer_size == 'merge':
							assert value_parallel, "Error: cannot merge policy net_arch twice!"
							# Merge the parallel lanes
							latent_value = tf.concat(latents_value+[flattened_observations_2,],1)
							value_parallel = False
						else:
							raise Exception("Error: unknown keyword {}".format(vf_layer_size))
					else:
						assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
						if value_parallel:
							for i in range(len(latents_value)):
								latents_value[i] = act_fun(linear(latents_value[i], "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))
						else:
							latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

				assert latent_policy is not None, "Error: we forgot to merge the policy!"
				assert latent_value is not None, "Error: we forgot to merge the value!"

				if not lstm_layer_constructed:
					raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

				self._value_fn = linear(latent_value, 'vf', 1)
				# TODO: why not init_scale = 0.001 here like in the feedforward
				self._proba_distribution, self._policy, self.q_value = \
					self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
		self._setup_init()