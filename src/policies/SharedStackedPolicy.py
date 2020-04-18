from stable_baselines.common.policies import ActorCriticPolicy, nature_cnn
import warnings
import tensorflow as tf
from itertools import zip_longest
import numpy as np

from stable_baselines.a2c.utils import linear

class SharedStackedPolicy(ActorCriticPolicy):
	"""
	Policy object that implements actor critic, using a feed forward neural network.

	:param sess: (TensorFlow session) The current TensorFlow session
	:param ob_space: (Gym Space) The observation space of the environment
	:param ac_space: (Gym Space) The action space of the environment
	:param n_env: (int) The number of environments to run
	:param n_steps: (int) The number of steps to run for each environment
	:param n_batch: (int) The number of batch to run (n_envs * n_steps)
	:param reuse: (bool) If the policy is reusable or not
	:param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
		(if None, default to [64, 64])
	:param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
		documentation for details).
	:param act_fun: (tf.func) the activation function to use in the neural network.
	:param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
	:param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
	:param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
	"""

	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_groups=2, group_size=4, n_first_layer_input_features=9, reuse=False, layers=None, net_arch=None,
				 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
		super(SharedStackedPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
												scale=(feature_extraction == "cnn"))

		self._kwargs_check(feature_extraction, kwargs)

		if layers is not None:
			warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
						  "(it has a different semantics though).", DeprecationWarning)
			if net_arch is not None:
				warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
							  DeprecationWarning)

		if net_arch is None:
			if layers is None:
				layers = [64, 64]
			net_arch = [dict(vf=layers, pi=layers)]

		with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
			if feature_extraction == "cnn":
				pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
			else:
				#flattened_inputs = tf.layers.flatten(self.processed_obs)
				# Split the observation in two sets: first layer features and second layer features
				pi_latent, vf_latent = mlp_extractor(tf.slice(self.processed_obs,[0,0,0,0],[-1,-1,-1,n_first_layer_input_features]),
													 tf.slice(self.processed_obs,[0,0,0,n_first_layer_input_features],[-1,-1,-1,-1]),
													 net_arch, act_fun, n_groups,group_size)

			self._value_fn = linear(vf_latent, 'vf', 1)

			self._proba_distribution, self._policy, self.q_value = \
				self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

		self._setup_init()

	def step(self, obs, state=None, mask=None, deterministic=False):
		if deterministic:
			action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
												   {self.obs_ph: obs})
		else:
			action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
												   {self.obs_ph: obs})
		return action, value, self.initial_state, neglogp

	def proba_step(self, obs, state=None, mask=None):
		return self.sess.run(self.policy_proba, {self.obs_ph: obs})

	def value(self, obs, state=None, mask=None):
		return self.sess.run(self.value_flat, {self.obs_ph: obs})

def mlp_extractor(observations_1, observations_2, net_arch, act_fun, n_groups, group_size):
	"""
	Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
	a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
	of them are shared between the policy network and the value network. It is assumed to be a list with the following
	structure:

	1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
	   If the number of ints is zero, there will be no shared layers.
	2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
	   It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
	   If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

	For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
	network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
	would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
	would be specified as [128, 128].

	:param flat_observations: (tf.Tensor) The observations to base policy and value function on.
	:param net_arch: ([int or dict]) The specification of the policy and value networks.
		See above for details on its formatting.
	:param act_fun: (tf function) The activation function to use for the networks.
	:return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
		If all layers are shared, then ``latent_policy == latent_value``
	"""

	# Split the observation into n_groups
	#dimensions_1 = tf.shape(observations_1)
	#assert dimensions_1[1] % n_groups == 0, "Error: we expected a dimension of [?,group_size*{},n_first_layer_features], but got {}".format(n_groups,dimensions_1)
	#group_size = dimensions_1[1] // n_groups
	latents = [tf.layers.flatten(tf.slice(observations_1,[0,0,i*group_size,0],[-1,-1,group_size,-1])) for i in range(n_groups)]

	flattened_observations_2 = tf.layers.flatten(observations_2)
	#latents_2 = [tf.slice(observations_2,[0,0,i*group_size,0],[-1,-1,group_size,-1]) for i in range(n_groups)]

	latent = None
	policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
	value_only_layers = []  # Layer sizes of the network that only belongs to the value network
	parallel = True	# Are we still building multiple parallel networks?
	# Iterate through the shared layers and build the shared parts of the network
	for idx, layer in enumerate(net_arch):
		if isinstance(layer, str):
			if layer == 'merge':
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
				raise Exception("Error: unknown keyword {}".format(layer))
		elif isinstance(layer, int):  # Check that this is a shared layer
			layer_size = layer
			if parallel:
				for i in range(len(latents)):
					latents[i] = act_fun(linear(latents[i], "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
			else:
				latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
		else:
			assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
			if 'pi' in layer:
				assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
				policy_only_layers = layer['pi']

			if 'vf' in layer:
				assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
				value_only_layers = layer['vf']
			break  # From here on the network splits up in policy and value network

	# Build the non-shared part of the network
	latents_policy = latents
	latent_policy = latent
	latents_value = latents
	latent_value = latent
	policy_parallel = parallel
	value_parallel = parallel
	for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
		if pi_layer_size is not None:
			if isinstance(pi_layer_size, str):
				if pi_layer_size == 'merge':
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

		if vf_layer_size is not None:
			if isinstance(vf_layer_size, str):
				assert vf_layer_size == 'merge', "Error: unknown keyword {}".format(vf_layer_size)
				assert value_parallel, "Error: cannot merge policy net_arch twice!"
				# Merge the parallel lanes
				latent_value = tf.concat(latents_value+[flattened_observations_2,],1)
				value_parallel = False
			else:
				assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
				if value_parallel:
					for i in range(len(latents_value)):
						latents_value[i] = act_fun(linear(latents_value[i], "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))
				else:
					latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

	assert latent_policy is not None, "Error: we forgot to merge the policy!"
	assert latent_value is not None, "Error: we forgot to merge the value!"

	return latent_policy, latent_value