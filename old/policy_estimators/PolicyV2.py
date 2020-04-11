import tensorflow as tf


from stable_baselines.common.policies import ActorCriticPolicy

class PolicyV2(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(PolicyV2, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            flat = tf.keras.layers.Flatten()(self.processed_obs)

            x = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc_0')(flat)
            pi_latent = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc_1')(x)

            x1 = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc_0')(flat)
            vf_latent = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc_1')(x1)

            value_fn = tf.keras.layers.Dense(1, name='vf')(vf_latent)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})