import silence_tensorflow.auto
import gym
gym.logger.set_level(40)

from src.agents.experiment import run_experiment, all_groups, include_cash, run_concurrent_experiment
from stable_baselines.common.policies import nature_cnn
from src.policies.SharedStackedPolicy import SharedStackedPolicy
from src.policies.SharedStackedRecurrentPolicy import SharedStackedRecurrentPolicy
from src.environments.StackedEnvBinary import StackedEnvBinary
from src.environments.StackedEnvDiff import StackedEnvDiff
from src.environments.StackedEnv import StackedEnv
import tensorflow as tf

group_count = 3
group_size = len(all_groups[0])

# +---------------------------------------------------------------------------------------+
# |                                  Super Experiment 1                                   |
# +---------------------------------------------------------------------------------------+
# | Super experiment testing combinations of:											  |
# |		- StackedEnv, StackedEnvDiff, StackedEnvBinary									  |
# |		- 0.0, 0.0002, 0.005 transaction costs											  |
# +---------------------------------------------------------------------------------------+
super_experiment_name = "experiment_1_a_retry"

first_layer_features = ['weights','EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
second_layer_features = []
n_permutations = 100
permutation_start_index = 0
ultimate_expert = False
expert_name = "no_second_layer"
pre_training = True
window = 1
reward_type="p&l"
policy = SharedStackedPolicy
net_arch=[60,10,'merge',30,dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10,])]
policy_kwargs = dict( net_arch=net_arch,
				 act_fun=tf.nn.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp")
n_environments = 3
steps_before_update = 512
agent_rebalances = False
total_timesteps = 10000

if __name__ == '__main__':
	for env_type in [StackedEnvDiff,]:
		for transaction_cost in [1e-11,]:
			print("Conducting experiment with the following parameters: {}, {}".format(str(env_type.__name__), str(transaction_cost)))

			experiment_name = "{}_{}_{}".format(super_experiment_name,str(env_type.__name__),str(transaction_cost))

			# Train the agent
			learning_rate = 1e-2
			train = True
			load = False
			fine_tune = False
			test = False
			load_name = experiment_name
			save_name = experiment_name

			print("	Training:")
			run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=n_environments,learning_rate=learning_rate,steps_before_update=steps_before_update,agent_rebalances=agent_rebalances,environment_type=env_type,verbose_experiment=False)

			# Fine-tune and test the agent
			learning_rate = 1e-2
			train = True
			load = True
			fine_tune = True
			test = True
			load_name = experiment_name
			save_name = "{}_{}".format(experiment_name,'{}')

			print("	Fine-tuning and testing:")
			run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=n_environments,learning_rate=learning_rate,steps_before_update=steps_before_update,agent_rebalances=agent_rebalances,environment_type=env_type,verbose_experiment=False)

