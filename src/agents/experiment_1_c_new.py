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
# |                                   Experiment 1 c                                      |
# +---------------------------------------------------------------------------------------+
# | Cfr. experiment 1 b, but with StackedEnv.											  |
# +---------------------------------------------------------------------------------------+

first_layer_features = ['weights','EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
second_layer_features = ['weights']
n_permutations = 100
permutation_start_index = 0
ultimate_expert = False
expert_name = ""
pre_training = True
window = 1
reward_type="p&l"
policy = SharedStackedPolicy
env_type = StackedEnv
#net_arch=[60,10,'merge_parallel',30,dict(pi=[20,'second_input',group_count*group_size+int(include_cash)],vf=[10,'second_input'])]
net_arch=[60,10,'merge',30,dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10,])]
#net_arch=[120,10,'merge_parallel',dict(pi=[20,'second_input',group_count*group_size+int(include_cash)],vf=[10,'second_input',])]
#net_arch=[60,10,'merge_parallel',30,'second_input',dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10,])]
#net_arch=[60,10,'merge',30,'lstm',dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10,])]
#net_arch=[60,10,'merge_parallel',30,'second_input',dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10,])]
#net_arch=[60,10,'merge_parallel',30,'second_input',30,30,dict(pi=[group_count*group_size+int(include_cash)],vf=[])]
policy_kwargs = dict( net_arch=net_arch,
				 act_fun=tf.nn.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp")
n_environments = 8
steps_before_update = 512
agent_rebalances = False


# Train the agent
transaction_cost = 0.0002
total_timesteps = 10000
learning_rate = 1e-3
train = True
load = False
fine_tune = False
test = False
load_name = "stacked_env"
save_name = "stacked_env"
if __name__ == '__main__':
	run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=n_environments,learning_rate=learning_rate,steps_before_update=steps_before_update,agent_rebalances=agent_rebalances,environment_type=env_type)

# Fine-tune and test the agent
transaction_cost = 0.0002
learning_rate = 1e-4
total_timesteps = 10000
train = True
load = True
fine_tune = True
test = True
load_name = "stacked_env"
save_name = "stacked_env_{}"
if __name__ == '__main__':
	run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=n_environments,learning_rate=learning_rate,steps_before_update=steps_before_update,agent_rebalances=agent_rebalances,environment_type=env_type)

