from src.agents.experiment import run_experiment, all_groups, include_cash
from stable_baselines.common.policies import nature_cnn
from src.policies.SharedStackedPolicy import SharedStackedPolicy
import tensorflow as tf
group_count = 3
group_size = len(all_groups[0])

first_layer_features = ['EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
second_layer_features = 'weights'
n_permutations = 10
permutation_start_index = 0
ultimate_expert = False
expert_name = "long_term_test_1"
pre_training = True
window = 1
reward_type="long_term_p&l_5"
policy = SharedStackedPolicy
#net_arch=[60,10,'merge_parallel',30,dict(pi=[20,'second_input',group_count*group_size+int(include_cash)],vf=[10,'second_input'])]
#net_arch=[60,10,'merge',30,dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10,])]
#net_arch=[120,10,'merge_parallel',dict(pi=[20,'second_input',group_count*group_size+int(include_cash)],vf=[10,'second_input',])]
net_arch=[60,10,'merge_parallel',30,dict(pi=[20,'second_input',group_count*group_size+int(include_cash)],vf=[10,'second_input',])]
policy_kwargs = dict( net_arch=net_arch,
				 act_fun=tf.nn.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp")
n_environments = 8
steps_before_update = 512*4
agent_rebalances = False


# Train the agent
transaction_cost = 0.01
total_timesteps = 200000
learning_rate = 1e-3
train = True
load = False
fine_tune = False
test = False
load_name = "testje_2"
save_name = "testje_2"
if __name__ == '__main__':
	run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=n_environments,learning_rate=learning_rate,steps_before_update=steps_before_update,agent_rebalances=agent_rebalances)

# Fine-tune the agent
transaction_cost = 0.01
learning_rate = 1e-4
total_timesteps = 200000
train = True
load = True
fine_tune = True
test = False
load_name = "testje_2"
save_name = "testje_2_{}"
if __name__ == '__main__':
	run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=n_environments,learning_rate=learning_rate,steps_before_update=steps_before_update,agent_rebalances=agent_rebalances)

# Test the agent
transaction_cost = 0.002
train = False
load = True
fine_tune = True
test = True
load_name = "testje_2_{}"
save_name = "testje_2_{}"
if __name__ == '__main__':
	run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=n_environments,learning_rate=learning_rate,steps_before_update=steps_before_update,agent_rebalances=agent_rebalances)

