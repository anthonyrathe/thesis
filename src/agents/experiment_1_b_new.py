from src.agents.experiment import run_experiment, all_groups, include_cash
from stable_baselines.common.policies import nature_cnn
from src.policies.SharedStackedPolicy import SharedStackedPolicy
import tensorflow as tf
group_count = 3
group_size = len(all_groups[0])

first_layer_features = ['EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
second_layer_features = []
n_permutations = 1
permutation_start_index = 0
ultimate_expert = False
expert_name = ""
window = 1
reward_type="p&l"
policy = SharedStackedPolicy
net_arch=[60,10,'merge',30,dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10])]
policy_kwargs = dict( net_arch=net_arch,
				 act_fun=tf.nn.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp")

# Train the base agent on extremely high transaction costs
pre_training = True
total_timesteps = 10000
transaction_cost = 1e-11
train = True
load = False
fine_tune = False
test = True
load_name = "test_high_transaction_costs"
save_name = "test_high_transaction_costs"
run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)

# Fine-tune the agent on extremely high transaction costs
total_timesteps = 20000
transaction_cost = 0.05
train = True
load = False
fine_tune = True
test = False
load_name = "test_high_transaction_costs"
save_name = "test_high_transaction_costs"
#run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)

# Test the fine-tuned agents on normal transaction costs
transaction_cost = 0.01
train = False
load = True
fine_tune = False
test = True
load_name = "test_high_transaction_costs"
save_name = "test_high_transaction_costs"
#run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)
