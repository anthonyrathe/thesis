from src.agents.experiment import run_experiment, all_groups, include_cash
from stable_baselines.common.policies import nature_cnn
from src.policies.SharedStackedPolicy import SharedStackedPolicy
import tensorflow as tf
group_count = 3
group_size = len(all_groups[0])

# Please note that this experiment originally used the 'profit_portfolio_old' method of BasicSimulator, instead of 'profit_portfolio'
# However, the obtained results are very similar in both cases
# Also, the discount factor (gamma) used to be 0.99 instead of 1.0

first_layer_features = ['EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
second_layer_features = []
n_permutations = 100
permutation_start_index = 0
transaction_cost = 0.00000000001
total_timesteps = 10000
ultimate_expert = False
expert_name = ""
pre_training = True
window = 1
reward_type="p&l"
policy = SharedStackedPolicy
net_arch=[60,10,'merge',30,dict(pi=[20,group_count*group_size+int(include_cash)],vf=[10])]
policy_kwargs = dict( net_arch=net_arch,
				 act_fun=tf.nn.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp")

# Train the base agent
train = True
load = False
fine_tune = False
test = False
load_name = "final_base_agent"
save_name = "final_base_agent"
run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)

# Fine-tune and test the agent
train = True
load = True
fine_tune = True
test = True
load_name = "final_base_agent"
save_name = "final_base_agent_{}"
run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)