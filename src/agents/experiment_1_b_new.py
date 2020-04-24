from src.agents.experiment import run_experiment

first_layer_features = ['EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
second_layer_features = 'weights'
n_permutations = 10
permutation_start_index = 0
ultimate_expert = False
expert_name = "test_transaction_costs_2_"
pre_training = True

# Train the base agent on extremely high transaction costs
total_timesteps = 10000
transaction_cost = 0.05
train = True
load = False
fine_tune = False
test = False
load_name = "final_base_agent_transaction_costs"
save_name = "final_base_agent_transaction_costs"
#run_experiment(train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)

# Fine-tune and the agent on extremely high transaction costs
total_timesteps = 20000
transaction_cost = 0.05
train = True
load = True
fine_tune = True
test = False
load_name = "final_base_agent_transaction_costs"
save_name = "final_base_agent_transaction_costs_{}"
run_experiment(train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)

# Test the fine-tuned agents on normal transaction costs
transaction_cost = 0.01
train = False
load = True
fine_tune = False
test = True
load_name = "final_base_agent_transaction_costs_{}"
save_name = "final_base_agent_transaction_costs_{}"
run_experiment(train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)
