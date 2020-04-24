from src.agents.experiment import run_experiment

first_layer_features = ['EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
second_layer_features = []
n_permutations = 100
permutation_start_index = 0
transaction_cost = 0.00000000001
total_timesteps = 10000
ultimate_expert = False
expert_name = ""

# Train the base agent
train = True
load = False
fine_tune = False
test = False
load_name = "final_base_agent"
save_name = "final_base_agent"
run_experiment(train,load,fine_tune,test,load_name,save_name,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)

# Fine-tune and test the agent
train = True
load = True
fine_tune = True
test = True
load_name = "final_base_agent"
save_name = "final_base_agent_{}"
run_experiment(train,load,fine_tune,test,load_name,save_name,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)