from src.agents.experiment import run_experiment

first_layer_features = ['random']
second_layer_features = []
n_permutations = 1
permutation_start_index = 0
transaction_cost = 0.00000000001
total_timesteps = 10000
ultimate_expert = False
expert_name = "random_input_"

# Train and test the agent
train = True
load = False
fine_tune = False
test = True
load_name = "random_input"
save_name = "random_input"
run_experiment(train,load,fine_tune,test,load_name,save_name,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features)
