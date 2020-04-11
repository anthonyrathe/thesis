from old.environments.BasicEnv import BasicEnv
import numpy as np
import pandas as pd

from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy, FeedForwardPolicy
from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from old.policy_estimators.PolicyV2 import PolicyV2
import itertools, math, random
import tensorflow as tf

train_test_split = 0.70
transaction_cost = 0.0035
overall_profit = True

#tickers = ['PEP','KO','KDP','MNST']
tickers = ['BBY','TGT','WMT','COST']
#tickers = ['FB','GOOGL','AAPL','AMZN']

#tickers = ['PEP','KO'] -> 1.552 VS 1.553
#tickers = ['PEP','KDP'] -> 2.244 VS 2.329
#tickers = ['PEP','MNST'] -> 1.211 VS 1.272
#tickers = ['KO','KDP'] -> 1.855 VS 1.850
#tickers = ['KO','MNST'] -> 1.360 VS 1.235
#tickers = ['KDP','MNST'] -> 0.0599 VS 0.0349

#tickers = ['KO','PEP'] -> 1.601 VS 1.553
#tickers = ['KDP','PEP'] -> 1.162 VS 2.329
#tickers = ['MNST','PEP'] -> 0.730 VS 1.272
#tickers = ['KDP','KO'] -> 0.876 VS 1.850
#tickers = ['MNST','KO'] -> 0.806 VS 1.235
#tickers = ['MNST','KDP'] -> 0.0333 VS 0.0349

# Custom MLP policy
class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[100, 100],
                                           layer_norm=True,
                                           feature_extraction="mlp",
										   act_fun=tf.nn.relu)

# Rolling std deviation, averages, exp mean, # consecutive movements, -> ts fresh, acceleration, velocity
# sharpe als reward
# convolution
# peer group identifier in input
# meer interessante trades uitlokken door meer gedecorreleerde aandelen in te voeren???

def get_env(tickers):
	return BasicEnv(tickers,transaction_cost,train_test_split,realized=False,overall_profit=overall_profit)
	#env_wrapped = DummyVecEnv([lambda: env])

def get_model(load,env,model_name="test"):
	if load:
		try:
			return SAC.load("./saved_models/{}".format(model_name), env, verbose=1)
		except:
			return SAC(CustomSACPolicy, env,verbose=1)
	else:
		return SAC(CustomSACPolicy, env, verbose=1)

def train_model(model,steps,model_name="test"):

	model.learn(total_timesteps=steps, log_interval=1)
	model.save("./saved_models/{}".format(model_name))
	return model


def test_model(models,env,transaction_cost=transaction_cost):
	test = True
	obs = env.reset(test=test)
	env.transaction_cost=transaction_cost

	env_base = BasicEnv(tickers,transaction_cost,train_test_split)
	env_base.reset(test=test)

	done = False
	profit = [1,]
	profit_base = [1,]
	dates = []
	actions = []
	while not done:
		# Get actions of all models in ensemble
		action = [model.predict(obs)[0] for model in models]
		# Perform softmax on actions
		action = [np.exp(x)/sum(np.exp(x)) for x in action]
		# Take average
		action = np.array(action).mean(axis=0)

		obs, rewards, done, info = env.step(action,test=test)
		obs_base, rewards_base, done_base, info_base = env_base.step(np.array([0,]+[1/len(tickers),]*len(tickers)),test=test)

		# Rescale rewards to actual profits
		#rewards=math.log(rewards)
		#rewards_base=math.log(rewards_base)
		if not env.overall_profit:rewards=profit[-1]*math.exp(rewards)
		if not env_base.overall_profit:rewards_base=profit_base[-1]*(rewards_base+1)

		profit.append(rewards)
		profit_base.append(rewards_base)
		dates.append(info['date'])
		# Perform softmax on action, as the basicsimulator will do so
		actions.append(np.exp(action)/sum(np.exp(action)))
		print("{}: {}% VS {}% base profit".format(info['date'],round((profit[-1]-1)*100,2), round((profit_base[-1]-1)*100,2)))

	risk_free = (1.02**((dates[-1]-dates[0]).days/365))**(1/len(dates))
	risk_free = np.array([1,]+[risk_free**i for i in range(1,len(dates)+1)])

	excess_returns = np.array(profit)-risk_free
	excess_returns_base = np.array(profit_base)-risk_free

	sharpe = excess_returns.mean()/excess_returns.std()
	sharpe_base = excess_returns_base.mean()/excess_returns_base.std()

	log = {'Date':dates,'Weights':actions,'Profit':profit[1:],'Base Profit':profit_base[1:]}

	print("Sharpe: {} VS {} base".format(sharpe,sharpe_base))
	return log

# Maybe not all combinations should be tried, as this might lead to overfitting
try_all_combinations = 5
load = True
train = True
# Train steps should be at least 500 so as to include all training data at least once
train_steps = 1000
# Ensembles make sense since the algorithm has demonstrated severe sensitivity to randomness (albeit from initialization or stochastic gradient descent)
ensemble = 1
test_name = "fixed_bug"

models = []
for i in range(ensemble):
	#combinations = list(itertools.combinations(tickers,2))
	combinations = list(itertools.permutations(tickers,len(tickers)))
	if try_all_combinations>0: combinations = random.sample(list(itertools.permutations(tickers,len(tickers))),try_all_combinations)

	print(combinations)
	start = True
	if train:
		for combination in combinations:
			print(combination)
			env = get_env(combination)
			model = get_model(not start or load,env)
			model = train_model(model,train_steps,model_name="{}_{}".format(test_name,i))
			start = False

	combination = combinations[-1]
	env = get_env(tickers)
	model = get_model(True,env,model_name="{}_{}".format(test_name,i))
	models.append(model)
	print(combination)
	#test_model(model,env)

log = test_model(models,env)
for ticker in range(len(tickers)):
	log[tickers[ticker]] = [row[ticker+1] for row in log['Weights']]
log['cash'] = [row[0] for row in log['Weights']]
pd.DataFrame(log).to_csv("./logs/{}.csv".format(test_name))
