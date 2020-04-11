from old.environments.MultiRewardEnv import MultiRewardEnv
from old.environments.PairEnv import PairEnv
from old.agents.peer_groups import get_peer_groups
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
reward_function = "sharpe_diff"
window_size = 10
include_cash = True

peer_group_aware = True
load = False
train = True
# Train steps should be at least 500 so as to include all training data at least once
train_steps = 1000
test_name = "stockpair_pga_single_layer_ws10_sharpediff_ext_features_100stocks"

#tickers = [['PEP','KO','MNST'],['BBY','TGT','WMT','COST'],['FB','GOOGL','AAPL','AMZN']]
#tickers = get_peer_groups(draw_random=100)
tickers = [['TJX'], ['NKE', 'VFC'], ['GM'], ['M'], ['DG'], ['MHK'], ['HD'], ['LEN'], ['CCL'], ['WHR'], ['YUM', 'SBUX'], ['ORLY', 'GPC', 'AN'], ['ADM'], ['CVS'], ['CLX', 'CL'], ['MKC', 'GIS'], ['MO'], ['EQT'], ['COP', 'MUR', 'WPX'], ['RIG', 'NE'], ['EOG', 'COG', 'BTU'], ['KMI'], ['STI', 'JPM', 'HBAN'], ['DFS'], ['PFG', 'MS', 'LM', 'GS', 'BEN', 'AMP'], ['HIG', 'CINF'], ['WY', 'PSA', 'AVB'], ['LIFE', 'GILD', 'BIIB'], ['PKI', 'ISRG', 'BDX', 'A'], ['PDCO'], ['ZTS', 'MRK'], ['NOC'], ['FAST'], ['PBI'], ['FLR'], ['APH'], ['WM'], ['SRCL', 'ROP', 'R', 'MMM', 'IR', 'EMR'], ['NSC'], ['NLSN'], ['AAPL'], ['ADS'], ['ACN', 'JBL'], ['AKAM', 'CRM', 'VRSN', 'NTAP', 'NFLX', 'FIS'], ['JNPR', 'CSCO'], ['MU', 'LSI', 'INTC', 'ALTR'], ['MSFT'], ['AVY'], ['OI', 'BLL'], ['SEE'], ['IFF'], ['ETR', 'DUK', 'AES', 'AEP'], ['CNP', 'NRG']]
tickers_flattened = [item for peer_group in tickers for item in peer_group]
print(len(tickers_flattened))
print('ABBV')


# Custom MLP policy
class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[100,100],
                                           layer_norm=True,
                                           feature_extraction="mlp",
										   act_fun=tf.nn.tanh)

# Rolling std deviation, averages, exp mean, # consecutive movements, -> ts fresh, acceleration, velocity
# sharpe als reward
# convolution
# peer group identifier in input
# meer interessante trades uitlokken door meer gedecorreleerde aandelen in te voeren???

def get_env(tickers,type='basic'):
	if type == 'basic':
		return MultiRewardEnv(tickers,transaction_cost,train_test_split,realized=False,reward=reward_function,include_cash=include_cash,window_size=window_size)
	elif type == 'pair':
		return PairEnv(tickers,transaction_cost,train_test_split,realized=False,reward=reward_function,include_cash=include_cash,window_size=window_size)

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


def test_model(combinations,model,env,transaction_cost=transaction_cost):
	test = True
	obs = env.reset(test=test)
	env.transaction_cost=transaction_cost

	env_base = get_env(tickers_flattened,type='pair')
	env_base.reset(test=test)

	done = False
	profit = [1,]
	profit_base = [1,]
	dates = []
	actions = []
	number_of_fields = int(obs.shape[0]/len(tickers_flattened))

	while not done:
		# Get actions for all pairs
		action_dict = {combination:model.predict(np.concatenate([obs[(tickers_flattened.index(combination[0]))*number_of_fields:(1+tickers_flattened.index(combination[0]))*number_of_fields],obs[(tickers_flattened.index(combination[1]))*number_of_fields:(1+tickers_flattened.index(combination[1]))*number_of_fields]]))[0] for combination in combinations}
		# Perform softmax on actions
		action_dict = {c:np.exp(a)/sum(np.exp(a)) for c, a in action_dict.items()}

		action = np.zeros(len(tickers_flattened)+int(include_cash))
		for combination in combinations:
			act = [0,]*(len(tickers_flattened)+int(include_cash))
			if include_cash: act[0] = action_dict[combination][0]
			act[int(include_cash)+tickers_flattened.index(combination[0])] = action_dict[combination][0+int(include_cash)]
			act[int(include_cash)+tickers_flattened.index(combination[1])] = action_dict[combination][1+int(include_cash)]
			action += np.array(act)
		action /= len(combinations)

		obs, _, done, info = env.step(action,test=test)
		if include_cash: _, _, _, _ = env_base.step(np.array([0,]+[1/len(tickers_flattened),]*len(tickers_flattened)),test=test)
		else: _, _, _, _ = env_base.step(np.array([1/len(tickers_flattened),]*len(tickers_flattened)),test=test)


		profit.append(env.profits[-1])
		profit_base.append(env_base.profits[-1])
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


combinations = tickers

print("{} combinations: {}".format(str(len(combinations)),combinations))
start = True
if train:
	count = 0
	for combination in random.sample(combinations,len(combinations)):
		print("{}: {}%".format(combination,count/len(combinations)*100))
		count += 1
		env = get_env(combination)
		model = get_model(not start or load,env)
		train_model(model,train_steps,model_name="{}".format(test_name))
		start = False


env = get_env(tickers_flattened,type='pair')
model = get_model(True,env,model_name="{}".format(test_name))


log = test_model(combinations,model,env)
for ticker in range(len(tickers_flattened)):
	log[tickers_flattened[ticker]] = [row[ticker+int(include_cash)] for row in log['Weights']]
log['cash'] = [row[0] for row in log['Weights']]
pd.DataFrame(log).to_csv("./logs/{}.csv".format(test_name))
