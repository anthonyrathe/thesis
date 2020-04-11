from src.environments.MultiRewardEnv import MultiRewardEnv
#from stable_baselines.sac.policies import FeedForwardPolicy
#from src.agents.StackedAgent import StackedAgent
#import tensorflow as tf
import numpy as np
import pandas as pd
import math

train_test_split = 0.70
transaction_cost = 0.0035
overall_profit = True
include_cash = False
reward_type="sharpe"
total_timesteps = 1000

test_name = "first_test"

#groups = [['PEP','KO','KDP','MNST'],['BBY','TGT','WMT','COST'],['FB','GOOGL','AAPL','AMZN']]
groups = [['C', 'JPM', 'TCF', 'UMBF'], ['PFE', 'JNJ', 'CAPR', 'AKRX'], ['VZ', 'T', 'FTR', 'SHEN'], ['INTC', 'XLNX', 'EMAN', 'KOPN'], ['LEA', 'MOD', 'DORM', 'STRT'], ['MCD', 'EAT', 'JACK', 'PZZA'], ['TGNA', 'NTN', 'CETV', 'SSP'], ['ZIXI', 'TCX', 'CSGS', 'RAMP'], ['KSU', 'CSX', 'UNP', 'NSC'], ['ARTNA', 'AWR', 'MSEX', 'YORW'], ['PCAR', 'SPAR', 'F', 'OSK'], ['TESS', 'ARW', 'TAIT', 'UUU'], ['GLBZ', 'TRST', 'GSBC', 'FCCO'], ['UAL', 'SKYW', 'ALK', 'AAL'], ['DAIO', 'TER', 'COHU', 'ITRI'], ['LNG', 'NJR', 'SJI', 'ATO'], ['NTIP', 'VHC', 'ACTG', 'REFR'], ['MUX', 'AUMN', 'VGZ', 'GSS'], ['FRHC', 'RJF', 'GBL', 'SIEB'], ['NBIX', 'BCRX', 'TECH', 'TTNP'], ['GHC', 'PRDO', 'STRA', 'ATGE'], ['FCAP', 'BYFC', 'PROV', 'CASH'], ['MARPS', 'SJT', 'TPL', 'PBT'], ['RIG', 'DO', 'VAL', 'PTEN'], ['WTS', 'CIR', 'PH', 'HLIO']]
tickers = list(np.array(groups).flatten())


env = MultiRewardEnv(groups,transaction_cost,train_test_split=train_test_split,realized=False,reward=reward_type,include_cash=include_cash,overall_profit=overall_profit)
print(env.historical_data.describe())

# Custom MLP policy
class CustomSACPolicy(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(CustomSACPolicy, self).__init__(*args, **kwargs,
										   layers=[20, 20],
										   layer_norm=True,
										   feature_extraction="mlp",
										   act_fun=tf.nn.relu)

def test_model(model,env,transaction_cost=transaction_cost):
	test = True
	obs = env.reset(test=test)
	env.transaction_cost=transaction_cost

	env_base = MultiRewardEnv(groups,transaction_cost,train_test_split)
	env_base.reset(test=test)

	done = False
	profit = [1,]
	profit_base = [1,]
	dates = []
	actions = []
	while not done:
		# Get actions of all models in ensemble
		action = model.predict(obs)[0]
		# Perform softmax on actions
		#action = [np.exp(x)/sum(np.exp(x)) for x in action]

		obs, rewards, done, info = env.step(action,test=test)
		obs_base, rewards_base, done_base, info_base = env_base.step(np.array([0,]+[1/len(tickers),]*len(tickers)),test=test)


		profit.append(env.profits[-1])
		profit_base.append(env_base.profits[-1])
		dates.append(info['date'])
		# Perform softmax on action, as the basicsimulator will do so
		#actions.append(np.exp(action)/sum(np.exp(action)))
		actions.append(action)
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

#TODO: random action sampling when learning hasn't started is very inefficient -> should be random delta instead of random portfolio
#TODO: add different data for second level
#TODO: add sector identifiers!!
model = StackedAgent(CustomSACPolicy, CustomSACPolicy, env,verbose=1,learning_starts=1,batch_size=1)
model.learn(total_timesteps=total_timesteps, log_interval=1)

log = test_model(model,env)
for ticker in range(len(tickers)):
	log[tickers[ticker]] = [row[ticker+1] for row in log['Weights']]
log['cash'] = [row[0] for row in log['Weights']]
pd.DataFrame(log).to_csv("./logs/{}.csv".format(test_name))