import pandas as pd
from old.environments.simulators.BasicSimulator import BasicSimulator
from os.path import dirname as dirname
import os
import numpy as np
import gym

class MultiRewardEnv(gym.Env):

	def __init__(self,tickers,transaction_cost,train_test_split=0.8,realized=False,reward="sharpe",include_cash=True,**kwargs):
		super(MultiRewardEnv, self).__init__()

		self.tickers = tickers
		self.historical_data = None
		self.transaction_cost = transaction_cost
		self.realized = realized
		self.reward = reward
		self.include_cash = include_cash

		self.train_test_split = train_test_split
		self.kwargs = kwargs

		input_fields = ['EV/EBITDA','P/E','P/B','D/E','Bias_EV/EBITDA_60','Bias_Price_28','R_Price','RSI']

		def create_input(df):
			return df[input_fields]

		for ticker in tickers:
			# Collect all historical data
			path = os.path.relpath("{}/palantirscreener/data/cleaned/v2/{}_1_1.csv".format(dirname(dirname(dirname(dirname(__file__)))),ticker))
			data = pd.read_csv(path,index_col=0,parse_dates=True)
			data = create_input(data)

			# Calculate velocity and acceleration
			window = 40
			for field in ['EV/EBITDA','P/E','P/B']:
				data['{}_v'.format(field)] = (data[field] - data[field].shift(window))
				data['{}_a'.format(field)] = (data['{}_v'.format(field)] - data['{}_v'.format(field)].shift(window))


			if self.historical_data is None:
				self.historical_data = data
			else:
				self.historical_data = pd.concat((self.historical_data,data),axis=1)

		self.historical_data = self.historical_data.dropna()

		self.observation_space = gym.spaces.Box(low=-np.Inf,high=np.Inf,shape=(len(self.historical_data.columns)+len(tickers),),dtype=np.float32)

		self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(len(tickers)+int(self.include_cash),),dtype=np.float32)


	def getState(self):
		data = self.historical_data.loc[self.date].values
		data = data.reshape((len(self.tickers),int(data.shape[0]/len(self.tickers))))
		portfolio = np.array(self.simulator.portfolio[1:])
		portfolio = portfolio.reshape((portfolio.shape[0],1))
		data = np.hstack((data,portfolio))

		return data

	def step(self,action,test=False):
		if not self.include_cash:
			action = np.insert(action,0,0.0)
		self.profits.append(self.simulator.profit_portfolio(self.date,action,realized=self.realized,overall_profit=True))
		dates = list(self.historical_data.index)
		if (test and self.date == dates[-1]) or (not test and self.date == dates[int(len(dates)*self.train_test_split)]):
			done = True
		else:
			self.date = dates[dates.index(self.date)+1]
			done = False

		if self.reward in ["sharpe","sharpe_diff","sortino","sortino_diff"]:
			window_size = 100
			if 'window_size' in self.kwargs.keys(): window_size = self.kwargs['window_size']
			index = max(len(self.dates)-window_size,0)
			dates = self.dates[index:]
			profits = self.profits[index:]
			risk_free = (1.02**((dates[-1]-dates[0]).days/365))**(1/len(profits))
			risk_free = np.array([1,]+[risk_free**i for i in range(1,len(profits))])

			excess_returns = np.array(profits)-risk_free

			sharpe = excess_returns.mean()/excess_returns.std()
			sortino = excess_returns.mean()/excess_returns[excess_returns<0].std()
		if self.reward == "sharpe":
			reward = sharpe
		elif self.reward == "sharpe_diff":
			reward = sharpe-self.sharpe
			self.sharpe = sharpe
		elif self.reward == "sortino":
			reward = sortino
		elif self.reward == "sortino_diff":
			reward = sortino-self.sortino
			self.sortino = sortino

		return self.getState().flatten(), reward, done, {'date':self.date}

	def reset(self,test=False):
		if test:
			return self.reset_test()

		self.date = min(self.historical_data.index)
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)


		self.profits = [1,]
		self.dates = [self.date,]
		self.sharpe = 0
		self.sortino = 0

		return self.getState().flatten()

	def reset_test(self):
		dates = list(self.historical_data.index)
		self.date = dates[int(len(dates)*self.train_test_split)+1]
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)

		self.profits = [1,]
		self.dates = [self.date,]
		self.sharpe = 0
		self.sortino = 0

		return self.getState().flatten()




# env = BasicEnv(['AAPL','AMZN'],0.001)
# print(env.reset())
# done = False
# while not done:
# 	state, reward, done, _ = env.step([1/3,1/3,1/3])
# 	#print(state)
# 	print(reward)