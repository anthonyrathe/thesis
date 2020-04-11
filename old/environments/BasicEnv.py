import pandas as pd
from old.environments.simulators.BasicSimulator import BasicSimulator
from os.path import dirname as dirname
import os
import numpy as np
import gym, math

class BasicEnv(gym.Env):

	def __init__(self,tickers,transaction_cost,train_test_split=0.8,realized=False,overall_profit=True):
		super(BasicEnv, self).__init__()

		self.tickers = tickers
		self.historical_data = None
		self.transaction_cost = transaction_cost
		self.realized = realized
		self.overall_profit = overall_profit

		self.train_test_split = train_test_split

		#input_fields = ['EV/EBITDA','P/E','P/B','D/E']
		#input_fields = ['EV/EBITDA','P/E','P/B','D/E','Bias_EV/EBITDA_60','Bias_Price_28','R_Price','RSI','revenueGrowth1y','earningsGrowth1y','EBITDAGrowth1y']
		input_fields = ['EV/EBITDA','P/E','P/B','D/E','Bias_EV/EBITDA_60','Bias_Price_28','R_Price','RSI']

		self.observation_space = gym.spaces.Box(low=-np.Inf,high=np.Inf,shape=(len(tickers)*(len(input_fields)+1),),dtype=np.float32)

		self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(len(tickers)+1,),dtype=np.float32)

		def create_input(df):
			return df[input_fields]

		for ticker in tickers:
			# Collect all historical data
			path = os.path.relpath("{}/palantirscreener/data/cleaned/v2/{}_1_1.csv".format(dirname(dirname(dirname(dirname(__file__)))),ticker))
			data = pd.read_csv(path,index_col=0,parse_dates=True)
			data = create_input(data)
			if self.historical_data is None:
				self.historical_data = data
			else:
				self.historical_data = pd.concat((self.historical_data,data),axis=1)

		self.historical_data = self.historical_data.dropna()

	def getState(self):
		data = self.historical_data.loc[self.date].values
		data = data.reshape((len(self.tickers),int(data.shape[0]/len(self.tickers))))
		portfolio = np.array(self.simulator.portfolio[1:])
		portfolio = portfolio.reshape((portfolio.shape[0],1))
		data = np.hstack((data,portfolio))

		return data

	def step(self,action,test=False):
		reward = self.simulator.profit_portfolio(self.date,action,realized=self.realized,overall_profit=self.overall_profit)
		#reward = math.exp(reward)
		dates = list(self.historical_data.index)
		if (test and self.date == dates[-1]) or (not test and self.date == dates[int(len(dates)*self.train_test_split)]):
			done = True
		else:
			self.date = dates[dates.index(self.date)+1]
			done = False

		return self.getState().flatten(), reward, done, {'date':self.date}

	def reset(self,test=False):
		if test:
			return self.reset_test()

		self.date = min(self.historical_data.index)
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)

		return self.getState().flatten()

	def reset_test(self):
		dates = list(self.historical_data.index)
		self.date = dates[int(len(dates)*self.train_test_split)+1]
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)
		return self.getState().flatten()




# env = BasicEnv(['AAPL','AMZN'],0.001)
# print(env.reset())
# done = False
# while not done:
# 	state, reward, done, _ = env.step([1/3,1/3,1/3])
# 	#print(state)
# 	print(reward)