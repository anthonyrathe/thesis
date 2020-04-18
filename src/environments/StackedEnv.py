import pandas as pd
from src.environments.simulators.BasicSimulator import BasicSimulator
from os.path import dirname as dirname
import os
import numpy as np
import gym
import talib

class StackedEnv(gym.Env):

	def __init__(self,groups,transaction_cost,train_test_split=0.8,realized=False,end_date=None,reward="sharpe",include_cash=True,clip_softmax=False,peer_normalize=[],normalize=[],window=1,a_space='box',step_size=1,**kwargs):
		super(StackedEnv, self).__init__()

		self.test = False
		self.clip_softmax = clip_softmax
		self.window = window
		self.step_size = step_size

		self.groups = groups
		assert len(set([len(group) for group in groups])) == 1
		self.group_size = len(groups[0])
		self.group_count = len(groups)
		self.tickers = []
		for group in groups:
			self.tickers += group

		self.historical_data = None
		self.transaction_cost = transaction_cost
		self.realized = realized
		self.reward = reward
		self.include_cash = include_cash

		self.train_test_split = train_test_split
		self.kwargs = kwargs

		technical_indicators = ['BBANDS_u','BBANDS_m','BBANDS_l','SAREXT','DEMA', 'EMA', 'SMA', 'TEMA','WMA','ADXR', 'APO',
								'AROON_down', 'AROON_up', 'CCI', 'CMO', 'MFI', 'MACD', 'MACD_signal', 'MACD_hist', 'MOM',
								'PLUS_DI', 'PPO', 'ROC', 'ROCP', 'RSI', 'STOCH_k', 'STOCH_d', 'STOCHF_k', 'STOCHF_d',
								'TRIX', 'ULTOSC', 'WILLR', 'AD', 'OBV', 'ATR', 'NATR','HT_DCPERIOD',
								'HT_SINE', 'HT_SINE_lead', 'HT_DCPHASE', 'HT_PHASOR_inphase', 'HT_PHASOR_quadrature',
								'HT_TRENDLINE']
		funda_technical_indicators = []
		first_layer_input_fields = ['EV/EBITDA','P/E','P/B','D/E','Bias_EV/EBITDA_60','Bias_Price_28','R_Price']
		second_layer_input_fields = ['EV/EBITDA','P/E','P/B','D/E']
		#first_layer_input_fields = technical_indicators
		input_fields = first_layer_input_fields+second_layer_input_fields

		def create_input(df):

			df['BBANDS_u'], df['BBANDS_m'], df['BBANDS_l'] = talib.BBANDS(df.close)

			df['SAREXT'] = talib.SAREXT(df.high, df.low)
			df['DEMA'] = talib.DEMA(df.close)
			df['EMA'] = talib.EMA(df.close)
			df['SMA'] = talib.SMA(df.close)
			df['TEMA'] = talib.TEMA(df.close)
			df['WMA'] = talib.WMA(df.close)
			df['ADXR'] = talib.ADXR(df.high, df.low, df.close)
			df['APO'] = talib.APO(df.close)
			df['AROON_down'], df['AROON_up'] = talib.AROON(df.high, df.low)
			df['CCI'] = talib.CCI(df.high, df.low, df.close)
			df['CMO'] = talib.CMO(df.close)
			df['MFI'] = talib.MFI(df.high, df.low, df.close, df.volume)
			df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df.close)
			df['MOM'] = talib.MOM(df.close)
			df['PLUS_DI'] = talib.PLUS_DI(df.high, df.low, df.close)
			df['PPO'] = talib.PPO(df.close)
			df['ROC'] = talib.ROC(df.close)
			df['ROCP'] = talib.ROCP(df.close)
			df['RSI'] = talib.RSI(df.close)
			df['STOCH_k'], df['STOCH_d'] = talib.STOCH(df.high, df.low, df.close)
			df['STOCHF_k'], df['STOCHF_d'] = talib.STOCHF(df.high, df.low, df.close)
			df['TRIX'] = talib.TRIX(df.close)
			df['ULTOSC'] = talib.ULTOSC(df.high, df.low, df.close)
			df['WILLR'] = talib.WILLR(df.high, df.low, df.close)
			df['AD'] = talib.AD(df.high, df.low, df.close, df.volume)
			df['OBV'] = talib.OBV(df.close, df.volume)
			df['ATR'] = talib.ATR(df.high, df.low, df.close)
			df['NATR'] = talib.NATR(df.high, df.low, df.close)
			df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df.close)
			df['HT_SINE'], df['HT_SINE_lead'] = talib.HT_SINE(df.close)
			df['HT_DCPHASE'] = talib.HT_DCPHASE(df.close)
			df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(df.close)

			# My metrics
			df['BBANDS_u'], df['BBANDS_m'], df['BBANDS_l'] = df['BBANDS_u']/df.close, df['BBANDS_m']/df.close, df['BBANDS_l']/df.close
			df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df.close)

			return df[input_fields]

		for group in groups:
			for ticker in group:
				# TODO: normalize compared to peers
				# Collect all historical data
				path = os.path.relpath("{}/BRIKSScreener/data/cleaned/v2/{}_1_1.csv".format(dirname(dirname(dirname(dirname(__file__)))),ticker))
				data = pd.read_csv(path,index_col=0,parse_dates=True)
				data = create_input(data)

				data = data.dropna()

				if self.historical_data is None:
					self.historical_data = data
				else:
					self.historical_data = pd.concat((self.historical_data,data),axis=1)
				self.historical_data = self.historical_data.dropna()

		end_of_train_set = int(len(self.historical_data.index)*self.train_test_split)
		for f in peer_normalize:
			field_indices = []
			if f in first_layer_input_fields:
				field_indices.append(first_layer_input_fields.index(f))
			if f in second_layer_input_fields:
				field_indices.append(second_layer_input_fields.index(f)+len(first_layer_input_fields))

			for group in range(self.group_count):
				for field in field_indices:
					indices = [(group*self.group_size+peer)*len(input_fields)+field for peer in range(self.group_size)]
					group_data = self.historical_data.iloc[:,indices]
					group_mean = group_data.mean(axis=1)
					group_sigma = group_data.std(axis=1)
					for peer in range(self.group_size):
						index = (group*self.group_size+peer)*len(input_fields)+field
						self.historical_data.iloc[:,index] = (self.historical_data.iloc[:,index]-group_mean)/group_sigma

		for field in normalize:
			field_indices = []
			if field in first_layer_input_fields:
				field_indices.append(first_layer_input_fields.index(field))
			if field in second_layer_input_fields:
				field_indices.append(second_layer_input_fields.index(field)+len(first_layer_input_fields))

			for ticker in range(len(self.tickers)):
				for field in field_indices:
					index = ticker*len(input_fields)+field
					mean = self.historical_data.iloc[:end_of_train_set,index].mean()
					sigma = self.historical_data.iloc[:end_of_train_set,index].std()
					self.historical_data.iloc[:,index] = (self.historical_data.iloc[:,index]-mean)/sigma


		if end_date is not None:
			self.historical_data = self.historical_data[self.historical_data.index<=end_date]
		self.historical_data = self.historical_data.dropna()


		self.feature_set_size = len(input_fields)+2
		self.first_layer_feature_set_size = len(first_layer_input_fields)+1
		self.second_layer_feature_set_size = len(second_layer_input_fields)+1


		self.observation_space = gym.spaces.Box(low=-np.Inf,high=np.Inf,shape=(self.window,len(self.tickers),self.first_layer_feature_set_size+self.second_layer_feature_set_size),dtype=np.float32)

		if a_space == 'box':
			self.action_space = gym.spaces.Box(low=0.0,high=1.0,shape=(len(self.tickers)+int(self.include_cash),),dtype=np.float32)
		elif a_space == 'binary':
			self.action_space = gym.spaces.MultiBinary(len(self.tickers)+int(self.include_cash))
		else:
			raise Exception("Unknown action space: {}".format(a_space))


	def getState(self):
		data = self.historical_data.loc[self.historical_data.index<=self.date].iloc[-self.window:].values
		data = data.reshape((self.window,len(self.tickers),int(data.shape[1]/len(self.tickers))))
		d = []
		for i in range(self.window):
			p = self.portfolios[-self.window+i][1:].reshape((-1,1))
			d.append(np.hstack((p,data[i,:,:self.first_layer_feature_set_size-1],p,data[i,:,self.first_layer_feature_set_size-1:])))
		data = np.array(d)
		return data



	def step(self,action,test=False):
		test = test or self.test
		if action is not None:
			if self.clip_softmax:
				# If action is -1, then weight should be set at 0
				# Only non-zero weights should be softmax-ed
				idx_zero = np.nonzero(action <= 0.0)[0]
				idx_rest = np.nonzero(action > 0.0)[0]
				action[idx_rest] = np.exp(action[idx_rest])/sum(np.exp(action[idx_rest]))
				action[idx_zero] = 0.0
			else:
				# Perform softmax on action, such that all weights sum to 1
				action = np.exp(action)/sum(np.exp(action))

			if not self.include_cash:
				action = np.insert(action,0,0.0)

			# If in some case the action returns an array full of zeros, then just put everything in cash
			if sum(action) == 0:
				action[0] = 1.0

		self.profits.append(self.simulator.profit_portfolio(self.date,new_portfolio=action,realized=self.realized,overall_profit=True))
		dates = list(self.historical_data.index)

		if (test and (dates.index(self.date) + self.step_size) >= len(dates)) or (not test and ((dates.index(self.date) + self.step_size) >= int(len(dates)*self.train_test_split))):
			done = True
		else:
			self.date = dates[dates.index(self.date)+self.step_size]
			done = False

		#if (test and self.date == dates[-self.step_size]) or (not test and self.date == dates[int(len(dates)*self.train_test_split)]):
		#	done = True
		#else:
		#	self.date = dates[dates.index(self.date)+self.step_size]
		#	done = False

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
		elif self.reward == "p&l":
			if len(self.profits) < 2:
				reward = 0
			else:
				reward = self.profits[-1]-self.profits[-2]

		self.portfolios.append(self.simulator.portfolio)

		return self.getState(), reward, done, {'date':self.date}

	def expert_step(self,lookahead=None,clip_softmax=True):

		if lookahead is None:
			lookahead = self.step_size

		dates = list(self.historical_data.index)
		lookahead_date = dates[dates.index(self.date)+lookahead]

		action = self.simulator.expert_portfolio(self.date,lookahead_date,clip_softmax=clip_softmax)

		self.date = dates[dates.index(self.date)+self.step_size]

		return action

	def reset(self,test=False):
		if test or self.test:
			return self.reset_test()

		self.date = self.historical_data.index[self.window-1]
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)


		self.profits = [1,]
		self.dates = [self.date,]
		self.sharpe = 0
		self.sortino = 0
		self.portfolios = [self.simulator.portfolio,]*self.window

		return self.getState()

	def reset_test(self):
		dates = list(self.historical_data.index)
		self.date = dates[int(len(dates)*self.train_test_split)+1]
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)

		self.profits = [1,]
		self.dates = [self.date,]
		self.sharpe = 0
		self.sortino = 0

		self.portfolios = [self.simulator.portfolio,]*self.window
		return self.getState()

	def test_mode(self):
		self.test = True

#env = StackedEnv(['AAPL','AMZN'],0.001)
# print(env.reset())
# done = False
# while not done:
# 	state, reward, done, _ = env.step([1/3,1/3,1/3])
# 	#print(state)
# 	print(reward)