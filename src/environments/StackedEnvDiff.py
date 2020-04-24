import pandas as pd
from src.environments.simulators.BasicSimulator import BasicSimulator
from src.environments.StackedEnv import StackedEnv
from os.path import dirname as dirname
import os
import numpy as np
import gym
import talib

class StackedEnvDiff(StackedEnv):

	def __init__(self,groups,transaction_cost,train_test_split=0.8,realized=False,end_date=None,reward="sharpe",include_cash=True,clip_softmax=False,first_layer_features=[],second_layer_features=[],peer_normalize=[],z_score_normalize=[],min_max_normalize=[],portfolio_normalize=[],window=1,a_space='box',step_size=1,train_end_date=None,**kwargs):
		super(StackedEnvDiff,self).__init__(groups,transaction_cost,train_test_split=train_test_split,realized=realized,end_date=end_date,reward=reward,include_cash=include_cash,clip_softmax=clip_softmax,first_layer_features=first_layer_features,second_layer_features=second_layer_features,peer_normalize=peer_normalize,z_score_normalize=z_score_normalize,min_max_normalize=min_max_normalize,portfolio_normalize=portfolio_normalize,window=window,a_space=a_space,step_size=step_size,train_end_date=train_end_date,**kwargs)

		self.a_space = a_space
		if a_space == 'box':
			self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(len(self.tickers)+int(self.include_cash),),dtype=np.float32)
		elif a_space == 'binary':
			self.action_space = gym.spaces.MultiBinary(len(self.tickers)+int(self.include_cash))
		else:
			raise Exception("Unknown action space: {}".format(a_space))

	def step(self,action,test=False,default_rebalance=False):

		if action is not None:
			if not self.include_cash:
				action = np.insert(action,0,0.0)

			if self.a_space == 'box':
				action = self.simulator.portfolio+action
			elif self.a_space == 'binary':
				# Interpret the binary vector as the desired positions
				action = action

		return super(StackedEnvDiff,self).step(action=action,test=test,default_rebalance=default_rebalance)

	def expert_step(self,lookahead=None,clip_softmax=True,ultimate_expert=False):

		if lookahead is None:
			lookahead = self.step_size

		dates = list(self.historical_data.index)
		lookahead_date = dates[dates.index(self.date)+lookahead]

		portfolio = self.simulator.portfolio
		desired_portfolio = self.simulator.expert_portfolio(self.date,lookahead_date,clip_softmax=clip_softmax,ultimate_expert=ultimate_expert)

		self.date = dates[dates.index(self.date)+self.step_size]

		return desired_portfolio-portfolio


#env = StackedEnv(['AAPL','AMZN'],0.001)
# print(env.reset())
# done = False
# while not done:
# 	state, reward, done, _ = env.step([1/3,1/3,1/3])
# 	#print(state)
# 	print(reward)