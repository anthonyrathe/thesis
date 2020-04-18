import pandas as pd
from src.environments.simulators.BasicSimulator import BasicSimulator
from src.environments.StackedEnv import StackedEnv
from os.path import dirname as dirname
import os
import numpy as np
import gym
import talib

class StackedEnvDiff(StackedEnv):

	def __init__(self,groups,transaction_cost,train_test_split=0.8,realized=False,end_date=None,reward="sharpe",include_cash=True,clip_softmax=False,peer_normalize=[],normalize=[],window=1,a_space='box',step_size=1,**kwargs):
		super(StackedEnvDiff,self).__init__(groups,transaction_cost,train_test_split=train_test_split,realized=realized,end_date=end_date,reward=reward,include_cash=include_cash,clip_softmax=clip_softmax,peer_normalize=peer_normalize,normalize=normalize,window=window,a_space=a_space,step_size=step_size,**kwargs)

		self.a_space = a_space
		if a_space == 'box':
			self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(len(self.tickers)+int(self.include_cash),),dtype=np.float32)
		elif a_space == 'binary':
			self.action_space = gym.spaces.MultiBinary(len(self.tickers)+int(self.include_cash))
		else:
			raise Exception("Unknown action space: {}".format(a_space))

	def step(self,action,test=False):

		if action is not None:
			if not self.include_cash:
				action = np.insert(action,0,0.0)

			if self.a_space == 'box':
				action = self.simulator.portfolio+action
			elif self.a_space == 'binary':
				# Interpret the binary vector as the desired positions
				action = action

		return super(StackedEnvDiff,self).step(action=action,test=test)

	def expert_step(self,lookahead=None,clip_softmax=True):

		if lookahead is None:
			lookahead = self.step_size

		dates = list(self.historical_data.index)
		lookahead_date = dates[dates.index(self.date)+lookahead]

		portfolio = self.simulator.portfolio
		desired_portfolio = self.simulator.expert_portfolio(self.date,lookahead_date,clip_softmax=clip_softmax)

		self.date = dates[dates.index(self.date)+self.step_size]

		return desired_portfolio-portfolio


#env = StackedEnv(['AAPL','AMZN'],0.001)
# print(env.reset())
# done = False
# while not done:
# 	state, reward, done, _ = env.step([1/3,1/3,1/3])
# 	#print(state)
# 	print(reward)