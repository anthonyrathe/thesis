import pandas as pd
from old.exceptions.IllegalPortfolioException import IllegalPortfolioException
import numpy as np
from os.path import dirname as dirname
import os, math
from src.environments.simulators.BasicSimulator import BasicSimulator

class BasicSimulatorDiff(BasicSimulator):

	def profit_portfolio(self,date,diff=None,realized=False,overall_profit=True):
		# Calculates profit for a portfolio at a certain date.
		# date is the date of which we use the opening price to calculate the profit of the current positions as well as
		# the cost basis of the new positions (cfr. new_portfolio).
		# Optionally updates portfolio and deducts transaction costs.
		# First entry of self.portfolio and new_portfolio is the cash position. The rest corresponds to the tickers, in order.
		# Assumes price data at certain date is available for all tickers or none.

		date = max(list(filter(lambda x: x<= date,self.prices.index)))
		current_portfolio = np.array(self.portfolio)*np.insert((self.prices.loc[date]/self.prices.loc[self.date]).values,0,1.0)
		profit_this_step = sum(current_portfolio)
		current_portfolio = current_portfolio/profit_this_step

		if (diff is None) or (date not in self.prices.index):
			new_portfolio = current_portfolio
			#print("{}: not changing portfolio".format(date))
			# TODO realized should take into account every position individually (an unchanged position does not influence
			# realized profits, a changed one does)
			if realized:
				if overall_profit:
					return self.profit
				else:
					return 0
		else:
			for i in range(len(diff)):
				if diff[i] < 0:
					diff[i] = -min(abs(diff[i]),current_portfolio[i])

			# ...


		#profit_this_step = (sum(((self.prices.loc[date]/self.prices.loc[self.date]).values-1)*((self.portfolio[1:])))+1) - sum(abs(np.array(new_portfolio[1:])-np.array(self.portfolio[1:])))*self.transaction_cost

		# 1. Calculate current portfolio
		# 2. Calculate profit from current portfolio
		# 3. Calculate current portfolio composition
		# 4. Calculate transaction costs to get from current portfolio composition to desired composition

		transaction_cost = sum(abs(np.array(new_portfolio[1:])-np.array(current_portfolio[1:])))*self.transaction_cost
		profit_this_step *= 1 - transaction_cost
		self.profit *= profit_this_step
		self.portfolio = new_portfolio
		self.date = date



		if overall_profit:
			return self.profit
		else:
			return math.log(profit_this_step)








#sim = BasicSimulator(['AAPL','AMZN'],0.001,datetime.datetime(2019,10,2))
#print(sim.prices)
#print(sim.profit_portfolio(datetime.datetime(2019,10,4),[0,1/2,1/2]))
#print(sim.profit_portfolio(datetime.datetime(2019,10,9),[1/2,0,1/2]))





