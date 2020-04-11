import pandas as pd
from old.exceptions.IllegalPortfolioException import IllegalPortfolioException
import numpy as np
from os.path import dirname as dirname
import os, math

class BasicSimulator:

	def __init__(self,tickers,transaction_cost,start_date):
		self.tickers = tickers
		self.prices = None
		self.transaction_cost = transaction_cost

		self.profit = 1
		self.date = start_date
		self.portfolio = np.array([0,]*(len(self.tickers)+1))

		for ticker in tickers:
			# Collect all open prices (open price is used for BUYs or SELLs, as we would place a market order just
			# before market open, which would then be filled at the open price)
			path = os.path.relpath("{}/BRIKSScreener/data/raw/prices/{}.csv".format(dirname(dirname(dirname(dirname(dirname(__file__))))),ticker))
			open = pd.read_csv(path,index_col=0,parse_dates=True)[['open']]
			if self.prices is None:
				self.prices = open
				self.prices.rename(columns={'open':ticker},inplace=True)
			else:
				self.prices[ticker] = open
		self.prices = self.prices.dropna()

	def profit_portfolio(self,date,new_portfolio=None,realized=False,overall_profit=True):
		# Calculates profit for a portfolio at a certain date.
		# date is the date of which we use the opening price to calculate the profit of the current positions as well as
		# the cost basis of the new positions (cfr. new_portfolio).
		# Optionally updates portfolio and deducts transaction costs.
		# First entry of self.portfolio and new_portfolio is the cash position. The rest corresponds to the tickers, in order.
		# Assumes price data at certain date is available for all tickers or none.

		if (new_portfolio is None) or (date not in self.prices.index) or (new_portfolio == self.portfolio).all():
			new_portfolio = self.portfolio
			#print("{}: not changing portfolio".format(date))
			# TODO realized should take into account every position individually (an unchanged position does not influence
			# realized profits, a changed one does)
			if realized:
				if overall_profit:
					return self.profit
				else:
					return 0

		#if sum(new_portfolio) != 1:
			#raise IllegalPortfolioException()
			# Softmax such that portfolio weights sum to 1
			# TODO
		new_portfolio = np.exp(new_portfolio)/sum(np.exp(new_portfolio))

		# Calculate new profit
		date = max(list(filter(lambda x: x<= date,self.prices.index)))

		profit_this_step = (sum(((self.prices.loc[date]/self.prices.loc[self.date]).values-1)*((self.portfolio[1:])))+1) - sum(abs(np.array(new_portfolio[1:])-np.array(self.portfolio[1:])))*self.transaction_cost
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





