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
		self.portfolio = np.array([1,]+[0,]*(len(self.tickers)))

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
		self.date = max(list(filter(lambda x: x<= start_date,self.prices.index)))

	def expert_portfolio_old(self,date,lookahead_date):
		# NOTE: this expert assumes zero transaction costs!
		expert_action = np.zeros(len(self.tickers)+1)

		price_deltas = (self.prices.loc[lookahead_date]/self.prices.loc[self.date]).values-1
		best_position = np.argmax(price_deltas)
		best_profit = price_deltas[best_position]


		if best_profit > 0:
			# Invest everyting in best position
			expert_action[best_position+1] = 1.0
		else:
			# Invest everyting in cash
			expert_action[0] = 1.0

		self.date = date
		return expert_action

	def expert_portfolio(self,date,lookahead_date,clip_softmax=True,ultimate_expert=False):

		expert_action = np.zeros(len(self.tickers)+1)

		price_deltas = (self.prices.loc[lookahead_date]/self.prices.loc[date]).values-1-self.transaction_cost

		if not ultimate_expert:
			if clip_softmax:
				# Set all declining stocks to zero
				# Assign weight to all other stocks according to the softmax function
				idx_negative = np.nonzero(price_deltas < 0.0)[0]
				idx_positive = np.nonzero(price_deltas >= 0.0)[0]
				expert_action[idx_positive+1] = np.exp(price_deltas[idx_positive])/sum(np.exp(price_deltas[idx_positive]))
				expert_action[idx_negative+1] = 0.0
			else:
				price_deltas = np.insert(price_deltas,0,0.0)
				expert_action = np.exp(price_deltas)/sum(np.exp(price_deltas))
		else:
			best_pos = np.argmax(price_deltas)
			if price_deltas[best_pos] > 0:
				expert_action[best_pos+1] = 1.0

		if expert_action.sum() == 0:
			# Invest everyting in cash
			expert_action[0] = 1.0

		return expert_action

	def profit_portfolio(self,date,new_portfolio=None,realized=False,overall_profit=True):
		# Calculates profit for a portfolio at a certain date.
		# date is the date of which we use the opening price to calculate the profit of the current positions as well as
		# the cost basis of the new positions (cfr. new_portfolio).
		# Optionally updates portfolio and deducts transaction costs.
		# First entry of self.portfolio and new_portfolio is the cash position. The rest corresponds to the tickers, in order.
		# Assumes price data at certain date is available for all tickers or none.

		if new_portfolio is not None:
			assert abs(new_portfolio.sum()-1) < 10-4, "Error: new portfolio should sum to 1, got {} which sums to {}".format(new_portfolio,new_portfolio.sum())

		date = max(list(filter(lambda x: x<= date,self.prices.index)))
		current_portfolio = np.array(self.portfolio)*np.insert((self.prices.loc[date]/self.prices.loc[self.date]).values,0,1.0)

		if (new_portfolio is None) or (date not in self.prices.index):
			new_portfolio = current_portfolio/sum(current_portfolio)
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

		#new_portfolio = np.exp(new_portfolio)/sum(np.exp(new_portfolio))


		#profit_this_step = (sum(((self.prices.loc[date]/self.prices.loc[self.date]).values-1)*((self.portfolio[1:])))+1) - sum(abs(np.array(new_portfolio[1:])-np.array(self.portfolio[1:])))*self.transaction_cost

		# 1. Calculate current portfolio
		# 2. Calculate profit from current portfolio
		# 3. Calculate current portfolio composition
		# 4. Calculate transaction costs to get from current portfolio composition to desired composition
		profit_this_step = sum(current_portfolio)
		if profit_this_step == 0.0:
			print(current_portfolio)
		current_portfolio = current_portfolio/profit_this_step
		volume = sum(abs(np.array(new_portfolio[1:])-np.array(current_portfolio[1:])))
		transaction_cost = volume*self.transaction_cost
		profit_this_step *= 1 - transaction_cost
		self.profit *= profit_this_step
		self.portfolio = new_portfolio
		self.date = date



		if overall_profit:
			return self.profit, volume
		else:
			return math.log(profit_this_step), volume








#sim = BasicSimulator(['AAPL','AMZN'],0.001,datetime.datetime(2019,10,2))
#print(sim.prices)
#print(sim.profit_portfolio(datetime.datetime(2019,10,4),[0,1/2,1/2]))
#print(sim.profit_portfolio(datetime.datetime(2019,10,9),[1/2,0,1/2]))





