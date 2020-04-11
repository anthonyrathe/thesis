import pandas as pd
import numpy as np
import os, random
from os.path import dirname as dirname


random.seed(123)

input_fields = ['EV/EBITDA','P/E','P/B','D/E','Bias_EV/EBITDA_60','Bias_Price_28','R_Price','RSI']
def has_data(series):
	return series.apply(lambda x: os.path.exists(os.path.relpath("{}/palantirscreener/data/cleaned/v2/{}_1_1.csv".format(dirname(dirname(dirname(dirname(__file__)))),x)))
									and pd.read_csv(os.path.relpath("{}/palantirscreener/data/cleaned/v2/{}_1_1.csv".format(dirname(dirname(dirname(dirname(__file__)))),x)),index_col=0,parse_dates=True)[input_fields].dropna().shape[0]>100)


def get_peer_groups(draw_random=None,as_list=True,verbose=False):
	data = pd.read_csv('{}/data/raw/tickers/s&p500_jan2014.csv'.format(dirname(dirname(dirname(__file__)))),delimiter=';')
	data = data[has_data(data['Ticker symbol'])]

	data['GICS Sub Industry'] = data['GICS Sector']+data['GICS Sub Industry'].replace(np.nan, '')

	if draw_random is not None:
		data = data.loc[random.sample(list(data.index),draw_random)]


	f = data.groupby(['GICS Sub Industry','Ticker symbol'])['GICS Sub Industry'].agg({'Frequency':'count'})
	f.sort_values('Frequency',ascending=False, inplace=True)

	d = {k:list(f.ix[k].index) for k in f.index.levels[0]}
	if verbose:
		for sector, tickers in d.items():
			print("{}: {}".format(sector, tickers))

	if as_list:
		return [v for v in d.values()]
	else:
		return d




