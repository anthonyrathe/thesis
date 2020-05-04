import pandas as pd
from src.environments.simulators.BasicSimulator import BasicSimulator
from src.utils.peer_groups import peer_group_ids, get_id_ticker
from os.path import dirname as dirname
import os, math
import numpy as np
import gym
import talib

class StackedEnv(gym.Env):

	def __init__(self,groups,transaction_cost,train_test_split=0.8,realized=False,end_date=None,reward="sharpe",include_cash=True,clip_softmax=False,first_layer_features=[],second_layer_features=[],peer_normalize=[],z_score_normalize=[],min_max_normalize=[],portfolio_normalize=[],window=1,a_space='box',step_size=1,train_end_date=None,default_rebalances=False,**kwargs):
		super(StackedEnv, self).__init__()

		self.test = False
		self.default_rebalances = default_rebalances
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

		self.kwargs = kwargs

		fundamentals_1 = ['EV/EBITDA','P/E','P/B','D/E','market_cap',
							'P/FCF','EV/EBITA']

		fundamentals_2 = ['QOE_adjusted','current_ratio_adjusted',
						  	'net_margin','operating_margin','EBITDA_margin',
						  	'D/A','tangible_asset_ratio','ROE','ROA','ROIC',
							'years_sales_outstanding','WACC',
							'quick_ratio_adjusted']

		fundamentals_derived_1 = ['{}_CAGR_{}y'.format(field,str(i))
											for field in 	['netincome','totalrevenue','ebit','FCF',
															'NOPAT','net_margin','operating_margin',
															'EBITDA']
											for i in 		[1,3,5]]

		fundamentals_derived_2 = ['{}_CAGR_{}y_to_PE'.format(field,str(i))
											for field in 	['netincome','ebit']
											for i in 		[1,3,5]]

		fundamentals_derived_3 = 	['EBITDA_CAGR_1y_to_EV/EBITDA',
									'EBITDA_CAGR_3y_to_EV/EBITDA',
									'EBITDA_CAGR_5y_to_EV/EBITDA']

		fundamental_technicals = ['{}_{}'.format(field,metric)
											for field in ['P/E','P/B','EV/EBITDA','P/FCF']
											for metric in [	'DEMA_ratio_adjusted',
															'HT_TRENDLINE_ratio_adjusted',
															'KAMA_ratio_adjusted',
															'MAMA_ratio_adjusted']]

		technicals = [	'close_DEMA_ratio_adjusted','close_HT_TRENDLINE_ratio_adjusted',
						'close_KAMA_ratio_adjusted','close_MAMA_ratio_adjusted',
						'SAR_ratio_adjusted','SAREXT_ratio_adjusted','ADX_adjusted',
						'DI_ratio_adjusted','ADXR_adjusted','APO','AROON_adjusted',
						'AROONOSC_adjusted','BOP','CCI_adjusted','CMO_adjusted','MACD_adjusted']

		doubles = ['{}_2'.format(f) for f in fundamentals_1 + fundamentals_2+ fundamentals_derived_1]

		macro_economics = ['tax_rate','10y_treasury_rate']

		#first_layer_input_fields = fundamentals_1 + fundamentals_2 + fundamentals_derived_1 + fundamentals_derived_2 + fundamentals_derived_3 + fundamental_technicals
		#second_layer_input_fields = technicals + doubles + macro_economics

		#first_layer_input_fields = fundamentals_1 + fundamentals_2 + fundamentals_derived_1 + fundamentals_derived_2 + fundamentals_derived_3 + fundamental_technicals + technicals
		#second_layer_input_fields = ['tax_rate',]

		first_layer_input_fields = ['EV/EBITDA','P/E','P/B','market_cap',
									'net_margin','operating_margin','EBITDA_margin',
									'P/FCF','D/A','tangible_asset_ratio','ROE','ROA','ROIC',
									'EV/EBITA','QOE_adjusted','quick_ratio_adjusted',
									'netincome_CAGR_5y_to_PE','ebit_CAGR_5y_to_PE','EBITDA_CAGR_5y_to_EV/EBITDA']+\
									['{}_{}'.format(field,metric)
											for field in ['P/E','EV/EBITDA']
											for metric in [	'HT_TRENDLINE_ratio_adjusted',
															'KAMA_ratio_adjusted',
															'MAMA_ratio_adjusted']]
		second_layer_input_fields = ['tax_rate',]

		#first_layer_input_fields = ['EV/EBITDA','P/E','P/B','D/E','EV/EBITDA_KAMA_ratio_adjusted']
		#second_layer_input_fields = ['EV/EBITDA','P/E','P/B','D/E']

		#first_layer_input_fields = ['EV/EBITDA','P/E','P/B','D/E','net_margin','EBITDA_margin','P/FCF','D/A','ROE','QOE_adjusted','EBITDA_CAGR_3y_to_EV/EBITDA','EV/EBITDA_KAMA_ratio_adjusted']
		#second_layer_input_fields = []
		self.first_weight_field = None
		if first_layer_features[0] in ('weights','open_positions'):
			if len(first_layer_features) > 1:
				first_layer_input_fields = first_layer_features[1:]
			else:
				first_layer_input_fields = []
			self.first_weight_field = first_layer_features[0]
		else:
			first_layer_input_fields = first_layer_features

		self.second_weight_field = None
		if len(second_layer_features)> 0 and second_layer_features[0] in ('weights','open_positions'):
			if len(second_layer_features) > 1:
				second_layer_input_fields = second_layer_features[1:]
			else:
				second_layer_input_fields = []
			self.second_weight_field = second_layer_features[0]
		else:
			second_layer_input_fields = second_layer_features

		if 'peer_group_ids' in first_layer_input_fields:
			first_layer_input_fields.remove('peer_group_ids')
			first_layer_input_fields += peer_group_ids

		if 'peer_group_ids' in second_layer_input_fields:
			second_layer_input_fields.remove('peer_group_ids')
			second_layer_input_fields += peer_group_ids

		input_fields = first_layer_input_fields+second_layer_input_fields

		def create_input(df,ticker):
			nb_days_in_year = 252
			nb_days_margin = 30

			# Generate peer group identifier
			if True:
				ticker_id = get_id_ticker(ticker)
				for id in peer_group_ids:
					df[id] = int(ticker_id == id)


			# Generate fundamentals
			if True:
				df['net_margin'] = df['netincome']/df['totalrevenue']
				df['operating_margin'] = df['ebit']/df['totalrevenue']
				df['EBITDA_margin'] = df['EBITDA']/df['totalrevenue']
				df['QOE_adjusted'] = (df['cashfromoperatingactivities']-df['netincome'])/(df['netincome'].abs())
				df['FCF'] = df['cashfromoperatingactivities']-df['capitalexpenditures']
				df['P/FCF'] = df['market_cap']/df['FCF']
				df['D/A'] = df['totalliabilities']/df['totalassets']
				#df['tax_rate'] = 1 - df['netincome']/df['incomebeforetaxes']
				df['tax_rate'] = pd.to_datetime(df.periodenddate,format="%m/%d/%Y").apply(date_to_tax_rate)
				df['NOPAT'] = (df['ebit']*(1-df['tax_rate']))
				df['tangible_asset_ratio'] = (df['totalassets']-df['intangibleassets'])/df['totalrevenue'] # Used as indicator to detect asset heavy businesses
				df['current_ratio_adjusted'] = df['totalcurrentassets']/df['totalcurrentliabilities']-1
				df['current_ratio_adjusted'] = df['current_ratio_adjusted'].fillna(value=10)
				df['quick_ratio_adjusted'] = (df['totalcurrentassets']-df['inventoriesnet'])/df['totalcurrentliabilities']-1
				df['quick_ratio_adjusted'] = df['quick_ratio_adjusted'].fillna(value=10)
				df['ROE'] = df['netincome']/df['totalstockholdersequity']
				df['ROA'] = df['netincome']/df['totalassets']
				df['ROIC'] = df['NOPAT']/(df['totalliabilities']+df['totalstockholdersequity'])
				df['years_sales_outstanding'] = df['totalreceivablesnet']/df['totalrevenue']
				df['10y_treasury_rate'] = pd.read_csv("{}/data/raw/yields/monthly_10y_US_treasury_yield.csv".format(dirname(dirname(dirname(__file__)))),index_col=0,parse_dates=True)['Rate']/100
				df['10y_treasury_rate'] = df['10y_treasury_rate'].ffill()
				df['S&P500'] = pd.read_csv("{}/data/raw/s&p_history/GSPC.csv".format(dirname(dirname(dirname(__file__)))),index_col=0,parse_dates=True)['Close']
				df['S&P500'] = df['S&P500'].ffill()
				df['beta'] = talib.BETA(df.close,df['S&P500'])
				df['cost_of_equity'] = df['10y_treasury_rate'] + df['beta']*(0.1-df['10y_treasury_rate'])
				df['cost_of_debt'] = df['10y_treasury_rate']*2
				df['WACC'] = (df['totalstockholdersequity']*df['cost_of_equity']+df['cost_of_debt']*df['totalliabilities']*(1-df['tax_rate']))/(df['totalstockholdersequity']+df['totalliabilities'])

			# Generate derivatives of fundamentals
			if True:

				fs = ['netincome','totalrevenue','ebit','FCF','NOPAT','net_margin','operating_margin','EBITDA']

				# Make sure we know where to fill nan's from
				first_nas = []
				for f in fs:
					df['shift'] = df[f].shift(nb_days_in_year*3+nb_days_margin)
					first_nas.append(df[f].index[~df['shift'].apply(np.isnan)].min())
				first_na = min(first_nas)

				for f in fs:
					df['{}_CAGR_1y'.format(f)] = df[f].pct_change(periods=nb_days_in_year+nb_days_margin)
					df['{}_CAGR_3y'.format(f)] = (df[f].pct_change(periods=nb_days_in_year*3+nb_days_margin)+1)**(1/3)-1
					df['{}_CAGR_5y'.format(f)] = (df[f].pct_change(periods=nb_days_in_year*5+nb_days_margin)+1)**(1/5)-1
					df.loc[first_na:,'{}_CAGR_1y'.format(f)].fillna(value=-1,inplace=True)
					df.loc[first_na:,'{}_CAGR_3y'.format(f)].fillna(value=-1,inplace=True)
					df.loc[first_na:,'{}_CAGR_5y'.format(f)].fillna(value=-1,inplace=True)

				df['netincome_CAGR_1y_to_PE'] = df['netincome_CAGR_1y']/df['P/E']
				df['netincome_CAGR_3y_to_PE'] = df['netincome_CAGR_3y']/df['P/E']
				df['netincome_CAGR_5y_to_PE'] = df['netincome_CAGR_5y']/df['P/E']

				df['ebit_CAGR_1y_to_PE'] = df['ebit_CAGR_1y']/df['P/E']
				df['ebit_CAGR_3y_to_PE'] = df['ebit_CAGR_3y']/df['P/E']
				df['ebit_CAGR_5y_to_PE'] = df['ebit_CAGR_5y']/df['P/E']

				df['EBITDA_CAGR_1y_to_EV/EBITDA'] = df['EBITDA_CAGR_1y']/df['EV/EBITDA']
				df['EBITDA_CAGR_3y_to_EV/EBITDA'] = df['EBITDA_CAGR_3y']/df['EV/EBITDA']
				df['EBITDA_CAGR_5y_to_EV/EBITDA'] = df['EBITDA_CAGR_5y']/df['EV/EBITDA']

			# Generate additional fundamentals
			if True:
				# Cfr. page 374 in Valuation - Measuring and Managing the Value of Companies (Fourth Edition, Wiley)
				df['EV/EBITA'] = (1-df['tax_rate'])*(1-df['NOPAT_CAGR_3y']/df['ROIC'])/(df['WACC']-df['NOPAT_CAGR_3y'])



			# Generate fundamental technicals derived from:
			#	- P/E
			#	- P/B
			#	- EV/EBITDA
			#	- P/FCF
			if True:
				for f in ['P/E','P/B','EV/EBITDA','P/FCF']:
					#field = df[f].abs()
					field = df[f]

					df['{}_DEMA_ratio_adjusted'.format(f)] = field/talib.DEMA(field,timeperiod=nb_days_in_year)-1
					df['{}_DEMA_ratio_adjusted'.format(f)].fillna(value=-1,inplace=True)
					#df['{}_DEMA_crossing'.format(f)] = (df['{}_DEMA_ratio'.format(f)]/(df['{}_DEMA_ratio'.format(f)].shift(5)) < 0).replace({True: 1, False: -1})

					df['{}_HT_TRENDLINE_ratio_adjusted'.format(f)] = field/talib.HT_TRENDLINE(field)-1
					df['{}_HT_TRENDLINE_ratio_adjusted'.format(f)].fillna(value=-1,inplace=True)

					df['{}_KAMA_ratio_adjusted'.format(f)] = field/talib.KAMA(field,timeperiod=nb_days_in_year)-1
					df['{}_KAMA_ratio_adjusted'.format(f)].fillna(value=-1,inplace=True)

					mama, fama = talib.MAMA(field)
					df['{}_MAMA_ratio_adjusted'.format(f)] = mama/fama-1
					df['{}_MAMA_ratio_adjusted'.format(f)].fillna(value=-1,inplace=True)




			# Generate technicals
			if True:
				df['close_DEMA_ratio_adjusted'] = df.close-talib.DEMA(df.close,timeperiod=30)

				df['close_HT_TRENDLINE_ratio_adjusted'] = df.close-talib.HT_TRENDLINE(df.close)

				df['close_KAMA_ratio_adjusted'] = df.close-talib.KAMA(df.close,timeperiod=30)

				mama, fama = talib.MAMA(df.close)
				df['close_MAMA_ratio_adjusted'] = mama-fama

				df['SAR_ratio_adjusted'] =df.close-talib.SAR(df.high,df.low)
				df['SAREXT_ratio_adjusted'] = df.close-talib.SAREXT(df.high,df.low)

				df['ADX_adjusted'] = talib.ADX(df.high,df.low,df.close,timeperiod=14) - 25
				df['DI_ratio_adjusted'] = talib.PLUS_DM(df.high,df.low,timeperiod=14)-talib.MINUS_DM(df.high,df.low,timeperiod=14)

				df['ADXR_adjusted'] = talib.ADXR(df.high,df.low,df.close,timeperiod=14) - 25

				df['APO'] = talib.APO(df.close,fastperiod=12,slowperiod=26)

				aroondown, aroonup = talib.AROON(df.high, df.low, timeperiod=14)
				df['AROON_adjusted'] = (aroonup-aroondown)/100

				df['AROONOSC_adjusted'] = talib.AROONOSC(df.high,df.low,timeperiod=14)/50

				df['BOP'] = talib.BOP(df.open,df.high,df.low,df.close)

				df['CCI_adjusted'] = talib.CCI(df.high,df.low,df.close,timeperiod=14)/100

				df['CMO_adjusted'] = talib.CMO(df.close,timeperiod=14)/50

				macd, macdsignal, _ = talib.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
				df['MACD_adjusted'] = macd-macdsignal


				# df['BBANDS_u'], df['BBANDS_m'], df['BBANDS_l'] = talib.BBANDS(df.close)
				# df['SAREXT'] = talib.SAREXT(df.high, df.low)
				# df['DEMA'] = talib.DEMA(df.close)
				# df['EMA'] = talib.EMA(df.close)
				# df['SMA'] = talib.SMA(df.close)
				# df['TEMA'] = talib.TEMA(df.close)
				# df['WMA'] = talib.WMA(df.close)
				# df['ADXR'] = talib.ADXR(df.high, df.low, df.close)
				# df['APO'] = talib.APO(df.close)
				# df['AROON_down'], df['AROON_up'] = talib.AROON(df.high, df.low)
				# df['CCI'] = talib.CCI(df.high, df.low, df.close)
				# df['CMO'] = talib.CMO(df.close)
				# df['MFI'] = talib.MFI(df.high, df.low, df.close, df.volume)
				# df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df.close)
				# df['MOM'] = talib.MOM(df.close)
				# df['PLUS_DI'] = talib.PLUS_DI(df.high, df.low, df.close)
				# df['PPO'] = talib.PPO(df.close)
				# df['ROC'] = talib.ROC(df.close)
				# df['ROCP'] = talib.ROCP(df.close)
				# df['RSI'] = talib.RSI(df.close)
				# df['STOCH_k'], df['STOCH_d'] = talib.STOCH(df.high, df.low, df.close)
				# df['STOCHF_k'], df['STOCHF_d'] = talib.STOCHF(df.high, df.low, df.close)
				# df['TRIX'] = talib.TRIX(df.close)
				# df['ULTOSC'] = talib.ULTOSC(df.high, df.low, df.close)
				# df['WILLR'] = talib.WILLR(df.high, df.low, df.close)
				# df['AD'] = talib.AD(df.high, df.low, df.close, df.volume)
				# df['OBV'] = talib.OBV(df.close, df.volume)
				# df['ATR'] = talib.ATR(df.high, df.low, df.close)
				# df['NATR'] = talib.NATR(df.high, df.low, df.close)
				# df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df.close)
				# df['HT_SINE'], df['HT_SINE_lead'] = talib.HT_SINE(df.close)
				# df['HT_DCPHASE'] = talib.HT_DCPHASE(df.close)
				# df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(df.close)

			# Generate random data for debugging
			if True:
				df['random'] = np.random.random(size=(len(df.index)))

			# Generate copies
			if True:
				for f in fundamentals_1 + fundamentals_derived_1 + fundamentals_2:
					df['{}_2'.format(f)] = df[f]

			return df[input_fields]

		for group in groups:
			for ticker in group:
				# Collect all historical data
				path = os.path.relpath("{}/BRIKSScreener/data/cleaned/v2/{}_1_1.csv".format(dirname(dirname(dirname(dirname(__file__)))),ticker))
				data = pd.read_csv(path,index_col=0,parse_dates=True,low_memory=False)
				data = create_input(data,ticker)

				data = data.dropna()

				#t = '...'
				#if ticker == t:
				#	self.historical_data.to_csv('./before.csv')
				#	data.to_csv('./change.csv')

				if self.historical_data is None:
					self.historical_data = data
				else:
					self.historical_data = pd.concat((self.historical_data,data),axis=1)
				self.historical_data = self.historical_data.dropna()

				#print(ticker)
				#print(self.historical_data.describe())
				#if ticker == t:
				#	self.historical_data.to_csv('./after.csv')
				#	return


		self.historical_data = self.historical_data.replace(np.inf,10e10).replace(-np.inf,-10e10)


		if train_end_date is None:
			self.end_of_train_set = int(len(self.historical_data.index)*train_test_split)
		else:
			dates = list(self.historical_data[self.historical_data.index<=train_end_date].index)
			first_date = dates[-1]
			self.end_of_train_set = dates.index(first_date)


		for f in peer_normalize:
			field_indices = []
			if f in first_layer_input_fields:
				field_indices.append(first_layer_input_fields.index(f))
			if f in second_layer_input_fields:
				field_indices.append(second_layer_input_fields.index(f)+len(first_layer_input_fields))
			#if field_indices == []: raise Warning("Couldn't find field {}".format(f))

			for group in range(self.group_count):
				for field in field_indices:
					indices = [(group*self.group_size+peer)*len(input_fields)+field for peer in range(self.group_size)]
					group_data = self.historical_data.iloc[:,indices]
					group_mean = group_data.mean(axis=1)

					#group_sigma = group_data.std(axis=1)
					group_sigma = pd.concat([group_data.std(axis=1),group_data.abs().max(axis=1)],axis=1).apply(replace_zero,axis=1)
					for peer in range(self.group_size):
						index = (group*self.group_size+peer)*len(input_fields)+field
						self.historical_data.iloc[:,index] = (self.historical_data.iloc[:,index]-group_mean)/group_sigma
						#self.historical_data.iloc[:,index].fillna(value=0.0,inplace=True)

		for f in portfolio_normalize:
			field_indices = []
			if f in first_layer_input_fields:
				field_indices.append(first_layer_input_fields.index(f))
			if f in second_layer_input_fields:
				field_indices.append(second_layer_input_fields.index(f)+len(first_layer_input_fields))
			#if field_indices == []: raise Warning("Couldn't find field {}".format(f))

			for field in field_indices:
				indices = [peer*len(input_fields)+field for peer in range(len(self.tickers))]
				group_data = self.historical_data.iloc[:,indices]
				group_mean = group_data.mean(axis=1)

				#group_sigma = group_data.std(axis=1)
				group_sigma = pd.concat([group_data.std(axis=1),group_data.abs().max(axis=1)],axis=1).apply(replace_zero,axis=1)
				for peer in range(len(self.tickers)):
					index = peer*len(input_fields)+field
					self.historical_data.iloc[:,index] = (self.historical_data.iloc[:,index]-group_mean)/group_sigma
					#self.historical_data.iloc[:,index].fillna(value=0.0,inplace=True)

		for field in z_score_normalize:
			field_indices = []
			if field in first_layer_input_fields:
				field_indices.append(first_layer_input_fields.index(field))
			if field in second_layer_input_fields:
				field_indices.append(second_layer_input_fields.index(field)+len(first_layer_input_fields))
			#if field_indices == []: raise Warning("Couldn't find field {}".format(field))

			for ticker in range(len(self.tickers)):
				for field in field_indices:
					index = ticker*len(input_fields)+field
					mean = self.historical_data.iloc[:self.end_of_train_set,index].mean()
					sigma = self.historical_data.iloc[:self.end_of_train_set,index].std()
					self.historical_data.iloc[:,index] = (self.historical_data.iloc[:,index]-mean)/sigma

		for field in min_max_normalize:
			field_indices = []
			if field in first_layer_input_fields:
				field_indices.append(first_layer_input_fields.index(field))
			if field in second_layer_input_fields:
				field_indices.append(second_layer_input_fields.index(field)+len(first_layer_input_fields))
			#if field_indices == []: raise Warning("Couldn't find field {}".format(field))

			for ticker in range(len(self.tickers)):
				for field in field_indices:
					index = ticker*len(input_fields)+field
					mxs = [x for x in self.historical_data.iloc[:self.end_of_train_set,index].abs() if x < 10e10]
					if mxs == []:
						mx = 0
					else:
						mx = max(mxs)
					if mx > 1:
						self.historical_data.iloc[:,index] = self.historical_data.iloc[:,index]/mx


		if end_date is not None:
			self.historical_data = self.historical_data[self.historical_data.index<=end_date]
		self.historical_data = self.historical_data.dropna()



		self.first_layer_feature_set_size = len(first_layer_input_fields)+(self.first_weight_field is not None)
		self.second_layer_feature_set_size = len(second_layer_input_fields)+(self.second_weight_field is not None)
		self.feature_set_size = self.first_layer_feature_set_size + self.second_layer_feature_set_size


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
			to_stack = []
			first_layer_has_weights = int(self.first_weight_field is not None)
			if self.first_weight_field is not None:
				if self.first_weight_field == 'open_positions':
					p = self.portfolios[-self.window+i][1:]
					p = np.array(list(map(lambda x: 1.0 if x > 0 else -1.0,p)))
					to_stack.append(p.reshape((-1,1)))
				else:
					p = self.portfolios[-self.window + i][1:].reshape((-1, 1))
					to_stack.append(p)

			to_stack.append(data[i,:,:self.first_layer_feature_set_size-first_layer_has_weights])

			if self.second_weight_field is not None:
				if self.second_weight_field == 'open_positions':
					p = self.portfolios[-self.window + i][1:]
					p = np.array(list(map(lambda x: 1.0 if x > 0 else -1.0, p)))
					to_stack.append(p.reshape((-1, 1)))
				else:
					p = self.portfolios[-self.window + i][1:].reshape((-1, 1))
					to_stack.append(p)

			if self.second_layer_feature_set_size > 1:
				to_stack.append(data[i,:,self.first_layer_feature_set_size-first_layer_has_weights:])

			d.append(np.hstack(to_stack))
			#p = self.portfolios[-self.window+i][1:].reshape((-1,1))
			#if self.second_layer_feature_set_size == 0:
			#	d.append(np.hstack((p,data[i,:,:self.first_layer_feature_set_size-1])))
			#elif self.second_layer_feature_set_size == 1:
			#	d.append(np.hstack((p,data[i,:,:self.first_layer_feature_set_size-1],p)))
			#else:
			#	d.append(np.hstack((p,data[i,:,:self.first_layer_feature_set_size-1],p,data[i,:,self.first_layer_feature_set_size-1:])))
		data = np.array(d)
		return data



	def step(self,action,test=False,clip_softmax=None):
		default_rebalance = self.default_rebalances
		if clip_softmax is None:
			clip_softmax = self.clip_softmax
		test = test or self.test
		if action is not None:
			if clip_softmax:
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

			self.last_action = action
		else:
			if default_rebalance:
				#action = self.simulator.portfolio
				action = self.last_action

		p, v = self.simulator.profit_portfolio(self.date,new_portfolio=action,realized=self.realized,overall_profit=True)
		self.profits.append(p)
		self.volumes.append(v)


		dates = list(self.historical_data.index)

		if (test and (dates.index(self.date) + self.step_size) >= len(dates)) or (not test and ((dates.index(self.date) + self.step_size) >= int(self.end_of_train_set))):
			done = True
			#print(self.date)
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
		elif self.reward.startswith('log(long_term_p&l_'):
			n = int(self.reward[len('log(long_term_p&l_'):-1])
			if len(self.profits) < 2:
				reward = math.log(1)
			elif len(self.profits) < n:
				reward = math.log((self.profits[-1]/self.profits[0])**(1/len(self.profits)))
			else:
				reward = math.log(self.profits[-1]/self.profits[-n]**(1/n))
		elif self.reward.startswith('long_term_p&l_'):
			n = int(self.reward[len('long_term_p&l_'):])
			if len(self.profits) < 2:
				reward = 1
			elif len(self.profits) < n:
				reward = (self.profits[-1]-self.profits[0])/len(self.profits)
			else:
				reward = (self.profits[-1]-self.profits[-n])/n


		self.portfolios.append(self.simulator.portfolio)

		return self.getState(), reward, done, {'date':self.date,'profit':self.profits[-1],'portfolio_start': action,'portfolio_end':self.simulator.portfolio,'volume':self.volumes[-1]}

	def expert_step(self,clip_softmax=True,ultimate_expert=False):

		lookahead = self.step_size

		dates = list(self.historical_data.index)
		lookahead_date = dates[dates.index(self.date)+lookahead]

		action = self.simulator.expert_portfolio(self.date,lookahead_date,clip_softmax=clip_softmax,ultimate_expert=ultimate_expert)

		self.date = dates[dates.index(self.date)+self.step_size]

		return action

	def reset(self,test=False):
		if test or self.test:
			return self.reset_test()

		self.date = self.historical_data.index[self.window-1]
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)


		self.profits = [1,]
		self.volumes = [0,]
		self.dates = [self.date,]
		self.sharpe = 0
		self.sortino = 0
		self.portfolios = [self.simulator.portfolio,]*self.window
		self.last_action = None

		return self.getState()

	def reset_test(self):
		dates = list(self.historical_data.index)
		self.date = dates[self.end_of_train_set+self.step_size+1]
		self.simulator = BasicSimulator(self.tickers,self.transaction_cost,self.date)

		self.profits = [1,]
		self.volumes = [0,]
		self.dates = [self.date,]
		self.sharpe = 0
		self.sortino = 0
		self.last_action = None

		self.portfolios = [self.simulator.portfolio,]*self.window
		return self.getState()

	def test_mode(self):
		self.test = True


def date_to_tax_rate(date):
	# Source: https://www.thebalance.com/corporate-income-tax-definition-history-effective-rate-3306024
	if date.year in range(1993,2018):
		return 0.35
	elif date.year >= 2018:
		return 0.21
	else:
		return None

def replace_zero(x):
	if  (x[0] == 0 or x[0] >= 10e10) and (x[1] == 0 or x[1] >= 10e10):
		return 1
	elif (x[0] == 0 or x[0] >= 10e10):
		return x[1]
	else:
		return x[0]



#env = StackedEnv(['AAPL','AMZN'],0.001)
# print(env.reset())
# done = False
# while not done:
# 	state, reward, done, _ = env.step([1/3,1/3,1/3])
# 	#print(state)
# 	print(reward)