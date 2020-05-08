from src.environments.StackedEnv import StackedEnv
from src.environments.StackedEnvDiff import StackedEnvDiff
from src.policies.SharedStackedPolicy import SharedStackedPolicy
from src.policies.SharedStackedRecurrentPolicy import SharedStackedRecurrentPolicy
from stable_baselines import A2C, PPO2
from stable_baselines.acktr import ACKTR
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
import tensorflow as tf
import gc
import numpy as np
import pandas as pd
from stable_baselines.common.policies import nature_cnn
import datetime
#from stable_baselines.gail import generate_expert_traj
from src.utils.record_expert import generate_expert_traj
import copy, os, math
from os.path import dirname
from tensorflow.keras.utils import Progbar
from src.pretraining.MultiDimensionalExpertDataset import MultiDimensionalExpertDataset
from math import sqrt
import warnings
from stable_baselines.common.policies import LstmPolicy
np.random.seed(1)


# +---------------------------------------------------------------------------------------+
# |                                   Experiment 	                                      |
# +---------------------------------------------------------------------------------------+
# | General framework for experiments													  |
# +---------------------------------------------------------------------------------------+

all_groups = [['C', 'JPM', 'TCF', 'UMBF'], ['PFE', 'JNJ', 'ABT', 'BMY'], ['INTC', 'XLNX', 'KOPN', 'MXIM'],
			  ['MCD', 'EAT', 'JACK', 'PZZA'], ['LH', 'AMS', 'DGX', 'PMD'], ['COHR', 'PKI', 'BIO', 'WAT'],
			  ['MMM', 'TFX', 'CRY', 'ATRI'], ['TRT', 'IVAC', 'ASYS', 'VECO'], ['GGG', 'FLS', 'ITT', 'IEX'],
			  ['AVX', 'HUBB', 'IIN', 'MRCY'], ['FLEX', 'CTS', 'IEC', 'SANM'], ['HDSN', 'KAMN', 'LAWS', 'WLFC'],
			  ['CIA', 'AAME', 'FFG', 'GL'], ['CIGI', 'FRPH', 'CTO', 'TRC'], ['NBIX', 'BCRX', 'TECH', 'TTNP'],
			  ['SCON', 'MSI', 'BKTI', 'VSAT'], ['LECO', 'CVR', 'SPXC', 'PFIN'], ['STRM', 'EBIX', 'UIS', 'JKHY'], ['MARPS', 'SJT', 'TPL', 'PBT'],
			  ['UVV', 'STKL', 'ANDE', 'PYX'], ['BZH', 'NVR', 'PHM', 'MTH'], ['MOD', 'DORM', 'STRT', 'SUP'],
			  ['PCAR', 'SPAR', 'F', 'OSK'], ['HLX', 'CLB', 'ENSV', 'RES'], ['BCPC', 'FMC', 'GRA', 'OLN']]

# Late filings:
#	- BMY: 	2002-12-15
#	- XLNX: 2006-08-10
#			2009-05-28
#	- KOPN: many
#	- MXIM: many <= 2008
#	- EAT: 	2008-06-30
#	- PZZA: 2019-02-28
#	- ...
# We assume that these late filings have negligable impact on model performance

all_groups = all_groups[:18]+all_groups[19:]

all_permutations = []
group_count = 3
group_size = len(all_groups[0])

groups = all_groups
while len(all_permutations) < 110:
	gs_i = np.random.choice(range(len(groups)),size=group_count,replace=False)
	gs = [groups[i] for i in gs_i]

	groups = [groups[i] for i in range(len(groups)) if i not in gs_i]
	if groups == []:
		groups = all_groups
	permutation = [np.random.permutation(x).tolist() for x in gs]
	permutation = np.random.permutation(permutation).tolist()
	if permutation not in all_permutations:
		all_permutations.append(permutation)

all_tickers = np.array(all_groups).flatten().tolist()
all_random_permutations = []
while len(all_random_permutations) < 110:
	random_permutation = []
	for _ in range(group_count):
		random_group_ids = np.random.choice(range(len(all_tickers)),size=group_size,replace=False)
		random_group = [all_tickers[i] for i in random_group_ids]

		all_tickers = [all_tickers[i] for i in range(len(all_tickers)) if i not in random_group_ids]

		if all_tickers == []:
			all_tickers = np.array(all_groups).flatten().tolist()

		random_permutation.append(random_group)

	all_random_permutations.append(random_permutation)


#print(all_permutations)
# Test whether there are any groups not included in the permutation set
for g in all_groups:
	for t in g:
		found = False
		for p in all_permutations:
			gs = [g2 for g2 in p if t in g2]
			if len(gs) > 0:
				found = True
				break

		if not found:
			print(g,t)

# +-------------------------------+
# |            Features           |
# +-------------------------------+
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

doubles = ['{}_2'.format(f) for f in fundamentals_1 + fundamentals_derived_1]

macro_economics = ['tax_rate','10y_treasury_rate']

#peer_normalize = fundamentals_1 + fundamentals_derived_1 + fundamentals_derived_2 + fundamentals_derived_3
#portfolio_normalize = doubles
#z_score_normalize = []
#min_max_normalize = fundamentals_1 + fundamentals_derived_1 + fundamentals_derived_2 + doubles

default_peer_normalize = fundamentals_1 + fundamentals_derived_1 + fundamentals_derived_2 + fundamentals_derived_3
default_portfolio_normalize = doubles
default_z_score_normalize = []
default_min_max_normalize = fundamentals_1 + fundamentals_derived_1 + fundamentals_derived_2 + doubles

# +-------------------------------+
# |        Train/Test data        |
# +-------------------------------+
train_test_split = 0.7
train_end_date = datetime.datetime(2014,1,1)
end_date = datetime.datetime(2020,1,1)

# +-------------------------------+
# |          Environment          |
# +-------------------------------+
overall_profit = True
clip_softmax = True

# +-------------------------------+
# |       Model parameters        |
# +-------------------------------+
RL_algo = PPO2
gamma = 1.0
include_cash = True
step_size = 1
action_space = 'box'

# +-------------------------------+
# |           Training            |
# +-------------------------------+
log_interval = 1

# +-------------------------------+
# |         Pre-Training          |
# +-------------------------------+
clip_expert_softmax = clip_softmax
pretrain_batch_size = 32


# +-------------------------------+
# |           Ensemble            |
# +-------------------------------+
deterministic = True
n_soft_ensemble = 1	# Number of predictions we should select and average (note that this will force non-deterministic actions)
					# Shouldn't be used for recurrent networks
n_hard_ensemble = 1	# Number of agents we should train

# +-------------------------------+
# |            General            |
# +-------------------------------+
tensorboard_log="./tensorboard_logs/test/"
full_tensorboard_log = False
start_method = None

# +-------------------------------+
# |            Testing            |
# +-------------------------------+
test_start_index = 0
n_tests = 1
log_suffix = "log_{}"
benchmark_rebalances = False


def sharpe(returns,risk_free=None):
	dates = returns.index
	if risk_free is None:
		risk_free = (1.02**((dates.max()-dates.min()).days/365))**(1/len(dates))
		risk_free = np.array([1,]+[risk_free**i for i in range(1,len(dates))])
	excess_returns = np.array(returns)-risk_free
	mu_excess_returns = excess_returns.mean()
	sigma_excess_returns = excess_returns.std()
	if mu_excess_returns < 0:
		return -sqrt(abs(mu_excess_returns)/sigma_excess_returns)
	else:
		return sqrt(mu_excess_returns/sigma_excess_returns)
def cagr(returns):
	return returns.iloc[-1]**(365.25/((returns.index.max()-returns.index.min()).days))-1
def max_drawdown(returns,years=None):
	if years is not None:
		returns = np.array(returns[returns.index>(returns.index.max()-datetime.timedelta(days=365*years))])
	return np.max((1-returns/np.maximum.accumulate(returns))) # end of the period
def calmar(returns):
	return cagr(returns)/max_drawdown(returns)
def sortino(returns,risk_free=None):
	dates = returns.index
	if risk_free is None:
		risk_free = (1.02**((dates.max()-dates.min()).days/365))**(1/len(dates))
		risk_free = np.array([1,]+[risk_free**i for i in range(1,len(dates))])
	excess_returns = np.array(returns)-risk_free
	mu_excess_returns = excess_returns.mean()
	sigma_excess_returns_negative = excess_returns[excess_returns<0].std()
	if mu_excess_returns < 0:
		return -sqrt(abs(mu_excess_returns)/sigma_excess_returns_negative)
	else:
		return sqrt(mu_excess_returns/sigma_excess_returns_negative)
def mean_upside(returns,risk_free=None):
	dates = returns.index
	if risk_free is None:
		risk_free = (1.02**((dates.max()-dates.min()).days/365))**(1/len(dates))
		risk_free = np.array([1,]+[risk_free**i for i in range(1,len(dates))])
	excess_returns = np.array(returns)/risk_free-1
	mu_excess_returns = excess_returns.mean()
	return mu_excess_returns
def std_upside(returns,risk_free=None):
	dates = returns.index
	if risk_free is None:
		risk_free = (1.02**((dates.max()-dates.min()).days/365))**(1/len(dates))
		risk_free = np.array([1,]+[risk_free**i for i in range(1,len(dates))])
	excess_returns = np.array(returns)/risk_free-1
	sigma_excess_returns = excess_returns.std()
	return sigma_excess_returns

def run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=1,learning_rate=0.01,steps_before_update=512,agent_rebalances=False,environment_type=StackedEnvDiff,n_epochs = 250,verbose_experiment=True,peer_group_aware=True,no_peer_normalization=False):

	if no_peer_normalization:
		peer_normalize = []
		portfolio_normalize = default_portfolio_normalize + default_peer_normalize
	else:
		portfolio_normalize = default_portfolio_normalize
		peer_normalize = default_peer_normalize
	z_score_normalize = default_z_score_normalize
	min_max_normalize = default_min_max_normalize

	def test_model(models,env,groups,transaction_cost=transaction_cost,progbar=None,perm=None):
		env.test_mode()
		env.transaction_cost=transaction_cost
		if n_environments > 1 and policy == SharedStackedRecurrentPolicy:
			vect_env = SubprocVecEnv([lambda: copy.deepcopy(env), ] * n_environments, start_method=start_method)
		else:
			vect_env = DummyVecEnv([lambda: env, ])
		obs = vect_env.reset()


		env_base = environment_type(groups,transaction_cost,train_test_split,realized=False,reward=reward_type,include_cash=True,overall_profit=overall_profit,end_date=end_date,clip_softmax=clip_softmax,first_layer_features=first_layer_features,second_layer_features=second_layer_features,peer_normalize=peer_normalize,z_score_normalize=z_score_normalize,min_max_normalize=min_max_normalize,portfolio_normalize=portfolio_normalize,window=window,a_space=action_space,step_size=step_size,train_end_date=train_end_date,default_rebalances=benchmark_rebalances)
		env_base.test_mode()
		env_base.reset()


		done = [False]
		profit = [1,]
		volume = [0,]
		profit_base = [1,]
		dates = []
		portfolios = []
		first_step = True

		# Expert track: env_copy = copy.deepcopy(env)
		while not done[0]:
			# Get action
			action = None
			for model in models:
				for _ in range(n_soft_ensemble):
					if action is None:
						action = model.predict(obs,deterministic=(deterministic and n_soft_ensemble <= 1))[0]
						# Expert track: action = env_copy.expert_step(clip_softmax=clip_expert_softmax,ultimate_expert=ultimate_expert)[None]
					else:
						action = np.concatenate((action,model.predict(obs,deterministic=(deterministic and n_soft_ensemble <= 1))[0]))
			#u, indices = np.unique(action, return_inverse=True)
			#action = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(action.shape),
			#						None, np.max(indices) + 1), axis=0)][None]

			if verbose_experiment: print(action[0])
			obs, _, done, info = vect_env.step(action)


			if first_step:
				obs_base, _, _, info_base = env_base.step(np.array([0,]+[1/len(tickers),]*len(tickers)))
				first_step = False
			else:
				obs_base, _, _, info_base = env_base.step(None)

			if not done[0]:
				profit.append(info[0]['profit'])
				volume.append(info[0]['volume'])
				profit_base.append(info_base['profit'])
				dates.append(info[0]['date'])
				portfolios.append(info[0]['portfolio_start'])
				if verbose_experiment: print("{}: {}% VS {}% base profit".format(info[0]['date'],round((profit[-1]-1)*100,2), round((profit_base[-1]-1)*100,2)))

		#risk_free = (1.02**((dates[-1]-dates[0]).days/365))**(1/len(dates))
		#risk_free = np.array([1,]+[risk_free**i for i in range(1,len(dates)+1)])

		#excess_returns = np.array(profit)-risk_free
		#excess_returns_base = np.array(profit_base)-risk_free

		#sharpe = excess_returns.mean()/excess_returns.std()
		#sharpe_base = excess_returns_base.mean()/excess_returns_base.std()

		log = {'Date':dates,'Weights':portfolios,'Profit':profit[1:],'Base Profit':profit_base[1:],'Volume':volume[1:]}

		df = pd.DataFrame(log).set_index('Date')
		if verbose_experiment:
			print("Sharpe: {} VS {} base".format(sharpe(df['Profit']),sharpe(df['Base Profit'])))
			print("CAGR: {} VS {} base".format(cagr(df['Profit']),cagr(df['Base Profit'])))
			print("MDD: {} VS {} base".format(-max_drawdown(df['Profit']),-max_drawdown(df['Base Profit'])))
			print("Calmar: {} VS {} base".format(calmar(df['Profit']),calmar(df['Base Profit'])))
			print("Turnover: {}".format(round(df['Volume'].sum(),2)))
		else:
			progbar.update(perm+1,[("Sharpe",sharpe(df['Profit'])),
								   ("Base Sharpe",sharpe(df['Base Profit'])),
								   ("CAGR",cagr(df['Profit'])),
								   ("Base CAGR",cagr(df['Base Profit'])),
								   ("MDD", -max_drawdown(df['Profit'])),
								   ("Base MDD", -max_drawdown(df['Base Profit'])),
								   ("Calmar", calmar(df['Profit'])),
								   ("Base Calmar", calmar(df['Base Profit'])),
								   ("Turnover", round(df['Volume'].sum(),2)),
								   ("Relative Sharpe",sharpe(df['Profit'],df['Base Profit'])),
								   ("Mean Upside",mean_upside(df['Profit'],df['Base Profit'])),])
		return log

	for test_round in range(test_start_index,test_start_index+n_tests):
		training = False

		if peer_group_aware:
			permutations = all_permutations[permutation_start_index:permutation_start_index+n_permutations]
		else:
			permutations = all_random_permutations[permutation_start_index:permutation_start_index + n_permutations]

		if not verbose_experiment:
			progbar = Progbar(len(permutations))
			progbar.update(0)
		for i in range(n_hard_ensemble):
			for perm in range(len(permutations)):
				groups = permutations[perm]
				if verbose_experiment:
					print("Started processing: {}".format(str(perm)))
					print(groups)
				tickers = list(np.array(groups).flatten())
				env = environment_type(groups,transaction_cost,train_test_split=train_test_split,realized=False,reward=reward_type,include_cash=include_cash,overall_profit=overall_profit,end_date=end_date,clip_softmax=clip_softmax,first_layer_features=first_layer_features,second_layer_features=second_layer_features,peer_normalize=peer_normalize,z_score_normalize=z_score_normalize,min_max_normalize=min_max_normalize,portfolio_normalize=portfolio_normalize,window=window,a_space=action_space,step_size=step_size,train_end_date=train_end_date,default_rebalances=agent_rebalances)
				if verbose_experiment:
					print(env.first_layer_feature_set_size)
					print(env.second_layer_feature_set_size)
					print(env.historical_data.describe())
				policy_kwargs['n_groups'] = env.group_count
				policy_kwargs['group_size'] = env.group_size
				policy_kwargs['n_first_layer_input_features'] = env.first_layer_feature_set_size
				policy_kwargs['second_layer'] = env.second_layer_feature_set_size > 0

				if n_environments > 1 and train:
					vect_env = SubprocVecEnv([lambda: copy.deepcopy(env), ] * n_environments,start_method=start_method)
				else:
					vect_env = DummyVecEnv([lambda: env, ])

				if n_hard_ensemble > 1:
					l_name = "./saved_models/{}_{}.model".format(load_name.format(str(perm+permutation_start_index)),str(i))
					s_name = "./saved_models/{}_{}.model".format(save_name.format(str(perm+permutation_start_index)),str(i))
				else:
					l_name = "./saved_models/{}.model".format(load_name.format(str(perm+permutation_start_index)))
					s_name = "./saved_models/{}.model".format(save_name.format(str(perm+permutation_start_index)))

				if training and not fine_tune:
					l_name = s_name

				if RL_algo == A2C:
					if load or training:
						model = A2C.load(l_name, vect_env, gamma=gamma, n_steps=steps_before_update, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
									 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=int(verbose_experiment), tensorboard_log=tensorboard_log,
									 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = A2C(policy, vect_env, gamma=gamma, n_steps=steps_before_update, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
									 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=int(verbose_experiment), tensorboard_log=tensorboard_log,
									 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				elif RL_algo == PPO2:
					if load or training:
						model = PPO2.load(l_name, vect_env, gamma=gamma, n_steps=steps_before_update, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
									 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
									 verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = PPO2(policy, vect_env, gamma=gamma, n_steps=steps_before_update, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
									 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
									 verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				elif RL_algo == ACKTR:
					if load or training:
						model = ACKTR.load(l_name, vect_env, gamma=gamma, nprocs=None, n_steps=steps_before_update, ent_coef=0.01, vf_coef=0.25,
										   vf_fisher_coef=1.0, learning_rate=learning_rate, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear',
										   verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, async_eigen_decomp=False,
										   kfac_update=1, gae_lambda=None, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = ACKTR(policy, vect_env, gamma=gamma, nprocs=None, n_steps=steps_before_update, ent_coef=0.01, vf_coef=0.25,
										   vf_fisher_coef=1.0, learning_rate=learning_rate, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear',
										   verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, async_eigen_decomp=False,
										   kfac_update=1, gae_lambda=None, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)

				if train:
					if pre_training:
						if verbose_experiment: print("Pre-training...")
						def create_expert(env):
							env_copy = copy.deepcopy(env)
							env_copy.reset()
							def expert(_obs):
								expert_action = env_copy.expert_step(clip_softmax=clip_expert_softmax,ultimate_expert=ultimate_expert)
								return expert_action
							return expert

						expert_path = os.path.relpath('{}/experts/{}{}_{}_{}_{}_{}.npz'.format(dirname(__file__),expert_name,window,environment_type.__name__,'U' if ultimate_expert else 'NU',("{}"*len(tickers)).format(*[t[0] for t in tickers]),str(transaction_cost)))
						if not os.path.isfile(expert_path):
							generate_expert_traj(create_expert(env), expert_path, copy.deepcopy(env), n_episodes=1,verbose=verbose_experiment)

						dataset = MultiDimensionalExpertDataset(expert_path=[expert_path], traj_limitation=-1, batch_size=pretrain_batch_size, verbose=int(verbose_experiment))
						model.pretrain(dataset,n_epochs=n_epochs)
					model.learn(total_timesteps=total_timesteps, log_interval=log_interval,seed=1)
					model.save(s_name)

				training = True

				if verbose_experiment: print("Done training {} -> {}!".format(l_name,s_name))

				if test:
					if not verbose_experiment:
						log = test_model([model,],env,groups,progbar=progbar,perm=perm)
					else:
						log = test_model([model, ], env, groups)
					for ticker in range(len(tickers)):
						log[tickers[ticker]] = [row[ticker+1] for row in log['Weights']]
					log['cash'] = [row[0] for row in log['Weights']]
					pd.DataFrame(log).to_csv("./logs/{}_{}_group_{}.csv".format(save_name,log_suffix.format(test_round),str(perm+permutation_start_index)))
				else:
					if not verbose_experiment:
						progbar.update(perm+1)

				gc.collect()
def run_concurrent_experiment(window,reward_type,policy,policy_kwargs,load,fine_tune,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features,n_environments=1,learning_rate=0.01,steps_before_update=512,agent_rebalances=False,environment_type=StackedEnvDiff,n_epochs = 250,verbose_experiment=True,peer_group_aware=True,no_peer_normalization=False):

	training = False

	if no_peer_normalization:
		peer_normalize = []
		portfolio_normalize = default_portfolio_normalize + default_peer_normalize
	else:
		portfolio_normalize = default_portfolio_normalize
		peer_normalize = default_peer_normalize
	z_score_normalize = default_z_score_normalize
	min_max_normalize = default_min_max_normalize

	if peer_group_aware:
		permutations = all_permutations[permutation_start_index:permutation_start_index + n_permutations]
	else:
		permutations = all_random_permutations[permutation_start_index:permutation_start_index + n_permutations]

	permutation_split = []
	for i in range(len(permutations)//n_environments):
		permutation_split.append(permutations[i*n_environments:(i+1)*n_environments])

	if len(permutations)%n_environments > 0:
		permutation_split.append(permutations[(len(permutations)//n_environments)*n_environments:])

	for i in range(n_hard_ensemble):
		if not verbose_experiment:
			progbar = Progbar(len(permutation_split))
			progbar.update(0)
		for perm in range(len(permutation_split)):
			permutations_subset = permutation_split[perm]
			if verbose_experiment:
				print("Started processing: {}".format(str(perm*n_environments)))
				print(permutations_subset)
			envs = [environment_type(groups,transaction_cost,train_test_split=train_test_split,realized=False,reward=reward_type,include_cash=include_cash,overall_profit=overall_profit,end_date=end_date,clip_softmax=clip_softmax,first_layer_features=first_layer_features,second_layer_features=second_layer_features,peer_normalize=peer_normalize,z_score_normalize=z_score_normalize,min_max_normalize=min_max_normalize,portfolio_normalize=portfolio_normalize,window=window,a_space=action_space,step_size=step_size,train_end_date=train_end_date,default_rebalances=agent_rebalances) for groups in permutations_subset]

			if verbose_experiment:
				print(envs[0].first_layer_feature_set_size)
				print(envs[0].second_layer_feature_set_size)
				print(envs[0].historical_data.describe())
			policy_kwargs['n_groups'] = envs[0].group_count
			policy_kwargs['group_size'] = envs[0].group_size
			policy_kwargs['n_first_layer_input_features'] = envs[0].first_layer_feature_set_size
			policy_kwargs['second_layer'] = envs[0].second_layer_feature_set_size > 0

			vect_env = SubprocVecEnv([lambda: copy.deepcopy(env) for env in envs],start_method=start_method)

			if n_hard_ensemble > 1:
				l_name = "./saved_models/{}_{}.model".format(load_name.format(str(perm+permutation_start_index)),str(i))
				s_name = "./saved_models/{}_{}.model".format(save_name.format(str(perm+permutation_start_index)),str(i))
			else:
				l_name = "./saved_models/{}.model".format(load_name.format(str(perm+permutation_start_index)))
				s_name = "./saved_models/{}.model".format(save_name.format(str(perm+permutation_start_index)))

			if training and not fine_tune:
				l_name = s_name

			if RL_algo == A2C:
				if load or training:
					model = A2C.load(l_name, vect_env, gamma=gamma, n_steps=steps_before_update, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
								 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=int(verbose_experiment), tensorboard_log=tensorboard_log,
								 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				else:
					model = A2C(policy, vect_env, gamma=gamma, n_steps=steps_before_update, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
								 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=int(verbose_experiment), tensorboard_log=tensorboard_log,
								 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
			elif RL_algo == PPO2:
				if load or training:
					model = PPO2.load(l_name, vect_env, gamma=gamma, n_steps=steps_before_update, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
								 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
								 verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				else:
					model = PPO2(policy, vect_env, gamma=gamma, n_steps=steps_before_update, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
								 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
								 verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
			elif RL_algo == ACKTR:
				if load or training:
					model = ACKTR.load(l_name, vect_env, gamma=gamma, nprocs=None, n_steps=steps_before_update, ent_coef=0.01, vf_coef=0.25,
									   vf_fisher_coef=1.0, learning_rate=learning_rate, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear',
									   verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, async_eigen_decomp=False,
									   kfac_update=1, gae_lambda=None, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				else:
					model = ACKTR(policy, vect_env, gamma=gamma, nprocs=None, n_steps=steps_before_update, ent_coef=0.01, vf_coef=0.25,
									   vf_fisher_coef=1.0, learning_rate=learning_rate, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear',
									   verbose=int(verbose_experiment), tensorboard_log=tensorboard_log, _init_setup_model=True, async_eigen_decomp=False,
									   kfac_update=1, gae_lambda=None, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)

			if pre_training:
				if verbose_experiment: print("Pre-training...")
				expert_paths = []
				for env in envs:
					def create_expert(env):
						env_copy = copy.deepcopy(env)
						env_copy.reset()
						def expert(_obs):
							expert_action = env_copy.expert_step(clip_softmax=clip_expert_softmax,ultimate_expert=ultimate_expert)
							return expert_action
						return expert

					expert_path = os.path.relpath('{}/experts/{}{}_{}_{}_{}_{}.npz'.format(dirname(__file__),expert_name,window,environment_type.__name__,'U' if ultimate_expert else 'NU',("{}"*len(env.tickers)).format(*[t[0] for t in env.tickers]),str(transaction_cost)))
					if not os.path.isfile(expert_path):
						generate_expert_traj(create_expert(env), expert_path, copy.deepcopy(env), n_episodes=1,verbose=verbose_experiment)

					expert_paths.append(expert_path)
				dataset = MultiDimensionalExpertDataset(expert_path=expert_paths, traj_limitation=-1, batch_size=pretrain_batch_size, verbose=int(verbose_experiment))
				model.pretrain(dataset,n_epochs=n_epochs)
			model.learn(total_timesteps=total_timesteps*len(permutations_subset), log_interval=log_interval,seed=1)
			model.save(s_name)

			training = True

			if verbose_experiment:
				print("Done training {} -> {}!".format(l_name,s_name))
			else:
				progbar.update(perm+1)

			gc.collect()
