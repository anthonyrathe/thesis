from src.environments.StackedEnv import StackedEnv
from src.environments.StackedEnvDiff import StackedEnvDiff
from src.policies.SharedStackedPolicy import SharedStackedPolicy
from src.policies.SharedStackedRecurrentPolicy import SharedStackedRecurrentPolicy
from stable_baselines import A2C, PPO2
from stable_baselines.acktr import ACKTR
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf
import numpy as np
import pandas as pd
from stable_baselines.common.policies import nature_cnn
import datetime
from stable_baselines.gail import generate_expert_traj
import copy, os, math
from os.path import dirname
from src.pretraining.MultiDimensionalExpertDataset import MultiDimensionalExpertDataset
np.random.seed(1)

# +---------------------------------------------------------------------------------------+
# |                                   Experiment 1 b                                      |
# +---------------------------------------------------------------------------------------+
# | Cfr. experiment 1 a, but with transaction costs.									  |
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

print(all_permutations)
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

peer_normalize = fundamentals_1 + fundamentals_derived_1 + fundamentals_derived_2 + fundamentals_derived_3
portfolio_normalize = doubles
z_score_normalize = []
min_max_normalize = fundamentals_1 + fundamentals_derived_1 + fundamentals_derived_2 + doubles

# +-------------------------------+
# |        Train/Test data        |
# +-------------------------------+
train_test_split = 0.7
train_end_date = datetime.datetime(2014,1,1)
end_date = datetime.datetime(2020,1,1)

# +-------------------------------+
# |          Environment          |
# +-------------------------------+
environment_type = StackedEnvDiff
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
learning_rate = 0.01
steps_before_update = 512 # Number of environment steps before network is updated -> should be large enough in order for network to learn
							# how to plan long-term, but small enough for the network to see short-term gains as well

# +-------------------------------+
# |         Pre-Training          |
# +-------------------------------+
n_epochs = 250 # Specify number of epochs to pre-train
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
		return -math.sqrt(abs(mu_excess_returns)/sigma_excess_returns)
	else:
		return math.sqrt(mu_excess_returns/sigma_excess_returns)
def cagr(returns):
	return returns.iloc[-1]**(365.25/((returns.index.max()-returns.index.min()).days))-1
def max_drawdown_last_3y(returns):
	returns = np.array(returns[returns.index>(returns.index.max()-datetime.timedelta(days=365*3))])
	return np.max((1-returns/np.maximum.accumulate(returns))) # end of the period
def calmar(returns):
	return cagr(returns)/max_drawdown_last_3y(returns)

def run_experiment(window,reward_type,policy,policy_kwargs,train,load,fine_tune,test,load_name,save_name,pre_training,ultimate_expert,expert_name,total_timesteps,transaction_cost,permutation_start_index,n_permutations,first_layer_features,second_layer_features):
	def test_model(models,env,groups,transaction_cost=transaction_cost):
		env.test_mode()
		env.transaction_cost=transaction_cost
		vect_env = DummyVecEnv([lambda: env,])
		obs = vect_env.reset()


		env_base = environment_type(groups,transaction_cost,train_test_split,realized=False,reward=reward_type,include_cash=True,overall_profit=overall_profit,end_date=end_date,clip_softmax=clip_softmax,first_layer_features=first_layer_features,second_layer_features=second_layer_features,peer_normalize=peer_normalize,z_score_normalize=z_score_normalize,min_max_normalize=min_max_normalize,portfolio_normalize=portfolio_normalize,window=window,a_space=action_space,step_size=step_size,train_end_date=train_end_date)
		env_base.test_mode()
		env_base.reset()


		done = False
		profit = [1,]
		volume = [0,]
		profit_base = [1,]
		dates = []
		portfolios = []
		first_step = True
		while not done:
			# Get action
			action = None
			for model in models:
				for _ in range(n_soft_ensemble):
					if action is None:
						action = model.predict(obs,deterministic=(deterministic and n_soft_ensemble <= 1))[0]
					else:
						action = np.concatenate((action,model.predict(obs,deterministic=(deterministic and n_soft_ensemble <= 1))[0]))
			#u, indices = np.unique(action, return_inverse=True)
			#action = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(action.shape),
			#						None, np.max(indices) + 1), axis=0)][None]

			print(action)
			obs, _, done, info = vect_env.step(action)

			if first_step:
				obs_base, _, _, _ = env_base.step(np.array([0,]+[1/len(tickers),]*len(tickers)))
				first_step = False
			else:
				obs_base, _, _, _ = env_base.step(None,default_rebalance=benchmark_rebalances)

			if not done:
				profit.append(vect_env.envs[0].profits[-1])
				volume.append(vect_env.envs[0].volumes[-1])
				profit_base.append(env_base.profits[-1])
				dates.append(info[0]['date'])
				portfolios.append(env.simulator.portfolio)
				print("{}: {}% VS {}% base profit".format(info[0]['date'],round((profit[-1]-1)*100,2), round((profit_base[-1]-1)*100,2)))

		risk_free = (1.02**((dates[-1]-dates[0]).days/365))**(1/len(dates))
		risk_free = np.array([1,]+[risk_free**i for i in range(1,len(dates)+1)])

		excess_returns = np.array(profit)-risk_free
		excess_returns_base = np.array(profit_base)-risk_free

		sharpe = excess_returns.mean()/excess_returns.std()
		sharpe_base = excess_returns_base.mean()/excess_returns_base.std()

		log = {'Date':dates,'Weights':portfolios,'Profit':profit[1:],'Base Profit':profit_base[1:],'Volume':volume[1:]}


		print("Sharpe: {} VS {} base".format(sharpe,sharpe_base))
		df = pd.DataFrame(log).set_index('Date')
		print("CAGR: {} VS {} base".format(cagr(df['Profit']),cagr(df['Base Profit'])))
		print("MDD: {} VS {} base".format(-max_drawdown_last_3y(df['Profit']),-max_drawdown_last_3y(df['Base Profit'])))
		print("Calmar: {} VS {} base".format(calmar(df['Profit']),calmar(df['Base Profit'])))
		print("Turnover: {}".format(round(df['Volume'].sum(),2)))
		return log

	for test_round in range(test_start_index,test_start_index+n_tests):
		training = False

		permutations = all_permutations[permutation_start_index:permutation_start_index+n_permutations]

		for i in range(n_hard_ensemble):
			for perm in range(len(permutations)):
				groups = permutations[perm]
				print("Started processing: {}".format(str(perm)))
				print(groups)
				tickers = list(np.array(groups).flatten())
				env = environment_type(groups,transaction_cost,train_test_split=train_test_split,realized=False,reward=reward_type,include_cash=include_cash,overall_profit=overall_profit,end_date=end_date,clip_softmax=clip_softmax,first_layer_features=first_layer_features,second_layer_features=second_layer_features,peer_normalize=peer_normalize,z_score_normalize=z_score_normalize,min_max_normalize=min_max_normalize,portfolio_normalize=portfolio_normalize,window=window,a_space=action_space,step_size=step_size,train_end_date=train_end_date)
				print(env.first_layer_feature_set_size)
				print(env.second_layer_feature_set_size)
				print(env.historical_data.describe())
				policy_kwargs['n_groups'] = env.group_count
				policy_kwargs['group_size'] = env.group_size
				policy_kwargs['n_first_layer_input_features'] = env.first_layer_feature_set_size
				policy_kwargs['second_layer'] = env.second_layer_feature_set_size > 0

				vect_env = DummyVecEnv([lambda: env,])

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
									 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=1, tensorboard_log=tensorboard_log,
									 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = A2C(policy, vect_env, gamma=gamma, n_steps=steps_before_update, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
									 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=1, tensorboard_log=tensorboard_log,
									 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				elif RL_algo == PPO2:
					if load or training:
						model = PPO2.load(l_name, vect_env, gamma=gamma, n_steps=steps_before_update, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
									 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
									 verbose=1, tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = PPO2(policy, vect_env, gamma=gamma, n_steps=steps_before_update, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
									 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
									 verbose=1, tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				elif RL_algo == ACKTR:
					if load or training:
						model = ACKTR.load(l_name, vect_env, gamma=gamma, nprocs=None, n_steps=steps_before_update, ent_coef=0.01, vf_coef=0.25,
										   vf_fisher_coef=1.0, learning_rate=learning_rate, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear',
										   verbose=1, tensorboard_log=tensorboard_log, _init_setup_model=True, async_eigen_decomp=False,
										   kfac_update=1, gae_lambda=None, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = ACKTR(policy, vect_env, gamma=gamma, nprocs=None, n_steps=steps_before_update, ent_coef=0.01, vf_coef=0.25,
										   vf_fisher_coef=1.0, learning_rate=learning_rate, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear',
										   verbose=1, tensorboard_log=tensorboard_log, _init_setup_model=True, async_eigen_decomp=False,
										   kfac_update=1, gae_lambda=None, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)

				if train:
					if pre_training:
						print("Pre-training...")
						def create_expert(env):
							env_copy = copy.deepcopy(env)
							env_copy.reset()
							def expert(_obs):
								expert_action = env_copy.expert_step(clip_softmax=clip_expert_softmax,ultimate_expert=ultimate_expert)
								return expert_action
							return expert

						expert_path = os.path.relpath('{}/experts/{}{}_{}_{}_{}_{}.npz'.format(dirname(__file__),expert_name,window,environment_type.__name__,'U' if ultimate_expert else 'NU',("{}"*len(tickers)).format(*[t[0] for t in tickers]),str(transaction_cost)))
						if not os.path.isfile(expert_path):
							generate_expert_traj(create_expert(env), expert_path, copy.deepcopy(env), n_episodes=1)

						dataset = MultiDimensionalExpertDataset(expert_path=expert_path, traj_limitation=-1, batch_size=pretrain_batch_size)
						model.pretrain(dataset,n_epochs=n_epochs)
					model.learn(total_timesteps=total_timesteps, log_interval=log_interval,seed=1)
					model.save(s_name)

				training = True

				print("Done training {} -> {}!".format(l_name,s_name))

				if test:
					log = test_model([model,],env,groups)
					for ticker in range(len(tickers)):
						log[tickers[ticker]] = [row[ticker+1] for row in log['Weights']]
					log['cash'] = [row[0] for row in log['Weights']]
					pd.DataFrame(log).to_csv("./logs/{}_{}_group_{}.csv".format(save_name,log_suffix.format(test_round),str(perm+permutation_start_index)))