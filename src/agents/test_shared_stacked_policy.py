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
import copy, os
from os.path import dirname
from src.pretraining.MultiDimensionalExpertDataset import MultiDimensionalExpertDataset


all_groups = [['C', 'JPM', 'TCF', 'UMBF'], ['PFE', 'JNJ', 'ABT', 'BMY'], ['INTC', 'XLNX', 'KOPN', 'MXIM'],
			  ['MCD', 'EAT', 'JACK', 'PZZA'], ['LH', 'AMS', 'DGX', 'PMD'], ['COHR', 'PKI', 'BIO', 'WAT'],
			  ['MMM', 'TFX', 'CRY', 'ATRI'], ['TRT', 'IVAC', 'ASYS', 'VECO'], ['GGG', 'FLS', 'ITT', 'IEX'],
			  ['AVX', 'HUBB', 'IIN', 'MRCY'], ['FLEX', 'CTS', 'IEC', 'SANM'], ['HDSN', 'KAMN', 'LAWS', 'WLFC'],
			  ['CIA', 'AAME', 'FFG', 'GL'], ['CIGI', 'FRPH', 'CTO', 'TRC'], ['NBIX', 'BCRX', 'TECH', 'TTNP'],
			  ['SCON', 'MSI', 'BKTI', 'VSAT'], ['LECO', 'CVR', 'SPXC', 'PFIN'], ['STRM', 'EBIX', 'UIS', 'JKHY'], ['MARPS', 'SJT', 'TPL', 'PBT'],
			  ['UVV', 'STKL', 'ANDE', 'PYX'], ['BZH', 'NVR', 'PHM', 'MTH'], ['MOD', 'DORM', 'STRT', 'SUP'],
			  ['PCAR', 'SPAR', 'F', 'OSK'], ['HLX', 'CLB', 'ENSV', 'RES'], ['BCPC', 'FMC', 'GRA', 'OLN']]
# faulty groups: 18 ['MARPS', 'SJT', 'TPL', 'PBT']
group_size = len(all_groups[0])


# +-------------------------------+
# |        Train/Test data        |
# +-------------------------------+
train_test_split = 0.7
end_date = datetime.datetime(2020,1,1)
group_start_index = 0
group_count = 24
n_group_shifts = 1

# +-------------------------------+
# |          Environment          |
# +-------------------------------+
environment_type = StackedEnvDiff
transaction_cost = 0.00000000001
overall_profit = True
clip_softmax = True

# +-------------------------------+
# |       Model parameters        |
# +-------------------------------+
RL_algo = PPO2
include_cash = True
step_size = 1
action_space = 'box'
peer_normalize = ['EV/EBITDA','P/E','P/B','D/E', 'Bias_EV/EBITDA_60','Bias_Price_28','R_Price']
normalize = ['BBANDS_u','BBANDS_m','BBANDS_l','SAREXT','DEMA', 'EMA', 'SMA', 'TEMA','WMA','ADXR', 'APO',
			 'AROON_down', 'AROON_up', 'CCI', 'CMO', 'MFI', 'MACD', 'MACD_signal', 'MACD_hist', 'MOM',
			 'PLUS_DI', 'PPO', 'ROC', 'ROCP', 'RSI', 'STOCH_k', 'STOCH_d', 'STOCHF_k', 'STOCHF_d',
			 'TRIX', 'ULTOSC', 'WILLR', 'AD', 'OBV', 'ATR', 'NATR','HT_DCPERIOD', 'HT_SINE', 'HT_SINE_lead',
			 'HT_DCPHASE', 'HT_PHASOR_inphase', 'HT_PHASOR_quadrature']+peer_normalize
normalize = peer_normalize
window = 1 # Amount of timesteps passed in one observation
reward_type="p&l"
#policy = SharedStackedRecurrentPolicy
#net_arch=[40,'lstm',10,'merge',30,dict(pi=[20,group_count*group_size+int(include_cash)],vf=[5])]
policy = SharedStackedPolicy
#net_arch=[40,10,'merge',30,dict(pi=[20,group_count*group_size+int(include_cash)],vf=[5])]
net_arch=[40,10,'merge',150,dict(pi=[40,group_count*group_size+int(include_cash)],vf=[10])]
#net_arch=['merge',400,200,100,dict(pi=[10,group_count*group_size+int(include_cash)],vf=[5])]
#net_arch=['merge',40*group_count,30*group_count,20,dict(pi=[20,'softmax'],vf=[10,3])]
policy_kwargs = dict( net_arch=net_arch,
				 act_fun=tf.nn.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp")
#policy_kwargs['n_lstm'] = 40

# +-------------------------------+
# |           Training            |
# +-------------------------------+
total_timesteps = 10000
log_interval = 1
learning_rate = 0.01
pre_training = True
n_epochs = 200 # Specify number of epochs to pre-train
clip_expert_softmax = clip_softmax
multi_training = 3

# +-------------------------------+
# |           Ensemble            |
# +-------------------------------+
deterministic = True
n_permutations = 1
full_permutation = True
n_soft_ensemble = 1	# Number of predictions we should select and average (note that this will force non-deterministic actions)
					# Shouldn't be used for recurrent networks
n_hard_ensemble = 1	# Number of agents we should train

# +-------------------------------+
# |            General            |
# +-------------------------------+
train = True
load = True
load_name = "golden_agent_all_groups_4"
save_name = "golden_agent_all_groups_5"
tensorboard_log="./tensorboard_logs/test/"
full_tensorboard_log = False

# +-------------------------------+
# |            Testing            |
# +-------------------------------+
test_start_index = 0
n_tests = 1
log_suffix = "log_{}"


#TODO: try different reward functions (log? actual absolute P&L per step? -> sum of rewards should have meaning)
#TODO:!!!! test simulator for correctness
#TODO: try different networks (softmax at the end)
#TODO: try different features (ts fresh?)
#TODO: play with some parameters (learning rate, ...)
#TODO: how many episodes should the RL agent be trained for? How much exploration/exploitation
#TODO: controleer alle code op correctheid (zijn de observations wel juist, worden de portfolio vectoren correct bijgehouden, ...)
#TODO: add peer group identifier
#TODO: train op permutaties!!
#TODO: try input scaling
#TODO: library: https://ta-lib.org/hdr_dw.html -> technical indicators
#TODO: ... check paper: Time-driven feature-aware jointly deep reinforcement learning for financial signal representation and algorithmic trading
#TODO: try including MULTIPLE timeframes in the input
# line 346 in stable_baselines.ppo2.ppo2 was added by me -> remove if necessary
#TODO: add 'technical' indicators of fundamental data (like MA of EV/EBITDA)
#TODO: concurrently perform supervised learning on actor-critic network -> target value is a one-hot vector encoding
#		the stock that will rise the most next day (could happen both on peer-group and on portfolio level)
# line 237 in stable_baselines.common.distributions -> removed final dense layer

def cagr(returns):
	return returns.iloc[-1]**(365.25/((returns.index.max()-returns.index.min()).days))-1
def max_drawdown_last_3y(returns):
	returns = np.array(returns[returns.index>(returns.index.max()-datetime.timedelta(days=365*3))])
	return np.max((1-returns/np.maximum.accumulate(returns))) # end of the period
def calmar(returns):
	return cagr(returns)/max_drawdown_last_3y(returns)

def test_model(models,env,groups,transaction_cost=transaction_cost):
	env.test_mode()
	env.transaction_cost=transaction_cost
	vect_env = DummyVecEnv([lambda: env,])
	obs = vect_env.reset()


	env_base = environment_type(groups,transaction_cost,train_test_split,realized=False,reward=reward_type,include_cash=True,overall_profit=overall_profit,end_date=end_date,clip_softmax=clip_softmax,peer_normalize=peer_normalize,normalize=normalize,window=window,a_space=action_space,step_size=step_size)
	env_base.test_mode()
	env_base.reset()


	done = False
	profit = [1,]
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
			obs_base, _, _, _ = env_base.step(None)

		if not done:
			profit.append(vect_env.envs[0].profits[-1])
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

	log = {'Date':dates,'Weights':portfolios,'Profit':profit[1:],'Base Profit':profit_base[1:]}


	print("Sharpe: {} VS {} base".format(sharpe,sharpe_base))
	df = pd.DataFrame(log).set_index('Date')
	print("CAGR: {} VS {} base".format(cagr(df['Profit']),cagr(df['Base Profit'])))
	print("MDD: {} VS {} base".format(-max_drawdown_last_3y(df['Profit']),-max_drawdown_last_3y(df['Base Profit'])))
	print("Calmar: {} VS {} base".format(calmar(df['Profit']),calmar(df['Base Profit'])))
	return log

#groups = [['PEP','KO','KDP','MNST'],['BBY','TGT','WMT','COST'],['FB','GOOGL','AAPL','AMZN']]
#groups = [['C', 'JPM', 'TCF', 'UMBF'], ['PFE', 'JNJ', 'CAPR', 'AKRX'], ['VZ', 'T', 'FTR', 'SHEN'], ['INTC', 'XLNX', 'EMAN', 'KOPN'], ['LEA', 'MOD', 'DORM', 'STRT'], ['MCD', 'EAT', 'JACK', 'PZZA'], ['TGNA', 'NTN', 'CETV', 'SSP'], ['ZIXI', 'TCX', 'CSGS', 'RAMP'], ['KSU', 'CSX', 'UNP', 'NSC'], ['ARTNA', 'AWR', 'MSEX', 'YORW'], ['PCAR', 'SPAR', 'F', 'OSK'], ['TESS', 'ARW', 'TAIT', 'UUU'], ['GLBZ', 'TRST', 'GSBC', 'FCCO'], ['UAL', 'SKYW', 'ALK', 'AAL'], ['DAIO', 'TER', 'COHU', 'ITRI'], ['LNG', 'NJR', 'SJI', 'ATO'], ['NTIP', 'VHC', 'ACTG', 'REFR'], ['MUX', 'AUMN', 'VGZ', 'GSS'], ['FRHC', 'RJF', 'GBL', 'SIEB'], ['NBIX', 'BCRX', 'TECH', 'TTNP'], ['GHC', 'PRDO', 'STRA', 'ATGE'], ['FCAP', 'BYFC', 'PROV', 'CASH'], ['MARPS', 'SJT', 'TPL', 'PBT'], ['RIG', 'DO', 'VAL', 'PTEN'], ['WTS', 'CIR', 'PH', 'HLIO']]
all_groups = all_groups[:18]+all_groups[19:]


for test_round in range(test_start_index,test_start_index+n_tests):
	training = False
	for group_round in range(n_group_shifts):


		groups = all_groups[group_start_index+group_round*group_count:group_start_index+(group_round+1)*group_count]
		tickers = list(np.array(groups).flatten())

		permutations = [groups,]
		np.random.seed(1)
		if n_permutations > 1:
			for _ in range(n_permutations-1):
				if full_permutation:
					permutation = [np.random.permutation(x).tolist() for x in groups]
				else:
					permutation = groups
				permutations.append(np.random.permutation(permutation).tolist())
		print(permutations)

		models = []
		for i in range(n_hard_ensemble):
			for groups in permutations:
				tickers = list(np.array(groups).flatten())
				env = environment_type(groups,transaction_cost,train_test_split=train_test_split,realized=False,reward=reward_type,include_cash=include_cash,overall_profit=overall_profit,end_date=end_date,clip_softmax=clip_softmax,peer_normalize=peer_normalize,normalize=normalize,window=window,a_space=action_space,step_size=step_size)
				print(env.historical_data.describe())
				policy_kwargs['n_groups'] = env.group_count
				policy_kwargs['group_size'] = env.group_size
				policy_kwargs['n_first_layer_input_features'] = env.first_layer_feature_set_size

				vect_env = DummyVecEnv([lambda: env,])


				if n_hard_ensemble > 1:
					l_name = "./saved_models/{}_{}.model".format(load_name,str(i))
					s_name = "./saved_models/{}_{}.model".format(save_name,str(i))
				else:
					l_name = "./saved_models/{}.model".format(load_name)
					s_name = "./saved_models/{}.model".format(save_name)

				if training:
					l_name = s_name

				if RL_algo == A2C:
					if load or training:
						model = A2C.load(l_name, vect_env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
									 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=1, tensorboard_log=tensorboard_log,
									 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = A2C(policy, vect_env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
									 learning_rate=learning_rate, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=1, tensorboard_log=tensorboard_log,
									 _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				elif RL_algo == PPO2:
					if load or training:
						model = PPO2.load(l_name, vect_env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
									 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
									 verbose=1, tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = PPO2(policy, vect_env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=learning_rate, vf_coef=0.5,
									 max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None,
									 verbose=1, tensorboard_log=tensorboard_log, _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
				elif RL_algo == ACKTR:
					if load or training:
						model = ACKTR.load(l_name, vect_env, gamma=0.99, nprocs=None, n_steps=20, ent_coef=0.01, vf_coef=0.25,
										   vf_fisher_coef=1.0, learning_rate=learning_rate, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear',
										   verbose=1, tensorboard_log=tensorboard_log, _init_setup_model=True, async_eigen_decomp=False,
										   kfac_update=1, gae_lambda=None, policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log)
					else:
						model = ACKTR(policy, vect_env, gamma=0.99, nprocs=None, n_steps=20, ent_coef=0.01, vf_coef=0.25,
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
								expert_action = env_copy.expert_step(clip_softmax=clip_expert_softmax)
								return expert_action
							return expert

						expert_path = os.path.relpath('{}/experts/{}_{}.npz'.format(dirname(__file__),environment_type.__name__,("{}"*len(tickers)).format(*[t[0] for t in tickers])))
						if not os.path.isfile(expert_path):
							generate_expert_traj(create_expert(env), expert_path, copy.deepcopy(env), n_episodes=1)

						dataset = MultiDimensionalExpertDataset(expert_path=expert_path, traj_limitation=-1, batch_size=32)
						model.pretrain(dataset,n_epochs=n_epochs)
					model.learn(total_timesteps=total_timesteps, log_interval=log_interval,seed=1)
					model.save(s_name)
				else:
					break

				training = True
			models.append(model)

		print("Done training!")

		log = test_model(models,env,permutations[-1])
		for ticker in range(len(tickers)):
			log[tickers[ticker]] = [row[ticker+1] for row in log['Weights']]
		log['cash'] = [row[0] for row in log['Weights']]
		pd.DataFrame(log).to_csv("./logs/{}_{}_group_{}.csv".format(save_name,log_suffix.format(test_round),str(group_round)))