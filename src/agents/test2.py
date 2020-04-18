from src.environments.StackedEnv import StackedEnv
import numpy as np

train_test_split = 0.7
transaction_cost = 0.00000000001
overall_profit = True
include_cash = True
reward_type="p&l"
a = np.array([[-1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.],
			  [-1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.],
			  [-1.,  1., -1., -1.,  1., -1.,  1.,  1.,  1.,  1., -1., -1.]])
u, indices = np.unique(a, return_inverse=True)
d = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(a.shape),
                                None, np.max(indices) + 1), axis=0)]
print(d)

groups = [['PEP','KO','KDP','MNST'],['BBY','TGT','WMT','COST'],['FB','GOOGL','AAPL','AMZN']]
tickers = list(np.array(groups).flatten())

env = StackedEnv(groups,transaction_cost,train_test_split=train_test_split,realized=False,reward=reward_type,include_cash=include_cash,overall_profit=overall_profit)
print(env.reset())