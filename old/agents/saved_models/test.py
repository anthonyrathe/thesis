import itertools, random

tickers = [['PEP','KO','KDP','MNST'],['BBY','TGT','WMT','COST'],['FB','GOOGL','AAPL','AMZN']]
peer_group_aware = True

peer_combinations = []
for peer_group in tickers:
	peer_combinations += list(itertools.permutations(peer_group,2))

if peer_group_aware:
	combinations = peer_combinations
else:
	combinations = random.sample(list(itertools.permutations([item for peer_group in tickers for item in peer_group],2)),len(peer_combinations))

print(combinations)