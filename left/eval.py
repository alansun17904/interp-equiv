import pickle
import pprint
from collections import Counter


settings = ["constant-ll-models", "corr-ll-models", "arch-ll-models", "corr-arch-ll-models"]

for s in settings:
	print(f"Stats for `{s}`")

	acc = pickle.load(open("src/data/reprs/" + s + "/eval-accuracy.pkl", "rb"))
	acc = list(acc.items())

	top = list(filter(lambda x: x[1][0] >= 0.8, acc))

	# get only the case_ids
	case_ids = [v[0][0] for v in top]
	case_ids = Counter(case_ids)

	pprint.pp(case_ids)