import pickle
from circuits_benchmark.utils.get_cases import get_cases


cases = get_cases()
cases = filter(lambda x: "CaseIOI" not in x.__class__.__name__, cases)
vocabs = [(case, case.get_vocab()) for case in cases]

admissible_tasks = dict()

# get the set of sets that all have the same vocab as
ref = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
tot = 0
for v in vocabs:
	if v[0].get_max_seq_len() != 10:
		continue

	if v[1] == ref:
		print(v[0].get_task_description())
		tot += 1

		print(v[0])

		admissible_tasks[int(v[0].get_name())] = v[0]

pickle.dump(admissible_tasks, open("admissible_tasks.pkl", "wb"))