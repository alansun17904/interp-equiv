"""Generates dataset for representation alignment, both "clean" and "corrupted"
such that all of the data points within the dataset share the same vocabulary.
"""

import pickle


def get_clean_data(case, samples: int):
	return case.get_clean_data(min_samples=samples, max_samples=samples)