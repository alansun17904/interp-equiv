#!/bin/bash

# python src/reprs.py src/data/constant-ll-models src/data/reprs/constant-ll-models
# python src/reprs.py src/data/corr-ll-models src/data/reprs/corr-ll-models
# python src/reprs.py src/data/arch-ll-models src/data/reprs/arch-ll-models
# python src/reprs.py src/data/corr-arch-ll-models src/data/reprs/corr-arch-ll-models

for i in {0..5}; do
	python3 src/algo.py 0 $i
done

for i in {0..5}; do
	python3 src/algo.py 1 $i
done

for i in {0..10}; do
	python3 src/algo.py 2 $i
done

for i in {0..5}; do
	python3 src/algo.py 3 $i
done

for i in {0..8}; do
	python3 src/algo.py 4 $i
done

for i in {0..5}; do
	python3 src/algo.py 5 $i
done


# python3 src/algo.py 0 0
# python3 src/algo.py 1 0
# python3 src/algo.py 2 0
# python3 src/algo.py 3 0
# python3 src/algo.py 4 0
# python3 src/algo.py 5 0

#python3 src/algo.py 0 1
#python3 src/algo.py 0 2
#python3 src/algo.py 0 3
#python3 src/algo.py 0 4
#python3 src/algo.py 0 5
#python3 src/algo.py 0 6

#python3 src/algo.py 1

#python3 src/algo.py 2
#python3 src/algo.py 3
#python3 src/algo.py 4
#python3 src/algo.py 5
