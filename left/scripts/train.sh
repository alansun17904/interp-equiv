#!/bin/bash

#torchrun --nproc_per_node=1 ./src/main.py train iit --epochs 30 -s 0.4 -iit 1 -b 1 --output-dir src/data/constant-ll-models
#torchrun --nproc_per_node=1 ./src/main.py train iit --epochs 30 -s 0.4 -iit 1 -b 1 --rand-correspondence --output-dir src/data/corr-ll-models
#torchrun --nproc_per_node=1 ./src/main.py train iit --epochs 30 -s 0.4 -iit 1 -b 1 --rand-architecture --output-dir src/data/arch-ll-models
./src/main.py train iit --epochs 30 -s 0.4 -iit 1 -b 1 --rand-correspondence --rand-architecture --output-dir src/data/corr-arch-ll-models
