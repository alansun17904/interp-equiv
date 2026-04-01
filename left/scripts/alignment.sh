#!/bin/bash

python3 src/alignment.py src/data/reprs/constant-ll-models src/data/alignment/constant-alignment.pkl
python3 src/alignment.py src/data/reprs/corr-ll-models src/data/alignment/corr-alignment.pkl
python3 src/alignment.py src/data/reprs/arch-ll-models src/data/alignment/arch-alignment.pkl
python3 src/alignment.py src/data/reprs/corr-arch-ll-models src/data/alignment/corr-arch-alignment.pkl