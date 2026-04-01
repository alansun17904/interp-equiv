#!/bin/bash

# python3 src/encoding.py 0
# python3 src/encoding.py 1
# python3 src/encoding.py 2
# python3 src/encoding.py 3
# python3 src/encoding.py 4
# python3 src/encoding.py 5


# for i in {0..5}; do
#   python3 src/encoding_interp.py 0 $i
# done

# for i in {0..5}; do
#   python3 src/encoding_interp.py 1 $i
# done

# for i in {0..10}; do
#   python3 src/encoding_interp.py 2 $i
# done

# for i in {0..5}; do
#   python3 src/encoding_interp.py 3 $i
# done

# for i in {0..8}; do
#   python3 src/encoding_interp.py 4 $i
# done

for i in {0..5}; do
  python3 src/encoding_interp.py 5 $i
done