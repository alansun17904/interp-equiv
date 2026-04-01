for i in {0..6}; do
    python3 src/task_encoding.py intervention $i True
done

for i in {0..6}; do
    python3 src/task_encoding.py intervention $i False
done

# for i in {0..6}; do
#     python3 src/task_encoding.py preinter $i True
