import os

for i in range(5):
    for j in range(5):
        print(f"Running with i={i}, j={j}")
        os.system(f"python3 train/neurons_learning.py {i} {j}")