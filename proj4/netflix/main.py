import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
n = len(X)
# TODO: Your code here

min_cost = []
for K in range(1,5):
  costs = []
  for seed in range(5):
    mixture, post = common.init(X, K, seed)
    mixture, _, cost = kmeans.run(X, mixture, post)
    costs.append((K, seed, cost))
    print(f"K={K}, seed={seed}, cost={cost:.2f}")
  min_cost.append(min(costs, key=lambda x: x[2]))

print("\nMinimun costs")
for k, seed, cost in min_cost:
  print(f"Cluster: {k}, seed: {seed}, cost: {cost:.2f}")

common.plot(X, mixture, post, "k-means")
