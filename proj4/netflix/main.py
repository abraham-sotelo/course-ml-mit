import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
n = len(X)

# Problem 2. K-means
def k_means():
  min_cost = []
  for K in range(1,5):
    costs = []
    for seed in range(5):
      mixture, post = common.init(X, K, seed)
      mixture, _, cost = kmeans.run(X, mixture, post)
      costs.append((K, seed, cost))
      print(f"K={K}, seed={seed}, cost={cost:.2f}")
    min_cost.append(min(costs, key=lambda x: x[2]))
    common.plot(X, mixture, post, f"k-means K={K}")

  print("\nMinimun costs")
  for k, seed, cost in min_cost:
    print(f"Cluster: {k}, seed: {seed}, cost: {cost:.2f}")


# Problem 3. Expectation–maximization algorithm
def naiveem():
  max_log_likelihood = []
  for K in range(1,5):
    log_likelihoods = []
    for seed in range(5):
      mixture, post = common.init(X, K, seed)
      mixture, post, log_likelihood = naive_em.run(X, mixture, post)
      log_likelihoods.append((K, seed, log_likelihood))
      print(f"K={K}, seed={seed}, log-likelihood={log_likelihood:.4f}")
    max_log_likelihood.append(max(log_likelihoods, key=lambda x: x[2]))
    common.plot(X, mixture, post, f"EM algorithm K={K}")

  print("\nMaximun log-likelihood")
  for k, seed, log_likelihood in max_log_likelihood:
    print(f"Cluster: {k}, seed: {seed}, log-likelihood: {log_likelihood:.5f}")


# Problem 5. Bayesian Information Criterion
def bic_em():
  bics = []
  for K in range(1,5):
    mixture, post = common.init(X, K)
    mixture, post, log_likelihood = naive_em.run(X, mixture, post)
    bic = common.bic(X, mixture, log_likelihood)
    print(f"K={K}, bic={bic:.6f}")
    bics.append((K, bic))
  k, best_bic = max(bics, key=lambda a:a[1])

  print("\nBest BIC")
  print(f"Cluster: {k}, Best BIC: {best_bic:.6f}")


# Problem 7. Implementing EM for matrix completion
def em():
  pass

#k_means()
#naiveem()
#bic_em()
#em()