import numpy as np

# Functions definitions------------------------------------
L1_norm = lambda a,b: np.sum(np.abs(a - b), axis=a.ndim-1)
L2_norm = lambda a,b: np.sum((a - b)**2, axis=a.ndim-1)

# K-medoids -----------------------------------------------
def k_medoids(x, centers, norm_func):
  z = centers.copy()
  k = len(z)  # Number of clusters
  old_z = -z  # Initialize old_z != z
  # Sanity check ------------------------------------------
  assert x.shape[1] == z.shape[1]
  
  while not np.array_equal(z, old_z):
    old_z = z[:]
    ''' Assign each xi to the closets zj ----------------------
    min(|xi - zj|^2)                                      '''
    dist = norm_func(x[:,np.newaxis,:], z[np.newaxis,:,:])
    C = np.where(dist[:, 0] < dist[:, 1], 0, 1)
    clusters = [x[C == j] for j in range(k)]

    ''' Find the best zj --------------------------------------
    Check which xi minimizes sum of distances
    '''
    for j in range(k):
      cluster = clusters[j]
      # If cluster has a single element do not actualize
      if len(cluster) == 1:
        continue
      
      dist_min = np.inf
      for temp_z in cluster:
        dist = 0
        for xi in cluster:
          dist += norm_func(xi, temp_z)
        if dist < dist_min:
          dist_min = dist
          z[j] = temp_z

  return (clusters, z)


def k_medoids_optimized(x, centers, norm_func):
  ''' This version uses:
  - Indices instead of copying the data into the clusters
  - Matrices operations instead of iterations             '''
  z = centers.copy()
  k = len(z)  # Number of clusters
  old_z = -z  # Initialize old_z != z
  # Sanity check ------------------------------------------
  assert x.shape[1] == z.shape[1]

  while not np.array_equal(z, old_z):
    old_z = z
    ''' Assign each xi to the closets zj ----------------------
    min(|xi - zj|^2)                                      '''
    dist = norm_func(x[:,np.newaxis,:], z[np.newaxis,:,:])
    C = np.argmin(dist, axis=1)
    cluster_indices = [np.where(C == j)[0] for j in range(k)]

    ''' Find the best zj --------------------------------------
    Check which xi minimizes sum of distances
    '''
    for j in range(k):
      cluster_idx = cluster_indices[j]
      # If cluster has a single element do not actualize
      if len(cluster_idx) == 1:
        continue

      dist = norm_func(x[cluster_idx,np.newaxis,:], x[np.newaxis,cluster_idx,:])
      total_distances = np.sum(dist, axis=1)
      best_idx_in_cluster = np.argmin(total_distances)
      z[j] = x[cluster_indices[j][best_idx_in_cluster]]

    return (cluster_indices, z)


# K-means -----------------------------------------------
def k_means(x, centers, norm_func):
  z = centers.copy()
  k = len(z)  # Number of clusters
  old_z = -z  # Initialize old_z != z
  # Sanity check ------------------------------------------
  assert x.shape[1] == z.shape[1]

  while not np.array_equal(z, old_z):
    old_z = z
    ''' Assign each xi to the closets zj --------------------
    min(|xi - zj|^2)                                      '''
    dist = norm_func(x[:,np.newaxis,:], z[np.newaxis,:,:])
    C = np.argmin(dist, axis=1)
    cluster_indices = [np.where(C == j)[0] for j in range(k)]

    ''' new zj = cluster centroid---------------'''
    for j in range(k):
      cluster_idx = cluster_indices[j]
      z[j] = np.median(x[cluster_idx,:], axis=0)

    return (cluster_indices, z)


# Testing--------------------------------------------------
# Input
data = np.array([[0,-6],[4,4],[0,0],[-5,2]])
centers_init = np.array([[-5.,2.],[0.,-6.]])

# K-medoids L1-norm
clusters_exp = [np.array([[4,4],[-5,2]]),np.array([[0,-6],[0,0]])]
centers_exp = np.array([[4,4],[0,-6]])
clusters_res, centers_res = k_medoids(data, centers_init, L1_norm)
assert np.array_equal(centers_exp, centers_res)
assert all(np.array_equiv(r, e) for r, e in zip(clusters_exp, clusters_res))
print("K-medoids with L1-norm correct")

# K-medoids optimized L1-norm
cluster_indices_exp = [np.array([1,3]), np.array([0,2])]
cluster_indices_res, centers_res = k_medoids_optimized(data, centers_init, L1_norm)
assert np.array_equal(centers_exp, centers_res)
assert all(np.array_equiv(r, e) for r, e in zip(cluster_indices_exp, cluster_indices_res))
print("K-medoids optimized with L1-norm correct")

# K-medoids L2-norm
cluster_indices_exp = [np.array([1,2,3]), np.array([0])]
centers_exp = np.array([[0,0],[0,-6]])
cluster_indices_res, centers_res = k_medoids_optimized(data, centers_init, L2_norm)
assert np.array_equal(centers_exp, centers_res)
assert all(np.array_equiv(r, e) for r, e in zip(cluster_indices_exp, cluster_indices_res))
print("K-medoids with L2-norm correct")

# K-means L1-norm
cluster_indices_exp = [np.array([1,3]), np.array([0,2])]
centers_exp = np.array([[-0.5,3],[0,-3]])
cluster_indices_res, centers_res = k_means(data, centers_init, L1_norm)
assert np.array_equal(centers_exp, centers_res)
assert all(np.array_equiv(r, e) for r, e in zip(cluster_indices_exp, cluster_indices_res))
print("K-means with L1-norm correct")
