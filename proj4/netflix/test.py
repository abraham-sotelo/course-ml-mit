import numpy as np
import em
import common
import vectorized_em

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

# Initialization ----------------------------------------------------
K = 4
n,d = X.shape
mixture, post = common.init(X, K)

mu_expected = np.array(
  [[2., 4., 5., 5., 0.],
   [3., 5., 0., 4., 3.],
   [2., 5., 4., 4., 2.],
   [0., 5., 3., 3., 3.]]
)
var_expected = np.array([5.93, 4.87, 3.99, 4.51])
p_expected = np.array([0.25, 0.25, 0.25, 0.25])

np.testing.assert_allclose(mixture.mu, mu_expected)
np.testing.assert_allclose(mixture.var, var_expected)
np.testing.assert_allclose(mixture.p, p_expected)
print("Init tests OK!")

# First run e-step --------------------------------------------------
post_expected = np.array(
  [[0.17713577, 0.12995693, 0.43161668, 0.26129062],
 [0.08790299, 0.35848927, 0.41566414, 0.13794359],
 [0.15529703, 0.10542632, 0.5030648,  0.23621184],
 [0.23290326, 0.10485918, 0.58720619, 0.07503136],
 [0.09060401, 0.41569201, 0.32452345, 0.16918054],
 [0.07639077, 0.08473656, 0.41423836, 0.42463432],
 [0.21838413, 0.20787523, 0.41319756, 0.16054307],
 [0.16534478, 0.04759109, 0.63399833, 0.1530658 ],
 [0.05486073, 0.13290982, 0.37956674, 0.43266271],
 [0.08779356, 0.28748372, 0.37049225, 0.25423047],
 [0.07715067, 0.18612696, 0.50647898, 0.23024339],
 [0.16678427, 0.07789806, 0.45643509, 0.29888258],
 [0.08544132, 0.24851049, 0.53837544, 0.12767275],
 [0.17773171, 0.19578852, 0.41091504, 0.21556473],
 [0.02553529, 0.1258932,  0.29235844, 0.55621307],
 [0.07604748, 0.19032469, 0.54189543, 0.1917324 ],
 [0.15623582, 0.31418901, 0.41418177, 0.1153934 ],
 [0.19275595, 0.13517877, 0.56734832, 0.10471696],
 [0.33228594, 0.02780214, 0.50397264, 0.13593928],
 [0.12546781, 0.05835499, 0.60962919, 0.20654801]]
)
log_likelihood_expected = -152.16319226209848
print("First e-step run -------------------------")
post, log_likelihood = vectorized_em.estep_compact(X, mixture)
assert post.shape == (n, K)
np.testing.assert_allclose(post, post_expected, atol=1e-8)
print(f"Output log_likelihood = {log_likelihood:.4f}")
np.testing.assert_almost_equal(log_likelihood, log_likelihood_expected)
print("First e-step OK --------------------------")
# First run m-step --------------------------------------------------
mu_expected = np.array([
    [2.38279095, 4.64102716, 3.73583539, 4.28989488, 2.17237898],
    [2.56629755, 4.6686168,  3.24084599, 3.88882023, 2.72874336],
    [2.45674721, 4.72686227, 3.55798344, 4.05614484, 2.5030405 ],
    [2.00305536, 4.7674522,  3.37388115, 3.7905181,  2.97986269]
])
var_expected = np.array([0.71489705, 0.64830186, 0.73650336, 0.85722393])
p_expected = np.array([0.13810266, 0.17175435, 0.46575794, 0.22438505])
print("First m-step run -------------------------")
mixture = vectorized_em.mstep(X, post, mixture)
np.testing.assert_allclose(mixture.p, p_expected, atol=1e-8)
assert mixture.mu.shape == mu_expected.shape
np.testing.assert_allclose(mixture.mu, mu_expected, atol=1e-8)
np.testing.assert_allclose(mixture.var, var_expected, atol=1e-8)
print("First m-step OK --------------------------")

# Run em-algorithm ------------------------------------------------
post_expected = np.array([
    [8.35114583e-01, 1.26066023e-01, 8.03346942e-03, 3.07859243e-02],
    [2.29595284e-04, 9.30406661e-01, 6.93634633e-02, 2.80840424e-07],
    [9.98723643e-01, 1.34234094e-04, 1.14212255e-03, 1.65905887e-14],
    [1.85331147e-04, 1.94115053e-03, 9.97873518e-01, 2.57285049e-14],
    [1.82091725e-08, 8.82200084e-01, 1.17730763e-01, 6.91351811e-05],
    [2.13395201e-14, 1.74763538e-08, 1.23289877e-04, 9.99876693e-01],
    [9.78452231e-01, 2.41596929e-05, 2.15236097e-02, 2.05795060e-14],
    [1.95291523e-06, 3.46537075e-03, 9.96532634e-01, 4.18625878e-08],
    [2.53995753e-04, 9.99058306e-01, 6.46220953e-04, 4.14767958e-05],
    [1.39755279e-03, 8.96199140e-01, 1.02340131e-01, 6.31761952e-05],
    [1.02964283e-05, 9.98438589e-01, 1.55110233e-03, 1.18280899e-08],
    [9.99175360e-01, 4.92298629e-07, 8.24147990e-04, 5.73816393e-13],
    [4.54696111e-06, 9.96705586e-01, 3.28986689e-03, 1.91139775e-10],
    [4.13182467e-02, 1.40457914e-05, 9.58667653e-01, 5.48560980e-08],
    [9.22358785e-14, 4.78927600e-06, 3.67220413e-07, 9.99994844e-01],
    [2.36604822e-04, 9.96136619e-01, 3.62659186e-03, 1.84275504e-07],
    [1.09042309e-01, 2.42442342e-01, 6.48515348e-01, 8.68166867e-11],
    [9.62134995e-01, 1.21159085e-04, 3.77438456e-02, 5.30337126e-16],
    [1.39885506e-04, 2.34579872e-06, 9.99672523e-01, 1.85246074e-04],
    [6.05773445e-01, 1.29236657e-02, 3.81302856e-01, 3.38895349e-08]
])
log_likelihood_expected = -84.98451993042474
mu_expected = np.array([
    [2.00570178, 4.99062403, 3.13772745, 4.00124767, 1.16193276],
    [2.99396416, 4.68350343, 3.00527213, 3.52422521, 3.08969957],
    [2.54539306, 4.20213487, 4.56501823, 4.55520636, 2.31130827],
    [1.01534912, 4.99975322, 3.49251807, 3.99998124, 4.99986013]
])
var_expected = np.array([0.25, 0.25, 0.44961685, 0.27930039])
p_expected = np.array([0.27660973, 0.35431424, 0.26752518, 0.10155086])
print("Run em-algorithm -------------------------")
mixture, post = common.init(X, K)
mixture, post, log_likelihood = vectorized_em.run(X, mixture, post)
np.testing.assert_allclose(post, post_expected, atol=1e-8)
print("post OK")
print(f"Output log_likelihood = {log_likelihood:.4f}")
assert log_likelihood == log_likelihood_expected
print("log_likelihood OK")
np.testing.assert_allclose(mixture.p, p_expected, atol=1e-8)
print("p OK")
assert mixture.mu.shape == mu_expected.shape
np.testing.assert_allclose(mixture.mu, mu_expected, atol=1e-8)
print("mu OK")
np.testing.assert_allclose(mixture.var, var_expected, atol=1e-8)
print("var OK")
print("Run em-algorithm OK-----------------------")

# Fill matrix -----------------------------------------------------
X_pred_expected = np.array([
    [2.,         5.,         3.,         3.94554203, 1.53247395],
    [3.,         5.,         3.11376,    4.,         3.],
    [2.,         4.98967752, 3.,         3.,         1.],
    [4.,         4.20321354, 4.,         5.,         2.],
    [3.,         4.,         3.18859109, 3.64540838, 4.],
    [1.,         4.99965498, 4.,         5.,         5.],
    [2.,         5.,         3.16858887, 4.01321529, 1.],
    [3.,         4.20380457, 5.,         4.,         3.],
    [2.99334056, 5.,         3.,         3.,         3.],
    [2.,         4.63458935, 3.16542905, 3.,         3.],
    [3.,         4.,         3.,         3.,         3.],
    [1.,         5.,         3.,         4.00170707, 1.],
    [4.,         5.,         3.,         4.,         3.],
    [1.,         4.,         4.50628741, 5.,         2.],
    [1.,         5.,         3.,         3.,         5.],
    [3.,         5.,         3.,         4.,         3.],
    [3.,         4.40437447, 4.03220151, 4.,         2.],
    [3.,         5.,         3.,         5.,         1.],
    [2.,         4.,         5.,         5.,         2.3116484 ],
    [2.,         5.,         4.,         4.,         2.]
])
rmse_expected = 0.3152301205749675
print("Run Fill matrix --------------------------")
X_pred = vectorized_em.fill_matrix(X, mixture)
np.testing.assert_allclose(X_pred, X_pred_expected, atol=1e-8)
print("Fill matrix OK")
np.testing.assert_almost_equal(vectorized_em.rmse(X_pred, X_gold), rmse_expected)
print("rmse OK")
print("All tests OK!")