import numpy as np
import matplotlib.pyplot as plt

# Givens
n = 10
n_train = 100
n_test = 1000
num_folds = 5
k = 5

# a - an arbitrary non-zero n-dimensional vector a
a = np.array([9, 8, 6, 5, 2, 4, 2, 1, 3, 7])
# An arbitrary Gaussian:
# with nonzero-mean m 
m = np.array([ 5.0,-3.0, 0.0, 1.0,-4.0, 2.0,-1.0, 5.0, -3.0, 0.0])
# with non-diagonal covariance matrix S for a n-dimensional random vector x
S = np.array([[1.5, 0.5, 0.2, 0.1, 0.3, 0.4, 0.3, 1.5, 1.5, 1.1],
              [0.5, 1.1, 0.6, 0.8, 0.2, 0.1, 0.0, 0.7, 0.5, 1.2],
              [0.2, 0.6, 1.0, 0.5, 0.4, 0.0, 0.9, 0.2, 0.3, 1.0],
              [0.1, 0.8, 0.5, 1.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.9],
              [0.3, 0.2, 0.4, 0.0, 0.7, 1.1, 0.5, 0.8, 0.4, 0.1],
              [0.4, 0.1, 0.0, 0.2, 1.1, 0.6, 1.1, 0.6, 0.2, 0.3],
              [0.3, 0.0, 0.9, 0.0, 0.5, 0.1, 1.5, 0.5, 0.4, 0.7],
              [1.5, 0.7, 0.2, 0.0, 0.8, 0.2, 0.5, 0.0, 1.2, 0.9],
              [1.5, 0.5, 0.3, 0.1, 0.3, 0.2, 0.4, 1.2, 1.2, 0.6],
              [1.1, 1.2, 1.0, 0.9, 0.1, 0.3, 0.7, 0.9, 0.6, 0.9]])

# Randomly Generated a potenial positive semidefinite matrix, code still failed :(
# S = np.array([[3.59, 3.12, 3.37, 2.92, 2.72, 2.57, 1.63, 3.31, 2.75, 3.06],
#             [3.12, 4.04, 3.81, 3.31, 3.1,  2.85, 2.71, 3.83, 3.25, 3.49],
#             [3.37, 3.81, 4.88, 3.46, 3.52, 2.91, 2.37, 3.83, 3.99, 3.64],
#             [2.92, 3.31, 3.46, 4.07, 3.51, 2.67, 2.69, 3.3,  3.08, 3.32],
#             [2.72, 3.1,  3.52, 3.51, 4.,   2.35, 2.04, 3.77, 2.73, 2.98],
#  [2.57, 2.85, 2.91, 2.67, 2.35, 2.64, 1.83, 3.,   2.67, 2.61],
#  [3.31, 3.83, 3.83, 3.3,  3.77, 3.,   2.22, 5.13, 3.13, 3.24],
#  [2.75, 3.25, 3.99, 3.08, 2.73, 2.67, 2.22, 3.13, 3.61, 3.01],
#  [3.06, 3.49, 3.64, 3.32, 2.98, 2.61, 2.59, 3.24, 3.01, 4.06]])


# Code to randomly Generate a potenial positive semidefinite matrix, code still failed :(
# A = np.random.rand(low = 0, high = 1.5, size=(10, 10))
#A = np.random.rand(10, 10)
#A = np.multiply(A, .25)
#S = np.dot(A, A.transpose())
# for i in range(10):
#     for j in range(10):
#         S[i][j] = round(S[i][j], 2)

def multivariate_gaussian_pdf(x, mu, sigma):
    """
    Returns likelihoods of all samples in array given mean and covariance
    :param x: Array of samples
    :param mean: Mean of multivariate distribution as 1-D matrix
    :param covariance: Covariance of multivariate distribution as 2-D matrix
    :param x_len: Length of x, helps speed up algorithm when this is called a lot
    :return: Array of likelihoods
    """

    ret_matrix = []
    dimensions = len(mu)
    normalization_constant = ((2 * math.pi) ** (-dimensions / 2)) * (
        np.linalg.det(sigma) ** -0.5
    )
    cov_inv = np.linalg.inv(sigma)
    for i in range(len(x)):
        mean_diff = np.subtract(x[i], mu)
        exponent = math.exp(
            np.matmul(np.matmul(-0.5 * np.transpose(mean_diff), cov_inv), mean_diff)
        )
        likelihood = normalization_constant * exponent
        ret_matrix.append(likelihood)
    return ret_matrix

def kfoldcrossvalidv2(training_data, k, perf_func, stop, training_labels = None, valid_labels = None, valid_data = None, order_start = 1, order_step = 1):
    # super useful row-wise merge! Thank you docummentation and stack exhange
    # https://stackoverflow.com/questions/21990345/merging-concatenating-arrays-with-different-elements/21990608#21990608
   	# Adding the [0] to the end of shape tell which element, the first in this case, that;s shape ashould be reported
    #   - In this case its the number of samples so one of 100, 200, 500, 1k, 2k, 5k
    #TODO: Doesn't work properly without endpoint param, need to investigate why
    partition_indices = np.r_[np.linspace(0, training_data.shape[0], num = k, endpoint = False, dtype=int), training_data.shape[0]]
    best_perf_orders = np.zeros(k)

    for k_under_test in range(k):
        temp_train = np.r_[training_data[:partition_indices[k_under_test]], training_data[partition_indices[k_under_test + 1]:]]
        temp_validate = training_data[partition_indices[k_under_test]:partition_indices[k_under_test + 1]]
        if training_labels is not None:
            temp_train_labels = np.r_[training_labels[:partition_indices[k_under_test]], training_labels[partition_indices[k_under_test + 1]:]]
        if valid_labels is not None:
            temp_validate_labels = training_labels[partition_indices[k_under_test]:partition_indices[k_under_test + 1]] 
        performance_decrement = 0
        last_perf = -2147483648 # smallest int in Java
        best_perf = -2147483648 # smallest int in Java
        best_perf_order = 0
        model_order = order_start

        while performance_decrement < stop:
            if training_labels is not None:
                curr_perf, nn_not_used = perf_func(temp_train, temp_validate, model_order, temp_train_labels, temp_validate_labels)
            else:
                #print("else")
                curr_perf, nn_not_used = perf_func(temp_train, temp_validate, model_order)
            
            if curr_perf <= last_perf:
                performance_decrement = performance_decrement + 1
            
            if curr_perf > best_perf:
                best_perf = curr_perf
                best_perf_order = model_order
            
            last_perf = curr_perf
            #print(str(model_order) + " model order for k " + str(k_under_test + 1) + "/" + str(k_under_test) + ", sample size = " + str(training_data.shape[0]) + ", performance = " + str(curr_perf))
            model_order += order_step
        best_perf_orders[k_under_test] = best_perf_order
        #print("K " + str(k_under_test + 1) + "/" + str(k) + " complete, sample size = " + str(training_data.shape[0]) + ", chosen order = " + str(best_perf_order))
    final_order = statistics.mean(best_perf_orders)
    #print("Sample size " + str(training_data.shape[0]) + " complete, chosen order = " + str(final_order))
    return final_order

#d_train_100_labels = np.empty(shape=100, dtype=int)
#d_train_100 = np.empty(shape=[100, 2])
#_train_100_labels = np.empty(shape=100, dtype=int)
#d_test_1000 = np.empty(shape=[1000, 2])
#α = 0
def generate_samples_v1(num_samples, num_dimensions, alphas):
    #num_samples = samples.shape
    return_array = []
    for sample in range(num_samples):
        # Draw Ntrain iid samples of n-dimensional samples of x from this Gaussian pdf
        x = np.random.multivariate_normal(m, S)
        # Draw Ntrain iid samples of an n-dimensional random variable z from a zero-mean, ε I-covariance-matrix Gaussian pdf (noise corrupting the input).
        z = np.random.multivariate_normal(np.zeros(num_dimensions), alphas * np.eye(num_dimensions))
        # Draw Ntrain iid samples of a scalar random variable v from a zero-mean, unit-variance Gaussian pdf.
        v = np.random.normal(0, 1)
        # Calculate Ntrain scalar values of a new random variable as follows y = a⊺(x+z)+v using the samples of x and v.
        y = np.add(np.matmul(np.transpose(a), np.add(x, z)), v)
        # 
        return_array.append([x, y])
    
    return_array = np.array(data, dtype='object')
    return return_array

def generate_samples_v2(num_samples, num_dimensions, α):
    #num_samples = samples.shape
    return_array = []
    for sample in range(num_samples):
        # Draw Ntrain iid samples of n-dimensional samples of x from this Gaussian pdf
        x = np.random.multivariate_normal(m, S)
        # Draw Ntrain iid samples of an n-dimensional random variable z from a zero-mean, ε I-covariance-matrix Gaussian pdf (noise corrupting the input).
        z = np.random.multivariate_normal(np.zeros(num_dimensions), α * np.eye(num_dimensions))
        # Draw Ntrain iid samples of a scalar random variable v from a zero-mean, unit-variance Gaussian pdf.
        v = np.random.normal(0, 1)
        # Calculate Ntrain scalar values of a new random variable as follows y = a⊺(x+z)+v using the samples of x and v.
        y = np.add(np.matmul(np.transpose(a), np.add(x, z)), v)
        # Add data to the array of samples
        return_array.append([x, y])
    # Recast the data into a Numpy Array
    return_array = np.array(return_array, dtype='object')
    # Return the newly generated array of samples
    return return_array
# _____-_________________-_________________-_________________-_________________-_________________-____________

def train_alpha(alpha_t):
	# Generate sets of training data
	d_train = generate_samples_v2(n_train, m.shape[0], alpha_t)
	d_test = generate_samples_v2(n_test, m.shape[0], alpha_t)
	
    # Choose a beta using k-fold cross validation
	beta_potential = kfoldcrossvalidv2(training_data = d_train, k = 10, perf_func = nn_train, stop = 100, order_start=0.00001, order_step=0.00001)

	llhood, best_weights = nn_train(d_train, d_train, beta_potential, (True, ))

	neg2llhood = -2 * weight_log_llhood(best_weights, d_test, beta_potential)
	# Find MSE of weights from the MAP-expected 0
	mean_square_error = np.mean([best_weights[dim] ** 2 for dim in range(best_weights.shape[0])])
	return beta_potential, neg2llhood, mean_square_error
# _____-_________________-_________________-_________________-_________________-_________________-____________


def nn_train(d_train, d_validate, beta, train_labels = None, validate_labels = None):
    # Weights based on derived MAP estimator
    # bigW = []
    bigW = (np.mean([np.array(sample[1] / (np.dot(np.r_[sample[0], 1], np.transpose(np.r_[sample[0], 1])) - 1.0 / beta) * np.r_[sample[0], 1]) for sample in d_train], axis=0))
    # print(str(type(bigW)))
    #bigW = np.array(bigW, dtype='object')
    # Return negative log likelihood of optimization weight results using validation data
    log_llhood = weight_log_llhood(bigW, d_validate, beta)
    return log_llhood, bigW

# _____-_________________-_________________-_________________-_________________-_________________-____________
def weight_log_llhood(weights, data, beta):
	# Prior log likelihood 
    #                      = ln[(2   π         ϐ)   ^    (-weight len     / 2)] -       (w*w_transpose)                  /   (2ϐ)
	prior_log_likelihood = np.log(2 * np.pi * beta) ** (-weights.shape[0] / 2) - np.dot(weights, np.transpose(weights)) / (2 * beta)
	# Sample log likelihood Formula
    #                          ln[1/sqrt(2π)]                 - mean(     1/2 *         (w *        [x_n, 1]_transpose - y_n)                   ^2)
	sample_log_likelihood = np.log(1.0 / np.sqrt(2 * np.pi)) - np.mean([0.5 * (np.dot(weights, np.transpose(np.r_[sample[0], 1])) - sample[1]) ** 2 for sample in data])
	# Weight log likelihood
    #       = ln[p(data|w)]     +      ln[p(w)]
	return prior_log_likelihood + sample_log_likelihood
# _____-_________________-_________________-_________________-_________________-_________________-____________


# np.trace - Returns the sum along diagonals of the array, in this case the variances
# np.logspace/np.linspace - Return evenly spaced numbers over a specified interval
# z_alphas = np.trace(S) / m.shape[0] * np.linspace(0, 10000, num = 50) # not super useful for displaying the data
# 10^(−3) × trace(Σ)/n to very large 10^(3) × trace(Σ)/n
alphas = np.logspace(-3, 3, num = 50) * np.trace(S) / m.shape[0]

training_reports = []  # Each tuple inside should be (beta_guess, neg2loglikelihood)
for alpha in alphas:
	training_reports.append(train_alpha(alpha))
	print("Alpha " + str(alph) + "evaluated")

# Recast the data into a Numpy Array
training_reports = np.array(training_reports)
