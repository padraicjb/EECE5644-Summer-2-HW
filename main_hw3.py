import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import operator
import sklearn as skl
#from sklearn import *
import scipy.optimize as sp_optimize
from sklearn.neural_network import MLPClassifier
# GIVENS:
# Classes:
q1_c1 = 0
q1_c2 = 1
q1_c3 = 3
q1_c4 = 4
q1_classes = np.array([q1_c1, q1_c2, q1_c4, q1_c4])

# Priors:
q1_p0 = 0.25
q1_p1 = 0.25
q1_p2 = 0.25
q1_p3 = 0.25
q1_priors = np.array([q1_p0, q1_p1, q1_p2, q1_p3])

# Weights (unneeded):
# q1_a1 = 0.25
# q1_a2 = 0.25
# q1_a3 = 0.25
# q1_a4 = 0.25
# q1_as = np.array([q1_a1, q1_a2, q1_a3, q1_a4])

# Mus:
q1_mu1 = np.array([1, 2, 3])
q1_mu2 = np.array([2, 3, 4])
q1_mu3 = np.array([3, 4, 1])
q1_mu4 = np.array([4, 2, 3])
q1_MUs = np.array([q1_mu1, q1_mu2, q1_mu3, q1_mu4])

# Sigmas
q1_sigma1 = np.array([[1.1,  0.35,   0.1],
                      [0.35, 1.2,    0.85],
                      [0.1,  0.85,   0.9]])
q1_sigma2 = np.array([[0.8,  0.4,    0.4],
                      [0.4,  0.8,    0.9],
                      [0.4,  0.9,    1.3]])
q1_sigma3 = np.array([[1.2,  0.45,   0.2],
                      [0.45, 1.2,    0.75],
                      [0.2,  0.75,   0.8]])
q1_sigma4 = np.array([[1.2,  0.45,   0.2],
                      [0.45, 1.2,    0.75],
                      [0.2,  0.75,   0.8]])
q1_Sigmas = [q1_sigma1, q1_sigma2, q1_sigma3, q1_sigma4]

# Storage - 100, 200, 500, 1000, 2000, 5000 samples and a test set with 100000 samples
d_train_100_labels = np.empty(shape=100, dtype=int)
d_train_100 = np.empty(shape=[100, 3])

d_train_200_labels = np.empty(shape=200, dtype=int)
d_train_200 = np.empty(shape=[200, 3])

d_train_500_labels = np.empty(shape=500, dtype=int)
d_train_500 = np.empty(shape=[500, 3])

d_train_1000_labels = np.empty(shape=1000, dtype=int)
d_train_1000 = np.empty(shape=[1000, 3])

d_train_2000_labels = np.empty(shape=2000, dtype=int)
d_train_2000 = np.empty(shape=[2000, 3])

d_train_5000_labels = np.empty(shape=5000, dtype=int)
d_train_5000 = np.empty(shape=[5000, 3])

d_valid_100000_labels = np.empty(shape=100000, dtype=int)
d_valid_100000 = np.empty(shape=[100000, 3])

# d_train_5000000_labels = np.empty(shape=5000000, dtype=int)
# d_train_5000000 = np.empty(shape=[5000000, 3])

def generate_sl(samps, labs):
    #not my best function design using global variables 
    for i in range(labs.size):
        temp_lab = np.random.rand()
        print
        if temp_lab <= q1_priors[0]:
            class_label = 0
            temp_samp = np.random.multivariate_normal(q1_MUs[0], q1_Sigmas[0])
        elif temp_lab > q1_priors[0] and temp_lab <= q1_priors[1] + q1_priors[0]:
            class_label = 1
            temp_samp = np.random.multivariate_normal(q1_MUs[1], q1_Sigmas[1])
        elif temp_lab > q1_priors[1] + q1_priors[0] and temp_lab <= q1_priors[2] + q1_priors[1] + q1_priors[0]:
            class_label = 2
            temp_samp = np.random.multivariate_normal(q1_MUs[2], q1_Sigmas[2])
        else:
            class_label = 3
            temp_samp = np.random.multivariate_normal(q1_MUs[3], q1_Sigmas[3])

        # Add the new label to the list of labels
        labs[i] = class_label
        samps[i][0] = temp_samp[0]
        samps[i][1] = temp_samp[1]
        samps[i][2] = temp_samp[2]

generate_sl(d_train_100, d_train_100_labels)
generate_sl(d_train_200, d_train_200_labels)
generate_sl(d_train_500, d_train_500_labels)
generate_sl(d_train_1000, d_train_1000_labels)
generate_sl(d_train_2000, d_train_2000_labels)
generate_sl(d_train_5000, d_train_5000_labels)
generate_sl(d_valid_100000, d_valid_100000_labels)
# generate_sl(d_train_5000000, d_train_5000000_labels)


def display_samples_v2(samps, labs):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection ="3d")    
    # ax1 = fig.add_subplot(projection='3d')
    # ax2 = fig.add_subplot(projection='3d')
    # ax3 = fig.add_subplot(projection='3d')

    # n rows, 2 columns
    # observatiuos on rows, features on columns

    labs_0s = np.argwhere(labs == 0)
    labs_1s = np.argwhere(labs == 1)
    labs_2s = np.argwhere(labs == 2)
    labs_3s = np.argwhere(labs == 3)
    ax.scatter3D(samps[labs_0s, 0], samps[labs_0s, 1], samps[labs_0s, 2], color="r", marker = '^', label="True Class 1")
    ax.scatter3D(samps[labs_1s, 0], samps[labs_1s, 1], samps[labs_1s, 2], color="b", marker = 'o', label="True Class 2")
    ax.scatter3D(samps[labs_2s, 0], samps[labs_2s, 1], samps[labs_2s, 2], color="g", marker = 'x', label="True Class 3")
    ax.scatter3D(samps[labs_3s, 0], samps[labs_3s, 1], samps[labs_3s, 2], color="orange", marker = '*', label="True Class 4")

    ax.legend()
    ax.set_title("Data set with {} samples".format(str(labs.size)))
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")

    plt.show()
# ----------------------------------------------------------------------------------------------------------
# UNCOMMENT TO SEE GRAPHS - The last graph may cosnume all your ram (it worked slowly on mine with 32GB of DDR4)
# display_samples_v2(d_train_100, d_train_100_labels)
# display_samples_v2(d_train_200, d_train_200_labels)
# display_samples_v2(d_train_500, d_train_500_labels)
# display_samples_v2(d_train_1000, d_train_1000_labels)
# display_samples_v2(d_train_2000, d_train_2000_labels)
# display_samples_v2(d_train_5000, d_train_5000_labels)
# display_samples_v2(d_valid_100000, d_valid_100000_labels)
# ---------------------------------------------------------------------------------------------------------------
#print(q1_priors.shape[0])
#print(q1_priors.size)

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

sample_lls = np.zeros(shape=[q1_priors.size, 100000])
#q1_priors.size is basically just the number of priors
for clss in range(q1_priors.size):
    # for each class 0, 1, 2, 3, build a 10k element array of liklihoods 
	sample_lls[clss] = np.multiply(
		q1_priors[clss], multivariate_gaussian_pdf(d_valid_100000, q1_MUs[clss], q1_Sigmas[clss]))

sample_decisions = np.zeros(100000)

for smpl in range(100000):
    # test = np.zeros(4)
    # test[0] = sample_lls[0][smpl]
    # test[1] = sample_lls[1][smpl]
    # test[2] = sample_lls[2][smpl]
    # test[3] = sample_lls[3][smpl]
    # max_test = max(test)
    # predicts the label of each sample by returning the Label for the gaussian component with the highest likelihood
    sample_decisions[smpl] = max(enumerate(sample_lls[:, smpl]), key = operator.itemgetter(1))[0]

tot_num_errors = 0.0
c1_errors = 0
c2_errors = 0
c3_errors = 0
c4_errors = 0
number_of_each_label = np.array((sum(d_valid_100000_labels == 0), sum(d_valid_100000_labels == 1), sum(d_valid_100000_labels == 2), sum(d_valid_100000_labels == 3),))

for i in range(100000):
    error_huh = sample_decisions[i] != d_valid_100000_labels[i]
    true_class = d_valid_100000_labels[i]
    if sample_decisions[i] != d_valid_100000_labels[i]:
        tot_num_errors += 1
    if error_huh and true_class == 0:
        c1_errors += 1
    if error_huh and true_class == 1:
        c2_errors += 1
    if error_huh and true_class == 2:
        c3_errors += 1
    if error_huh and true_class == 3:
        c4_errors += 1

empirical_minimum_error = tot_num_errors / 100000
empirical_minimum_c1_error = c1_errors/number_of_each_label[0]
empirical_minimum_c2_error = c2_errors/number_of_each_label[1]
empirical_minimum_c3_error = c3_errors/number_of_each_label[2]
empirical_minimum_c4_error = c4_errors/number_of_each_label[3]


def print_emp_error():
    print("Empirical Minimum Error Probability: " + str(empirical_minimum_error))
    print("Empirical Class 1 Minimum Error Probability: " + str(round(empirical_minimum_c1_error, 5)))
    print("Empirical Class 2 Minimum Error Probability: " + str(round(empirical_minimum_c2_error, 5)))
    print("Empirical Class 3 Minimum Error Probability: " + str(round(empirical_minimum_c3_error, 5)))
    print("Empirical Class 4 Minimum Error Probability: " + str(round(empirical_minimum_c4_error, 5)))

def kfoldcrossvalid(training_data, training_labels, valid_data, valid_labels, k, perf_func, stop, order_start = 1, order_step = 1):
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
        temp_train_labels = np.r_[training_labels[:partition_indices[k_under_test]], training_labels[partition_indices[k_under_test + 1]:]]
        temp_validate_labels = training_labels[partition_indices[k_under_test]:partition_indices[k_under_test + 1]] 
        performance_decrement = 0
        last_perf = -2147483648 # smallest int in Java, should work here
        best_perf = -2147483648 # smallest int in Java, should work here
        best_perf_order = 0
        model_order = order_start

        while performance_decrement < stop:
            curr_perf, nn_not_used = perf_func(temp_train, temp_validate, model_order, temp_train_labels, temp_validate_labels)
            
            if curr_perf <= last_perf:
                performance_decrement = performance_decrement + 1
            
            if curr_perf > best_perf:
                best_perf = curr_perf
                best_perf_order = model_order
            
            last_perf = curr_perf
            print(str(model_order) + " model order for k " + str(k_under_test + 1) + "/" + str(k_under_test) + ", sample size = " + str(training_data.shape[0]) + ", performance = " + str(curr_perf))
            model_order += order_step
        best_perf_orders[k_under_test] = best_perf_order
        print("K " + str(k_under_test + 1) + "/" + str(k) + " complete, sample size = " + str(training_data.shape[0]) + ", chosen order = " + str(best_perf_order))
    final_order = statistics.mean(best_perf_orders)
    print("Sample size " + str(training_data.shape[0]) + " complete, chosen order = " + str(final_order))
    return final_order

#--------------------------------------------------------------------------------------------------------
def nn_train(d_train, d_valid, num_perceptrons, d_train_labels, d_valid_labels):
    #use Scikit learn because Lord knows I am not implementing that myself
    nn = MLPClassifier(hidden_layer_sizes=(num_perceptrons,), activation='relu', solver='adam',
	                                      alpha=1e-6, max_iter=6000, shuffle=True, tol=1e-4, verbose=False,
	                                      warm_start=False, early_stopping=False, n_iter_no_change=10)
    #d_train_recast = d_train.tolist()
    # print(d_train.shape)
    # print("Features Data Type: " + str(type(d_train[0][0])))
    # print(d_train_labels.shape)
    # print("Lables Data Type: " + str(type(d_train_labels[0])))
    nn.fit(d_train, d_train_labels)
    d_valid_predict = nn.predict(d_valid)
    # yndarray, shape (n_samples,) or (n_samples, n_classes)
    # The predicted classes.
    
    #d_valid_predict = nn.predict_log_proba(d_valid)               
    # log_y_probndarray of shape (n_samples, n_classes)
    # The predicted log-probability of the sample for each class in the model,
    # where classes are ordered as they are in self.classes_.
    # Equivalent to log(predict_proba(X)).

    #d_valid_predict = nn.predict_proba(d_valid)  
    # y_probndarray of shape (n_samples, n_classes)
    # The predicted probability of the sample for each class in the model,
    # where classes are ordered as they are in self.classes_.

    #log_llhood = np.sum(d_valid_predict)

    # def nll_loss(parameters, X, y, sigma=1):
    # mu_pred = X.dot(parameters)
    
    # # Compute log-likelihood function, setting mu=0 as we're estimating it
    # log_lld = np.sum(norm.logpdf(y - mu_pred, 0, sigma))
    # # Return NLL
    # return -log_lld


    num_correct = 0
    for sample in range(d_valid.shape[0]):
        if d_valid_predict[sample] == d_valid_labels[sample]:
            num_correct += 1


   
    log_llhood = np.log(num_correct / d_valid.shape[0])
    
    #                      misclassification rate <- 
    #misclassification_rate =  ((d_valid.shape[0]- num_correct) / d_valid.shape[0])
    #                      classification reate
    #classification_rate =  (num_correct / d_valid.shape[0])
    
    # "How well parameters being fgot to the model describe the data"
    # models the distribution over the data, unlike discrete -ness of 
    # can look at all samples at once, whole data set

    return log_llhood, nn

def training_report(d_train_tr, d_train_labels_tr, k, num_runs, d_valid_tr, d_valid_labels_tr):
    # Use 10-fold cross validation for each training dataset to determine best number of perceptrons
                          
    perceptrons = kfoldcrossvalid(d_train_tr, d_train_labels_tr, d_valid_tr, d_valid_labels_tr, k, nn_train, 3)
    perceptrons = int(np.round(perceptrons, decimals=0))

    # Train the data using the number of perceptrons found in cross validation
	# Train multiple times and take best performance
    best_train_nn = None
    best_train_perf = -2147483648 # smallest int in Java
    for unused in range(num_runs):
        # works Better than a while loop with its own iterator I think
        perf, nn = nn_train(d_train_tr, d_valid_tr, perceptrons, d_train_labels_tr, d_valid_labels_tr)
        if perf > best_train_perf:
            #update the best after a run
            best_train_perf = perf
            best_train_nn = nn

	# Apply MLP to test data and get error probability
    d_validate_predictions_tr = best_train_nn.predict(d_valid_tr)
    num_correct_tr = 0
    for sample in range(d_valid_tr.shape[0]):
        if d_validate_predictions_tr[sample] == d_valid_labels_tr[sample]:
            num_correct_tr += 1
    
    best_log_llhood = np.log(num_correct_tr / d_valid_tr.shape[0])    

    error_prob = 1 - np.exp(best_log_llhood)
    print(str(d_train_tr.shape[0]) + " samples error probability:" + str(error_prob))
    return error_prob, perceptrons


kfolds = 10
training_runs = 5

# generate_sl(d_train_100, d_train_100_labels)
# generate_sl(d_train_200, d_train_200_labels)
# generate_sl(d_train_500, d_train_500_labels)
# generate_sl(d_train_1000, d_train_1000_labels)
# generate_sl(d_train_2000, d_train_2000_labels)
# generate_sl(d_train_5000, d_train_5000_labels)
# generate_sl(d_valid_100000, d_valid_100000_labels)

print_emp_error()

d_train_100_error, d_train_100_perceptrons = training_report(d_train_100, d_train_100_labels, kfolds, training_runs,
                                                    d_valid_100000, d_valid_100000_labels)

d_train_200_error, d_train_200_perceptrons = training_report(d_train_200, d_train_200_labels, kfolds, training_runs,
                                                     d_valid_100000, d_valid_100000_labels)

d_train_500_error, d_train_500_perceptrons = training_report(d_train_500, d_train_500_labels, kfolds, training_runs,
                                                    d_valid_100000, d_valid_100000_labels)

d_train_1000_error, d_train_1000_perceptrons = training_report(d_train_1000, d_train_1000_labels, kfolds, training_runs,
                                                     d_valid_100000, d_valid_100000_labels)

d_train_2000_error, d_train_2000_perceptrons = training_report(d_train_2000, d_train_2000_labels, kfolds, training_runs,
                                                    d_valid_100000, d_valid_100000_labels)

d_train_5000_error, d_train_5000_perceptrons = training_report(d_train_5000, d_train_5000_labels, kfolds, training_runs,
                                                     d_valid_100000, d_valid_100000_labels)

d_train_5000000_error, d_train_5000000_perceptrons = training_report(d_train_5000000, d_train_5000000_labels, kfolds, training_runs,
                                                     d_valid_100000, d_valid_100000_labels)

all_error_probs = [d_train_100_error, d_train_200_error, d_train_500_error, d_train_1000_error, d_train_2000_error, d_train_5000_error]
num_samples_per_test = [100, 200, 500, 1000, 2000, 5000]

# Added this to debug graphs without needing to run ALL the rest of the code 
with open('error_prob.csv', 'w+') as my_file:
    np.savetxt(my_file, all_error_probs)
plt.plot(num_samples_per_test, all_error_probs, 'b')
# "baseline"
plt.plot([num_samples_per_test[0], num_samples_per_test[-1]], [empirical_minimum_error] * 2, 'r')
plt.xscale('log')
plt.title('Minimum Classification Error with Varying Training Sizes')
plt.xlabel('Training Data Samples')
plt.ylabel('Minimum Error')
plt.legend(['Training Min Error', 'Estimated Theoretical Min Error'])
plt.show()

all_perceptrons = [d_train_100_perceptrons, d_train_200_perceptrons, d_train_500_perceptrons, d_train_1000_perceptrons, d_train_2000_perceptrons, d_train_5000_perceptrons]

# Added this to debug graphs without needing to run ALL the rest of the code 
with open('perceptrons.csv', 'w+') as my_file:
     np.savetxt(my_file, all_perceptrons)
# PLOT the Optimal Perceptrons Vesus Number of Training Samples
plt.plot(num_samples_per_test, all_perceptrons, 'b')
plt.xscale('log')
plt.title('Optimal Perceptrons Vesus Number of Training Samples')
plt.xlabel('Training Data Samples')
plt.ylabel('Chosen Perceptrons')
plt.show()