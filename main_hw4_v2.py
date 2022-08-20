import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import operator
import sklearn as skl
#from sklearn import *
import scipy.optimize as sp_optimize
from sklearn.neural_network import MLPClassifier
from matplotlib.ticker import MaxNLocator
#
from scipy.stats import multivariate_normal as mvn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold 
from sklearn.svm import SVC, LinearSVC

q1_c1 = 0
q1_c2 = 1
q1_classes = np.array([q1_c1, q1_c2])

# d_train_1000_labels = np.empty(shape=1000, dtype=int)
# d_train_1000 = np.empty(shape=[1000, 3])

# d_valid_100000_labels = np.empty(shape=100000, dtype=int)
# d_valid_100000 = np.empty(shape=[100000, 3])
mean = np.array([0,0])
sigma = np.array([[1,0],
          		 [0,1]])

def generate_samples_v1(num_samples):
    #num_samples = samples.shape
    return_array = []
    return_labels = []
    for sample in range(num_samples):
        if np.random.rand() < 0.5:
            r = 2
            n = np.random.multivariate_normal(mean, sigma)
            theta =  np.random.uniform(low = -180, high = 180, size=None)
            middle = np.array([[math.sin(math.radians(theta))],[math.cos(math.radians(theta))]])
            x = middle*r + np.atleast_2d(n).T
            return_array.append(x)
            return_labels.append(-1)
        else:
            r = 4
            n = np.random.multivariate_normal(mean, sigma)
            theta =  np.random.uniform(low = -180, high = 180, size=None)
            middle = np.array([[math.sin(math.radians(theta))],[math.cos(math.radians(theta))]])
            x = middle*r + np.atleast_2d(n).T
            return_array.append(x)
            return_labels.append(1)

    return_array = np.array(return_array)
    return_labels = np.array(return_labels)

    return_array = np.reshape(return_array, (num_samples, 2))
    return return_array, return_labels


def get_radii(samps, labs):
    labels_0s = np.argwhere(labs == -1)
    labels_1s = np.argwhere(labs == 1)
    samples_0s = []
    samples_1s = []

    for l in labels_0s:
        samples_0s.append(samps[l])
    for l in labels_1s:
        samples_1s.append(samps[l])
    samples_0s = np.array(samples_0s, dtype = 'object')
    samples_1s = np.array(samples_1s, dtype = 'object')      
    samps_by_labels = [samples_0s, samples_1s]
    return_array = []
    for sett in samps_by_labels:
        small_radius = 2147483648
        large_radius = 0
        running_total = 0
        for sample in sett:
            if small_radius > math.sqrt((sample[0][0] ** 2) + (sample[0][1] ** 2)):
                small_radius = math.sqrt((sample[0][0] ** 2) + (sample[0][1] ** 2))
            if large_radius < math.sqrt((sample[0][0] ** 2) + (sample[0][1] ** 2)):
                large_radius = math.sqrt((sample[0][0] ** 2) + (sample[0][1] ** 2))
            running_total = running_total + math.sqrt((sample[0][0] ** 2) + (sample[0][1] ** 2)) 
        return_array.append(small_radius)
        return_array.append(large_radius)
        return_array.append(math.sqrt(((running_total/sett.size) ** 2) + ((running_total/sett.size) ** 2)))

    return return_array

    

def display_samples_v2(samps, labs):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)    

    labels_0s = np.argwhere(labs == -1)
    labels_1s = np.argwhere(labs == 1)

    ax.scatter(samps[labels_1s, 0], samps[labels_1s, 1], color="b", marker = 'o', label="True Class 2")
    ax.scatter(samps[labels_0s, 0], samps[labels_0s, 1], color="r", marker = '^', label="True Class 1")

    radii = get_radii(samps, labs)

    small_circle_red = plt.Circle((0, 0), radius = radii[0], color='lightcoral', fill = False)
    ax.add_patch(small_circle_red)
    large_circle_red  = plt.Circle((0, 0), radius = radii[1], color='crimson', fill = False)
    ax.add_patch(large_circle_red)
    #average_circle_red  = plt.Circle((0, 0), radius = radii[2], color='red', fill = False, linestyle = "--", linewidth = 1.0)
    #ax.add_patch(average_circle_red)

    small_circle_blue = plt.Circle((0, 0), radius = radii[3], color='dodgerblue', fill = False)
    ax.add_patch(small_circle_blue)
    large_circle_blue  = plt.Circle((0, 0), radius = radii[4], color='darkblue', fill = False)
    ax.add_patch(large_circle_blue)
    #average_circle_blue  = plt.Circle((0, 0), radius = radii[5], color='blue', fill = False, linestyle = "--", linewidth = 1.0)
    #ax.add_patch(average_circle_blue)   
    
    plt.legend(loc='upper left')
    plt.title("Data set with {} samples".format(str(labs.size)))
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

#display_samples_v2(d_train_1000, d_train_1000_labels)
#display_samples_v2(d_train_100000, d_train_100000_labels)

def kfoldcrossvalid_hw4(training_data, training_labels, valid_data, valid_labels, k, perf_func, stop_increase, order_start = 1, order_step = 1, stop_improvement = 10):
    # super useful row-wise merge! Thank you docummentation and stack exhange
    # https://stackoverflow.com/questions/21990345/merging-concatenating-arrays-with-different-elements/21990608#21990608
   	# Adding the [0] to the end of shape tell which element, the first in this case, that;s shape ashould be reported
    partition_indices = np.r_[np.linspace(0, training_data.shape[0], num = k, endpoint = False, dtype=int), training_data.shape[0]]
    best_perf_orders = np.zeros(k)
    #ptron_hist = []

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

        while performance_decrement < stop_increase:
            curr_perf, nn_not_used = perf_func(temp_train, temp_validate, model_order, temp_train_labels, temp_validate_labels)
            
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
    return final_order, best_perf_orders

# def my_mse(nn, d_valid, d_valid_labels):
#     # same prediction from SciKitLearn function as in homework 3
#     predicted_values = nn.predict(d_valid)
    
#     running_total = 0
#     for sample in range(len(d_valid_labels)):
#         #MSE Function: (Sum of all (Label - prediction) squared ) / (Num of events)
#         running_total += (d_valid_labels[sample] - predicted_values[sample]) ** 2
#     mse = running_total / len(d_valid_labels)
#     return mse

#nn_train(d_train, d_valid, num_perceptrons, d_train_labels, d_valid_labels):
def nn_train(d_train, d_valid, num_perceptrons, d_train_labels, d_valid_labels):
    #nnv2 = MLPRegressor(hidden_layer_sizes = num_perceptrons, activation='logistic', solver='sgd', alpha=1e-6, max_iter=50000, shuffle=True, tol=1e-5, verbose=False, n_iter_no_change=20)
    nn = MLPClassifier(hidden_layer_sizes = num_perceptrons, activation='relu', solver='adam', alpha=1e-6, max_iter=6000, shuffle=True, tol=1e-4, verbose=False, warm_start=False, early_stopping=False, n_iter_no_change=10)
    nn.fit(d_train, d_train_labels)

   #d_valid = np.reshape(d_valid, (d_valid.size, 2))
    d_valid_predict = nn.predict(d_valid)

    # skl.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True) - Mean squared error regression loss
    
    num_correct = 0
    for sample in range(d_valid.shape[0]):
        if d_valid_predict[sample] == d_valid_labels[sample]:
            num_correct += 1
            
    log_classify_err = np.log(num_correct / d_valid.shape[0])
    return log_classify_err, nn

def training_report(d_train_tr, d_train_labels_tr, k, num_runs, d_valid_tr, d_valid_labels_tr):
    # Use 10-fold cross validation for each training dataset to determine best number of perceptrons
                          
    perceptrons, perceptron_history = kfoldcrossvalid_hw4(training_data = d_train_tr, training_labels = d_train_labels_tr, valid_data = d_valid_tr, valid_labels = d_valid_labels_tr,
                                                        k = 10, perf_func = nn_train, stop_increase = 7, order_start = 1, order_step=1, stop_improvement = 10)

    perceptrons = int(np.round_(perceptrons, decimals=0))  # Convert to nearest integer
    
    for i in range(perceptron_history.size):
        perceptron_history[i] = int(np.round_(perceptron_history[i], decimals=0))
    
    perceptron_history = perceptron_history.astype(int)

    # Train the data using the number of perceptrons found in cross validation
	# Train multiple times and take best performance
    best_train_nn = None
    best_train_perf = -2147483648 # smallest int in Java
    performance_history = []
    for unused in range(num_runs):
        # works Better than a while loop with its own iterator I think
        print("here")
        perf, nn = nn_train(d_train_tr, d_valid_tr, perceptrons, d_train_labels_tr, d_valid_labels_tr)
        if perf > best_train_perf:
            #update the best after a run
            best_train_perf = perf
            best_train_nn = nn
        performance_history.append((unused, perf))


	# Apply MLP to test data and get error probability
    d_validate_predictions_tr = best_train_nn.predict(d_valid_tr)
    num_correct_tr = 0
    for sample in range(d_valid_tr.shape[0]):
        if d_validate_predictions_tr[sample] == d_valid_labels_tr[sample]:
            num_correct_tr += 1
    
    # best_log_llhood = np.log(num_correct_tr / d_valid_tr.shape[0])
    
    # misclassification_rate  
    error_prob =  ((d_valid_tr.shape[0] - num_correct_tr) / d_valid_tr.shape[0])   

    #print(str(d_train_tr.shape[0]) + " samples error probability:" + str(error_prob))
    return error_prob, perceptrons, perceptron_history, performance_history


def plot_preceptrons(data):
    plt.hist(x = data, histtype = 'bar', align = 'left', rwidth = 1, bins = range(min(data), max(data) + 2, 1))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Perceptrons History During 10-Fold CV')
    plt.ylabel('Number of Perceptrons')
    plt.xlabel('Number of Times Chosen')
    plt.show()

def plot_performance(data):
    print("Plot Performance!")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    data_x = []
    data_y = []
    for i in range(len(data)):
        data_x.append(data[i][0])
        data_y.append(data[i][1])  

    ax.scatter(data_x, data_y, color="b", marker = 'o')
  
    plt.title('Performance per Run')
    plt.xlabel('Run Number')
    plt.ylabel('Performance')
    plt.show()

# Generate Samples
d_train_1000, d_train_1000_labels = generate_samples_v1(1000)
d_train_100000, d_train_100000_labels = generate_samples_v1(100000)
print("Samples Generated!")
# Run MLP and K Fo
# perceptrons, perceptron_history = kfoldcrossvalid_hw4(training_data = d_train_1000, training_labels = d_train_1000_labels, valid_data = d_valid_100000, valid_labels = d_valid_100000_labels,
#                                                         k = 10, perf_func = nn_train, stop_increase = 7, order_start = 1, order_step=1, stop_improvement = 10)
error_prob, perceptrons, perceptron_history, performance_history = training_report(d_train_1000, d_train_1000_labels, 10, 5, d_train_100000, d_train_100000_labels)
print("MLP and K-Fold Done")

plot_preceptrons(perceptron_history)
plot_performance(performance_history)

print("Done")