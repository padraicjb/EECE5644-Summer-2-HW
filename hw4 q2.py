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
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, LinearSVC

mean = np.array([0,0])
sigma = np.array([[1,0],
          		 [0,1]])

def plot_classifier_predictions(classifier, n_samples=100):
    # Create coordinate matrices determined by the sample space
    xx, yy = np.meshgrid(np.linspace(-4, 4, n_samples), np.linspace(-4, 4, n_samples))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Matrix of predictions on a grid of samples
    y_pred = classifier.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.2)

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

def display_samples_v3(samps, labs, classifier, n_samples=100):
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
    
    #xx, yy = np.meshgrid(np.linspace(-4, 4, n_samples), np.linspace(-4, 4, n_samples))
    xx, yy = np.meshgrid(np.linspace(-9, 9, n_samples), np.linspace(-9, 9, n_samples))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Matrix of predictions on a grid of samples
    y_pred = classifier.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.2)


    plt.legend(loc='upper left')
    plt.title("Data set with {} samples".format(str(labs.size)))
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

def get_xys(ar, size):
    xs =[]
    ys = []
    for data_point in range(size):
        xs.append(ar[data_point][0])
        ys.append(ar[data_point][1])
    return xs, ys

d_train_1000, d_train_1000_labels = generate_samples_v1(1000)
d_validate_100000, d_validate_100000_labels = generate_samples_v1(100000)

d_train_1000_xs, d_train_1000_ys = get_xys(d_train_1000, 1000)
d_validate_100000_xs, d_validate_100000_ys = get_xys(d_validate_100000, 100000)

# rbf_svc = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=0.7, C=1.0))
# rbf_svc.fit(d_train_1000, d_train_1000_labels)
# rbf_svc.predict(d_validate_100000)

def gsearch_report(svc_pipeline, d_train_1000, d_train_1000_labels, d_validate_100000, d_validate_100000_labels, optimizer='grid_search', n_iter=None):
    print("Performing GridSearch")
    #  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    battleship = GridSearchCV(estimator = svc_pipeline, param_grid = {'C': np.logspace(-2, 2),'gamma': np.logspace(-2,2),'kernel': ['rbf']}, cv = 10, refit = True, return_train_score = False,
                            scoring = 'accuracy')
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV.fit
    battleship.fit(d_train_1000, d_train_1000_labels)
    
   
    # Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data
    hit_or_miss = battleship.best_estimator_
    # Mean cross-validated score of the best_estimator
    best_score = battleship.best_score_
    print("Mean cross-validated score of the best_estimator: " + str(best_score))

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    # Array of scores of the estimator for each run of the cross validation.
    scores = cross_val_score(hit_or_miss, d_train_1000, d_train_1000_labels, cv=10)
    print("Cross-validation Score: ")
    print(scores)
    print("Average Score: " + str(scores.mean()))
    # Parameter setting that gave the best results on the hold out data.
    print("Parameter settings that gave the best results on the hold out data.")
    print(battleship.best_params_)
    

    print("Best Estimator Score: " + str(hit_or_miss.score(d_validate_100000, d_validate_100000_labels)))
    
    battleship.fit(d_validate_100000, d_validate_100000_labels)
    predictions = battleship.predict(d_validate_100000)
    
    return battleship, hit_or_miss, predictions


#display_samples_v3(d_train_1000, d_train_1000_labels, rbf_svc, n_samples=500)
gridsearchobject, estimatorobject, predictions = gsearch_report(SVC(), d_train_1000, d_train_1000_labels, d_validate_100000, d_validate_100000_labels)