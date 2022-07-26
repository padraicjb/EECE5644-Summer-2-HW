
import matplotlib.pyplot as plt # For general plotting
from mpl_toolkits import mplot3d
import numpy as np
import math
from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sys import float_info
np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

# Borrow Prof Mark's Graph Settings
plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title
# ---------------------------GIVENS------------------------------------
# PDF:
# p(x) = p(x|L = 0)p(L = 0)+ p(x|L = 1)p(L = 1) 

# Priors:
# p(L = 0) = 0.65
# p(L = 1) = 0.35
p0 = 0.65
p1 = 0.35
priors = np.array([p0,p1])

Classes = np.array([0,1])

# Class-Conditional PDFs:
# p(x|L = 0) = N(x|μ0,Σ0)
# μ0 - mu_naught - column vector
# Σ0 - SIGMA_naught - 2d array
mu_naught = np.array([-0.5, -0.5, -0.5])
SIGMA_naught = np.array([[ 1.0,-0.5, 0.3],
                         [-0.5, 1.0,-0.5],
                         [ 0.3,-0.5, 1.0]])

# p(x|L = 1) = N(x|μ1,Σ1)
# μ0 - mu_one - column vector 
# Σ0 - SIGMA_one - 2D array
mu_one = np.array([1.0,1.0,1.0])
SIGMA_one = np.array([[ 1.0,0.3,-0.2],
                      [ 0.3,1.0, 0.3],
                      [-0.2,0.3, 1.0]])

mus = np.array([mu_naught, mu_one])
SIGMAs = np.array([SIGMA_naught, SIGMA_one])

# int for the total number of samples
Num_Samples = 10000
N = Num_Samples

#-------------------------------------------------------
#TODO: UNSURE IF I NEED YET
# Determine dimensionality from mixture PDF parameters
mu = np.array([[-1, 0],
               [1, 0],
               [0, 1]])

n = mu.shape[1]

#-------------------------------------------------------
# Generate Samples with Labels
samples = np.empty((Num_Samples, 3))
labels = np.empty(Num_Samples)
for i in range(Num_Samples):
    # Generate a random and figure out which label it goes under
    if np.random.rand() < p0:
        class_label = 0
    else:
        class_label = 1
    # Add the new label to the list of labels
    labels[i] = class_label
    # Generate a sample to go with the the label we just made and stored
    if class_label == 0:
        temp = np.random.multivariate_normal(mu_naught, SIGMA_naught)
        samples[i][0] = temp[0]
        samples[i][1] = temp[1]
        samples[i][2] = temp[2]
    else:
        temp = np.random.multivariate_normal(mu_one, SIGMA_one)
        samples[i][0] = temp[0]
        samples[i][1] = temp[1]
        samples[i][2] = temp[2]

def display_samples():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    marker_shapes = 'd+'
    marker_colors = 'rk'
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    for i in range(Num_Samples):
        l = int(labels[i])
        ax.scatter3D(samples[i][0], samples[i][1], samples[i][2], marker = marker_shapes[l],color=marker_colors[l],label="True Class {}".format(l) )
    #ax.legend(scatterpoints=1)
    plt.show()

#Uncomment for 3D Scatterplot of Sample Data
#display_samples()

Nums_samps_per = np.array([sum(labels == l) for l in np.array(range(0, 2))])
#print("Number of samples from Class 1: {:d}, Class 2: {:d}, ".format(Nums_samps_per[0], Nums_samps_per[1]))

# Determine the PDF

# arrray w/dim 0:2 |
# 0 - array 0:10k  |
# 1 - array 0:10k \/            array\/                                                                        \/loop 4 tot num priors
class_conditional_pdfs = np.array([multivariate_normal.pdf(x = samples, mean = mus[c], cov = SIGMAs[c]) for c in range(len(priors))])
# p(L=r|x) = p(L=r)p(x|L=r)
#          = prior*conditional

#Straight From HW0
class_priors = np.diag(priors)
#print(priors)
#print(class_priors)
#                  diag - returns the extracted diagonal or a constructed diagonal.
class_posteriors = class_priors.dot(class_conditional_pdfs) # from L02a_Expected Risk... Page 1 

decisions = np.argmax(class_posteriors, axis = 0)
#                /\ index of max value per row (probably), axis - 
conf_matrix = confusion_matrix(decisions, labels)

correct_class_samples = np.sum(np.diag(conf_matrix))

def display_confusion_matrix():
    #print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))
    conf_display = ConfusionMatrixDisplay.from_predictions(decisions, labels, display_labels=['1', '2'], colorbar=False)
    plt.ylabel('Predicted Labels')
    plt.xlabel('True Labels')
    plt.show()
#Uncomment for Graph of Confusion Matrix
#display_confusion_matrix()

# Loss Matrix - from L02a_Expected Risk... page 2
LAMBDA = np.ones((len(priors), len(priors))) - np.identity(len(priors))
#print(LAMBDA) 0 1
#              1 0 

discriminant_score_erm = np.log(class_conditional_pdfs[1]) - np.log(class_conditional_pdfs[0])
#                      # probabilty density fx depending on prob
                    # ratio of probs in log space 
#print(np.size(discriminant_score_erm))
gamma_map = (LAMBDA[1,0] - LAMBDA[0,0]) / (LAMBDA[0,1] - LAMBDA[1,1]) * priors[0] / priors[1]
# scaler = 1.85
#print(gamma_map)

decisions_map = discriminant_score_erm >= np.log(gamma_map)
#                                          0.2
# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability - says negative is negative    CORRECT
ind_00_map = np.argwhere((decisions_map==0) & (labels==0))
# list of indicies where True Negative
#print(np.size(ind_00_map))
#print(ind_00_map)
p_00_map = len(ind_00_map) / Nums_samps_per[0]
#rate 
#probability -> Number of instances of neg neg / total number of negs
# False Positive Probability - says positive is negative   INCORRECT
ind_10_map = np.argwhere((decisions_map==1) & (labels==0))
p_10_map = len(ind_10_map) / Nums_samps_per[0]
# False Negative Probability - says negative is positive   INCORRECT
ind_01_map = np.argwhere((decisions_map==0) & (labels==1))
p_01_map = len(ind_01_map) / Nums_samps_per[1]
# True Positive Probability - says positive is positive    CORRECT
ind_11_map = np.argwhere((decisions_map==1) & (labels==1))
p_11_map = len(ind_11_map) / Nums_samps_per[1]

# Probability of error for MAP classifier
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nums_samps_per.T / Num_Samples)
def display_MAP():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')


    ax.scatter3D(samples[ind_00_map, 0], samples[ind_00_map, 1], marker = 'o',color = 'g', label="Correct Class 0")
    ax.scatter3D(samples[ind_10_map, 0], samples[ind_10_map, 1], marker = 'o', color = 'r', label="Incorrect Class 0")
    ax.scatter3D(samples[ind_01_map, 0], samples[ind_01_map, 1], marker = '+', color = 'k', label="Incorrect Class 1")
    ax.scatter3D(samples[ind_11_map, 0], samples[ind_11_map, 1], marker = '+', color = 'b',label="Correct Class 1")

    plt.legend()
    plt.title("MAP Decisions (RED incorrect)")
    plt.tight_layout()
    plt.show()
#display_MAP()
#From HW0 Solution Set
def estimate_roc(discriminant_score, label):
    # Generate ROC curve samples
    # Takes:
    # discriminant_score - discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])
    # label- 0 or 1 (I think)
    # Returns:
    # roc - a
    # gamma 
    nums_labels = np.array((sum(label == 0), sum(label == 1)))
    # Sorting necessary so the resulting FPR and TPR axes plot threshold probabilities in order as a line
    sorted_score = sorted(discriminant_score)
    #sort by defualt in ascending order
    # the ratio in log space + more likely in label 1, - more likely in label 0 higher/lower value indicates the confidence in that prediction
    # Use gamma values that will account for every possible classification split
    gammas = ([sorted_score[0] - float_info.epsilon] + sorted_score + [sorted_score[-1] + float_info.epsilon])
                                #float_info is the smallest number
    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= g for g in gammas]
    #taking array of disc score (log ratios) 
    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    #indices was 0 but labeled - False Positive
    #indexes of False Positives 
    p10 = [len(inds)/nums_labels[0] for inds in ind10]
    # rate of false positives 
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/nums_labels[1] for inds in ind11]
    #tru pos rate
    ind01 = [np.argwhere((d==0) & (label==1)) for d in decisions]
    p01 = [len(inds)/nums_labels[1] for inds in ind01] 
    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))
    # false postive * class prob for that class
    gamma_errors = [len(np.argwhere((d==1) & (label==0)))*priors[1] + len(np.argwhere((d==0) & (label==1)))*priors[0] for d in decisions]
    best_gamma = [gammas[np.argmin(gamma_errors)],p10[np.argmin(gamma_errors)],p11[np.argmin(gamma_errors)]]
    return roc, gammas, best_gamma

roc_erm, gammas, bgamma = estimate_roc(discriminant_score_erm, labels)
#receiver operating characteristic curve _ empirical risk minimization

roc_map = np.array((p_10_map, p_11_map))


def display_ROC_MAP():
    fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
    ax_roc.plot(roc_erm[0], roc_erm[1])  # <- graphs the entire line, the ERM
    ax_roc.plot(roc_map[0], roc_map[1], 'rx', label="Minimum Pr(error) MAP: ({x},{y})".format(x = str(round(roc_map[0],2)), y = str(round(roc_map[1],2))), markersize=16) # <- graphs the single best point, the MAP
    ax_roc.plot(bgamma[1], bgamma[2], 'bx', label="Emperical Minimum Pr(error) MAP: ({x},{y})".format(x = str(round(bgamma[1],2)), y = str(round(bgamma[2],2))), markersize=16) 
    ax_roc.legend()
    ax_roc.set_xlabel(r"Probability of false alarm $p(D=1|L=0)$")
    ax_roc.set_ylabel(r"Probability of correct decision $p(D=1|L=1)$")
    plt.grid(True)
    plt.show()
# Uncomment to display the ROC and MAP
#display_ROC_MAP()

#----------------------------------
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
	normalization_constant = ((2 * math.pi) ** (-dimensions / 2)) * (np.linalg.det(sigma) ** -0.5)
	cov_inv = np.linalg.inv(sigma)
	for i in range(len(x)):
		mean_diff = np.subtract(x[i], mu)
		exponent = math.exp(np.matmul(np.matmul(-0.5 * np.transpose(mean_diff), cov_inv), mean_diff))
		likelihood = normalization_constant * exponent
		ret_matrix.append(likelihood)
	return ret_matrix

# ----- Part B --------- 
sigma_part_b = [np.eye(3)*s for s in SIGMAs]
ccp_part_b = np.array([multivariate_normal.pdf(x = samples, mean = mus[c], cov = sigma_part_b[c]) for c in range(len(priors))])
discriminant_score_erm_b = np.log(ccp_part_b[1]) - np.log(ccp_part_b[0])
roc_erm_b, gammas_b, bgamma_b = estimate_roc(discriminant_score_erm_b, labels)

decisions_map_b = discriminant_score_erm_b >= np.log(gamma_map)
#                                          0.2
# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability - says negative is negative    CORRECT
ind_00_map_b = np.argwhere((decisions_map_b==0) & (labels==0))
# list of indicies where True Negative
#print(np.size(ind_00_map_b))
#print(ind_00_map_b)
p_00_map_b = len(ind_00_map_b) / Nums_samps_per[0]
#rate 
#probability -> Number of instances of neg neg / total number of negs
# False Positive Probability - says positive is negative   INCORRECT
ind_10_map_b = np.argwhere((decisions_map_b==1) & (labels==0))
p_10_map_b = len(ind_10_map_b) / Nums_samps_per[0]
# False Negative Probability - says negative is positive   INCORRECT
ind_01_map_b = np.argwhere((decisions_map_b==0) & (labels==1))
p_01_map_b = len(ind_01_map_b) / Nums_samps_per[1]
# True Positive Probability - says positive is positive    CORRECT
ind_11_map_b = np.argwhere((decisions_map_b==1) & (labels==1))
p_11_map_b = len(ind_11_map_b) / Nums_samps_per[1]

roc_map_b = np.array((p_10_map_b, p_11_map_b))



fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
ax_roc.plot(roc_erm_b[0], roc_erm_b[1])  
ax_roc.plot(roc_map_b[0], roc_map_b[1], 'rx', label="Minimum Pr(error) MAP: ({x},{y})".format(x = str(round(roc_map_b[0],2)), y = str(round(roc_map_b[1],2))), markersize=16) # <- graphs the single best point, the MAP
ax_roc.plot(bgamma_b[1], bgamma_b[2], 'bx', label="Emperical Minimum Pr(error) MAP: ({x},{y})".format(x = str(round(bgamma_b[1],2)), y = str(round(bgamma_b[2],2))), markersize=16) 
ax_roc.legend()
ax_roc.set_xlabel(r"Probability of false alarm $p(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $p(D=1|L=1)$")
plt.grid(True)
#plt.show()
#likelihood_ratios_naive_bayesian = divide(multivariate_gaussian_pdf(samples_q1, m1, diag(diag(cov0))),
#                                          multivariate_gaussian_pdf(samples_q1, m0, diag(diag(cov1))))

def perform_lda(X_LDA, mu_LDA, Sigma_LDA, C_LDA=2):
    """  Fisher's Linear Discriminant Analysis (LDA) on data from two classes (C=2).

    In practice the mean and covariance parameters would be estimated from training samples.
    
    Args:
        X_LDA: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.
        mu_LDA: Mean vector [C, n].
        Sigma_LDA: Covariance matrices [C, n, n].

    Returns:
        w: Fisher's LDA project vector, shape [n, 1].
        z: Scalar LDA projections of input samples, shape [N, 1].
    """

    mu_LDA = np.array([mu_LDA[i].reshape(-1, 1) for i in range(C_LDA)])
    cov_lda = np.array([Sigma_LDA[i].T for i in range(C_LDA)])

    # Determine between class and within class scatter matrix
    Sb = (mu_LDA[1] - mu_LDA[0]).dot((mu_LDA[1] - mu_LDA[0]).T)
    Sw = cov_lda[0] + cov_lda[1]

    # Regular eigenvector problem for matrix Sw^-1 Sb
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]

    # Extract corresponding sorted eigenvectors
    U = U[:, idx]

    # First eigenvector is now associated with the maximum eigenvalue, mean it is our LDA solution weight vector
    w = U[:, 0]

    # Scalar LDA projections in matrix form
    z = X_LDA.dot(w)

    return w, z
 
value, discriminant_score_lda_c = perform_lda(X_LDA = samples, mu_LDA = mus, Sigma_LDA = SIGMAs, C_LDA = 2)
#print('discriminant_score_lda_c', discriminant_score_lda_c)

# Estimate the ROC curve for this LDA classifier
roc_lda_c, gamma_lda_c, best_gamma_lda = estimate_roc(-1 * discriminant_score_lda_c, labels)
#print('shape roc_lda_c', roc_lda_c.shape)

# ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR
#                                                                           ******       
prob_error_lda_c = (np.array((roc_lda_c[0,:], 1 - roc_lda_c[1,:]))).T.dot(Nums_samps_per.T /Num_Samples)
# dot product is multiplying by the emperical probabilites (close to the given p0 p1)
# print((np.array((roc_lda_c[0,:], 1 - roc_lda_c[1,:]))).T.shape) # 10002 x 2
# print("Frist two")
# print((np.array((roc_lda_c[0,:], 1 - roc_lda_c[1,:]))).T[0:3,:])
# print("Last two")
# print((np.array((roc_lda_c[0,:], 1 - roc_lda_c[1,:]))).T[-3:,:])

print(Nums_samps_per.T /Num_Samples) # [0.6433, 0.3567]

# Min prob error
min_prob_error_lda = np.min(prob_error_lda_c)
min_ind_lda = np.argmin(prob_error_lda_c)

# Display the estimated ROC curve for LDA and indicate the operating points
# with smallest empirical error probability estimates (could be multiple)
fig_roc, ax_roc_lda = plt.subplots(figsize=(10, 10))

ax_roc_lda.plot(roc_lda_c[0], roc_lda_c[1], 'b:')
ax_roc_lda.plot(roc_lda_c[0, min_ind_lda], roc_lda_c[1, min_ind_lda], 'r.', label = "Minimum Pr(error) LDA", markersize = 16)
ax_roc_lda.set_title("ROC Curves for ERM and LDA")
ax_roc_lda.legend()

#plt.show()
def disc_score_hist():
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.hist(discriminant_score_erm - discriminant_score_erm_b)
    ax.set_title('ERM vs. Naive Bayes error')
    ax = axes[1]
    ax.hist(discriminant_score_erm - discriminant_score_lda_c)
    ax.set_title('ERM vs. LDA error')
    plt.show()    
# disc_score_hist()
def disc_score_scatter():
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.scatter(discriminant_score_erm, discriminant_score_erm_b)
    ax.set_xlabel('ERM')
    ax.set_ylabel('Naive Bayes')
    ax.set_title('ERM vs. Naive Bayes Error')
    ax = axes[1]
    ax.scatter(discriminant_score_erm, discriminant_score_lda_c)
    ax.set_xlabel('ERM')
    ax.set_ylabel('Fisher LDA')
    ax.set_title('ERM vs. LDA Error')
    plt.show()

# ------------------------------------------------------------------------------

# X = [ 00 01
#       10 11 ]
labels_q2 = [1,2,3,4]
P0_q2 = 0.2
P1_q2 = 0.25
P2_q2 = 0.25
P3_q2 = 0.3
priors_q2 = [P0_q2, P1_q2, P2_q2, P3_q2]

#mu_naught = np.array([-0.5, -0.5, -0.5])
#SIGMA_naught = np.array([[ 1.0,-0.5, 0.3],
 #                        [-0.5, 1.0,-0.5],
  #                       [ 0.3,-0.5, 1.0]])


# MU dimensions should be 2 x 1
mu0_q2 = np.array([0,-0])
mu1_q2 = np.array([1,-1])
mu2_q2 = np.array([2,-2])
mu3_q2 = np.array([3,-3])
mus_q2 = [mu0_q2, mu1_q2, mu2_q2, mu3_q2]

# Sigma dimenstions should be 2 x 2
# "a mean vector containing all the class-conditioned pdf parameters should be e.g. shape (C, n)
# where C is the # classes and n the dimensionality of X"
sigma0_q2 = np.array([[0.5, 0],
                     [0, 0.5]])
sigma1_q2 = np.array([[0.5, 0],
                     [0, 0.5]])
sigma2_q2 = np.array([[0.5, 0],
                     [0, 0.5]])
sigma3_q2 = np.array([[0.5, 0],
                     [0, 0.5]])
sigmas_q2 = [sigma0_q2, sigma1_q2, sigma2_q2, sigma3_q2]

# Generate Samples
samples_q2 = np.empty((Num_Samples, 2))
labels_q2 = np.empty(Num_Samples)

class_probs = np.random.rand(Num_Samples, 1)

for i in range(Num_Samples):
    # Generate a random and figure out which label it goes under
    if class_probs[i] < P0_q2:
        class_label = 0
    elif class_probs[i] >= P0_q2 and class_probs[i] < P1_q2 :
        class_label = 1
    elif class_probs[i] >= P0_q2 + P1_q2 and class_probs[i] < P0_q2 + P1_q2 + P2_q2:
        class_label = 2
    else:
        class_label = 3
    # Add the new label to the list of labels
    labels_q2[i] = class_label
    # Generate a sample to go with the the label we just made and stored
    temp = np.random.multivariate_normal(mus_q2[class_label], sigmas_q2[class_label])
    samples_q2[i][0] = temp[0]
    samples_q2[i][1] = temp[1]

Nums_samps_per_q2 = np.array([sum(labels_q2 == l) for l in np.array(range(0, 4))])
#print("Number of samples from Class 1: {:d}, Class 2: {:d}, ".format(Nums_samps_per[0], Nums_samps_per[1]))

# Determine the PDF

# arrray w/dim 0:2 |
# 0 - array 0:10k  |
# 1 - array 0:10k \/            array\/                                                                        \/loop for tot num priors
class_conditional_pdfs_q2 = np.array([multivariate_normal.pdf(x = samples_q2, mean = mus_q2[c], cov = sigmas_q2[c]) for c in range(len(priors_q2))])
# p(L=r|x) = p(L=r)p(x|L=r)
#          = prior*conditional

#Straight From HW0
class_priors_q2 = np.diag(priors_q2)
#print(priors_q2)
#print(class_priors_q2)
#                  diag - returns the extracted diagonal or a constructed diagonal.
class_posteriors_q2 = class_priors_q2.dot(class_conditional_pdfs_q2) # from L02a_Expected Risk... Page 1 

decisions_q2 = np.argmax(class_posteriors_q2, axis = 0)
#                /\ index of max value per row (probably), axis - 
conf_matrix_q2 = confusion_matrix(decisions_q2, labels_q2)

correct_class_samples_q2 = np.sum(np.diag(conf_matrix_q2))

def display_confusion_matrix():
    #print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))
    conf_display = ConfusionMatrixDisplay.from_predictions(decisions_q2, labels_q2, display_labels=['1', '2', '3', '4'], colorbar=False)
    plt.ylabel('Predicted Labels')
    plt.xlabel('True Labels')
    plt.show()
display_confusion_matrix()

def display_samples_q2():
    # fig = plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(10, 10))
    #ax = fig.add_subplot(projection='3d')
    marker_colors = list('rbgk')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    ax.scatter(samples_q2[:, 0], samples_q2[:, 1], c=[marker_colors[int(k)] for k in labels_q2])
    for k, mu in enumerate(mus_q2):
        ax.text(mu[0], mu[1], str(k))
    
    plt.show()
    
display_samples_q2()

# discriminant_score_erm_q2 = np.log(class_conditional_pdfs_q2[1]) - np.log(class_conditional_pdfs_q2[0])
# #                      # probabilty density fx depending on prob
#                     # ratio of probs in log space 
# #print(np.size(discriminant_score_erm_q2))
# gamma_map = (LAMBDA[1,0] - LAMBDA[0,0]) / (LAMBDA[0,1] - LAMBDA[1,1]) * priors[0] / priors[1]
# # scaler = 1.85
# #print(gamma_map)

# decisions_map = discriminant_score_erm_q2 >= np.log(gamma_map)
# #                                          0.2
# # Get indices and probability estimates of the four decision scenarios:
# # (true negative, false positive, false negative, true positive)

# # True Negative Probability - says negative is negative    CORRECT
# ind_00_map = np.argwhere((decisions_map==0) & (labels==0))
# # list of indicies where True Negative
# #print(np.size(ind_00_map))
# #print(ind_00_map)
# p_00_map = len(ind_00_map) / Nums_samps_per[0]
# #rate 
# #probability -> Number of instances of neg neg / total number of negs
# # False Positive Probability - says positive is negative   INCORRECT
# ind_10_map = np.argwhere((decisions_map==1) & (labels==0))
# p_10_map = len(ind_10_map) / Nums_samps_per[0]
# # False Negative Probability - says negative is positive   INCORRECT
# ind_01_map = np.argwhere((decisions_map==0) & (labels==1))
# p_01_map = len(ind_01_map) / Nums_samps_per[1]
# # True Positive Probability - says positive is positive    CORRECT
# ind_11_map = np.argwhere((decisions_map==1) & (labels==1))
# p_11_map = len(ind_11_map) / Nums_samps_per[1]


LAMBDA = [[0, 1, 2, 3],
          [1, 0, 1, 2],
          [2, 1, 0, 1],
          [3, 2, 1, 0]]