# import ml_helpers as ml
import numpy as np
import matplotlib.pyplot as plt
import math
from sys import float_info

# import scipy.optimize as sp_optimize

# GIVENS:
# Classes:
q1_c1 = 0
q1_c2 = 1
q1_classes = np.array([q1_c1, q1_c2])

# Priors:
q1_p0 = 0.65
q1_p1 = 0.35
q1_priors = np.array([q1_p0, q1_p1])

# Weights(?):
q1_a1 = 0.5
q1_a2 = 0.5
q1_as = np.array([q1_a1, q1_a2])

# Mus:
q1_mu1 = np.array([3, 0])
q1_mu2 = np.array([0, 3])
q1_mu3 = np.array([2, 2])
q1_MUs = np.array([q1_mu1, q1_mu2, q1_mu3])

# Sigmas
q1_sigma1 = np.array([[2, 0], [0, 1]])
q1_sigma2 = np.array([[1, 0], [0, 2]])
q1_sigma3 = np.array([[1, 0], [0, 1]])
q1_Sigmas = [q1_sigma1, q1_sigma2, q1_sigma3]

# Storage
d_train_20_labels = np.empty(shape=20, dtype=int)
d_train_20 = np.empty(shape=[20, 2])
d_train_200_labels = np.empty(shape=200, dtype=int)
d_train_200 = np.empty(shape=[200, 2])
d_train_1000_labels = np.empty(shape=1000, dtype=int)
d_train_1000 = np.empty(shape=[1000, 2])
d_valid_10000_labels = np.empty(shape=10000, dtype=int)
d_valid_10000 = np.empty(shape=[10000, 2])


def generate_sl(samps, labs, prior0, A1, A2, Mus, Sigmas):
    for i in range(labs.size):
        # Generate a random and figure out which label it goes under
        if np.random.rand() < prior0:
            class_label = 0
        else:
            class_label = 1
        # Add the new label to the list of labels
        labs[i] = class_label

        if class_label == 0:
            if np.random.rand() < 0.5:
                temp = np.random.multivariate_normal(Mus[0], Sigmas[0])
            else:
                temp = np.random.multivariate_normal(Mus[1], Sigmas[1])
            # ary    = scalar wght * gaussian array  + scalar wght * gaussian array
            samps[i][0] = temp[0]
            samps[i][1] = temp[1]
            # samples[i][2] = temp[2]
        else:
            temp = np.random.multivariate_normal(Mus[2], Sigmas[2])
            samps[i][0] = temp[0]
            samps[i][1] = temp[1]
            # samples[i][2] = temp[2]


generate_sl(d_train_20, d_train_20_labels, q1_p0, q1_a1, q1_a2, q1_MUs, q1_Sigmas)
generate_sl(d_train_200, d_train_200_labels, q1_p0, q1_a1, q1_a2, q1_MUs, q1_Sigmas)
generate_sl(d_train_1000, d_train_1000_labels, q1_p0, q1_a1, q1_a2, q1_MUs, q1_Sigmas)
generate_sl(d_valid_10000, d_valid_10000_labels, q1_p0, q1_a1, q1_a2, q1_MUs, q1_Sigmas)

# Useful to have
class_0_samples = np.zeros([10000, 2])
class_1_samples = np.zeros([10000, 2])
class_0_samples[d_valid_10000_labels == 0] = d_valid_10000[d_valid_10000_labels == 0]
class_1_samples[d_valid_10000_labels == 1] = d_valid_10000[d_valid_10000_labels == 1]

# for index in range(10000):
#     if d_valid_10000_labels[index] == 0:
#         class_0_samples[index] = d_valid_10000[index]
#     else:
#         class_1_samples[index] = d_valid_10000[index]


def display_samples(samps, labs):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    # n rows, 2 columns
    # observatiuos on rows, features on columns

    labs_0s = np.argwhere(labs == 0)
    # samps_0s = np.array(samps[zero_index] for zero_index in labs_0s)
    labs_1s = np.argwhere(labs == 1)
    # samps_1s = np.array(samps[one_index] for one_index in labs_1s)
    ax.scatter(samps[labs_0s, 0], samps[labs_0s, 1], color="r", label="True Class 1")
    ax.scatter(samps[labs_1s, 0], samps[labs_1s, 1], color="k", label="True Class 2")

    ax.legend()
    ax.set_title("Data set with {} samples".format(str(labs.size)))
    ax.set_xlabel("x0")
    ax.set_xlabel("x1")
    plt.show()


# display_samples(d_train_20, d_train_20_labels)
# display_samples(d_train_200, d_train_200_labels)
# display_samples(d_train_1000, d_train_1000_labels)
display_samples(d_valid_10000, d_valid_10000_labels)


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


def plot_theo_bound(axs):
    # defines the fineness of grid used to determine the boundary
    # basically want to check likelihood ratios distributed around the validation data
    xy_field = 50

    # x_points - 0:50 - even distribution of points from the min and max of the validate set
    x_points = np.linspace(min(d_valid_10000[:, 0]), max(d_valid_10000[:, 0]), xy_field)
    # y_points - 0:50 - even distribution of points from the min and max of the validate set
    y_points = np.linspace(min(d_valid_10000[:, 1]), max(d_valid_10000[:, 1]), xy_field)

    # 2500,2 empty, to be immediately filled
    xy_array = np.zeros([xy_field * xy_field, 2])
    for i in range(xy_field):
        # iterrates up to 50 for all of the samples from the valid set in the x dimension
        for j in range(xy_field):
            # iterrates up to 50 for all of the samples from the valid set in the y dimension
            # builds up coordinate pairs of all xs and ys
            xy_array[i * xy_field + j][0] = x_points[i]
            xy_array[i * xy_field + j][1] = y_points[j]

    # Creates the grid of test points to a grid of liklihood ratios
    # Not Log Scale! (last assignment I did it logscale)
    likelihood_ratios_grid = np.divide(
        multivariate_gaussian_pdf(xy_array, q1_mu3, q1_sigma3),
        np.multiply(q1_a1, multivariate_gaussian_pdf(xy_array, q1_mu1, q1_sigma1))
        + np.multiply(q1_a2, multivariate_gaussian_pdf(xy_array, q1_mu2, q1_sigma2)),
    )

    # Creates empty 50x50
    z_grid = np.zeros([xy_field, xy_field])
    # Converts the liklihood list into a 50 50 2d array
    for i in range(xy_field):
        for j in range(xy_field):
            z_grid[i][j] = likelihood_ratios_grid[j * xy_field + i]

    x_points, y_points = np.meshgrid(x_points, y_points)
    return axs.contour(
        x_points, y_points, z_grid, [q1_p1 / q1_p0], colors=["k"], linewidths=2
    )


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
    # sort by defualt in ascending order
    # the ratio in log space + more likely in label 1, - more likely in label 0 higher/lower value indicates the confidence in that prediction
    # Use gamma values that will account for every possible classification split
    gammas = (
        [sorted_score[0] - float_info.epsilon]
        + sorted_score
        + [sorted_score[-1] + float_info.epsilon]
    )
    # float_info is the smallest number
    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= g for g in gammas]
    # taking array of disc score (log ratios)
    ind10 = [np.argwhere((d == 1) & (label == 0)) for d in decisions]
    # indices was 0 but labeled - False Positive
    # indexes of False Positives
    p10 = [len(inds) / nums_labels[0] for inds in ind10]
    # rate of false positives
    ind11 = [np.argwhere((d == 1) & (label == 1)) for d in decisions]
    p11 = [len(inds) / nums_labels[1] for inds in ind11]
    # tru pos rate
    ind01 = [np.argwhere((d == 0) & (label == 1)) for d in decisions]
    p01 = [len(inds) / nums_labels[1] for inds in ind01]
    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))
    # false postive * class prob for that class
    gamma_errors = [
        len(np.argwhere((d == 1) & (label == 0))) * q1_priors[1]
        + len(np.argwhere((d == 0) & (label == 1))) * q1_priors[0]
        for d in decisions
    ]
    best_gamma = [
        gammas[np.argmin(gamma_errors)],
        p10[np.argmin(gamma_errors)],
        p11[np.argmin(gamma_errors)],
    ]
    return roc, gammas, best_gamma


def question1_part1(
    p0_est,
    p1_est,
    a1_est,
    mu01_est,
    sigma01_est,
    a2_est,
    mu02_est,
    sigma02_est,
    mu03_est,
    sigma03_est,
):
    # # Print estimates of parameters
    # print("p0: %s, p1: %s" % (str(p0_est), str(p1_est)))
    # print("nw01_hat: %s, m01_hat: %s")
    # print(cov01_hat: %s\nw02_hat: %s, m02_hat: %s, "
    #     "cov02_hat: %s\nm1_hat: %s, cov1_hat: %s"

    #         str(a1_est),
    #         str(mu01_est),
    #         str(sigma01_est),
    #         str(a2_est),
    #         str(mu02_est),
    #         str(sigma02_est),
    #         str(mu03_est),
    #         str(sigma03_est),
    #     )
    # )

    # Calculate likelihood ratios of all samples using knowledge of pdf. Then, generate ROC curve

    likelihood_ratios = np.log(
        multivariate_gaussian_pdf(d_valid_10000, mu03_est, sigma03_est)
    ) - np.log(
        (
            np.multiply(
                a1_est,
                multivariate_gaussian_pdf(d_valid_10000, mu01_est, sigma01_est),
            )
            + np.multiply(
                a2_est,
                multivariate_gaussian_pdf(d_valid_10000, mu02_est, sigma02_est),
            )
        )
    )
    LAMBDA = np.ones((len(q1_priors), len(q1_priors))) - np.identity(len(q1_priors))
    gamma_map = (
        (LAMBDA[1, 0] - LAMBDA[0, 0])
        / (LAMBDA[0, 1] - LAMBDA[1, 1])
        * q1_priors[0]
        / q1_priors[1]
    )
    # scaler = 1.85
    # print(gamma_map)

    decisions_map = likelihood_ratios >= np.log(gamma_map)
    #                                          0.2
    # Get indices and probability estimates of the four decision scenarios:
    # (true negative, false positive, false negative, true positive)

    # True Negative Probability - says negative is negative    CORRECT
    ind_00_map = np.argwhere((decisions_map == 0) & (d_valid_10000_labels == 0))
    # list of indicies where True Negative
    # print(np.size(ind_00_map))
    # print(ind_00_map)
    p_00_map = len(ind_00_map) / class_0_samples.size
    # rate
    # probability -> Number of instances of neg neg / total number of negs
    # False Positive Probability - says positive is negative   INCORRECT
    ind_10_map = np.argwhere((decisions_map == 1) & (d_valid_10000_labels == 0))
    p_10_map = len(ind_10_map) / class_0_samples.size
    # False Negative Probability - says negative is positive   INCORRECT
    ind_01_map = np.argwhere((decisions_map == 0) & (d_valid_10000_labels == 1))
    p_01_map = len(ind_01_map) / class_1_samples.size
    # True Positive Probability - says positive is positive    CORRECT
    ind_11_map = np.argwhere((decisions_map == 1) & (d_valid_10000_labels == 1))
    p_11_map = len(ind_11_map) / class_1_samples.size

    roc, gammas, b_gamma = estimate_roc(likelihood_ratios, d_valid_10000_labels)
    roc_map = np.array((p_10_map, p_11_map))

    fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
    ax_roc.plot(roc[0], roc[1])  # <- graphs the entire line, the ERM
    ax_roc.plot(
        roc_map[0],
        roc_map[1],
        "rx",
        label="Minimum Pr(error) MAP: ({x},{y})".format(
            x=str(round(roc_map[0], 2)), y=str(round(roc_map[1], 2))
        ),
        markersize=16,
    )  # <- graphs the single best point, the MAP
    ax_roc.plot(
        b_gamma[1],
        b_gamma[2],
        "bx",
        label="Emperical Minimum Pr(error) MAP: ({x},{y})".format(
            x=str(round(b_gamma[1], 2)), y=str(round(b_gamma[2], 2))
        ),
        markersize=16,
    )
    ax_roc.legend()
    ax_roc.set_xlabel(r"Probability of false alarm $p(D=1|L=0)$")
    ax_roc.set_ylabel(r"Probability of correct decision $p(D=1|L=1)$")
    # plt.grid(True)
    # plt.show()

    # Plot theoretical and estimated decision boundaries on top of validation data
    # xy_grid_num = 100
    # x_grid = np.linspace(
    #     min(d_validate_20000[:, 0]), max(d_validate_20000[:, 0]), xy_grid_num
    # )
    # y_grid = np.linspace(
    #     min(d_validate_20000[:, 1]), max(d_validate_20000[:, 1]), xy_grid_num
    # )
    # xy_array = np.zeros([xy_grid_num * xy_grid_num, 2])
    # for i in range(xy_grid_num):
    #     for j in range(xy_grid_num):
    #         xy_array[i * xy_grid_num + j][0] = x_grid[i]
    #         xy_array[i * xy_grid_num + j][1] = y_grid[j]
    # likelihood_ratios_grid = np.divide(
    #     multivariate_gaussian_pdf(xy_array, mu03_est, sigma03_est),
    #     np.multiply(a1_est, multivariate_gaussian_pdf(xy_array, mu01_est, sigma01_est))
    #     + np.multiply(
    #         a2_est, multivariate_gaussian_pdf(xy_array, mu02_est, sigma02_est)
    #     ),
    # )

    xy_field = 50

    x_points = np.linspace(min(d_valid_10000[:, 0]), max(d_valid_10000[:, 0]), xy_field)
    y_points = np.linspace(min(d_valid_10000[:, 1]), max(d_valid_10000[:, 1]), xy_field)
    xy_array = np.zeros([xy_field * xy_field, 2])
    for i in range(xy_field):
        for j in range(xy_field):
            xy_array[i * xy_field + j][0] = x_points[i]
            xy_array[i * xy_field + j][1] = y_points[j]
    likelihood_ratios_grid = np.divide(
        multivariate_gaussian_pdf(xy_array, q1_mu3, q1_sigma3),
        np.multiply(q1_a1, multivariate_gaussian_pdf(xy_array, q1_mu1, q1_sigma1))
        + np.multiply(q1_a2, multivariate_gaussian_pdf(xy_array, q1_mu2, q1_sigma2)),
    )
    z_grid = np.zeros([xy_field, xy_field])
    for i in range(xy_field):
        for j in range(xy_field):
            z_grid[i][j] = likelihood_ratios_grid[j * xy_field + i]

    x_points, y_points = np.meshgrid(x_points, y_points)

    fig_bound, ax_bound = plt.subplots(figsize=(10, 10))

    ax_bound.scatter(
        [row[0] for row in class_0_samples],
        [row[1] for row in class_0_samples],
        s=1,
        c="blue",
    )
    ax_bound.scatter(
        [row[0] for row in class_1_samples],
        [row[1] for row in class_1_samples],
        s=1,
        c="red",
    )
    ax_bound.set_title("Boundary for Deciding Classes 0 (Blue) and 1 (Red)")
    contour_est = ax_bound.contour(
        x_points, y_points, z_grid, [b_gamma[0]], colors=["lime"], linewidths=2
    )
    contour_theoretical = plot_theo_bound(ax_bound)
    # ontour_est_legend, _ = contour_est.legend_elements()
    # contour_theoretical_legend, _ = contour_theoretical.legend_elements()
    # plt.legend(
    #     [contour_est_legend[0], contour_theoretical_legend[0]],
    #     ["Estimated Boundary", "Theoretical Boundary"],
    #     bbox_to_anchor=(1.1, 1),
    # )
    plt.show()
    print("Albert")


question1_part1(
    q1_p0, q1_p1, q1_a1, q1_mu1, q1_sigma1, q1_a1, q1_mu2, q1_sigma2, q1_mu3, q1_sigma3
)

def estimate_parameters(gausses, sample, inits, threshold):
    #threshold is chosen through trial and error

    #store the samples locally in the functions so I can use them as I see fit
    samps = samples
    #useful variable to have on hand
    num_samps = len(samples)
    #placeholder values that I need to be small
    big_log = -1000000000000
    max_priors   = [0] * gausses
    max_mus      = [0] * gausses
    max_sigmas   = [0] * gausses

    for i in range(inits):
        #TODO: Why do I need to shuffle these
        #random.shuffle(samps)
        #at this point I do not know what the priors are supposed to be, so make them all equal 
        priors = [1 / gausses] * gausses
        #                                                                                 #need axis paramter to get correct mean
        mus = [mean(samps[round(G*num_samps/gausses):round((G+1)*num_samps/gausses-1)],axis=0,) for G in range(gausses)]
        sigmas = [cov(transpose(samps[round(i*num_samps/gausses):round((i+1)*num_samps/gausses-1)])) for G in range(gausses)]

        #Boolean to track if I am done iterating or not
        convergance_huh = False
        while not convergance_huh:
            # generate the liklihoods given an estimate of the Gaussian Varaibles, works for gaussian mixture model
            # G by samples or samples by G
            potenial_class_llhoods = [(multiply(priors[G],multivariate_gaussian_pdf(samps, mus[i], sigmas[i], num_samps)))for G in range(gausses)]
            # sums up the columns of the potential class llhoods because of the axis 0 optional parameter
            pcl_col_sums = sum(potenial_class_llhoods, axis=0)
            # normalized (unsure if mathatically correct term)
            # divides each element by the sum of the column in question 
            class_llhoods_per_samp = [[potenial_class_llhoods[i][j] / pcl_col_sums[j] for j in range(num_samps)]for i in range(gausses)]
            #create new potential prior and means based on the new class llhoods
            new_priors = [mean(class_llhoods_per_samp[i]) for i in range(gausses)]
            new_mus = [divide(sum([multiply(samps[j], class_llhoods_per_samp[i][j])for j in range(num_samps)],axis=0), sum(class_llhoods_per_samp[i])) for i in range(num_gaussians)]
            new_sigmas = [
                add(
                    divide(
                        sum(
                            [multiply(class_llhoods_per_samp[i][j], outer((subtract(samps[j], new_mus[i])),
                                        transpose(subtract(samps[j], new_mus[i])))) for j in range(num_samps)],
                            axis=0,
                        ),
                        sum(class_llhoods_per_samp[i]),
                    ),
                    0.0000000001 * identity(len(samples[0]))) for i in range(gausses)]
            
            # Did we converge on this iteration?
            # basically looking at average error
            if (
                mean(absolute(subtract(new_priors, priors)))
                + mean(absolute(subtract(new_mus, mus)))
                + mean(absolute(subtract(new_sigmas, sigmas)))
                < threshold
            ):
                convergance_huh = True
            # Reassign the prior, mu, and sigmas since we just spent all that time calculating them 
            priors = new_priors
            mus = new_mus
            sigmas = new_sgimas
            print("Albert")

        prob_dens_funcs = zeros(num_samps)

        for G in range(gausses):
            store_pdf = add(
                prob_dens_funcs,
                multiply(
                    priors[G], multivariate_gaussian_pdf(samps, mus[G], sigmas[G])
                ),
            )
            prob_dens_funcs = store_pdf
        log_likelihood = sum(log(prob_dens_funcs))
        #first time through this will trip because of the value chosen to initilize max_log_likelihood
        if log_likelihood > max_log_likelihood:
            max_log_likelihood = log_likelihood
            max_priors = priors
            max_mus = mus
            max_sigmas = sigmas

    return max_priors, max_mus, max_sigmas, max_log_likelihood

print("Hello World!")
