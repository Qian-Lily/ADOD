import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import make_blobs

from adod import ADOD

# Probabilistic-based
from pyod.models.ecod import ECOD
from pyod.models.abod import ABOD
from pyod.models.sos import SOS

# Linear Model-based
from pyod.models.kpca import KPCA
from pyod.models.ocsvm import OCSVM

# Proximity-based
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.knn import KNN

# Ensembles-based
from pyod.models.dif import DIF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lscp import LSCP

# Neural Networks-based
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.alad import ALAD
from pyod.models.lunar import LUNAR

# Set random seed for reproducibility
np.random.seed(42)
random_state = 42

# Generate synthetic data: 3 Gaussian blobs with varying densities
n_samples = 500
outliers_fraction = 0.15

# Create meshgrid for plotting decision functions
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))

# Calculate the number of inliers and outliers
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1

# Define centers and standard deviations for the Gaussian blobs
centers = [[-3.6, -2.6], [0, 2], [3.5, -1.5]]
cluster_std = [0.6, 1.2, 0.3]

# Generate inliers using make_blobs
X, y = make_blobs(n_samples=n_inliers, centers=centers, cluster_std=cluster_std, random_state=random_state)

# Generate uniform random outliers
X_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, 2))
X = np.vstack([X, X_outliers])

# Print data statistics
print('Number of inliers: %i' % n_inliers)
print('Number of outliers: %i' % n_outliers)
print('Ground truth shape is {shape}. Outlier are 1 and inliers are 0.\n'.format(shape=ground_truth.shape))
print(ground_truth, '\n')

# Define a dictionary of outlier detection models
classifiers = {
    'Adaptive Density-based Outlier Detection (ADOD)': ADOD(contamination=outliers_fraction),
    'Empirical Cumulative Distribution Functions (ECOD)': ECOD(
        contamination=outliers_fraction),
    'Fast Angle-based Outlier Detector (FastABOD)': ABOD(
        contamination=outliers_fraction, method='fast'),
    'Stochastic Outlier Selection (SOS)': SOS(contamination=outliers_fraction),
    'Kernel Principal Component Analysis (KPCA)': KPCA(
        contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Local Outlier Factor (LOF)': LOF(contamination=outliers_fraction),
    'Connectivity-based Outlier Factor (COF)': COF(contamination=outliers_fraction, method="fast"),
    'K Nearest Neighbors (kNN)': KNN(contamination=outliers_fraction, method='largest'),
    'Deep Isolation Forest (DIF)': DIF(contamination=outliers_fraction, random_state=random_state),
    'Feature Bagging (FB)': FeatureBagging(contamination=outliers_fraction, random_state=random_state),
    'Locally Selective Combination of Parallel (LSCP)': LSCP(
        detector_list=[LOF(n_neighbors=15), LOF(n_neighbors=20),
                       LOF(n_neighbors=25), LOF(n_neighbors=35)], contamination=outliers_fraction,
        random_state=random_state),
    'Multiple-Objective Generative Adversarial Active Learning (MO-GAAL)': MO_GAAL(
        contamination=outliers_fraction),
    'Adversarially learned anomaly detection (ALAD)': ALAD(contamination=outliers_fraction),
    'Unifying Local Outlier Detection Methods via Graph Neural Networks (LUNAR)': LUNAR(
        contamination=outliers_fraction),
}

# Display all detectors
for i, clf in enumerate(classifiers.keys()):
    print('Model', i + 1, clf)

# Fit the models with the generated data and compare model performances
plt.figure(figsize=(15, 20))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    print()
    print(i + 1, 'fitting', clf_name)

    clf.fit(X)
    scores_pred = clf.decision_scores_ * -1
    y_pred = clf.labels_

    # Calculate the decision threshold based on the outlier fraction
    threshold = percentile(scores_pred, 100 * outliers_fraction)

    # Count the number of misclassified points
    n_errors = (y_pred != ground_truth).sum()

    # Compute the decision function for each point on the meshgrid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)

    # Create a subplot for each classifier
    subplot = plt.subplot(5, 3, i + 1)

    if threshold > Z.min():
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)
    else:
        pass

    # Draw a contour line at the threshold which separates inliers from outliers
    a = subplot.contour(xx, yy, Z, levels=[threshold],
                        linewidths=2, colors='red')
    if threshold < Z.max():
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    else:
        pass

    # Plot white points for true inliers
    b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white', s=20, edgecolor='k')
    # Plot black points for true outliers
    c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black', s=20, edgecolor='k')

    subplot.axis('tight')
    subplot.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Add legend to the last subplot for clarity
    if i == len(classifiers) - 1:
        subplot.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=16),
            loc='lower right')

    clf_name_in_brackets = clf_name[clf_name.find("(") + 1:clf_name.find(")")]
    subplot.set_xlabel("%s (errors: %d)" % (clf_name_in_brackets, n_errors), fontsize=18)
    subplot.set_xlim((-7, 7))
    subplot.set_ylim((-7, 7))

plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04, wspace=0.02, hspace=0.1)
plt.show()
