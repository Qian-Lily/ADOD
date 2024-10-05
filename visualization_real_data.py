import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adod import ADOD
from pyod.utils.utility import standardizer
from umap import UMAP

# List of dataset filenames
npz_file_list = [
    'musk_ODDS',  # 3062*166
    'magic_gamma_ADBench',  # 19020*10
    'pima_ODDS',  # 768*8
]

# Iterate through the dataset filenames
for dataset_name in npz_file_list:
    file_path = os.path.join('datasets', dataset_name + '.npz')
    data = np.load(file_path)

    X = data['X']
    y = data['y'].ravel()

    # Remove duplicate items
    X_df = pd.DataFrame(X)
    X_df.drop_duplicates(keep='first', inplace=True)
    X = X_df.values
    y = y[X_df.index]

    n_outliers = np.count_nonzero(y)
    outliers_fraction = n_outliers / len(y)

    X_norm = standardizer(X)

    clf = ADOD(contamination=outliers_fraction)
    clf.fit(X_norm)
    test_scores = clf.decision_scores_

    threshold = np.percentile(test_scores, 100 * (1 - outliers_fraction))
    labels = (test_scores > threshold).astype(int)

    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X)

    # Store UMAP results and labels by dataset name for plotting later
    if dataset_name == 'musk_ODDS':
        X_umap_musk = X_umap
        y_musk = y
        labels_musk = labels
    elif dataset_name == 'magic_gamma_ADBench':
        X_umap_magic_gamma = X_umap
        y_magic_gamma = y
        labels_magic_gamma = labels
    elif dataset_name == 'pima_ODDS':
        X_umap_pima = X_umap
        y_pima = y
        labels_pima = labels

# Set up a 2x3 grid of subplots
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# Plotting for the musk dataset
scatter1 = ax[0, 0].scatter(X_umap_musk[:, 0], X_umap_musk[:, 1], c=y_musk, cmap='bwr', alpha=0.75, s=10)
ax[0, 0].set_title('musk', fontsize=20)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])

scatter2 = ax[1, 0].scatter(X_umap_musk[:, 0], X_umap_musk[:, 1], c=labels_musk, cmap='bwr', alpha=0.75, s=10)
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])

# Plotting for the magic_gamma dataset
scatter3 = ax[0, 1].scatter(X_umap_magic_gamma[:, 0], X_umap_magic_gamma[:, 1], c=y_magic_gamma, cmap='bwr', alpha=0.75,
                            s=10)
ax[0, 1].set_title('magic_gamma', fontsize=20)
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])

scatter4 = ax[1, 1].scatter(X_umap_magic_gamma[:, 0], X_umap_magic_gamma[:, 1], c=labels_magic_gamma, cmap='bwr',
                            alpha=0.75, s=10)
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])

# Plotting for the pima dataset
scatter5 = ax[0, 2].scatter(X_umap_pima[:, 0], X_umap_pima[:, 1], c=y_pima, cmap='bwr', alpha=0.75, s=10)
ax[0, 2].set_title('pima', fontsize=20)
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

scatter6 = ax[1, 2].scatter(X_umap_pima[:, 0], X_umap_pima[:, 1], c=labels_pima, cmap='bwr', alpha=0.75, s=10)
ax[1, 2].set_xticks([])
ax[1, 2].set_yticks([])

# Add legends to the pima plots showing the true and predicted labels
handles, labels = scatter5.legend_elements(prop="colors")
legend5 = ax[0, 2].legend(handles, ["True Inliers", "True Outliers"], prop={'size': 15})
ax[0, 2].add_artist(legend5)

handles, labels = scatter6.legend_elements(prop="colors")
legend6 = ax[1, 2].legend(handles, ["Predicted Inliers", "Predicted Outliers"], prop={'size': 15})
ax[1, 2].add_artist(legend6)

plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.show()
