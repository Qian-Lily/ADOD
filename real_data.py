import os
import pandas as pd
import numpy as np
from pyod.utils.utility import standardizer, precision_n_scores
from sklearn.metrics import roc_auc_score, average_precision_score
from time import time

from adod import ADOD
from adod_original import ADOD_Original

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

# 32 real datasets
npz_file_list = [
    'Hepatitis_ADBench',  # 80*19
    'wine_ODDS',  # 129*13
    'lympho_ODDS',  # 148*18
    'WPBC_ADBench',  # 198*33
    'Stamps_ADBench',  # 340*9
    'WDBC_ADBench',  # 367*30
    'wbc_ODDS',  # 378*30
    'arrhythmia_ODDS',  # 452*274
    'pima_ODDS',  # 768*8
    'vowels_ODDS',  # 1456*12
    'cardio_ODDS',  # 1831*21
    'musk_ODDS',  # 3062*166
    'Waveform_ADBench',  # 3443*21
    'speech_ODDS',  # 3686*400
    'thyroid_ODDS',  # 3772*6
    'PageBlocks_ADBench',  # 5393*10
    'satimage-2_ODDS',  # 5803*36
    'satellite_ODDS',  # 6435*36
    'pendigits_ODDS',  # 6870*16
    'annthyroid_ODDS',  # 7200*6
    'mnist_ODDS',  # 7603*100
    'mammography_ODDS',  # 11183*6
    'magic_gamma_ADBench',  # 19020*10
    'campaign_ADBench',  # 41188*62
    'shuttle_ODDS',  # 49097*9
    'smtp_ODDS',  # 95156*3
    'backdoor_ADBench',  # 95329*196
    'celeba_ADBench',  # 202599*39
    'fraud_ADBench',  # 284807*29
    'cover_ODDS',  # 286048*10
    'census_ADBench',  # 299285*500
    'http_ODDS',  # 567498*3
]

n_ite = 10

# Iterate through each dataset
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
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    X_norm = standardizer(X)

    # Define a dictionary mapping classifier names to their instances
    classifiers = {
        'ADOD': ADOD(contamination=outliers_fraction),
        'ADOD_Original': ADOD_Original(contamination=outliers_fraction),
        'ECOD': ECOD(contamination=outliers_fraction),
        'FastABOD': ABOD(contamination=outliers_fraction, method='fast'),
        'SOS': SOS(contamination=outliers_fraction),
        'KPCA': KPCA(contamination=outliers_fraction),
        'OCSVM': OCSVM(contamination=outliers_fraction),
        'LOF': LOF(contamination=outliers_fraction),
        'COF': COF(contamination=outliers_fraction, method="fast"),
        'kNN': KNN(contamination=outliers_fraction, method='largest'),
        'DIF': DIF(contamination=outliers_fraction),
        'FB': FeatureBagging(contamination=outliers_fraction),
        'LSCP': LSCP(detector_list=[LOF(n_neighbors=15), LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=35)],
                     contamination=outliers_fraction),
        'MO-GAAL': MO_GAAL(contamination=outliers_fraction),
        'ALAD': ALAD(contamination=outliers_fraction),
        'LUNAR': LUNAR(contamination=outliers_fraction),
    }

    results = {clf_name: {'roc': [], 'prn': [], 'pr': [], 'duration': []} for clf_name in classifiers}

    # Process each classifier
    for clf_name, clf in classifiers.items():
        print(f"\n... Processing {dataset_name} using {clf_name} ...")

        # Run 10 iterations for each classifier
        for i in range(n_ite):
            print(f'Iteration {i + 1}')

            # Set the random_state if the classifier supports it
            if hasattr(clf, 'random_state'):
                clf.random_state = i

            t0 = time()
            clf.fit(X_norm)
            test_scores = clf.decision_scores_
            t1 = time()
            duration = round(t1 - t0, ndigits=4)

            # Calculate performance metrics
            roc = roc_auc_score(y, test_scores) if not np.isnan(test_scores).any() else np.nan
            prn = precision_n_scores(y, test_scores) if not np.isnan(test_scores).any() else np.nan
            pr = average_precision_score(y, test_scores) if not np.isnan(test_scores).any() else np.nan

            print('{clf_name} ROC:{roc:.3f}, precision @ rank n:{prn:.3f}, average precision: {pr:.3f}, execution time: '
                  '{duration:.3f}s'.format(clf_name=clf_name, roc=roc, prn=prn, pr=pr, duration=duration))

            # Store results in the dictionary
            results[clf_name]['roc'].append(roc)
            results[clf_name]['prn'].append(prn)
            results[clf_name]['pr'].append(pr)
            results[clf_name]['duration'].append(duration)

        # Calculate mean and standard deviation of results
        mean_roc = np.mean(results[clf_name]['roc'])
        std_roc = np.std(results[clf_name]['roc'])
        mean_prn = np.mean(results[clf_name]['prn'])
        std_prn = np.std(results[clf_name]['prn'])
        mean_pr = np.mean(results[clf_name]['pr'])
        std_pr = np.std(results[clf_name]['pr'])
        mean_duration = np.mean(results[clf_name]['duration'])
        std_duration = np.std(results[clf_name]['duration'])

        print(f'\n{clf_name} Average ROC: {mean_roc:.3f} ± {std_roc:.3f}, '
              f'Average P@N: {mean_prn:.3f} ± {std_prn:.3f}, '
              f'Average AP: {mean_pr:.3f} ± {std_pr:.3f}, '
              f'Average Execution Time: {mean_duration:.3f} ± {std_duration:.3f} seconds')
