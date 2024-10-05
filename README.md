# ADOD

This repository contains the source code for the paper titled "ADOD: Adaptive Density Outlier Detection." It includes implementations of our proposed methods along with 14 comparative baselines using 32 real-world datasets.

## Requirements

Before running the code, ensure you have Python 3.9.19 installed along with the required libraries. You can install all dependencies by running the following command:

```
pip install -r requirements.txt
```

#### Key Dependencies and Their Usage

- **[faiss-gpu](https://github.com/facebookresearch/faiss)**: Used for efficient nearest neighbor search.
- **[umap-learn](https://umap-learn.readthedocs.io/en/latest/)**: Used for dimensionality reduction for visualization of real datasets.
- **[pyod](https://github.com/yzhao062/pyod)**: A comprehensive Python toolkit for detecting outliers in multivariate data. We use 14 outlier detection algorithms from different categories provided by this library as baselines to compare the performance of our proposed methods.

## **Datasets**

We use 32 real-world datasets for our experiments. These datasets are sourced from two repositories:

- **20 datasets** from the [ODDS repository](https://odds.cs.stonybrook.edu/)
- **12 datasets** from the [ADBench repository](https://github.com/Minqi824/ADBench/tree/main)

All datasets are located in the `datasets` folder. The suffix `_ODDS` or `_ADBench` in the dataset filenames indicates their respective sources.

#### Data Preprocessing

- **Deduplication**: Deduplication was performed for datasets containing duplicate entries to ensure data uniqueness.
- **Standardization**: All datasets were standardized before conducting outlier detection. This preprocessing step scales the features to have zero mean and unit variance.

The experiments utilize the entire dataset to ensure comprehensive evaluation of outlier detection performance.

## Implemented Algorithms

- **ADOD**: Our primary algorithm that uses nearest neighbor search, implemented in `adod.py`.
- **ADOD_Original**: Referred to as ADOD* in the paper, this is the original algorithm without nearest neighbor search, implemented in `adod_original.py`.

## Running the Code

### 1. Evaluate on Real Datasets

Run the `real_data.py` script to evaluate our algorithm and 14 baselines on 32 real-world datasets using PRC, P@N, and AP metrics.

```
python real_data.py
```

- All models are employed with their default parameters to ensure a fair comparison. For the LSCP algorithm, we used LOF with $n_neighbors$ set to ${15, 20, 25, 30}$ as the `detector_list`.

- Execution time is measured.

- Each dataset is run ten times with `random_state` set from 0 to 9, reporting the mean and standard deviation of the results.

### 2. Generate Decision Boundary Plots

Run the `synthetic_decision_boundary.py` script to generate decision boundary comparison plots of 15 algorithms on synthetic datasets with three Gaussian clusters of different densities.

```
python synthetic_decision_boundary.py
```

- This script trains the detector using the entire synthetic dataset and then computes decision function values on the entire 2D plane to draw decision boundaries.

### 3. Visualize Real Datasets

Run the `visualization_real_data.py` script to visualize 3 real datasets, using UMAP for dimensionality reduction, and compare true labels with labels predicted by our algorithm.

```
python visualization_real_data.py
```

## Results 

Results related to Average Precision (AP) scores are stored in the `results_ap` folder. This includes:

- AP Scores on real datasets
- Critical Difference Diagrams of AP
- Parameter Sensitivity Analysis of AP
