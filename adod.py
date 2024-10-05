import numpy as np
import scipy.stats as st
from numba import njit, prange
import faiss
from numpy import percentile
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


@njit(parallel=True)
def binary_search_perplexity_knn(knn_dists, perplexity):
    """
    Efficiently find the sigma values (scale of the Gaussian kernel) for each data point using binary search.

    Parameters:
    - knn_dists: distances to k-nearest neighbors for each data point.
    - perplexity: desired perplexity which roughly measures the number of effective nearest neighbors.

    Returns:
    - sigma: sigma values for each data point.
    """
    n = len(knn_dists)
    sigma = np.empty(n, dtype=np.float32)
    logU = np.log(perplexity)
    tol = 1e-5

    for i in prange(n):
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf
        Di = knn_dists[i, 1:]

        for iter in range(50):
            P = np.exp(-Di * beta)
            sumP = np.sum(P)
            H = np.log(sumP) + beta * np.sum(Di * P) / sumP
            Hdiff = H - logU

            if np.abs(Hdiff) <= tol:
                break
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf or betamax == -np.inf:
                    beta = beta * 2.
                else:
                    beta = (beta + betamax) / 2.
            else:
                betamax = beta
                if betamin == np.inf or betamin == -np.inf:
                    beta = beta / 2.
                else:
                    beta = (beta + betamin) / 2.

        sigma[i] = np.sqrt(1 / beta)

    return sigma


@njit(parallel=True)
def compute_mutual_neighbor_knn(knn_dists, knn_indices, boundary):
    """
    Computes a mutual neighbor graph where an edge exists if two points are mutual neighbor within a specified boundary.

    Parameters:
    - knn_dists: distances to k-nearest neighbors for each data point.
    - knn_indices: indices to k-nearest neighbors for each data point.
    - boundary: neighborhood boundary for each data point.

    Returns:
    - mutual_neighbor: boolean array indicating mutual neighbor graph.
    """
    n = len(knn_dists)
    mutual_neighbor = np.zeros((n, n), dtype=np.bool_)

    for i in prange(n):
        for j_index in range(knn_dists.shape[1]):
            j = knn_indices[i, j_index]
            if knn_dists[i, j_index] <= min(boundary[i], boundary[j]):
                mutual_neighbor[i, j] = True
                mutual_neighbor[j, i] = True
            elif knn_dists[i, j_index] > boundary[i]:
                break

    np.fill_diagonal(mutual_neighbor, False)

    return mutual_neighbor


@njit(parallel=True)
def compute_mutual_neighbor_knn_new(knn_dists, knn_indices, boundary_new, boundary):
    """
    Computes a mutual neighbor graph for new data points in relation to existing data points.

    Parameters:
    - knn_dists: distances from a new point to its k-nearest neighbors in the existing data.
    - knn_indices: indices from a new point to its k-nearest neighbors in the existing data.
    - boundary_new: neighborhood boundary for each new data point.
    - boundary: neighborhood boundary for each existing data point.

    Returns:
    - mutual_neighbor: boolean array indicating mutual neighbor graph between new and existing data points.
    """
    n_new = len(knn_dists)
    n_exist = len(boundary)
    mutual_neighbor = np.zeros((n_new, n_exist), dtype=np.bool_)

    for i in prange(n_new):
        for j_index in range(knn_dists.shape[1]):
            j = knn_indices[i, j_index]
            if knn_dists[i, j_index] <= min(boundary_new[i], boundary[j]):
                mutual_neighbor[i, j] = True
            elif knn_dists[i, j_index] > boundary_new[i]:
                break

    return mutual_neighbor


@njit(parallel=True)
def compute_local_density(mutual_neighbor, boundary):
    """
    Calculates the local density for each point based on the count of mutual neighbors within boundary.

    Parameters:
    - mutual_neighbor: boolean array indicating mutual neighbor graph.
    - boundary: neighborhood boundary for each data point.

    Returns:
    - local_density: local density for each data point.
    """
    n = len(boundary)
    local_density = np.empty(n, dtype=np.float32)

    for i in prange(n):
        n_neighbors = np.sum(mutual_neighbor[i])
        local_density[i] = (n_neighbors + 1) / boundary[i]
    return local_density


@njit(parallel=True)
def compute_outlier_scores_knn(local_density, mutual_neighbor, knn_dists, knn_indices):
    """
    Computes outlier scores based on the inverse of local density and the weighted average of the inverse
    densities of neighbors, where weights are inversely proportional to the distances.

    Parameters:
    - local_density: local density for each data point.
    - mutual_neighbor: boolean array indicating mutual neighbor graph.
    - knn_dists: distances to k-nearest neighbors for each data point.
    - knn_indices: indices to k-nearest neighbors for each data point.

    Returns:
    - outlier_scores: outlier scores.
    """
    n = len(local_density)
    outlier_scores = np.empty(n, dtype=np.float32)
    local_density_inverse = (1.0 / local_density).astype(np.float32)

    for i in prange(n):
        mutual_neighbors = np.where(mutual_neighbor[i])[0]

        valid_neighbors = []
        distances = []

        for neighbor in mutual_neighbors:
            index_in_knn_array = np.where(knn_indices[i] == neighbor)[0]

            if index_in_knn_array.size > 0:
                index_in_knn = index_in_knn_array[0]
                distance = knn_dists[i][index_in_knn]

                if distance > 0:
                    valid_neighbors.append(neighbor)
                    distances.append(distance)

        if len(distances) > 0:
            distances = np.array(distances)

            weights = 1 / distances
            weights /= np.sum(weights)

            valid_neighbors = np.array(valid_neighbors)
            weighted_scores_diff = np.dot(weights, local_density_inverse[valid_neighbors] - local_density_inverse[i])
            outlier_scores[i] = local_density_inverse[i] + weighted_scores_diff
        else:
            outlier_scores[i] = local_density_inverse[i]

    return outlier_scores


@njit(parallel=True)
def compute_outlier_scores_knn_new(local_density_new, local_density_exist, mutual_neighbor_new, knn_dists, knn_indices):
    """
    Computes outlier scores for new data points relative to existing data points by utilizing the local density
    differences and distance-weighted averages.

    Parameters:
    - local_density_new: local density for each new data point.
    - local_density_exist: local density for existing data points.
    - mutual_neighbor_new: boolean array indicating mutual neighbor graph between new and existing data points.
    - knn_dists: distances from a new point to its k-nearest neighbors in the existing data.
    - knn_indices: indices from a new point to its k-nearest neighbors in the existing data.

    Returns:
    - outlier_scores: outlier scores for new data points.
    """
    n = len(local_density_new)
    outlier_scores = np.zeros(n, dtype=np.float32)

    local_density_new_inverse = (1.0 / local_density_new).astype(np.float32)
    local_density_exist_inverse = (1.0 / local_density_exist).astype(np.float32)

    for i in prange(n):
        mutual_neighbors = np.where(mutual_neighbor_new[i])[0]

        valid_neighbors = []
        distances = []

        for neighbor in mutual_neighbors:
            index_in_knn_array = np.where(knn_indices[i] == neighbor)[0]

            if index_in_knn_array.size > 0:
                index_in_knn = index_in_knn_array[0]
                distance = knn_dists[i][index_in_knn]

                if distance > 0:
                    valid_neighbors.append(neighbor)
                    distances.append(distance)

        if len(distances) > 0:
            distances = np.array(distances)

            weights = 1 / distances
            weights /= np.sum(weights)

            valid_neighbors = np.array(valid_neighbors)
            weighted_scores_diff = np.dot(weights,
                                          local_density_exist_inverse[valid_neighbors] - local_density_new_inverse[i])
            outlier_scores[i] = local_density_new_inverse[i] + weighted_scores_diff
        else:
            outlier_scores[i] = local_density_new_inverse[i]
    return outlier_scores


class ADOD:
    """Adaptive Density Outlier Detection (ADOD).

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, contamination=0.1, probability=0.999, perplexity=None):
        self.probability = probability
        self.contamination = contamination
        self.perplexity = perplexity

    def fit(self, X):
        """
        Fits the ADOD model to the provided dataset X.

        Parameters:
        - X: data to fit the model

        Returns:
        - self: the instance itself
        """
        X = check_array(X)
        n, d = X.shape

        if self.perplexity is None:
            self.perplexity = int(np.sqrt(n)) * 2

        self.n_neighbors = min(n, 3 * self.perplexity, 2048)

        # Initialize and fill a FAISS index for fast nearest neighbor search
        gpures = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        self.index = faiss.GpuIndexFlatL2(gpures, d, flat_config)
        self.index.add(X.astype('float32'))

        # Find k-nearest neighbors
        knn_dists, knn_indices = self.index.search(X, self.n_neighbors)

        self.sigma = binary_search_perplexity_knn(knn_dists, self.perplexity)
        knn_dists = np.sqrt(knn_dists)
        self.boundary = st.norm.ppf(self.probability, loc=0, scale=self.sigma)
        mutual_neighbor = compute_mutual_neighbor_knn(knn_dists, knn_indices, self.boundary)
        self.local_density_ = compute_local_density(mutual_neighbor, self.boundary)
        self.decision_scores_ = compute_outlier_scores_knn(self.local_density_, mutual_neighbor, knn_dists, knn_indices)

        # Determine the threshold for labeling outliers
        self.threshold_ = percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        return self

    def decision_function(self, X):
        """
        Applies the ADOD model to new data X to compute the outlier decision function.

        Parameters:
        - X: new data for which to compute the decision function.

        Returns:
        - decision_scores_new: outlier scores for the new data.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        # Perform search for the nearest neighbors in the new data
        knn_dists, knn_indices = self.index.search(X, self.n_neighbors)

        sigma_new = binary_search_perplexity_knn(knn_dists, self.perplexity)
        knn_dists = np.sqrt(knn_dists)
        boundary_new = st.norm.ppf(self.probability, loc=0, scale=sigma_new)
        mutual_neighbor_new = compute_mutual_neighbor_knn_new(knn_dists, knn_indices, boundary_new, self.boundary)
        local_density_new = compute_local_density(mutual_neighbor_new, boundary_new)
        decision_scores_new = compute_outlier_scores_knn_new(local_density_new, self.local_density_,
                                                             mutual_neighbor_new, knn_dists, knn_indices)

        return decision_scores_new

    def predict(self, X):
        """
        Predicts labels for the new data X using the fitted ADOD model.

        Parameters:
        - X: new data for which to predict labels.

        Returns:
        - labels: predicted labels (0 for inliers, 1 for outliers).
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        pred_score = self.decision_function(X)

        if isinstance(self.contamination, (float, int)):
            labels = (pred_score > self.threshold_).astype('int').ravel()

        return labels
