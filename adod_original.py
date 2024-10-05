import numpy as np
import scipy.stats as st
from numba import njit, prange
from numpy import percentile
from sklearn.utils import check_array


@njit()
def square_dist(x1, x2):
    """
    Calculates the squared Euclidean distance between two vectors.

    Parameters:
    - x1: First vector.
    - x2: Second vector.

    Returns:
    - result: Squared Euclidean distance between the two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return result


@njit(parallel=True)
def square_dist_matrix(X):
    """
    Computes the pairwise squared Euclidean distance matrix for all pairs of rows in X.

    Parameters:
    - X: An array of shape (n_samples, n_features) representing the data.

    Returns:
    - dist_matrix: A symmetric matrix of shape (n_samples, n_samples) where element (i, j) is the
                   squared distance between rows i and j of X.
    """
    n = X.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in prange(n - 1):
        for j in range(i + 1, n):
            dist = square_dist(X[i], X[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


@njit(parallel=True)
def binary_search_perplexity(dissimilarity, perplexity):
    """
    Find the sigma values (scale of the Gaussian kernel) for each data point using binary search.

    Parameters:
    - dissimilarity: Precomputed dissimilarity matrix (squared Euclidean distances).
    - perplexity: desired perplexity which roughly measures the number of effective nearest neighbors

    Returns:
    - sigma: sigma values for each data point.
    """
    n = len(dissimilarity)
    sigma = np.empty(n, dtype=np.float32)
    logU = np.log(perplexity)
    tol = 1e-5

    for i in prange(n):
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf
        Di = np.concatenate((dissimilarity[i, :i], dissimilarity[i, i + 1:]))

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
def compute_mutual_neighbor(dissimilarity, boundary):
    """
    Computes a mutual neighbor graph where an edge exists if two points are mutual neighbor within a specified boundary.

    Parameters:
    - dissimilarity: Precomputed dissimilarity matrix.
    - boundary: neighborhood boundary for each data point.

    Returns:
    - mutual_neighbor: boolean array indicating mutual neighbor graph
    """
    n = dissimilarity.shape[0]
    mutual_neighbor = np.zeros((n, n), dtype=np.bool_)

    for i in prange(n):
        for j in range(i, n):
            if dissimilarity[i, j] <= min(boundary[i], boundary[j]):
                mutual_neighbor[i, j] = True
                mutual_neighbor[j, i] = True

    np.fill_diagonal(mutual_neighbor, False)

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
def compute_outlier_scores(local_density, mutual_neighbor, dissimilarity):
    """
    Computes outlier scores based on the inverse of local density and the weighted average of the inverse
    densities of neighbors, where weights are inversely proportional to the distances.

    Parameters:
    - local_density: local density for each data point.
    - mutual_neighbor: boolean array indicating mutual neighbor graph.
    - dissimilarity: Precomputed dissimilarity matrix.

    Returns:
    - outlier_scores: outlier scores.
    """
    n = len(local_density)
    outlier_scores = np.empty(n, dtype=np.float32)
    local_density_inverse = (1.0 / local_density).astype(np.float32)

    for i in prange(n):
        mutual_neighbors = np.where(mutual_neighbor[i])[0]
        distances = dissimilarity[i, mutual_neighbors]

        valid_mask = distances > 0
        valid_neighbors = mutual_neighbors[valid_mask]
        valid_distances = distances[valid_mask]

        if valid_distances.size > 0:
            weights = 1 / valid_distances
            weights /= np.sum(weights)

            weighted_scores_diff = np.dot(weights, local_density_inverse[valid_neighbors] - local_density_inverse[i])
            outlier_scores[i] = local_density_inverse[i] + weighted_scores_diff
        else:
            outlier_scores[i] = local_density_inverse[i]

    return outlier_scores


class ADOD_Original:
    """Original Adaptive Density Outlier Detection (ADOD) without nearest neighbor search.

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

        dissimilarity = square_dist_matrix(X)
        self.sigma = binary_search_perplexity(dissimilarity, self.perplexity)
        dissimilarity = np.sqrt(dissimilarity)
        self.boundary = st.norm.ppf(self.probability, loc=0, scale=self.sigma)
        mutual_neighbor = compute_mutual_neighbor(dissimilarity, self.boundary)
        self.local_density_ = compute_local_density(mutual_neighbor, self.boundary)
        self.decision_scores_ = compute_outlier_scores(self.local_density_, mutual_neighbor, dissimilarity)

        self.threshold_ = percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        return self
