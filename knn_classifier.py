import warnings
import numpy as np
from numpy.typing import ArrayLike


class kNN_Classifier:
    def __init__(self, k: int) -> None:
        """
        Initialize the kNN Classifier

        Args:
            - k (int): Number of considered nearest neighbors
        """
        # Validate k
        self.k = validate_k(k)

        self.features = None
        self.labels = None

    def fit(self, features: ArrayLike, labels: ArrayLike) -> None:
        """
        Fit the kNN Classifier with provided training data

        Args:
            - features (ArrayLike): Training features, shape (num_observations, num_variables)
            - labels (ArrayLike): Training labels, shape (num_observations,)

        Raises:
            - Warning: If k is greater than the number of training samples in features.
        """
        # Validate the provided training data (`features` and `labels`)
        self.features, self.labels, self.k = validate_knn_fit_inputs(features, labels, self.k)

    def predict(self, X: ArrayLike) -> list:
        """
        Predict labels for given input data using the kNN classifier

        Args:
            - X (ArrayLike): Input data, shape (num_observations, num_variables)

        Returns:
            - predictions (list): Predicted labels for the provided input data based on training data
        """
        # Validate the provided input data (`X`)
        X = validate_knn_predict_inputs(X, self.features)

        # Calculate the pairwise distances between the input data and the training data
        distances = np.array([np.linalg.norm(self.features - x, axis=1) for x in X])
        # Note: sklearn's pairwise_distances could also be used, other distance metrics could also then be implemented
        # distances = pairwise_distances(X, self.features, metric='euclidean')

        # Find the labels of each k-nearest neighbor for each observation in input X
        k_nearest_indices_sorted = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.labels[k_nearest_indices_sorted]

        # Use majority voting to get the most common nearest neighbor using np.unique()
        predictions = []
        for labels in k_nearest_labels:
            unique_labels, unique_counts = np.unique(labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(unique_counts)]
            predictions.append(most_common_label)
        # Note: using np.bincounts would be more efficient, if np.unique is not a strict requirement:
        # predictions = [np.argmax(np.bincounts(labels)) for labels in k_nearest_labels]

        return predictions


def validate_k(k: int) -> int:
    """
    Validate the value of k (number of nearest neighbors)

    Args:
        - k (int): Number of considered nearest neighbors

    Raises:
        - TypeError: If k is not an integer
        - ValueError: If k is not a positive integer

    Returns:
        - k (int): The input k, but checked to make sure it was a positive integer
    """
    if not isinstance(k, int):
        raise TypeError(f"k must be an integer. Got type {type(k).__name__} instead")

    if k <= 0:
        raise ValueError(f"{k} is not a valid value for k, k must be a positive integer")

    return k


def validate_knn_fit_inputs(features: ArrayLike, labels: ArrayLike, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Validates training data to ensure compatibility
    Converts inputs to NumPy Arrays

    Parameters:
            - features (ArrayLike): Training features, shape (num_observations, num_variables)
            - labels (ArrayLike): Training labels, shape (num_observations,)
            - k (int): Number of considered nearest neighbors
    Returns:
        - (features (np.ndarray), labels (np.ndarray), k (int)): The original features, targets, and k values,
                                                                 but validated and converted to np.ndarrays and an int, respectfully

    Raises:
        - ValueError: If features and labels have mismatched dimensions or invalid shapes
        - ValueError: If k is greater than the number of training samples
    """
    features = np.asarray(features)
    labels = np.asarray(labels)

    # Validate Dimensions
    if len(features) != len(labels):
        raise ValueError(
            f"The number of samples in X and y must be equal. Got X: {features.shape} and y: {labels.shape} instead.")
    if len(features) == 0:
        raise ValueError(f"features must contain at least one observation label pair.")

    if features.ndim != 2:
        raise ValueError(
            f"X must be a 2D array with shape (num_observations, num_variables). Got shape {features.shape} instead.")
    if labels.ndim != 1:
        raise ValueError(f"y must be a 1D array with shape (num_observations,). Got shape {labels.shape} instead.")

    # Validate k size
    if k > len(features):
        warnings.warn(
            f"The number of neighbors (k={k}) is larger than the number of training observations {len(features)}.\n"
            f"To avoid computational errors, setting k to {len(features)}"
        )
        k = len(features)

    return features, labels, k


def validate_knn_predict_inputs(X: ArrayLike, features: np.ndarray) -> np.ndarray:
    """
    Validates input data to ensure compatibility
    Converts input data to NumPy Array

    Parameters:
            - X (ArrayLike): Input data, shape (num_observations, num_variables)
            - features (np.ndarray): Training features, shape (num_observations, num_variables)

    Returns:
        - X (np.ndarray): The input X, but validated converted to a np.ndarray

    Raises:
        - ValueError: If X is not a 2D array or the number of variables for X differs from features
    """
    X = np.asarray(X)

    # Validate dimensions
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array with shape (n_samples, n_features). Got shape {X.shape} instead.")
    if X.shape[1] != features.shape[1]:
        raise ValueError(
            f"Number of variable features in X ({X.shape[1]}) does not match the training data ({features.shape[1]})."
        )

    return X
