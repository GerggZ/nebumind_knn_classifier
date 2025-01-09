import warnings
import numpy as np
from numpy.typing import ArrayLike


class kNN_Clasifier:
    def __init__(self, k: int) -> None:
        """
        Initialization for a kNN classifier

        Args:
            - k (int): Number of considered nearest neighbors

        Raises:
            - ValueError: If k is not a positive integer
            - TypeError: If k is not of type int
        """
        # Validate the value provided for k (`Number of Neighbors`)
        try:
            k = int(k)
        except (TypeError, ValueError):
            raise TypeError(f"k must be an integer or convertible to an integer; got {type(k).__name__} instead")
        if k <= 0:
            raise ValueError(f"{k} is not a valid value for k, k must be a positive integer")
        self.k = k

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """
        Fitting function for the kNN classifier

        Args:
            - X_train (ArrayLike): Training features (num_observations, num_variables)
            - y_train (ArrayLike): Training labels (num_observations,)

        Raises:
            - ValueError: If X_train and y_train have different numbers of observations.
            - ValueError: If X_train contains no observations (i.e., has a length of 0)
            - ValueError: If X_train is not a 2D array or y_train is not a 1D array
            - Warning: If k is greater than the number of training samples in X_train.
        """
        # Validate the provided training data (`X_train` and `y_train`)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if len(X_train) != len(y_train):
            raise ValueError(f"The number of samples in X and y must be equal. Got X: {X_train.shape} and y: {y_train.shape} instead.")
        if len(X_train) == 0:
            raise ValueError(f"X_train must contain at least one observation label pair.")
        if X_train.ndim != 2:
            raise ValueError(f"X must be a 2D array with shape (num_observations, num_variables). Got shape {X_train.shape} instead.")
        if y_train.ndim != 1:
            raise ValueError(f"y must be a 1D array with shape (num_observations,). Got shape {y_train.shape} instead.")
        if self.k > len(X_train):
            warnings.warn(
                f"The number of neighbors (k={self.k}) is larger than the number of training observations {len(X_train)}.\n"
                f"To avoid computational errors, setting k to {len(X_train)}"
            )
            self.k = len(X_train)

        # Assign training data
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X: ArrayLike) -> list:
        """
        Prediction for the kNN classifier

        Args:
            - X (ArrayLike): Data to be fitted (num_observations, num_variables)

        Returns:
            - predictions (list): Assumed labels for the provided data based on the fitted data (X_train, y_train)

        Raises:
            - ValueError: If X is not a 2D array
            - ValueError: If the number of observations for X differs from X_train
        """
        # Validate the provided data (`X`)
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array with shape (n_samples, n_features). Got shape {X.shape} instead.")
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match the training data ({self.X_train.shape[1]})."
            )

        # Calculate all of the pairwise distances between the input data and the training data
        distances = np.array([np.linalg.norm(self.X_train - x, axis=1) for x in X])
        # Note: sklearn's pairwise_distances could also be used and provide more versatility
        # distances = pairwise_distances(X, self.X_train, metric='euclidean')

        # Find the labels of each k-nearest neighbor for each observation in X
        k_nearest_indices_sorted = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices_sorted]

        # Use majority voting to get the most common nearest neighbor using np.unique()
        predictions = []
        for labels in k_nearest_labels:
            unique_labels, unique_counts = np.unique(labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(unique_counts)]
            predictions.append(most_common_label)

        # Note: using np.bincounts or collections.Counter may be more efficient than using np.unique:
        # predictions = [Counter(labels).most_common(1)[0][0] for labels in k_nearest_labels]
        # predictions = [np.argmax(np.bincounts(labels)) for labels in k_nearest_labels]
        return predictions

