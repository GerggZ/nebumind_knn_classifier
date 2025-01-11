import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris as load_iris_dataset
from sklearn.model_selection import train_test_split, KFold

from knn_classifier import kNN_Clasifier

# Constant Variables
RANDOM_STATE = 42  # Random number used with the sklearn library to ensure reproducible dataset splits
MAX_NUM_NEIGHBORS = 75  # The maximum number of neighbors (`k`) considered to evaluate the k-NN classifier
MAX_NUM_FOLDS = 10  # The maximum number of folds used for k-fold cross-validation

# Plot Colors
TRAIN_ACCURACY_COLOR = "#E08226"
TEST_ACCURACY_COLOR = "#37A9AB"
OPTIMAL_TEST_ACCURACY_COLOR = "#9467BD"
OPTIMAL_K_COLOR = "#A0A0A0"


def train_and_evaluate_knn_classifier(
        num_neighbors: int,
        train_features: np.ndarray, train_labels: np.ndarray,
        test_features: np.ndarray, test_labels: np.ndarray
) -> tuple[float, float]:
    """
    Trains a k-NN classifier and evaluates it on the train and test datasets.

    Args:
        num_neighbors (int): Number of neighbors for the k-NN classifier.
        train_features (np.ndarray): Features for training.
        train_labels (np.ndarray): Labels for training.
        test_features (np.ndarray): Features for testing.
        test_labels (np.ndarray): Labels for testing.

    Returns:
        tuple[float, float]: Train accuracy and test accuracy.
    """
    # Initialize/train the k-NN Classifier
    knn_classifier = kNN_Clasifier(k=num_neighbors)
    knn_classifier.fit(X_train=train_features, y_train=train_labels)

    # Calculate predictions for training and testing data
    train_predictions = knn_classifier.predict(X=train_features)
    test_predictions = knn_classifier.predict(X=test_features)

    # Calculate the accuracy
    train_accuracy = np.mean(train_predictions == train_labels)
    test_accuracy = np.mean(test_predictions == test_labels)

    return train_accuracy, test_accuracy


def evaluate_knn_accuracy_by_k() -> pd.DataFrame:
    """
    Compares our k-NN classifier accuracy for varying numbers of neighbors (k)

    Returns:
        - pd.DataFrame: Results with the k values and corresponding training and testing accuracies
    """
    # Load Dataset
    iris_dataset = load_iris_dataset()
    features, labels = iris_dataset.data, iris_dataset.target

    # Split dataset into training and testing, using 67% for training
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=0.67, random_state=RANDOM_STATE
    )

    # Analyze kNN Efficacy over a range of nearest neighbors values
    results = []
    for num_neighbors in range(1, MAX_NUM_NEIGHBORS + 1):
        train_accuracy, test_accuracy = train_and_evaluate_knn_classifier(
            num_neighbors, train_features, train_labels, test_features, test_labels
        )
        results.append({
            "Num Neighbors": num_neighbors,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy
        })

    return pd.DataFrame(results)


def cross_validate_knn_with_folds() -> pd.DataFrame:
    """
    Performs k-fold cross-validation for a variety of fold amounts
    Additionally tests each fold amount for a variety of number of neighbors (k)

    Returns:
        - pd.DataFrame: Results with the k values and corresponding training and testing accuracies
    """
    # Load Dataset
    iris_dataset = load_iris_dataset()
    features, labels = iris_dataset.data, iris_dataset.target

    # Analyze kNN efficacy over a range of kFolds
    results = []
    for num_folds in tqdm(range(2, MAX_NUM_FOLDS + 1), total=MAX_NUM_FOLDS - 1, desc="Testing various kFolds and Number of Neighbors amounts"):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)

        for training_indices, testing_indices in kf.split(features):
            train_features, train_labels = features[training_indices], labels[training_indices]
            test_features, test_labels = features[testing_indices], labels[testing_indices]

            # Analyze each kFolds over a range of nearest neighbor values
            for num_neighbors in range(1, MAX_NUM_NEIGHBORS + 1):
                train_accuracy, test_accuracy = train_and_evaluate_knn_classifier(
                    num_neighbors, train_features, train_labels, test_features, test_labels
                )
                results.append({
                    "Num Neighbors": num_neighbors,
                    "Num of Folds": num_folds,
                    "Train Accuracy": train_accuracy,
                    "Test Accuracy": test_accuracy
                })

    results_df = pd.DataFrame(results)

    # Group results by each `Num Folds` and `Num Neighbors` combination
    # Calculate the average training and testing accuracy for each group
    results_folds_averaged_df = results_df.groupby(["Num of Folds", "Num Neighbors"])[["Test Accuracy", "Train Accuracy"]].mean()

    # Regroup results by each `Num Neighbors`
    # Calculate the average training and testing accuracy again for each new group
    results_neighbors_averaged_df = results_folds_averaged_df.groupby("Num Neighbors")[["Test Accuracy", "Train Accuracy"]].mean()

    return results_neighbors_averaged_df.reset_index()


def plot_results(results_step2_df: pd.DataFrame, results_step3_df: pd.DataFrame) -> None:
    """
    Plots k-NN accuracy results for step 2 and step 3

    Args:
        results_step2_df (pd.DataFrame): Results from step 2
        results_step3_df (pd.DataFrame): Results from step 3
    """
    # Calculate the optimal k values (and corresponding test accuracies)
    optimal_k_step2 = results_step2_df.loc[results_step2_df["Test Accuracy"].idxmax(), "Num Neighbors"]
    optimal_k_step3 = results_step3_df.loc[results_step3_df["Test Accuracy"].idxmax(), "Num Neighbors"]
    optimal_accuracy_step2 = results_step2_df.loc[results_step2_df["Num Neighbors"] == optimal_k_step2, "Test Accuracy"].iloc[0]
    optimal_accuracy_step3 = results_step3_df.loc[results_step3_df["Num Neighbors"] == optimal_k_step3, "Test Accuracy"].iloc[0]

    # Plot all the data!
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # 1 row, 2 columns

    axs[0].axhline(
        y=optimal_accuracy_step2, color=OPTIMAL_TEST_ACCURACY_COLOR, linestyle="--", linewidth=1.5,
        label=f"Optimal Test Accuracy {optimal_accuracy_step2*100:.2f} @ k={optimal_k_step2}"
    )
    axs[0].axvline(x=optimal_k_step2, color=OPTIMAL_K_COLOR, linestyle="--", linewidth=1)
    axs[0].plot(
        results_step2_df["Num Neighbors"], results_step2_df["Train Accuracy"],
        label="Train Accuracy", marker="o", markersize=4, color=TRAIN_ACCURACY_COLOR
    )
    axs[0].plot(
        results_step2_df["Num Neighbors"], results_step2_df["Test Accuracy"],
        label="Test Accuracy", marker="o", markersize=4, color=TEST_ACCURACY_COLOR
    )
    axs[0].set_title("k-NN Accuracy (Step 2)", fontsize=14)
    axs[0].set_xlabel("Number of Neighbors (k)", fontsize=11)
    axs[0].set_ylabel("Accuracy", fontsize=11)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].axhline(
        y=optimal_accuracy_step3, color=OPTIMAL_TEST_ACCURACY_COLOR, linestyle="--", linewidth=1.5,
        label=f"Optimal Test Accuracy {optimal_accuracy_step3*100:.2f} @ k={optimal_k_step3}"
    )
    axs[1].axvline(x=optimal_k_step3, color=OPTIMAL_K_COLOR, linestyle="--", linewidth=1)
    axs[1].plot(
        results_step3_df["Num Neighbors"], results_step3_df["Train Accuracy"],
        label="Train Accuracy", marker="o", markersize=4, color=TRAIN_ACCURACY_COLOR
    )
    axs[1].plot(
        results_step3_df["Num Neighbors"], results_step3_df["Test Accuracy"],
        label="Test Accuracy", marker="o", markersize=4, color=TEST_ACCURACY_COLOR
    )
    axs[1].set_title("k-NN Accuracy with K-Fold Cross Validation (Step 3)", fontsize=14)
    axs[1].set_xlabel("Number of Neighbors (k)", fontsize=11)
    axs[1].set_ylabel("Accuracy", fontsize=11)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    step2_results_df = evaluate_knn_accuracy_by_k()
    step3_results_df = cross_validate_knn_with_folds()

    plot_results(step2_results_df, step3_results_df)


if __name__ == "__main__":
    main()
