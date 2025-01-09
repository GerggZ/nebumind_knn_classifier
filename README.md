# k-NN Classifier Implementation

## Overview
This project implements a k-Nearest Neighbors (k-NN) classifier in Python using the `numpy` library. The classifier is evaluated on the Iris dataset from the `sklearn` library and includes k-fold cross-validation for tuning the `k` parameter.

The project is divided into two main components:
1. Implementation of the `k-NN` classifier (`knn_classifier.py`).
2. Evaluation of the classifier and cross-validation (`main.py`).

## Features
1. Custom `k-NN` classifier implemented in `knn_classifier.py` using the `numpy` library.
2. Evaluation of the classifier on the Iris dataset for varying values of `k` in `main.py`.
3. Accuracy measurement for both training and testing datasets.
4. K-fold cross-validation to optimize the choice of `k`.

## Requirements
The project requires the following dependencies:

```plaintext
matplotlib==3.10.0
numpy==2.2.1
pandas==2.2.3
scikit_learn==1.6.0
tqdm==4.67.1
```

## Python Version
This project was made using **Python 3.11**. Ensure you have the correct version installed by running:

```bash
python --version
```
If another version of python must be used, take note that the versioning for the dependencies may change.

## Installation

### Setup
1. **Copy Files**: Download or copy the `knn_classifier.py` and `main.py` files into a directory of your choice.
2. **Install Dependencies**: Ensure you have the required dependencies installed (see project files for details).

## Configuration

The following variables can be modified at the top of the `main.py` file to customize the behavior of the program:

### Constant Variables

- `RANDOM_STATE`: Random number seed used with the `sklearn` library to ensure reproducible dataset splits (default: `42`).
- `MAX_NUM_NEIGHBORS`: The maximum number of neighbors (`k`) considered when evaluating the k-NN classifier (default: `75`).
- `MAX_NUM_FOLDS`: The maximum number of folds used for k-fold cross-validation (default: `10`).

### Plot Colors
Used for visualizing accuracy results in plots:
- `TRAIN_ACCURACY_COLOR`: Color for training accuracy in plots.
- `TEST_ACCURACY_COLOR`: Color for testing accuracy in plots.
- `OPTIMAL_TEST_ACCURACY_COLOR`: Color for marking the optimal testing accuracy in plots.
- `OPTIMAL_K_COLOR`: Color for marking the optimal value of `k` in plots.

## Usage

### Step 1: Develop a k-NN Classifier (`knn_classifier.py`)
- The `knn_classifier` function:
  - Accepts the following inputs:
    - `train_features`: Feature set of the training data.
    - `class_labels`: Class labels of the training data.
    - `test_features`: Features of the testing dataset.
    - `num_neighbors`: Size of the neighborhood (`k`).
  - Implements:
    - Neighbor identification using `numpy.argsort` and `numpy.linalg.norm`.
    - Class predictions via majority voting using `numpy.unique`.

### Step 2: Evaluate the Classifier (`main.py`)
- The `main.py` script contains the `evaluate_classifier` function, which:
  1. Uses the Iris dataset to evaluate the classifier for varying values of `k`.
  2. Measures accuracy for both the training and testing datasets.

### Step 3: Perform Cross-Validation (`main.py`)
- The `main.py` script also includes the `perform_cross_validation` function, which:
  1. Implements k-fold cross-validation to tune the value of `k`.
  2. Divides training data into `k` folds.
  3. Uses `k-1` folds for training and the remaining fold for validation.
  4. Averages accuracy across folds to determine the optimal value of `k`.

### Example Execution
When you run `main.py`, it will:
- Import the k-nn classifer `knn_classifer` class function from `knn_classifier.py` (Step 1)
- Execute the `evaluate_classifier` function (Step 2)
  - Evaluate the classifier over a range of `k` values
- Execute the `perform_cross_validation` function (Step 3)
  - Evaluate the classifier over a range of `k` values
  - Evaluate the classifer over a range of fold amounts and fold permutations
- Visualize the results in `plot_results` using the `matplotlib` library

## Contact
For questions or suggestions, please contact [gzinniel@gmail.com](mailto:gzinniel@gmail.com).

