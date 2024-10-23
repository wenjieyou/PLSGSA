# PLS-based Gene (Feature) Subset Augmentation (PLSGSA)

## Overview

This repository implements a **PLS-based Feature Subset Augmentation** algorithm, which uses **PLS-based Multi-Perturbation Ensemble Gene Selection (MpEGS-PLS)** to generate a series of feature subsets. The algorithm evaluates these subsets based on their recognition performance (e.g., MCC, accuracy, and AUC), and extracts the top-performing features for further analysis.

Highlights:

- A novel gene subset augmentation strategy that employs different perturbation mechanisms to identify differentially expressed gene subsets is proposed. The proposed strategy can identify a large number of different gene subsets, surpassing the traditional method of generating single gene subset.

- The multiple sene measurement alsorithm based on partial least squares considers sene-phenotype and sene-sene correlations to a certain extent, which can identify differentially expressed genes, including weak signals genes.

- Gene networks and association analysis are constructed on a large number of subsets obtained by the augmentation algorithm to mine and identify tumor-specific genes. The proposed method can also cope with the challenges of high-dimensional data sets and limited samples in various fields.


### Key Features:
- **PLS-based Feature Selection**: Uses MpEGS-PLS to perturb the training data and select important features.
- **Performance Evaluation**: Selects the top-k features that yield the best classification performance (using metrics like MCC, accuracy, and AUC).
- **Customizable Classifiers**: Supports various machine learning models, including SVM, KNN, Decision Trees, etc.
- **Scalable & Efficient**: For limitted sample, large-scale gene subsets are generated through feature subset augmentation for subsequent analysis and processing (deep learning).


## Installation

To get started, clone this repository:

```bash
git clone https://github.com/wenjieyou/PLSGSA.git
cd PLSFSA
```

## Usage

To run the feature subset augmentation process, you can use the provided script `main_PLSGSA.py`. The script supports standard machine learning workflows for training and testing with feature subset selection.

```bash
python main_PLSGSA.py
```

You can modify the script to fit your data or change the classifier. For example, by default, it uses a linear SVM, but you can easily switch to other classifiers like K-Nearest Neighbors, Decision Trees, or Random Forest.

### Function Breakdown

- **`plsgsa`**: This function handles the core feature selection and evaluation logic. It iterates through multiple runs, selecting top-k features and evaluating them based on the test set's recognition performance.

Example usage of the `plsgsa` function:
```python
max_perf, lst_best_ids = plsgsa(trn, ytrn, tst, ytst, clf, max_k=10, max_nRun=10, nB=2000)
```

### Input Data Format

The script expects the input data. Ensure that your dataset contains the following keys:
- `trn`: Training data matrix
- `ytrn`: Training labels
- `tst`: Testing data matrix
- `ytst`: Testing labels

## Output

The results will be saved in `.npz` format, including:
- **`max_perf`**: A matrix containing the best performance metrics (accuracy, MCC, AUC) for each run.
- **`lst_best_ids`**: A list of the best feature indices for each run, corresponding to the maximum MCC value.

## Example

```python
# Load data
data = sio.loadmat('dat/BCdat.mat')
trn, ytrn, tst, ytst = data['trn'], data['ytrn'].ravel(), data['tst'], data['ytst'].ravel()

# Standardize data
scaler = StandardScaler().fit(trn)
trn, tst = scaler.transform(trn), scaler.transform(tst)

# Instantiate SVM classifier
clf = SVC(kernel='linear', C=1.0, probability=True)

# Run PLS-based feature subset augmentation
max_perf, lst_best_ids = plsgsa(trn, ytrn, tst, ytst, clf, max_k=10, max_nRun=10, nB=2000)

# Save results
np.savez('results.npz', max_perf=max_perf, lst_best_ids=np.array(lst_best_ids, dtype=object))
```
