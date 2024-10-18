# -*- coding: utf-8 -*-
"""
PLS-based Feature Subset Augmentation: PLSFSA
Uses PLS-based multi-perturbation integrated gene selection (MpEGS-PLS) to 
generate a series of feature subsets. 
Recognition performance of the test set is used as the evaluation criterion.

Reference:
    W. You, Z. Yang, and G. Ji, PLS-based gene subset augmentation and tumor-specific gene 
    identification, Computers in Biology and Medicine, Volume 174, 2024, 108434, 
    https://doi.org/10.1016/j.compbiomed.2024.108434.
    
@author: wenjie
"""

import scipy.io as sio
import numpy as np
from plsfs import * 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time

def plsgsa(trn, ytrn, tst, ytst, clf, max_k=10, max_nRun=10, nB=2000):
    """
    PLS-based Feature Subset Augmentation (PLSFSA)
    Args:
        trn: Training data
        ytrn: Training labels
        tst: Testing data
        ytst: Testing labels
        clf: Classifier model (e.g., SVC)
        max_k: Maximum number of selected features
        max_nRun: Maximum number of runs
        nB: Number of bootstrap samples for feature selection
    Returns:
        max_perf: Array of best performance metrics (accuracy, MCC, AUC) for each run
        lst_best_ids: List of best feature indices for each run
    """
    max_perf = np.zeros((max_nRun, 3))  # Initialize performance storage
    lst_best_ids = []  # List to store best feature indices

    for nRun in range(max_nRun):
        # Feature selection (MpEGS-PLS)
        idx_feat, w_feat = mpegs_pls(trn, ytrn, nB, 2)
        max_k_idx = idx_feat[:max_k].tolist()  # Top-k feature indices

        # Arrays to store performance metrics
        acc_sel = np.zeros(max_k)
        mcc_sel = np.zeros(max_k)
        auc_sel = np.zeros(max_k)

        for k in range(max_k):
            trn_sel_X = trn[:, max_k_idx[:k+1]]
            tst_sel_X = tst[:, max_k_idx[:k+1]]
            clf.fit(trn_sel_X, ytrn)
            yp_sel = clf.predict(tst_sel_X)
            yprob_sel = clf.predict_proba(tst_sel_X)[:, 1]

            # Calculate performance metrics for selected features
            mcc_sel[k] = metrics.matthews_corrcoef(ytst, yp_sel)
            acc_sel[k] = metrics.accuracy_score(ytst, yp_sel)
            auc_sel[k] = metrics.roc_auc_score(ytst, yprob_sel)

        best_id = int(np.argmax(mcc_sel))  # Find best feature subset
        max_res = [acc_sel[best_id], mcc_sel[best_id], auc_sel[best_id]]
        max_perf[nRun, :] = max_res

        best_ids = idx_feat[:best_id+1]  # Best feature indices
        print(f'Run {nRun}, Best MCC={max_res[1]},\nFeature List: {best_ids}')

        lst_best_ids.append(best_ids)

    return max_perf, lst_best_ids


start = time.time()

# Load data
data = sio.loadmat('dat/BCdat.mat')
trn = data['trn']
ytrn = data['ytrn'].ravel()
tst = data['tst']
ytst = data['ytst'].ravel()

# Data standardization
stdScale = StandardScaler().fit(trn)
trn = stdScale.transform(trn)
tst = stdScale.transform(tst)

# Instantiate classifier (SVM in this case)
clf = SVC(kernel='linear', C=1.0, probability=True)

# Call the plsgsa function. 
# For example, the following settings will generate 20,000 "shopping baskets"
max_perf, lst_best_ids = plsgsa(trn, ytrn, tst, ytst, clf, 
                                max_k=10, max_nRun=20000, nB=2000)

# Save results
np.savez('Result_BCdat_MpEGS2000_top10_time20000', max_perf=max_perf, 
         lst_best_ids=np.array(lst_best_ids, dtype=object))

end = time.time()
print(f'\n CPU运行时间：{end-start} 秒')

