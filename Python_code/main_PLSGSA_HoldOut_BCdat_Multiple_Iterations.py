# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:22:10 2022

Gene (Feature) Subset Augmentation (PLSGSA)
         Using PLS-based multi-perturbation ensemble gene selection MpEGS-PLS to generate a series of feature subsets
         These subsets are all based on the recognition performance of the test set (such as limiting the top 20 genes, the maximum recognition rate of topk)
         Subsequent, using a large number of feature subsets to mine: association analysis, topological network, etc.
		 
@author: wenjie
"""

import scipy.io as sio
import numpy as np
from plsfs import * 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time

start = time.time()

# Read in matlab data, cell format
data = sio.loadmat('dat/BreastCancerData.mat');
trn = data['trn']    # Read dictionary objects
ytrn = data['ytrn'].ravel()
tst = data['tst']
ytst = data['ytst'].ravel()

# Data standardization: normalize the training set before applying the rules to the test set
# StandardScaler:Make the processed data conform to the standard normal distribution, 
# that is, the mean is 0 and the standard deviation is 1
stdScale = StandardScaler().fit(trn)  # Generate normalization rules
trn = stdScale.transform(trn)     # Apply the rules to the training set
tst = stdScale.transform(tst)       # Apply the rules to the test set    
    
# Instantiating Shallow Learning Models: SVM Classifiers
# clf = LinearDiscriminantAnalysis()
clf = SVC(kernel='linear',C=1.0,probability=True)
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = DecisionTreeClassifier(criterion='gini',max_depth=5)
# clf = RandomForestClassifier(criterion='gini',n_estimators=500)

topk = 10
rnd_time = 20000   # Random number, similar to the number of shopping carts generated

max_perf = np.zeros((rnd_time,3))
lst_topkid = []

for i in range(rnd_time):
    
    # Feature Selector (to test our algorithm)
    # idx_feat, vip_feat = plsfrc(trn,ytrn,2)
    idx_feat, w_feat = mpegs_pls(trn,ytrn,2000,2)
    topk_idx = idx_feat[:topk].tolist() # topk is the maximum number of selected features   
    acc_sel = np.zeros(topk)
    mcc_sel = np.zeros(topk)   
    auc_sel = np.zeros(topk)
         
    for k in range(topk):
        trn_sel_X = trn[:,topk_idx[:k+1]]
        tst_sel_X = tst[:,topk_idx[:k+1]]
        clf.fit(trn_sel_X, ytrn)
        yp_sel = clf.predict(tst_sel_X)
        yprob_sel = clf.predict_proba(tst_sel_X)[:,1]
        # Calculate the ACC value of the first k features, and the corresponding feature index
        mcc_sel[k] = metrics.matthews_corrcoef(ytst,yp_sel)        
        acc_sel[k] = metrics.accuracy_score(ytst,yp_sel)      
        auc_sel[k] = metrics.roc_auc_score(ytst,yprob_sel)
    
    max_id = int(np.argmax(mcc_sel))   # Locate the position of the maximum
    max_res = [acc_sel[max_id],mcc_sel[max_id],auc_sel[max_id]]  # maximum value
    max_perf[i,:] = max_res
    
    # The position of the maximum value, the corresponding feature list, that is, the list of important features (index number), 
    # the importance from large to small.
    max_topkid = idx_feat[:max_id+1]
    print('The {}th random, maximum MCC={},\n feature list: {}'.format(i,max_res[1],max_topkid))
    
    lst_topkid.append(max_topkid)
    # The above lst_topkid (maximum ACC corresponding feature index list, the list format can be used for association analysis)
    
# ## save result
np.savez('Result_BCdat_MpEGS2000_top10_time20000',max_perf=max_perf,lst_topkid=np.array(lst_topkid,dtype=object))

end = time.time()

print('\n CPU running time: %s second'%(end-start))
