# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:22:10 2022

特征子集增强 （PLS-based Feature Subset Augmentation: PLSFSA）
        利用 基于PLS的多扰动集成基因选择 MpEGS-PLS 实现 生成一系列的 特征子集
        这些子集均以测试集的识别性能为标准(如限定前20个基因，topk的最大识别率)
        后继，利用所得大量的特征子集进行挖掘：关联分析，拓扑网络 等
        
Feature Subset Augmentation (PLS-based Feature Subset Augmentation: PLSFSA)
         Using PLS-based multi-perturbation integrated gene selection MpEGS-PLS to generate a series of feature subsets
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

# 读入 matlab 数据 cell 格式
data = sio.loadmat('dat/BreastCancerData.mat');
trn = data['trn']    # 读取字典对象，字典是另一种可变容器模型
ytrn = data['ytrn'].ravel()
tst = data['tst']
ytst = data['ytst'].ravel()

# 数据标准化：先对训练集标准化再将规则用于测试集
# StandardScaler使得经过处理的数据符合标准正态分布，即均值为0，标准差为1
stdScale = StandardScaler().fit(trn)  # 生成标准化规则
trn = stdScale.transform(trn)     # 将规则应用于训练集
tst = stdScale.transform(tst)       # 将规则应用于测试集    
    
# 实例化浅层学习模型：SVM分类器
# clf = LinearDiscriminantAnalysis()
clf = SVC(kernel='linear',C=1.0,probability=True)
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = DecisionTreeClassifier(criterion='gini',max_depth=5)
# clf = RandomForestClassifier(criterion='gini',n_estimators=500)

topk = 10
rnd_time = 20000   # 随机次数，类似产生 购物栏 数目

max_perf = np.zeros((rnd_time,3))
lst_topkid = []

for i in range(rnd_time):
    
    # 特征选择器  (测试我们的算法)
    # idx_feat, vip_feat = plsfrc(trn,ytrn,2)
    idx_feat, w_feat = mpegs_pls(trn,ytrn,2000,2)
    topk_idx = idx_feat[:topk].tolist() # topk 为选取特征的最大个数   
    acc_sel = np.zeros(topk)
    mcc_sel = np.zeros(topk)   
    auc_sel = np.zeros(topk)
         
    for k in range(topk):
        trn_sel_X = trn[:,topk_idx[:k+1]]
        tst_sel_X = tst[:,topk_idx[:k+1]]
        clf.fit(trn_sel_X, ytrn)
        yp_sel = clf.predict(tst_sel_X)
        yprob_sel = clf.predict_proba(tst_sel_X)[:,1]
        # 计算前k个特征的 ACC值，和对应的特征索引
        mcc_sel[k] = metrics.matthews_corrcoef(ytst,yp_sel)        
        acc_sel[k] = metrics.accuracy_score(ytst,yp_sel)      
        auc_sel[k] = metrics.roc_auc_score(ytst,yprob_sel)
    
    max_id = int(np.argmax(mcc_sel))   # 定位出最大值的位置
    max_res = [acc_sel[max_id],mcc_sel[max_id],auc_sel[max_id]]  # 最大值
    max_perf[i,:] = max_res
    
    # 最大值的位置，对应的特征列表，也即 重要特征的列表(索引号)，重要性从大至小
    max_topkid = idx_feat[:max_id+1]
    print('第{}次随机，最大MCC={},\n特征列表：{}'.format(i,max_res[1],max_topkid))
    
    lst_topkid.append(max_topkid)
    # 上面的 lst_topkid (最大ACC对应特征索引列表, 列表格式可用于关联分析)
    
# ## 保存结果
np.savez('Result_BCdat_MpEGS2000_top10_time20000',max_perf=max_perf,lst_topkid=np.array(lst_topkid,dtype=object))

end = time.time()

print('\n CPU运行时间：%s 秒'%(end-start))
