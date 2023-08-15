# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:42:54 2022

基于 PLS 特征选择 
    1.(弱选择器 plsfrc == plsranking(Matlab))
    2.基于双重扰动(样本放回抽样,特征局部抽样)的集成特征选择--mpegs_pls 
    结果返回每个变量的集成pls-vip值(evip), 注意有两种方法：权值和排名
    其中样本抽样时服从总体中类的分布

@author: wenjie
"""

import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.utils.extmath import randomized_svd
import random
np.seterr(divide='ignore',invalid='ignore')


def mpegs_pls(dat, ylab, nit, nfac):
    """
     基于双重扰动(样本放回抽样,特征局部抽样)的PLS集成特征选择。
     结果返回每个变量的集成vip值(evip),
     其中样本抽样时服从总体中类的分布
    """
    s_num, f_num = dat.shape
    D = int(np.sqrt(f_num))
    
    V = np.zeros((1,f_num))   # 每次基于PLS的VIP系数
    nsel = np.ones((1,f_num)) # number of selected 选中的次数(初始化1保证分母非零)
    
    for k in range(nit):
        # 对特征扰动，从特征空间中随机(部分)抽样，抽样率为sqrt()
        f_sel = random.sample(range(f_num),D)
        # f_sel = f_sel.sort()
        dat_tmp = dat[:,f_sel]
        
        # 对样本扰动，从样本集中依类别分布进行随机放回抽样，
        class_label = np.unique(ylab)    # % 类别标签总数，类别个数
        s_sel= []
        # 抽样保证类平衡, 也即样本放回抽样后的类分布同总体分布
        for i in range(class_label.shape[0]):
            s_tmp = np.argwhere(ylab.reshape(s_num) == class_label[i])
            s_tmp_num = s_tmp.shape[0]
            sid_tmp = np.random.choice(s_tmp.reshape(s_tmp_num), 
                                       s_tmp_num, replace=True)  # 放回抽样
            s_sel = s_sel + sid_tmp.tolist()
            
        X = dat_tmp[s_sel,:]   
        y = ylab[s_sel]
        
        vip = plsvip(X, y, nfac)   # 基于特征权值
        V[:,f_sel] = V[:,f_sel] + vip
        nsel[:,f_sel] = nsel[:,f_sel] + 1
        
    evip = (V/nsel).ravel()   
    w_feat = np.sort(evip)[::-1]
    idx_feat = np.argsort(evip)[::-1]
        
    return idx_feat, w_feat      


def plsfrc(trn, ytrn, nfac):    
    """
    PLSFRC - PLS-based Feature Ranker for Classification
    
    Parameters
    ----------
  TRN  -  training examples
  YTRN - training labels
   NFAC - number of  latent variables (factors), NFAC defaults to 'number of categories'.
   TRN is a data matrix whose rows correspond to points (or observations) and whose
   columns correspond to features (or predictor variables). YTRN is a column vector
   of response values or class labels for each observations in TRN.  TRN and YTRN
   must have the same number of rows.


   IDX_FEAT  -  indices of columns in TRN ordered by feature importance.
   VIP_FEAT - feature weights with large positive weights assigned to important feature.

   Code by: Wenjie, You, 2022.09.22
   
   Example:       
       data = sio.loadmat('dat/BCdat.mat')
       trn = data['trn']    
       ytrn = data['ytrn']
       rank_feat, vip_feat = plsfrc(trn,ytrn,2)
   
   Reference:
   [1] G. Ji, Z. Yang, and W. You, PLS-based Gene Selection and Identification
         of Tumor-Specific Genes, IEEE Transactions on Systems, Man, Cybernetics C,
         Application Review, vol. 41, no. 6, pp. 830-841, Nov. 2011.
   [2] W. You, Z. Yang, and G. Ji, PLS-based Recursive Feature Elimination
         for High-imensional Small Sample.
   [3] https://github.com/rmarkello/pyls.git

    """
        
    m = ytrn.shape[0]
    
    if nfac is None:
        nfac = np.unique(ytrn).shape[0]
        
    class_label = np.unique(ytrn)    # % 类别标签编码(哑变量:类别-1)
    Y = np.zeros((m, class_label.shape[0]-1),dtype=int)    
    for i in range(class_label.shape[0]-1):
        cls_label_vec = np.tile(class_label[i], m)
        Y[:,i] = (ytrn.reshape(m) == cls_label_vec)
        
    X = trn
    pctvar, W = plsreg(X, Y, nfac);        
    vip = X.shape[1]*pctvar[1]@((W**2).T)/np.sum(pctvar[1])
    
    vip_feat = np.sort(vip)[::-1]
    idx_feat = np.argsort(vip)[::-1]
        
    return idx_feat, vip_feat

def plsvip(trn, ytrn, nfac):    
    """    
    利用PLS计算VIP指标，其中ytrn类别标签编码。
    结果返回 vip (每个变量的 vip 值),  
    """    
        
    m = ytrn.shape[0]
    
    if nfac is None:
        nfac = np.unique(ytrn).shape[0]
        
    class_label = np.unique(ytrn)    # % 类别标签编码(哑变量:类别-1)
    Y = np.zeros((m, class_label.shape[0]-1),dtype=int)    
    for i in range(class_label.shape[0]-1):
        cls_label_vec = np.tile(class_label[i], m)
        Y[:,i] = (ytrn.reshape(m) == cls_label_vec)
        
    X = trn
    pctvar, W = plsreg(X, Y, nfac);        
    vip = X.shape[1]*pctvar[1]@((W**2).T)/np.sum(pctvar[1])
    
    return vip



def plsreg(X, Y, ncomp):   
    
    X0 = (X - X.mean(axis=0, keepdims=True))
    Y0 = (Y - Y.mean(axis=0, keepdims=True))       
    dict0 = simpls(X0, Y0, ncomp)
    x_loadings = dict0['x_loadings']
    y_loadings = dict0['y_loadings']
    W = dict0['x_weights']
    # percent variance explained for both X and Y for all components
    pctvar = [np.sum(x_loadings ** 2, axis=0) / np.sum(X0 ** 2),
              np.sum(y_loadings ** 2, axis=0) / np.sum(Y0 ** 2)
              ]
    return pctvar, W


def simpls(X, Y, n_components=None, seed=0):
    """
    Performs partial least squares regression with the SIMPLS algorithm

    Parameters
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is observations and `B` is features
    Y : (S, T) array_like
        Input data matrix, where `S` is observations and `T` is features
    n_components : int, optional
        Number of components to estimate. If not specified will use the
        smallest of the input X and Y dimensions. Default: None
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed to use for random number generation. Helps ensure reproducibility
        of results. Default: None

    Returns
    -------
    results : dict
        Dictionary with x- and y-loadings / x-weights       

    """

    X, Y = np.asanyarray(X), np.asanyarray(Y)
    if n_components is None:
        n_components = min(len(X) - 1, X.shape[1])

    # center variables and calculate covariance matrix
    X0 = (X - X.mean(axis=0, keepdims=True))
    Y0 = (Y - Y.mean(axis=0, keepdims=True))
    Cov = X0.T @ Y0

    # to store outputs
    x_loadings = np.zeros((X.shape[1], n_components))
    y_loadings = np.zeros((Y.shape[1], n_components))
    x_weights = np.zeros((X.shape[1], n_components))
    V = np.zeros((X.shape[1], n_components))

    for comp in range(n_components):
        ci, si, ri = svd(Cov, n_components=1, seed=seed)
        ti = X0 @ ri
        normti = np.linalg.norm(ti)
        x_weights[:, [comp]] = ri / normti        
        ti /= normti        
        x_loadings[:, [comp]] = X0.T @ ti  # == X0.T @ X0 @ x_weights
        qi = Y0.T @ ti
        y_loadings[:, [comp]] = qi 
        
        vi = x_loadings[:, [comp]]
        for repeat in range(2):
            for j in range(comp):
                vj = V[:, [j]]
                vi = vi - ((vj.T @ vi) * vj)
        vi /= np.linalg.norm(vi)
        V[:, [comp]] = vi
   
        Cov = Cov - (vi @ (vi.T @ Cov))
        Vi = V[:, :comp]
        Cov = Cov - (Vi @ (Vi.T @ Cov))
   
    return dict(
        x_weights=x_weights,
        x_loadings=x_loadings,
        y_loadings=y_loadings,
    )


def svd(crosscov, n_components=None, seed=None):
    """
    Calculates the SVD of `crosscov` and returns singular vectors/values

    Parameters
    ----------
    crosscov : (B, T) array_like
        Cross-covariance (or cross-correlation) matrix to be decomposed
    n_components : int, optional
        Number of components to retain from decomposition
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    U : (B, L) `numpy.ndarray`
        Left singular vectors from singular value decomposition
    d : (L, L) `numpy.ndarray`
        Diagonal array of singular values from singular value decomposition
    V : (J, L) `numpy.ndarray`
        Right singular vectors from singular value decomposition
    """

    seed = check_random_state(seed)
    crosscov = np.asanyarray(crosscov)

    if n_components is None:
        n_components = min(crosscov.shape)
    elif not isinstance(n_components, int):
        raise TypeError('Provided `n_components` {} must be of type int'
                        .format(n_components))

    # run most computationally efficient SVD
    if crosscov.shape[0] <= crosscov.shape[1]:
        U, d, V = randomized_svd(crosscov.T, n_components=n_components,
                                 random_state=seed, transpose=False)
        V = V.T
    else:
        V, d, U = randomized_svd(crosscov, n_components=n_components,
                                 random_state=seed, transpose=False)
        U = U.T

    return U, np.diag(d), V

