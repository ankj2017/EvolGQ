 import numpy as np
 from numpy import linalg as LA
 import pandas as pd
 import statsmodels.api as sm
 from statsmodels.formula.api import ols
 import math
 import warnings
 import statistics 
 def lm (xrow, xcol, yrow, ycol, array):
     df = pd.DataFrame(array)
     return ols(y=df.iloc[yrow,ycol], x=df.iloc[xrow,xcol])
 def lapply(lista, func):
     return list(map(func = func, iterables = lista))
 def AlphaRep (cor_matrix, sample_size):
     if (np.sum(cor_matrix.diagonal())!= cor_matrix.shape[0]):
         raise Exception ("Matrices do not appear to be correlation matrices.")
    vec = np.tril(cor_matrix)
    var_erro = (1-vec.mean()**2)/(sample_size-2)
    var_vec = np.var(vec)
    return (var_vec - var_erro)/var_vec
def cov2cor(covariance):
    vec = np.sqrt(np.diag(covariance))
    outer_vec = np.outer(vec, vec)
    correlation = covariance / outer_vec
    correlation[covariance == 0] = 0
    return correlation
def is_symmetric(a):
    return np.all(np.abs(a-a.T) ==0)
def BootstrapRep (ind_data, ComparisonFunc, iterations=1000, sample_size = ind_data.shape[0],correlation = False,
                 parallel = False):
    if (correlation):
        StatFunc = np.corrcoef
        c2v = cov2cor
    else:
        StatFunc = np.cov
        c2v = lambda x: x
    repeatability = BootstrapStrat(ind_data, iterations, ComparisonFunc = lambda x,y: ComparisonFunc(c2v(x), 
                                                                                c2v(y)),
                                    StatFunc = StatFunc, sample_size = sample_size, parallel = parallel)
    return (repeatability[:,2].mean())
def BootstrapStrat (ind_data, iterations,
                           ComparisonFunc, 
                           StatFunc, sample_size = ind_data.shape[0], parallel = False):
    if (is_symmetric(ind_data)):
        raise Exception("input appears to be a matrix, use residuals.")
    c_matrix = StatFunc(ind_data)
    n_ind = ind_data.shape[0]
    populations = list(map(lambda x: ind_data[np.random.choice(1:n_ind, sample_size, replace=True)]),1:iterations)
    comparisons = pd.DataFrame(list(map(lambda x: doComparisonBS(x,c_matrix, ComparisonFunc, StatFunc, ind_data, sample_size, n_ind),populations)))                                                                   
    return comparisons
def doComparisonBS(x,c_matrix, ComparisonFunc, StatFunc, ind_data, sample_size, n_ind):
    while (True):
        try:
            out = ComparisonFunc (c_matrix, StatFunc(x))
        except:
            print("Error in BootStrap sample, trying again")
        if (True in np.isnan(np.array(out))):
            x = ind_data[np.random.choice(1:n_ind, sample_size, replace = True)]
        else:
            break
    return out
def BootstrapR2 (ind_data, iterations=1000):
    it_r2 = BootstrapStrat(ind_data, iterations, ComparisonFunc = (lambda x,y: y),StatFunc = lambda x: CalcR2(np.corrcoef(x)),parallel = False)
    return it_r2[:,2]
def CalcAVG (cor_hypothesis, cor_matrix, MHI = True, landmark_dim = None):
    if (np.array_equal(cor_hypothesis, cor_hypothesis.astype(bool))):
        raise Exception("modularity hypothesis matrix should be binary")
    if (landmark_dim is not None):
        if (landmark_dim is not 2 or landmark_dim is not 3):
            raise Exception("landmark_dim should be either 2 or 3 dimensions")
        num_traits = cor_matrix.shape[0]
        n_land = num_traits/landmark_dim
        withinLandMat = CreateWithinLandMat(n_land, landmark_dim)
        cor_hypothesis[withinLandMat]=2
    index = np.tril(cor_hypothesis)
    avg_plus = np.mean(np.tril(cor_matrix)[index==1])
    avg_minus = np.mean(np.tril(cor_matrix)[index==0])
    if (MHI):
        avg_index = (avg_plus-avg_minus)/CalcEigenVar(cor_matrix, sd = True, rel= False)
        output = [avg_plus, avg_minus, avg_index]
    else:
        avg_ratio = avg_plus/avg_minus
        output = [avg_plus, avg_minus, avg_ratio]
    if (landmark_dim is not None):
        output.append(np.mean(np.tril(cor_matrix)[index==2]))
    return output
def CalcEigenVar(matrix, sd = False, rel = True, sample = None):
    if not is_symmetric(matrix)):
        raise Exception("covariance matrix must be symmetric.")
    eigenv = LA.eig(matrix)[0]
    m = np.mean(eigenv)
    n = matrix.shape[0]
    sqrd = (eigenv-m)**2
    obs = sum(sqrd)/n
    max = (n-1)*sum(eigenv)**2/(n**2)
    if sample is not None:
        obs = obs - (max/sample)
        max = max - (max/sample)
    if (sd):
        obs = math.sqrt(obs)
        max = math.sqrt(max)
    if (rel):
        Evar = obs/max
    else:
        Evar = obs
    return Evar
def CalcICV(cov_matrix):
    if not is_symmetric(matrix)):
        raise Exception("covariance matrix must be symmetric.")
    if (np.sum(cor_matrix.diagonal())== cor_matrix.shape[0]):
        warnings.warn("Matrix appears to be a correlation matrix! Only covariance matrices should be used for ICV.")
        eVals = LA.eig(cov_matrix)[0]
        ICV = np.std(eVals)/np.mean(eVals)
        return ICV
def CalcR2(c_matrix):
    cor_matrix = cov2cor(c_matrix)
    return np.mean(LA.matrix_power(np.tril(cor_matrix),2))

def CalcR2CVCorrected (ind_data, cv_level = 0.06, iterations = 1000, **kwargs, default=True):
    if default:
        cv = lambda x: np.std(x)/np.mean(x)
        def Stats (x):
            cov_matrix = np.var(x)
            cor_matrix = cov2cor(cov_matrix)
            return [CalcR2(cor_matrix),cv(LA.eig(cov_matrix)[0]),np.mean(np.apply_along_axis(cv, 1, x))]
        it_stats = BootstrapStrat(ind_data, iterations, ComparisonFunc = lambda x,y: y, StatFunc = Stats)[:,-1]
        lm_r2 = lm(xrow = slice(len(it_stats)),xcol = 3,yrow = slice(len(it_stats)),ycol = 1,array=ind_data )
        lm_evals_cv = lm(xrow = slice(len(it_stats)),xcol = 3,yrow = slice(len(it_stats)),ycol = 2,array=ind_data )
        adjusted_r2 = lm_r2.params + [1,cv_level]
        adjusted_evals_cv = lm_evals_cv.params+[1,cv_level]
        adjusted_integration = adjusted_r2+adjusted_evals_cv
        models = [lm_r2, lm_evals_cv]
        return [adjusted_integration,models,it_stats]
    else:
        cv = lambda x: np.std(x)/np.mean(x)
        return 0
        #finish this func
def CalcRepeatability (ID, ind_data):
    models_list = np.apply_along_axis(lambda vec: ols(y = vec, x = pd.factorize(pd.DataFrame(ID))),axis=1, ind_data)
    models_list  = lapply(models_list,sm.stats.anova_lm())
    def rep_itself (lm_model):
        msq = lm_model.mean_sq
        s2a = (msq[0]-msq[1])/2
        return s2a/(s2a+msq[1])
    return lapply(models_list,rep_itself)
def CalculateMatrix(linear_m):
    cov_matrix = np.var(linear_m.resid)*((linear_m.resid.shape[0]-1)/linear_m.df_resid)
    return cov_matrix
def BayesianCalculateMatrix(linear_m, samples=None, nu=None, S_0=None, **kwargs):
    return 0
def ComparisonMap (matrix_list, MatrixCompFunc, repeat_vector=None, **kwargs):
    n_matrix = len(matrix_list)
    return 0
def CreateHypotMatrix (modularity_hypot):
    if modularity_hypot.shape is None:
        return np.outer(modularity_hypot, modularity_hypot)
    num_hyp = modularity_hypot.shape[1]
    num_traits = modularity_hypot.shape[0]
    hyp_list_func = np.vectorize(lambda x: np.outer(x,x))
    m_hyp_list = hyp_list_func(modularity_hypot)
    #finish this
#finish CombineHypot, Partition2Hypot
#DriftTest
#EigenTensorDecomp
def ExtendMatrix(cov_matrix, var_cut_off = 1e-4, ret_dim = None):
    p = cov_matrix.shape[0]
    if (p<10):
        warnings.warn("matrix is too small")
    eigen_cov_matrix = LA.eigen(cov_matrix)
    eVal, eVec = eigen_cov_matrix
    grad = np.empty(p-2)
    tr_cov_matrix = sum(eVal)
    for (i in range(p-2)):
        grad[i] = abs(eVal[i]/tr_cov_matrix - 2*(eVal[i+1]/tr_cov_matrix) + 
        eVal[i+2]/tr.cov_matrix)
    var_grad = np.empty(p-6)
    for i in range(p-6):
        var_grad[i] = variance(grad[i:(i+4)])
    if ret_dim is not None:
        ret_dim = 4+ np.where(var_grad<var_cut_off)[0]
    eVal[eVal<eVal[ret_dim]] = eVal[ret_dim]
    extended_cov_matrix = eVec+np.diag(eVal)+np.transpose(eVal)
    #var_grad dataframe implementation
    return [extended_cov_matrix, var_grad, eVal]
def KrzCor_default(cov_x, cov_y, ret_dim = None, **kwargs):
    if ret_dim is not None:
        ret_dim = round(cov_x.shape[0]/2-1)
    eVec_x = LA.eigen(cov_x)[0]
    eVec_y = LA.eigen(cov_y)[0]
    return [sum((np.transpose(eVec_x[:,:ret_dim]+eVec_y[:,:ret_dim]))**2))/ret_dim]
def KrzCor_list(cov_x, cov_y = None, ret_dim = None, repeat_vector = None, **kwargs):
    if cov.y is None:
        output = ComparisonMap(cov_x, lambda x,y: [KrzCor_default(x,y,ret_dim), np.nan],
                               repeat_vector = repeat_vector)
        output = output[[0]]
    else:
        output = SingleComparisonMap(cov_x, cov_y, lambda x,y: [KrzCor_default(x,y,ret_dim), np.nan])
        output = output[:,-len(output)]
    return output
#KrzCor.mcmc_sample
def KrzProjection_default (cov_x, cov_y, ret_dim_1 = None, ret_dim_2 = None, **kwargs):
    num_traits = cov_x.shape[0]
    if ret_dim_1 is None:
        ret_dim_1 = round(num_traits/2-1)
    if ret_dim_2 is None:
        ret_dim_2 = round(num_traits/2-1)
    eigen_cov_x = LA.eigen(cov_x)
    eVal_1,eVec_1 = eigen_cov_x
    eVar_1 = np.transpose(np.asarray(list(map(lambda x:eVec_1[:,x]*math.sqrt(eVal_1[x]),0:num_traits)))))
    eVec_2 = LA.eigen(cov_y)[1]
    def SumSQ (x):
        return sum(x**2)   
    def MapProjection(x):
        SumSQ (np.transpose(np.asarray(list(map(lambda n: eVar_1[:,x]+eVec_2[:,n],0:ret_dim_2)))))
    ProjectionNorms = np.asarray(list(map(MapProjection,0:ret_dim_1)))
    output = [sum(ProjectionNorms)/sum(eVal_1),ProjectionNorms/eval_1[0:ret_dim_1]]
    return output
#KrzProjection.list
#KrzSubspac
#MINT
#MantelCor
def MantelModTest_default (cor_hypothesis, cor_matrix, permutations = 1000, MHI = False,
                            landmark_dim = None, withinLandmark = False, **kwargs):
    if (np.array_equal(cor_hypothesis, cor_hypothesis.astype(bool))):
        raise Exception("modularity hypothesis matrix should be binary")
    mantel_output = MantelCor(cor_matrix, cor_hypothesis, permutations, MHI, landmark_dim,
    withinLandmark = False, mod = True, **kwargs)
    output = [mantel_output, CalcAVG(cor_hypothesis,cor_matrix,MHI,landmark_dim)]
    return output
def MantelModTest_list (cor_hypothesis, cor_matrix, permutations = 1000, MHI = False,
                            landmark_dim = None, withinLandmark = False, **kwargs):
    output = SingleComparisonMap(cor_hypothesis,cor_matrix,lambda x,y: MantelModTest_default(x,y,permutations,MHI,landmark_dim,withinLandmark))
    return output
#Matrixcompare
#def MatrixDistance_default (cov_x,cov_y,distance=[]
#Matrixdistancelist
#Riemann dist
#Overlap dist
#Matrix Stats
#Mean matrix
#Montecarlo
#Multi Mahala
#Multi drift


