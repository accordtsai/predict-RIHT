# Development and validation of a machine learning model of radiation-induced hypothyroidism with clinical and dose–volume features
# Mu-Hung Tsai et al.
# https://doi.org/10.1016/j.radonc.2023.109911

# This code reproduces the model and external validation results from the paper. 

# Requirements:
# - scikit-survival (sksurv) 0.21.0
# Run line below to install with conda:
# conda install -c conda-forge scikit-survival=0.21.0

import numpy as np
import pickle
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import *

# Load training and validation data
# We are using g1_event and g1_time for grade ≥1 RIHT; change y_devel and y_extval lines below to g2_event and g2_time for grade ≥2 RIHT
devel = pd.read_csv('data_training.csv') # (378, 269)
y_devel = Surv.from_dataframe(event='g1_event', time='g1_time', data=devel)
x_devel = devel.drop(columns=['StudyID','g1_event','g1_time','g2_event','g2_time','Thyroid_sphere','Thyroid_modal'])

extval = pd.read_csv('data_validation.csv') # (49, 269)
y_extval = Surv.from_dataframe(event='g1_event', time='g1_time', data=extval)
x_extval = extval.drop(columns=['StudyID','g1_event','g1_time','g2_event','g2_time','Thyroid_sphere','Thyroid_modal'])

# Define traditional NTCP models
def boomsma(r): 
    # t_mean in Gy
    s = 0.011 + (0.062 * r['Thyroid_mean'] / 100) + (-0.19 * r['Thyroid_volume'])
    ntcp = 1/(1 + np.exp(-s))
    return ntcp

def ronjom_logistic(r): 
    # t_mean in Gy
    s = -2.019 + (0.0821* r['Thyroid_mean'] / 100) - (0.189 * r['Thyroid_volume'])
    ntcp = 1/(1 + np.exp(-s))
    return ntcp

def ronjom_mixture(r): 
    # t_mean in Gy
    s = -1.235 + (0.1162* r['Thyroid_mean'] / 100) - (0.2873 * r['Thyroid_volume'])
    ntcp = 1/(1 + np.exp(-s))
    return ntcp

def bak_LKB(r):
    # Copied from https://sourceforge.net/p/ntcpcalc/code-0/HEAD/tree/ntcpcalc.py
    TD50=44.1
    m=0.26
    dx=0.1
    #Lyman-Kutcher-Burman Model:
    # solving integral over lung doses numerically
    # and return NTCP for all models
    # uses effective dose in LKB model
    t=((r['Thyroid_mean'] / 100) - TD50) / (m*TD50)
    num_range=np.arange(-999,t,dx)
    sum_ntcp=0.
    for dummy in range(len(num_range)):
        sum_ntcp+=np.exp(-1*num_range[dummy]**2/2)*dx
    return 1./np.sqrt(2*np.pi)*sum_ntcp 

def cella1(r): 
    s = -1.83 + (0.038* r['Thyroid_V30']) + (-2.32 * (r['Sex']-1)) # Note: Sex is 1 or 2 in this dataset
    ntcp = 1/(1 + np.exp(-s))
    return ntcp

def cella2(r): 
    absolute_t30 = r['Thyroid_V30'] / 100 * r['Thyroid_volume']
    s = 1.94 + (0.26* absolute_t30) + (-2.21 * (r['Sex']-1)) + (-0.27 * r['Thyroid_volume'])
    ntcp = 1/(1 + np.exp(-s))
    return ntcp

def luo(r): 
    # s_max: maximum dose of the pituitary; default to mean value as reported in the paper
    s_max = r['Sella_nondeformed_max'] / 100
    chemotherapy = 1 if (r['Induction'] + r['Concurrent'] + r['Adjuvant'] > 0) else 0
    s = -2.695 + (0.05* r['Thyroid_V50']) + (-0.026 * s_max) + (1.28 * (r['Sex']-1)) + (2.902 * chemotherapy)
    ntcp = 1/(1 + np.exp(-s))
    return ntcp
    
def luo_default(r): 
    # s_max: maximum dose of the pituitary; default to mean value as reported in the paper
    s_max = 37.856
    chemotherapy = 1
    s = -2.695 + (0.05* r['Thyroid_V50']) + (-0.026 * s_max) + (1.28 * (r['Sex']-1)) + (2.902 * chemotherapy)
    ntcp = 1/(1 + np.exp(-s))
    return ntcp

## Helper functions to compute and score all traditional NTCP models at once
def compute_trad_ntcp(r, StudyID=False):
    devel = r.copy() 
    devel['boomsma'] = devel.apply(boomsma, axis=1)
    devel['ronjom_log'] = devel.apply(ronjom_logistic, axis=1)
    devel['ronjom_mix'] = devel.apply(ronjom_mixture, axis=1)
    devel['bak'] = devel.apply(bak_LKB, axis=1)
    devel['cella1'] = devel.apply(cella1, axis=1)
    devel['cella2'] = devel.apply(cella2, axis=1)
    devel['luo'] = devel.apply(luo, axis=1)
    devel['luo_default'] = devel.apply(luo_default, axis=1)
    if StudyID:
        return devel[['StudyID','boomsma','ronjom_log','ronjom_mix','bak','cella1','cella2','luo','luo_default']]
    else:
        return devel[['boomsma','ronjom_log','ronjom_mix','bak','cella1','cella2','luo','luo_default']]

def score_trad_models(x_val, y_val, x_train, y_train, eval_times, brier_times, brier_score_at=[1,2,3,4,5]):
    results = {
        'c-index':{},
        'mAUC':{},
        'iBS':{}
    }
    
    aucname = [('AUC@' + str(at)) for at in eval_times]
    briername = [('BS@' + str(at)) for at in brier_score_at]
    for an in aucname:
        results[an]={}
    for bn in briername:
        results[bn]={}
    
    ntcp = compute_trad_ntcp(x_val)
    y_event, y_time = zip(*y_val)
    
    for name in ntcp.columns:
        risk_scores = ntcp[[name]].squeeze().tolist()
        results['c-index'][name] = concordance_index_censored(y_event, y_time, risk_scores)[0]
        
        auc_scores, results['mAUC'][name] = cumulative_dynamic_auc(y_train, y_val, risk_scores, eval_times)
        for i, score in enumerate(auc_scores):
            results[aucname[i]][name] = auc_scores[i]
        risk_scores = np.array(risk_scores)
        risk_scores_2d = np.repeat(np.array(risk_scores)[:, np.newaxis], len(brier_score_at), axis=1)
        _, brier_scores = brier_score(y_train, y_val, risk_scores_2d, brier_score_at)
        for i, score in enumerate(brier_scores):
            results[briername[i]][name] = brier_scores[i]
        risk_scores_2d = np.repeat(np.array(risk_scores)[:, np.newaxis], len(brier_times), axis=1)
        results['iBS'][name] = integrated_brier_score(y_train, y_val, risk_scores_2d, brier_times)
    
    ordered_columns = ['iBS'] + briername + ['c-index','mAUC'] + aucname
    res = pd.DataFrame.from_dict(results)
    return res[ordered_columns]

## Helper functions to fit and score all machine learning models at once
def fitMLmodels(modeldict, x_train, y_train):
    for name in modeldict.keys():
        modeldict[name].fit(x_train, y_train)
    return

def scoreMLmodels(modeldict, x_val, y_val, x_train, y_train, eval_times, brier_times, brier_score_at=[1,2,3,4,5]):
    results = {
        'c-index':{},
        'mAUC':{},
        'iBS':{}
    }
    
    aucname = [('AUC@' + str(at)) for at in eval_times]
    briername = [('BS@' + str(at)) for at in brier_score_at]
    for an in aucname:
        results[an]={}
    for bn in briername:
        results[bn]={}
    
    for name in modeldict.keys():
        m = modeldict[name]
        results['c-index'][name] = m.score(x_val, y_val)
        risk_scores = m.predict(x_val)
        auc_scores, results['mAUC'][name] = cumulative_dynamic_auc(y_train, y_val, risk_scores, eval_times)
        for i, score in enumerate(auc_scores):
            results[aucname[i]][name] = auc_scores[i]
        survs = m.predict_survival_function(x_val)
        preds = np.array([fn(brier_score_at) for fn in survs])
        _, brier_scores = brier_score(y_val, y_val, preds, brier_score_at)
        for i, score in enumerate(brier_scores):
            results[briername[i]][name] = brier_scores[i]
        preds_integrated = np.asarray([[fn(t) for t in brier_times] for fn in survs])
        results['iBS'][name] = integrated_brier_score(y_train, y_val, preds_integrated, brier_times)
    
    ordered_columns = ['iBS'] + briername + ['c-index','mAUC'] + aucname
    res = pd.DataFrame.from_dict(results)
    return res[ordered_columns]

# Define evaluation timepoints
eval_times = np.arange(1, 6, 1)
eval_times_len = len(eval_times)
brier_times = np.arange(1, 5, 0.01)
brier_score_at = [1,2,3,4,5]
random_state=2020

# Define ML models
final={}
final['GB'] = GradientBoostingSurvivalAnalysis(n_estimators=10, 
                                       dropout_rate=0.1, 
                                       learning_rate=1.0, 
                                       max_depth=4,
                                       random_state=random_state,
                                       subsample=0.5)
final['RF'] = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=35,
                           n_jobs=-1,
                           random_state=random_state)
final['Cox'] = CoxnetSurvivalAnalysis(fit_baseline_model=True,
                                     l1_ratio=0.8, alpha_min_ratio=0.1)

# Define thyroid-only ML models
cols = x_devel.columns
tcols = cols[96:]
x_devel_t = x_devel[tcols]
x_extval_t = x_extval[tcols]

final_t={}
final_t['GB-thyroid'] = GradientBoostingSurvivalAnalysis(n_estimators=10, 
                                       dropout_rate=0.1, 
                                       learning_rate=1.0, 
                                       max_depth=4,
                                       random_state=random_state,
                                       subsample=0.5)
final_t['RF-thyroid'] = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=35,
                           n_jobs=-1,
                           random_state=random_state)
final_t['Cox-thyroid'] = CoxnetSurvivalAnalysis(fit_baseline_model=True,
                                     l1_ratio=0.8, alpha_min_ratio=0.1)

# Train ML models using development cohort; score using external validation cohort
m1 = score_trad_models(x_extval, y_extval, x_devel, y_devel, eval_times, brier_times, brier_score_at=brier_score_at)

fitMLmodels(final, x_devel, y_devel)
m2 = scoreMLmodels(final, x_extval, y_extval, x_devel, y_devel, eval_times, brier_times)

fitMLmodels(final_t, x_devel_t, y_devel)
m3 = scoreMLmodels(final_t, x_extval_t, y_extval, x_devel_t, y_devel, eval_times, brier_times)

metric = pd.concat([m2, m3, m1])

# Reproduce external validation results (Table 3)
print(metric.round(3))

# Save final models
with open('models/GB.pkl','wb') as fh:
    pickle.dump(final['GB'], fh)

with open('models/RF.pkl','wb') as fh:
    pickle.dump(final['RF'], fh)

with open('models/Cox.pkl','wb') as fh:
    pickle.dump(final['Cox'], fh)

# Save thyroid-only models for incorporation into treatment planning systems
with open('models/GB-thyroid.pkl','wb') as fh:
    pickle.dump(final_t['GB-thyroid'], fh)

with open('models/RF-thyroid.pkl','wb') as fh:
    pickle.dump(final_t['RF-thyroid'], fh)

with open('models/Cox-thyroid.pkl','wb') as fh:
    pickle.dump(final_t['Cox-thyroid'], fh)