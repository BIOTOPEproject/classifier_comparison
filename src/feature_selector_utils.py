# import necessary libraries for pipeline:
import numpy as np
import feature_importance_utils
from sklearn.ensemble import RandomForestClassifier
import boruta_py
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# ensure reproducibility by seeding random processes:
rs = 1

# perform no feature selection:
def biotope_select(y, X):
    return [ f for f in list(X.columns) if 'danger_sign' != f and '_woho' not in f and '_risc' not in f and '_perch' not in f and '_prepare' not in f ]

# perform no feature selection:
def biotope_composite_danger_sign_select(y, X):
    danger_sign_features = ["weight_for_age", "oxygen_saturation", "vomiting_everything", "convulsions", "lethargy", "unconciousness", "unable_to_feed", "stridor"]
    return [ f for f in list(X.columns) if f not in danger_sign_features and '_woho' not in f and '_risc' not in f and '_perch' not in f and '_prepare' not in f ]

# perform no feature selection:
def biotope_exclude_danger_sign_and_features_select(y, X):
    danger_sign_features = ["weight_for_age", "oxygen_saturation", "vomiting_everything", "convulsions", "lethargy", "unconciousness", "unable_to_feed", "stridor"]+["danger_sign"]
    return [ f for f in list(X.columns) if f not in danger_sign_features and '_woho' not in f and '_risc' not in f and '_perch' not in f and '_prepare' not in f ]

# perform WHO selection:
def who_select(y, X):
    return [ f for f in list(X.columns) if '_woho' in f ]

# perform WHO feature selection:
def who_features_select(y, X):
    return [ f.split('_woho')[0] for f in list(X.columns) if '_woho' in f ]

# perform RISC feature selection:
def risc_select(y, X):
    return [ f for f in list(X.columns) if '_risc' in f ]

# perform RISC feature selection:
def risc_features_select(y, X):
    return [ f.split('_risc')[0] for f in list(X.columns) if '_risc' in f ]

# perform PERCH selection:
def perch_select(y, X):
    return [ f for f in list(X.columns) if '_perch' in f ]

# perform PERCH feature selection:
def perch_features_select(y, X):
    return [ f.split('_perch')[0] for f in list(X.columns) if '_perch' in f ]

# perform PREPARE selection:
def prepare_select(y, X):
    return [ f for f in list(X.columns) if '_prepare' in f ]

# perform PREPARE feature selection:
def prepare_features_select(y, X):
    return [ f.split('_prepare')[0] for f in list(X.columns) if '_prepare' in f ]

# perform Boruta feature selection:
def biotope_boruta_select(y, X, random_state=rs):
    X = X[biotope_select(y, X)]
    rf = RandomForestClassifier(random_state=random_state)
    boruta = boruta_py.BorutaPy(rf, n_estimators='auto')
    fit = boruta.fit(X.values, y.values)
    selected = fit.support_
    return X.columns[selected]

# perform correlation-based feature selection:
def biotope_corr_select(y, X, cutoff_point=0.90):
    X = X[biotope_select(y, X)]
    corr = spearmanr(X).correlation
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    dist_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(dist_matrix))
    cluster_ids = hierarchy.fcluster(dist_linkage, (cutoff_point+10e-05), criterion="distance")
    impurity = feature_importance_utils.compute_impurity_importance(y, X)
    median_impurity = impurity.groupby(by=[1]).median()
    median_impurity_clusters = median_impurity.assign(cluster_id=list(cluster_ids))
    cluster_id_to_feature_ids = { idx: {'score': -np.Inf, 'feature': None } for idx in range(1,(max(list(cluster_ids))+1)) }
    for idx in range(median_impurity_clusters.shape[0]):
        if median_impurity_clusters[2][idx] >= cluster_id_to_feature_ids[median_impurity_clusters['cluster_id'][idx]]['score']:
            cluster_id_to_feature_ids[median_impurity_clusters['cluster_id'][idx]]['feature'] = median_impurity_clusters.index[idx]
            cluster_id_to_feature_ids[median_impurity_clusters['cluster_id'][idx]]['score'] = median_impurity_clusters[2][idx]
    selected_features_names = [ d['feature'] for d in cluster_id_to_feature_ids.values() ]
    return selected_features_names

# assess near-zero variability of given feature `x`:
def nzv_indv(x):
    idx = None
    value_counts = x.value_counts()
    if (len(set(x)) >= 1) and ((value_counts.shape[0]/len(x) > 0.1) and (value_counts.iloc()[0]/value_counts.iloc()[1] < 95/5)):
        idx = x.name
    return idx

# perform near-zero variability feature selection:
def biotope_nzv_select(y, X):
    X = X[biotope_select(y, X)]
    return [ nzv_indv(X[c]) for c in X.columns if nzv_indv(X[c]) != None ]

# dictionary of feature selection models:
feature_selectors = { 'biotope': biotope_select, 'biotope_composite_danger_sign': biotope_composite_danger_sign_select, 'biotope_exclude_danger_sign_and_features': biotope_exclude_danger_sign_and_features_select, 'biotope_boruta': biotope_boruta_select, 'biotope_corr': biotope_corr_select, 'biotope_nzv': biotope_nzv_select, 'who': who_select, 'risc': risc_select, 'perch': perch_select, 'prepare': prepare_select, 'who_features': who_features_select, 'risc_features': risc_features_select, 'perch_features': perch_features_select, 'prepare_features': prepare_features_select }
