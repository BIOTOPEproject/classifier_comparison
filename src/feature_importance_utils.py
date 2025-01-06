# import necessary libraries for pipeline:
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.inspection import permutation_importance

# ensure reproducibility by seeding random processes:
rs = 1

# compute cross-validated impurity importances:
def compute_impurity_importance(y, X, n_folds=5, n_estimators=1000, random_state=rs):

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    impurity_importances = []

    # use random stratified folds to preserve class imbalance distribution across `n_folds` folds:
    cv_out = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i,f_out in enumerate(cv_out.split(X, y)):

        f_out_idx_train = list(f_out[0]) # training data
        f_out_train_cov = X.iloc[f_out_idx_train] # features
        f_out_train_lab = y.iloc[f_out_idx_train] # response

        rf.fit(f_out_train_cov, f_out_train_lab) # fit model on training data

        for j in range(len(f_out_train_cov.columns)):
            impurity_importances.append([i,f_out_train_cov.columns[j],rf[-1].feature_importances_[j]])

    return pd.DataFrame(impurity_importances)

# compute cross-validated permutation importances:
def compute_permutation_importance(y, X, n_folds=5, n_repeats=10, n_estimators=1000, random_state=rs):

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    permutation_importances = []

    # use random stratified folds to preserve class imbalance distribution across `n_folds` folds:
    cv_out = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i,f_out in enumerate(cv_out.split(X, y)):

        f_out_idx_train = list(f_out[0]) # training data
        f_out_train_cov = X.iloc[f_out_idx_train] # features
        f_out_train_lab = y.iloc[f_out_idx_train] # response

        f_out_idx_test = list(f_out[1]) # testing data
        f_out_test_cov = X.iloc[f_out_idx_test] # features
        f_out_test_lab = y.iloc[f_out_idx_test] # response

        rf.fit(f_out_train_cov, f_out_train_lab) # fit model on training data
        result = permutation_importance(rf, f_out_test_cov, f_out_test_lab, n_repeats=n_repeats, random_state=random_state)

        for j in range(len(f_out_test_cov.columns)):
            permutation_importances.append([i,f_out_test_cov.columns[j],result.importances_mean[j]])

    return pd.DataFrame(permutation_importances)
