# import necessary libraries for pipeline:
import argparse
import random
import numpy as np
import helper_utils
import feature_selector_utils
import classifier_utils
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# command line arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--in_file")
parser.add_argument("--feature_selector_parameterisation")
parser.add_argument("--classifier_parameterisation")
parser.add_argument("--n_folds")
parser.add_argument("--exclude_who_danger")
parser.add_argument("--normalise")
parser.add_argument("--out_file")
args = parser.parse_args()

in_file = args.in_file
feature_selector_parameterisation = args.feature_selector_parameterisation
classifier_parameterisation = args.classifier_parameterisation
n_folds = int(args.n_folds)
exclude_who_danger = int(args.exclude_who_danger)
normalise = int(args.normalise)
out_file = args.out_file

# ensure reproducibility by seeding random processes:
rs = 1
random.seed(rs)
np.random.seed(rs)

feature_selector_in_use = feature_selector_utils.feature_selectors[feature_selector_parameterisation]
classifier_in_use = classifier_utils.classifiers[classifier_parameterisation]

X, y = helper_utils.process_data(in_file, exclude_who_danger=exclude_who_danger) # tidy input data (predictors and target)
if normalise != 0:
    X = helper_utils.normalise_data(X)

selected = [ f for f in feature_selector_in_use(y, X) if f in X.columns ] # select features using `feature_selector_parameterisation`
X_selected = X[selected] # subset selected features for training data

# clear output file:
with open(out_file, "w") as outfile:
    outfile.write(f"")

learning_curve = {}
f_out0_idx = []
# use random stratified folds to preserve class imbalance distribution across `n_folds` folds:
cv_out0 = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
cv_out1 = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
for i,f_out0 in enumerate(cv_out0.split(X, y)):

    learning_curve[i] = {}

    f_out0_idx = f_out0_idx + list(f_out0[1]) # cumulatively add to training/validation data
    f_out0_cov = X_selected.iloc[f_out0_idx]
    f_out0_lab = y.iloc[f_out0_idx]

    for j,f_out1 in enumerate(cv_out1.split(f_out0_cov, f_out0_lab)):

        f_out1_idx_train = list(f_out1[0]) # training data
        f_out1_train_cov = f_out0_cov.iloc[f_out1_idx_train] # features
        f_out1_train_lab = f_out0_lab.iloc[f_out1_idx_train] # response

        f_out1_idx_test = list(f_out1[1]) # testing data
        f_out1_test_cov = f_out0_cov.iloc[f_out1_idx_test] # features
        f_out1_test_lab = f_out0_lab.iloc[f_out1_idx_test] # response

        classifier_in_use.fit(f_out1_train_cov, f_out1_train_lab) # fit model on training data
        prob = [ e[1] for e in classifier_in_use.predict_proba(f_out1_test_cov) ] # transfrom into probability on validation data
        learning_curve[i]["n"] = len(f_out0_idx)
        learning_curve[i]["auc"] = roc_auc_score(f_out1_test_lab, prob)
        learning_curve[i]["prauc"] = average_precision_score(f_out1_test_lab, prob)

        with open(out_file, "a") as outfile:
            outfile.write(f"{learning_curve[i]['n']}\t{learning_curve[i]['auc']}\t{learning_curve[i]['prauc']}\n")
