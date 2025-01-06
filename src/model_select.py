# import necessary libraries for pipeline:
import argparse
import random
import helper_utils
import feature_selector_utils
import classifier_utils
import numpy as np
from sklearn import model_selection

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

test_results = { n: {} for n in range(n_folds) }
# use random stratified folds to preserve class imbalance distribution across `n_folds` folds:
cv_out = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
for i,f_out in enumerate(cv_out.split(X, y)):

    f_out_idx_train = list(f_out[0]) # training data
    f_out_train_cov = X.iloc[f_out_idx_train] # features
    f_out_train_lab = y.iloc[f_out_idx_train] # response

    f_out_idx_test = list(f_out[1]) # testing data
    f_out_test_cov = X.iloc[f_out_idx_test] # features
    f_out_test_lab = y.iloc[f_out_idx_test] # response

    selected = [ f for f in feature_selector_in_use(f_out_train_lab, f_out_train_cov) if f in X.columns ] # select features using `feature_selector_parameterisation`
    f_out_train_cov_selected = f_out_train_cov[selected] # subset selected features for training data
    f_out_test_cov_selected = f_out_test_cov[selected] # subset selected features for validation data

    classifier_in_use.fit(f_out_train_cov_selected, f_out_train_lab) # fit model on training data
    prob = [ e[1] for e in classifier_in_use.predict_proba(f_out_test_cov_selected) ] # transform into probability on validation data
    test_results[i] = { 'selector': feature_selector_parameterisation, 'classifier': classifier_parameterisation, 'score': prob, 'label': list(f_out_test_lab), 'selected': selected }

with open(out_file, "w") as outfile:
    for i in test_results:
        for n in range(len(test_results[i]['score'])):
            outfile.write(f"{i}\t{test_results[i]['selector']}:{test_results[i]['classifier']}:{n_folds}:{exclude_who_danger}:{normalise}\t{test_results[i]['score'][n]}\t{test_results[i]['label'][n]}\n")
