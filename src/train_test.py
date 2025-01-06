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

X_train, y_train = helper_utils.process_data(in_file, cohort_id=2, exclude_who_danger=exclude_who_danger) # training cohort
X_test, y_test = helper_utils.process_data(in_file, cohort_id=1, exclude_who_danger=exclude_who_danger) # testing cohort

# must normalise `X_test` prior to normalising `X_train` so that we're substracting the unnormalised mean and dividing by the unnormalised standard deviation!!!
if normalise != 0:
    X_test = helper_utils.normalise_data(X_test, normaliser=X_train)
    X_train = helper_utils.normalise_data(X_train)

selected = [ f for f in feature_selector_in_use(y_train, X_train) if f in X_train.columns ] # select features using `feature_selector_parameterisation`
X_train_selected = X_train[selected] # subset selected features for training data
classifier_in_use.fit(X_train_selected, y_train) # fit model on training data (only need to train once for each `feature_selector_parameterisation` and `classifier_parameterisation` combination)

test_results = { n: {} for n in range(n_folds) }
# use random stratified folds to preserve class imbalance distribution across `n_folds` folds:
cv_out = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
for i,f_out in enumerate(cv_out.split(X_test, y_test)):

        f_out_idx_test = list(f_out[1]) # testing data
        f_out_test_cov = X_test.iloc[f_out_idx_test] # features
        f_out_test_lab = y_test.iloc[f_out_idx_test] # response
        f_out_test_cov_selected = f_out_test_cov[selected] # subset selected features for validation data

        prob = [ e[1] for e in classifier_in_use.predict_proba(f_out_test_cov_selected) ] # transfrom into probability on validation data
        test_results[i] = { 'selector': feature_selector_parameterisation, 'classifier': classifier_parameterisation, 'score': prob, 'label': list(f_out_test_lab), 'selected': selected }

with open(out_file, "w") as outfile:
    for i in test_results:
        for n in range(len(test_results[i]['score'])):
            outfile.write(f"{i}\t{test_results[i]['selector']}:{test_results[i]['classifier']}:{n_folds}:{exclude_who_danger}:{normalise}\t{test_results[i]['score'][n]}\t{test_results[i]['label'][n]}\n")
