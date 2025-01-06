# import necessary libraries for pipeline:
import argparse
import random
import numpy as np
import helper_utils
import feature_selector_utils
import feature_importance_utils

# command line arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--in_file")
parser.add_argument("--feature_selector_parameterisation")
parser.add_argument("--n_folds")
parser.add_argument("--exclude_who_danger")
parser.add_argument("--normalise")
parser.add_argument("--out_file")
args = parser.parse_args()

in_file = args.in_file
feature_selector_parameterisation = args.feature_selector_parameterisation
n_folds = int(args.n_folds)
exclude_who_danger = int(args.exclude_who_danger)
normalise = int(args.normalise)
out_file = args.out_file

# ensure reproducibility by seeding random processes:
rs = 1
random.seed(rs)
np.random.seed(rs)

X, y = helper_utils.process_data(in_file, exclude_who_danger=exclude_who_danger) # tidy input data (predictors and target)
if normalise != 0:
    X = helper_utils.normalise_data(X)

feature_selector_in_use = feature_selector_utils.feature_selectors[feature_selector_parameterisation]
selected = [ f for f in feature_selector_in_use(y, X) if f in X.columns ] # select features using `feature_selector_parameterisation`
X_selected = X[selected] # subset selected features for training data
X_selected = X_selected.sort_index(axis=1)

impurity_importances = feature_importance_utils.compute_impurity_importance(y, X_selected, n_folds=n_folds) # compute importances
impurity_importances.to_csv(out_file, sep="\t", index=False, header=False)
