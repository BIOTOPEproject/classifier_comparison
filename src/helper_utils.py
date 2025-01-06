# import necessary libraries for pipeline:
import math
import itertools
import pandas as pd

# generate Cartesian search grid from dictionary:
def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def process_data(in_file, cohort_id=2, exclude_who_danger=0):
    dat = pd.read_csv(in_file, sep="\t") # read data
    if exclude_who_danger != 0:
        dat = dat[dat['danger_sign'] != 1].reset_index(drop=True)
    not_cohort = [ i[0] for i in enumerate(dat['cohort']) if i[1] != cohort_id ]; dat = dat.drop(not_cohort).reset_index(drop=True)
    X = dat[[ c for c in dat.columns if c not in ['target', 'admitted_hospital', 'child_still_alive', 'id', 'cohort'] ]].reset_index(drop=True)
    y = dat['target'] # response vector
    idx_nan = [ i[0] for i in enumerate(list(y)) if math.isnan(i[1]) ]; X = X.drop(idx_nan).reset_index(drop=True); y = y.drop(idx_nan).reset_index(drop=True)
    y = y.astype(int)
    return [X, y] # return predictors and target

def normalise_data(X, normaliser=None, exclude=[]):
    if type(normaliser) != pd.core.frame.DataFrame:
        normaliser = X
    exclude = [ c for c in X.columns if c in ["id", "cohort"]+exclude ]
    for c in X.columns:
        if (exclude == None or c not in exclude) and not (normaliser[c].isin([0,1]).all()):
            mu = normaliser[c].mean()
            sigma = normaliser[c].std()
            X[c] = (X[c] - mu) / sigma
    return X
