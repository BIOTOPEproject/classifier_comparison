# import necessary libraries for pipeline:
import helper_utils
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ensure reproducibility by seeding random processes:
rs = 1

class WHO_classifier:
    def __init__(self):
        self.method = "who"

    # for processing data for WHO method
    def calibrate(self, v):
        if v > 1:
            return 1
        else:
            return v

    def fit(self, X, y):
        self.fit_flag = "done"

    def predict_proba(self, X):
        self.prob = [ self.calibrate(v) for v in X.sum(axis=1) ]
        return [ [(1-p), p] for p in self.prob ]

class RISC_classifier:
    def __init__(self):
        self.method = "risc"

    def fit(self, X, y):
        self.fit_flag = "done"

    def predict_proba(self, X):
        self.prob = X.sum(axis=1)/17
        return [ [(1-p), p] for p in self.prob ]

class PERCH_classifier:
    def __init__(self):
        self.method = "perch"

    def fit(self, X, y):
        self.fit_flag = "done"

    def predict_proba(self, X):
        self.prob = X.sum(axis=1)/17
        return [ [(1-p), p] for p in self.prob ]

class PREPARE_classifier:
    def __init__(self):
        self.method = "prepare"

    def fit(self, X, y):
        self.fit_flag = "done"

    def predict_proba(self, X):
        self.prob = X.sum(axis=1)/17
        return [ [(1-p), p] for p in self.prob ]

# dictionary of classifier models:
classifiers = {}

# naive Bayes models:
nb_param_grid = helper_utils.expand_grid({'alpha': [ 10**e for e in range(-1,2) ]})
for p in range(nb_param_grid.shape[0]):
    classifiers[','.join(['nb',str(nb_param_grid.iloc[p]['alpha'])])] = naive_bayes.BernoulliNB(alpha=nb_param_grid.iloc[p]['alpha'])

# SVM models:
svm_param_grid = helper_utils.expand_grid({'C': [ 10**e for e in range(-1,2) ], 'gamma': [ 10**e for e in range(-1,2) ], 'kernel': ['rbf']})
for p in range(svm_param_grid.shape[0]):
    classifiers[",".join(["svm",str(svm_param_grid.iloc[p]['C']),str(svm_param_grid.iloc[p]['gamma']),str(svm_param_grid.iloc[p]['kernel'])])] = SVC(random_state=rs, C=svm_param_grid.iloc[p]['C'], gamma=svm_param_grid.iloc[p]['gamma'], kernel=svm_param_grid.iloc[p]['kernel'], probability=True)

# neural_network models:
mlp_param_grid = helper_utils.expand_grid({'layer1': [10,50,100]})
for p in range(mlp_param_grid.shape[0]):
    classifiers[",".join(["mlp",str(mlp_param_grid.iloc[p]['layer1'])])] =  MLPClassifier(random_state=rs, activation='logistic', max_iter=10000, hidden_layer_sizes=(mlp_param_grid.iloc[p]['layer1']))

mlp_param_grid = helper_utils.expand_grid({'layer1': [10,50,100], 'layer2': [10,50,100]})
for p in range(mlp_param_grid.shape[0]):
    classifiers[",".join(["mlp",str(mlp_param_grid.iloc[p]['layer1']),str(mlp_param_grid.iloc[p]['layer2'])])] =  MLPClassifier(random_state=rs, activation='logistic', max_iter=10000, hidden_layer_sizes=(mlp_param_grid.iloc[p]['layer1'], mlp_param_grid.iloc[p]['layer2']))

mlp_param_grid = helper_utils.expand_grid({'layer1': [10,50,100], 'layer2': [10,50,100], 'layer3': [10,50,100]})
for p in range(mlp_param_grid.shape[0]):
    classifiers[",".join(["mlp",str(mlp_param_grid.iloc[p]['layer1']),str(mlp_param_grid.iloc[p]['layer2']),str(mlp_param_grid.iloc[p]['layer3'])])] =  MLPClassifier(random_state=rs, activation='logistic', max_iter=10000, hidden_layer_sizes=(mlp_param_grid.iloc[p]['layer1'], mlp_param_grid.iloc[p]['layer2'], mlp_param_grid.iloc[p]['layer3']))

# random forest models:
classifiers["rf"] =  RandomForestClassifier(n_estimators=1000, random_state=rs)

# WHO method:
classifiers["who"] =  WHO_classifier()

# RISC method:
classifiers["risc"] =  RISC_classifier()

# PERCH method:
classifiers["perch"] =  PERCH_classifier()

# PREPARE method:
classifiers["prepare"] =  PREPARE_classifier()
