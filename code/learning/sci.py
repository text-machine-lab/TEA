from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score

from multiprocessing import cpu_count

import numpy as np

import math

def train( featsDict, Y, do_grid=False, ovo=False):
    '''
    train()
        train a single classifier for a given data and label set

    @param featsDict: List of dictionary representations of every data entry
    @param Y: labels for each data entry
    @param do_grid: use grid search? (default False)
    @param ovo: use one-vs-one classification? (default False results in one-vs-rest classification)
    '''

    #vectorize dictionary data
    vec = DictVectorizer()
    X = vec.fit_transform(featsDict).toarray()

    # Search space
    C_range     = 10.0 ** np.arange( -5, 9 )
    gamma_range = 10.0 ** np.arange( -5, 9 )

    clf = None

    # one-vs-one?
    func_shape = 'ovr'
    if ovo:
        func_shape = 'ovo'

    # Grid search?
    if do_grid:

        print "training model [GRID SEARCH ON]"

        estimates = SVC(kernel='poly', max_iter=1000, decision_function_shape=func_shape, class_weight='balanced')
        parameters = [{'C':C_range, 'gamma':gamma_range}]

        # Find best classifier
        clf = GridSearchCV(estimates,
                           parameters,
                           scoring='f1_weighted',
                           n_jobs=5,
                           verbose=10)
        clf.fit(X, Y)

    else:

        print "training model [GRID SEARCH OFF]"
        clf = SVC(kernel='poly', max_iter=1000, decision_function_shape=func_shape, class_weighted='balanced')
        clf.fit(X, Y)

    return clf, vec

