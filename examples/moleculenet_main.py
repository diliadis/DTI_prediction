import moleculenet_preprocessing as mnp
import ml_methods
from scipy import sparse
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import LogisticRegression


def main():

    m = mnp.moleculenet_dataset(dataset_name = 'sider')
    X_train, y_train, X_test, y_test, X_val, y_val = m.get_dataset()

    num_threads = -1
    num_ccs = 100
    ccru_version = 'eccru'

    model1 = SVC()
    model2 = LogisticRegression()
    model3 = MultinomialNB()

    ensemble_model = ml_methods.parallel_ecc_train(X_train, y_train, num_threads,
                                                   num_ccs,
                                                   ccru_version, model1, X_val=X_val, y_val=y_val)

    results = ml_methods.parallel_ecc_test(ensemble_model, X_test, y_test,
                                           num_threads,
                                           num_ccs, ccru_version)

    ml_methods.get_mean_auROC(y_test, results)