import ezzat_preprocessing as ezzatp
import ml_methods
from scipy import sparse
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    interaction_matrix, drug_feature_vectors, target_feature_vectors = ezzatp.load_dataset()
    num_folds = 5
    fold_features, fold_labels = ezzatp.get_folds(interaction_matrix, drug_feature_vectors, num_folds=num_folds)

    num_threads = -1
    num_ccs = 50
    ccru_version = 'eccru'

    model1 = SVC()
    model2 = LogisticRegression()
    model3 = MultinomialNB()

    auROC_score_per_fold = []

    for i in range(5):

        X_train = np.concatenate((np.delete(fold_features, i)), axis=0)
        y_train = np.concatenate((np.delete(fold_labels, i)), axis=0)

        X_test = fold_features[i]
        y_test = fold_labels[i]

        ensemble_model = ml_methods.parallel_ecc_train(X_train, y_train, num_threads,
                                                       num_ccs,
                                                       ccru_version, model1)

        results = ml_methods.parallel_ecc_predict(ensemble_model, X_test, y_test,
                                               num_threads,
                                               num_ccs, ccru_version)

        auROC_score_per_fold.append(ml_methods.get_mean_auROC(y_test, results, averaging='macro'))

    print('The average auROC over the '+str(num_folds)+' folds is '+str(np.mean(auROC_score_per_fold)))