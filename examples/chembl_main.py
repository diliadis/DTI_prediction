import ml_methods
import sys
import cotraining
import chembl_preprocessing
import cotraining
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def main3(args):

    setting = 'standard'
    num_threads = 5
    num_ccs = 5
    ccru_version = 'eccru'

    model1 = SVC(gamma='auto', kernel='linear')
    model2 = LogisticRegression(solver='liblinear')
    model3 = MultinomialNB()

    sampled_dict, train_outputfile, test_outputfile, feature_freq_limit = chembl_preprocessing.main(
        '-i output.txt -c 0 --featurefreqlimit=100 --otrainfile=chembl_datasets/train_set.csv --otestfile=chembl_datasets/test_set.csv --samplesize=200000'.split(
            ' '))

    folds = [0, 1, 2]
    auc_result = 0
    for fold in folds:

        if setting == 'co_training':
            X_train, y_train, X_test, y_test, X_unlabeled, y_unlabeled = chembl_preprocessing.get_train_test_set_from_fold(sampled_dict, fold, feature_freq_limit, train_outputfile, test_outputfile, setting)

            # if you don't want to use explicit inactivity information
            y_test[y_test == -1] = 0
            y_train[y_train == -1] = 0
            y_unlabeled[y_unlabeled == -1] = 0

            averaged_votes, aucRoc_results_per_view_dict, auPR_results_per_view_dict = cotraining.main(X_train, y_train, X_test, y_test, X_unlabeled, y_unlabeled, 10, 'eccru', model2, num_threads, num_ccs, num_of_views=2)
            auc_result = auc_result + ml_methods.get_mean_auROC(y_test, averaged_votes)

        elif setting == 'standard':

            X_train, y_train, X_test, y_test = chembl_preprocessing.get_train_test_set_from_fold(sampled_dict, fold, feature_freq_limit, train_outputfile, test_outputfile, setting)

            # if you don't want to use explicit inactivity information
            y_test[y_test == -1] = 0
            y_train[y_train == -1] = 0

            ensemble_model = ml_methods.parallel_ecc_train(X_train, y_train, num_threads, num_ccs, ccru_version, model2)
            test_results = ml_methods.parallel_ecc_predict(ensemble_model, X_test, y_test, num_threads, num_ccs)
            auc_result = auc_result + ml_methods.get_mean_auROC(y_test, test_results)

    print('mean_auc: ' + str(auc_result / len(folds)))



if __name__ == "__main__":
   main3(sys.argv[1:])