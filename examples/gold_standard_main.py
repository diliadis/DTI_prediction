import ml_methods
import sys
import golden_datasets_preprocessing
import cotraining
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def main(args):
    num_threads = 5
    num_ccs = 5
    num_folds = 5
    setting = 'standard'

    model1 = SVC(gamma='auto', kernel='linear')
    model2 = LogisticRegression(solver='liblinear')
    model3 = MultinomialNB()

    data_types = ['nuclear_receptor', 'enzyme', 'ion_channel', 'GPCR']

    results_dict = {}

    for data_type in data_types:
        features_filename = 'gold_standard_datasets/' + data_type + '_features'
        folds = golden_datasets_preprocessing.k_fold_split(features_filename, num_folds)

        targets_per_compound_dict = golden_datasets_preprocessing.load_dataset(
            'gold_standard_datasets/' + data_type + '_dataset' + '.txt')

        auc_results = []
        pr_results = []
        ccru_versions = ['standard', 'eccru', 'eccru2', 'eccru3']
        for ccru_version in ccru_versions:
            results = {}
            auc_result = 0.0
            pr_result = 0.0
            fold_counter = 0
            for fold in folds:
                fold_counter += 1
                print('========= Preparing fold: ' + str(fold_counter) + '=========')
                if setting == 'standard':
                    X_train, y_train, X_test, y_test = golden_datasets_preprocessing.get_train_test_set_from_fold(
                        targets_per_compound_dict, fold, features_filename, 'standard', 0.9)

                    if ccru_version == 'eccru2' or ccru_version == 'eccru3':
                        ensemble_model, indexes_list = ml_methods.parallel_ecc_train(X_train, y_train, num_threads,
                                                                                     num_ccs,
                                                                                     ccru_version, model2)
                        results[fold_counter] = ml_methods.parallel_ecc_test(ensemble_model, X_test, y_test,
                                                                             num_threads,
                                                                             num_ccs, ccru_version, indexes_list)
                    else:
                        ensemble_model = ml_methods.parallel_ecc_train(X_train, y_train, num_threads, num_ccs, ccru_version)
                        results[fold_counter] = ml_methods.parallel_ecc_test(ensemble_model, X_test, y_test,
                                                                             num_threads,
                                                                             num_ccs, ccru_version, model2)

                elif setting == 'co_training':
                    X_train, y_train, X_test, y_test, Un_set_X, Un_set_y = golden_datasets_preprocessing.get_train_test_set_from_fold(targets_per_compound_dict, fold, features_filename, 'co-training', 0.9)

                    results[fold_counter], aucRoc_results_per_view_dict, auPR_results_per_view_dict = cotraining.main(X_train, y_train, X_test, y_test, Un_set_X, Un_set_y, 10, 'eccru', model2, num_threads, num_ccs, num_of_views=2)

                auc_result = auc_result + ml_methods.get_mean_auROC(y_test, results[fold_counter])
                pr_result = pr_result + ml_methods.get_mean_auc_pr(y_test, results[fold_counter])
            print('mean_auc: ' + str(auc_result / fold_counter))
            print('mean_pr_auc: ' + str(pr_result / fold_counter))
            auc_results.append(auc_result / fold_counter)
            pr_results.append(pr_result / fold_counter)

        results_dict[data_type] = (auc_results.copy(), pr_results.copy())


if __name__ == "__main__":
   main(sys.argv[1:])


