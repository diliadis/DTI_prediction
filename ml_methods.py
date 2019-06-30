from sklearn.datasets import make_multilabel_classification
import os
from sklearn.multioutput import ClassifierChain
from custom_classifier_chain_v2 import ClassifierChain_with_random_undesampling as CCRU
import numpy as np
import random
from joblib import Parallel, delayed
from sklearn.externals import joblib
from sklearn. metrics import roc_auc_score
import math
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from hypopt import GridSearch
from sklearn.utils import resample


def train_model(X_train, y_train, seed, ccru_version, base_classifier, X_val, y_val, feature_subsets_per_cc = []):
    pid = os.getpid()
    print('The id of '+str(seed)+' is :'+str(pid))
    # print('Train ecc: '+str(seed)+' started')

    if ccru_version == 'standard':
        model = ClassifierChain(base_classifier, order='random', random_state=seed)
    elif ccru_version == 'eccru' or ccru_version == 'eccru2' or ccru_version == 'eccru3':
        model = CCRU(base_classifier, order='random', random_state=seed)
    elif ccru_version == 'binary_relevance':
        model = SVC(gamma='auto', kernel='linear')
    else:
        print('Cannot recoginize ccru version!!!!')

    class_1 = 1
    class_2 = 0
    if -1 in y_train:
        class_2 = -1

    if ccru_version == 'binary_relevance':

        class_1_counter = np.count_nonzero(y_train[:, 0] == class_1)
        class_2_counter = np.count_nonzero(y_train[:, 0] == class_2)
        # class_1_counter = y_train.flatten().tolist()[0].count(class_1)
        # class_2_counter = y_train.flatten().tolist()[0].count(class_2)

        if class_1_counter <= class_2_counter:
            minority_class = class_1
            majority_class = class_2
            minority_counter = class_1_counter
        else:
            minority_class = class_2
            majority_class = class_1
            minority_counter = class_2_counter

        sampled_index = [index for index, label in enumerate(y_train) if label == minority_class]
        sampled_y = [minority_class] * minority_counter

        temp_sampled_index = [index for index, label in enumerate(y_train) if label == majority_class]

        sampled_index.extend(random.sample(temp_sampled_index, minority_counter))
        sampled_y.extend([majority_class] * minority_counter)
        print('Train binary_relevance: ' + str(seed) + ' started')

        print('training on ' + str(len(sampled_y)))
        if len(feature_subsets_per_cc) != 0:
            trained_model = model.fit(X_train[np.array(sampled_index), feature_subsets_per_cc[seed]], y_train, X_val, y_val)
        else:
            trained_model = model.fit(X_train[np.array(sampled_index), :], sampled_y)
    else:
        print('Train ecc: ' + str(seed) + ' started')
        if len(feature_subsets_per_cc) != 0:
            trained_model = model.fit(X_train[:, feature_subsets_per_cc[seed]], y_train, X_val, y_val)
        else:
            trained_model = model.fit(X_train, y_train, X_val, y_val)
    print('Train model: ' + str(seed) + ' ended')
    return trained_model


def predict_results(X_test, seed, trained_model, ccru_version, feature_subsets_per_cc=[]):
    if ccru_version == 'binary_relevance':
        print('Predict for label: '+str(seed)+' started')
        y_pred = trained_model.predict_proba(X_test)
        y_pred = y_pred[:, 1]
        print('predict_'+str(seed)+'_done')
    elif ccru_version == 'eccru' or ccru_version == 'eccru2' or ccru_version == 'eccru3' or ccru_version == 'standard':
        if len(feature_subsets_per_cc) == 0:
            y_pred = trained_model.predict(X_test)
        else:
            y_pred = trained_model.predict(X_test[:, feature_subsets_per_cc[seed]])
        print('predict_'+str(seed)+'_done')
    else:
        y_pred = None

    return y_pred


def get_mean_auROC(y_test, prob_result, averaging='micro'):
    final_score = 0
    if averaging == 'micro':
        results_list = []
        contains_negative = False
        if y_test.__contains__(-1):
            contains_negative = True

        for i in range(0, prob_result.shape[1]):
            compute_label = False
            if contains_negative == False:
                if np.count_nonzero(y_test[:, i] == 1) !=0:
                    compute_label = True
            else:
                if np.count_nonzero(y_test[:, i] == 1) != 0 and np.count_nonzero(y_test[:, i] == -11) != 0:
                    compute_label = True

            if compute_label:
                score = roc_auc_score(np.array(y_test[:, i]).flatten(), prob_result[:, i])
                print(str(i)+'  score: '+str(score))
                results_list.append(score)

        print('mean roc_auc: ' + str(np.mean(results_list)))
        final_score = np.mean(results_list)
    elif averaging == 'macro':
        flat_prob_result = prob_result.ravel()
        flat_y_test = y_test.ravel()
        final_score = roc_auc_score(flat_y_test, flat_prob_result)

    return final_score


def get_mean_auc_pr(y_test, prob_result):
    results_list = []
    contains_negative = False
    if y_test.__contains__(-1):
        contains_negative = True

    for i in range(0, prob_result.shape[1]):
        compute_label = False
        if contains_negative == False:
            if np.count_nonzero(y_test[:, i] == 1) !=0:
                compute_label = True
        else:
            if np.count_nonzero(y_test[:, i] == 1) != 0 and np.count_nonzero(y_test[:, i] == -11) != 0:
                compute_label = True

        if compute_label:
            score = average_precision_score(np.array(y_test[:, i]).flatten(), prob_result[:, i])
            print(str(i)+'  score: '+str(score) + '    num_of_actives:'+str(np.count_nonzero(y_test[:, i] == 1)))
            results_list.append(score)

    print('mean auc_pr: ' + str(np.mean(results_list)))
    return np.mean(results_list)


# create a random subsampled feature set for every member of the ensemble
def get_random_feature_subsamples(num_ccs, num_features, feature_subsampling_ratio):
    feature_subsets_per_cc = []
    for cc in range(num_ccs):
        temp_subset = np.random.choice(num_features, int(feature_subsampling_ratio * num_features),
                                       replace=False)
        temp_subset.sort()
        feature_subsets_per_cc.append(temp_subset)
    return feature_subsets_per_cc


def parallel_ecc_train(X_train, y_train, num_threads, num_ccs, ccru_version, base_classifier, X_val=None, y_val=None, feature_subsampling_ratio=None):
    feature_subsets_per_cc = []
    if ccru_version == 'eccru2' or ccru_version == 'eccru3':
        c_value = num_ccs
        theta_max = 10
        theta_min = 0.5
        c_theta_max = c_value * theta_max
        c_theta_min = c_value * theta_min
        m = []
        c = []

        class_1 = 1
        class_2 = 0
        if -1 in y_train:
            class_2 = -1

        for i in range(0, y_train.shape[1]):
            class_1_counter = np.count_nonzero(y_train[:, 0] == class_1)
            class_2_counter = np.count_nonzero(y_train[:, 0] == class_2)
            # class_1_counter = y_train[:, i].flatten().tolist().count(class_1)
            # class_2_counter = y_train[:, i].flatten().tolist().count(class_2)

            if class_1_counter <= class_2_counter:
                m.append(class_1_counter)
            else:
                m.append(class_2_counter)

        # calculate the sum of all the labels minority class counts
        minorities_sum = sum(m)

        # calculate the number of classifiers that should be constructed for each label
        if ccru_version == 'eccru2':
            for i in range(0, y_train.shape[1]):
                # print('m['+str(i)+'] = '+str(m[i]))
                min_c = min(math.trunc((minorities_sum * c_value) / (m[i] * y_train.shape[1])), c_theta_max)
                if min_c == 0:
                    min_c = 1
                c.append(min_c)
        elif ccru_version == 'eccru3':
            for i in range(0, y_train.shape[1]):
                min_c = min( max( math.trunc(((minorities_sum * c_value) / (m[i] * y_train.shape[1]))), c_theta_min), c_theta_max)
                if min_c == 0:
                    min_c = 1
                c.append(int(min_c))

        cn = c.copy()
        indexes_list = []

        # calculate what labels should be in each cc. These label indexes will be used to select the corresponding columns from the general y_train
        for i in range(0, c_value * theta_max):
            labels_indexes = []
            for j in range(0, y_train.shape[1]):
                if cn[j] > 0:
                    labels_indexes.append(j)
                    cn[j] = cn[j] - 1
            if len(labels_indexes) < 2:
                break
            else:
                indexes_list.append(labels_indexes)

        print('Going to train '+str(len(indexes_list))+' CCs')
        print('----'+ccru_version+'-----TRAIN-WITH-PROCESSES----------')
        if feature_subsampling_ratio is not None:
            feature_subsets_per_cc = get_random_feature_subsamples(len(indexes_list), X_train.shape[1], feature_subsampling_ratio)
        model = Parallel(n_jobs=num_threads, verbose=10)(delayed(train_model)(X_train, y_train[:, indexes_list[seed]], seed, ccru_version, base_classifier, X_val, y_val, feature_subsets_per_cc) for seed in range(len(indexes_list)))

    elif ccru_version == 'standard' or ccru_version == 'eccru':
        print('----'+ccru_version+'-----TRAIN-WITH-PROCESSES----------')
        if feature_subsampling_ratio is not None:
            feature_subsets_per_cc = get_random_feature_subsamples(num_ccs, X_train.shape[1], feature_subsampling_ratio)
        model = Parallel(n_jobs=num_threads, verbose=10)(delayed(train_model)(X_train, y_train, seed, ccru_version, base_classifier, X_val, y_val, feature_subsets_per_cc) for seed in range(num_ccs))

    elif ccru_version == 'binary_relevance':
        print('----'+ccru_version+'-----TRAIN-WITH-PROCESSES----------')
        if feature_subsampling_ratio is not None:
            feature_subsets_per_cc = get_random_feature_subsamples(y_train.shape[1], X_train.shape[1], feature_subsampling_ratio)
        model = Parallel(n_jobs=num_threads)(delayed(train_model)(X_train, y_train[:, seed], seed, ccru_version, base_classifier, X_val, y_val, feature_subsets_per_cc) for seed in range(y_train.shape[1]))

    if len(feature_subsets_per_cc) != 0:
        if ccru_version == 'eccru2' or ccru_version == 'eccru3':
            return model, indexes_list, feature_subsets_per_cc
        else:
            return model, feature_subsets_per_cc
    else:
        if ccru_version == 'eccru2' or ccru_version == 'eccru3':
            return model, indexes_list,
        else:
            return model


def parallel_ecc_predict(model, X_test, y_test, num_threads, num_ccs, ccru_version, indexes_list=[], feature_subsets_per_cc = []):
    ensemble_votes = np.zeros((y_test.shape[0], y_test.shape[1]))
    if ccru_version == 'eccru2' or ccru_version == 'eccru3':
        # q-dimensional counter
        cc = np.zeros(y_test.shape[1])
        # output vectors
        print('----' + ccru_version + '-----PREDICT-WITH-PROCESSES----------')
        # count how many times each label has been used in a cc
        for indexes in indexes_list:
            for index in indexes:
                cc[index] += 1

        Y_pred_chains = Parallel(n_jobs=num_threads, verbose=10)(delayed(predict_results)(X_test, seed, model[seed], ccru_version, feature_subsets_per_cc) for seed in range(len(indexes_list)))

        for cc_index in range(0, len(indexes_list)):  # for every CCRU
            instances_indexes_list, labels_indexes_list = np.nonzero(
                Y_pred_chains[cc_index] == 1)  # get two arrays with all the indexes of the positions of predicted 1s

            for label_index in range(0, len(labels_indexes_list)):  # update the label indexes with the correct indexes
                ind = labels_indexes_list[label_index]
                labels_indexes_list[label_index] = indexes_list[cc_index][ind]

            for i in range(0, len(instances_indexes_list)):
                ensemble_votes[instances_indexes_list[i], labels_indexes_list[i]] += 1

        for i in range(0, y_test.shape[1]):
            ensemble_votes[:, i] = ensemble_votes[:, i] / cc[i]

    elif ccru_version == 'standard' or ccru_version == 'eccru':
        print('----'+ccru_version+'-----PREDICT-WITH-PROCESSES----------')
        Y_pred_chains = Parallel(n_jobs=num_threads, verbose=10)(delayed(predict_results)(X_test, seed, model[seed], ccru_version, feature_subsets_per_cc) for seed in range(num_ccs))
        for result in Y_pred_chains:
            ensemble_votes = np.add(result, ensemble_votes)
        ensemble_votes = ensemble_votes/num_ccs # returns the average confidence of the ensemble on each label of each test instance

    elif ccru_version == 'binary_relevance':
        print('----'+ccru_version+'-----PREDICT-WITH-PROCESSES----------')
        dense_X_test = X_test.todense()
        ensemble_votes = Parallel(n_jobs=num_threads)(delayed(predict_results)(dense_X_test, seed, model[seed], ccru_version, feature_subsets_per_cc) for seed in range(y_test.shape[1]))
        ensemble_votes = np.asarray(ensemble_votes)
        ensemble_votes = ensemble_votes.transpose()

    return ensemble_votes


def predict_on_large_dataset(ensemble_model, X_test, y_test, ccru_version, num_threads, num_ccs):
    start = 0
    step_size = 2000
    finish = step_size
    step_counter = 1
    final_result = []
    while finish < y_test.shape[0]:
        print('------------------------------------------Step: ' + str(step_counter))
        step_counter += 1
        results = parallel_ecc_predict(ensemble_model, X_test[start: finish, :], y_test[start: finish, :], num_threads,
                                               num_ccs, ccru_version)
        if start == 0:
            final_result = results.copy()
        else:
            final_result = np.concatenate((final_result, results), axis=0)
        start = finish
        finish = start + step_size
    results = parallel_ecc_predict(ensemble_model, X_test[start: y_test.shape[0], :],
                                           y_test[start: y_test.shape[0], :], num_threads, num_ccs, ccru_version)
    final_result = np.concatenate((final_result, results), axis=0)
    return final_result

def save_obj(obj, name):
    joblib.dump(obj, name+'.joblib')


def load_obj(name):
    return joblib.load(name+'.joblib')
