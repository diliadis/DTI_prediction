from scipy.sparse import csr_matrix
import sys
import getopt
import pickle
from pathlib import Path
import random
import numpy as np
import smtplib
from email.mime.text import MIMEText
import time


def main(argv):

    inputfile = ''
    train_outputfile = ''
    test_outputfile = ''
    cluster_id = 0
    feature_freq_limit = 0
    sample_size = 0
    try:
        opts, args = getopt.getopt(argv, "hi:c:", ["ifile=", "testcluster=", "featurefreqlimit=", "otrainfile=", "otestfile=", "samplesize="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -c <clusterid> --featurefreqlimit <frequency_limit> --otrainfile <trainoutputfile> --otestfile <testoutputfile> --samplesize<sample_size>')
        sys.exit(2)
    for opt, arg in opts:
        print('opt: ' + opt + '   arg: ' + arg)
        if opt == '-h':
            print('test.py -i <inputfile> -c <clusterid> --featurefreqlimit <frequency_limit> --otrainfile <trainoutputfile> --otestfile <testoutputfile> --samplesize<sample_size>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-c", "--testcluster="):
            cluster_id = int(arg)
        elif opt in ("--featurefreqlimit="):
            feature_freq_limit = int(arg)
        elif opt in ("--otrainfile="):
            train_outputfile = arg
        elif opt in ("--otestfile="):
            test_outputfile = arg
        elif opt in ("--samplesize="):
            sample_size = int(arg)

    print(' ')
    my_file = Path("chembl_datasets/pkl_files/dataset.pkl")
    if my_file.is_file():
        print('File: '+str(my_file)+' already exists')
        start_time = time.time()
        dataset_dict = load_obj('chembl_datasets/pkl_files/dataset')
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print('File: '+str(my_file)+' is missing')
        start_time = time.time()
        dataset_dict = load_dataset(inputfile)
        print("--- %s seconds ---" % (time.time() - start_time))
        save_obj(dataset_dict, 'chembl_datasets/pkl_files/dataset')

    print(' ')
    if sample_size < len(dataset_dict):
        my_file = Path('chembl_datasets/pkl_files/sampled_dataset'+str(sample_size)+'.pkl')
        if my_file.is_file():
            print('File: '+str(my_file)+' already exists')
            start_time = time.time()
            sampled_dict = load_obj('chembl_datasets/pkl_files/sampled_dataset'+str(sample_size))
            print("--- %s seconds ---" % (time.time() - start_time))
        else:
            print('File: '+str(my_file)+' is missing')
            start_time = time.time()
            sampled_dict = random_sampling(dataset_dict, sample_size)
            print("--- %s seconds ---" % (time.time() - start_time))
            save_obj(sampled_dict, 'chembl_datasets/pkl_files/sampled_dataset'+str(sample_size))
    else:
        sampled_dict = dataset_dict.copy()

    return sampled_dict, train_outputfile, test_outputfile, feature_freq_limit


def get_train_test_set_from_fold(sampled_dict, cluster_id, feature_freq_limit, train_outputfile, test_outputfile, mode):
    # Split the original sampled dataset to train and test.
    # If the dataset will be used by a co_training method, an additional dataset has to be created, called the
    # unlabeled set.
    train_test_split('chembl_datasets/cluster1.info.txt', cluster_id, sampled_dict, mode, 0)
    number_of_lines('chembl_datasets/train_split.csv')
    number_of_lines('chembl_datasets/test_split.csv')
    if mode == 'co_training':
        number_of_lines('chembl_datasets/unlabeled_split.csv')

    # calculate the frequency of every ECFP feature in the train set and filter out every feature with frequency under the specified limit
    features_frequencies_dict = calculate_feature_frequencies('chembl_datasets/train_split.csv', feature_freq_limit)

    # After the filtering, the features that remain have non sequencial ids.
    # This causes a problem in the dataset creation step because the id of a feature will be equal to its position
    # in the final sparse array. To mediate that we map the initial id space to a new sequencial space.
    map_features('chembl_datasets/train_split.csv', features_frequencies_dict, train_outputfile)
    map_features('chembl_datasets/test_split.csv', features_frequencies_dict, test_outputfile)

    # If the dataset has to be used by a co_training method, an aditional feature id mapping has to be applied in the
    # unlabeled set
    if mode == 'co_training':
        map_features('chembl_datasets/unlabeled_split.csv', features_frequencies_dict, 'chembl_datasets/unlabeled_set.csv')


    samples_per_label_limit = 15
    # load the active the inactive compounds for every target so that you can filter targets with less than a
    # specified limit of actives and inactives.
    act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict = load_target_activities('chembl_datasets/targetActivities.txt')

    # load the features of the train, test and unlabeled(for the co_training version) datasets
    train_features = load_features('chembl_datasets/train_set.csv')
    print(str(len(train_features))+' compounds in train_features dictionary')
    test_features = load_features('chembl_datasets/test_set.csv')
    print(str(len(test_features))+' compounds in test_features dictionary')
    if mode == 'co_training':
        unlabeled_features = load_features('chembl_datasets/unlabeled_set.csv')
        print(str(len(unlabeled_features))+' compounds in unlabeled_features dictionary')
    print(' ')

    train_compounds_list = list(train_features.keys())
    test_compounds_list = list(test_features.keys())
    if mode == 'co_training':
        unlabeled_compounds_list = list(unlabeled_features.keys())

    targets_over_limit = []
    train_act_inact_targets_per_compound_dict = load_labels(act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict, train_compounds_list, samples_per_label_limit, targets_over_limit)
    test_act_inact_targets_per_compound_dict = load_labels(act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict, test_compounds_list, samples_per_label_limit, targets_over_limit)

    if mode == 'co_training':
        unlabeled_act_inact_targets_per_compound_dict = load_labels(act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict, unlabeled_compounds_list, samples_per_label_limit, targets_over_limit)

    targets_dict = {}
    features_dict = {}

    # Generate the final train test and unlabeled (for the co_training version) sets
    X_train, y_train = create_sparse_matrix(train_act_inact_targets_per_compound_dict,
                                             test_act_inact_targets_per_compound_dict, targets_dict, features_dict,
                                             'train', train_features)

    X_test, y_test = create_sparse_matrix(train_act_inact_targets_per_compound_dict,
                                           test_act_inact_targets_per_compound_dict, targets_dict, features_dict,
                                           'test', test_features)

    if mode == 'co_training':
        X_unlabeled, y_unlabeled = create_sparse_matrix(train_act_inact_targets_per_compound_dict,
                                            unlabeled_act_inact_targets_per_compound_dict, targets_dict, features_dict,
                                           'test', unlabeled_features)

    print('X_train.shape: '+str(X_train.shape))
    print('y_train.shape: ' + str(y_train.shape))
    print('X_test.shape: ' + str(X_test.shape))
    print('y_test.shape: ' + str(y_test.shape))
    if mode == 'co_training':
        print('X_unlabeled.shape: ' + str(X_unlabeled.shape))
        print('y_unlabeled.shape: ' + str(y_unlabeled.shape))

    y_train2 = y_train.todense()
    y_test2 = y_test.todense()
    if mode == 'co_training':
        y_unlabeled2 = y_unlabeled.todense()

    if mode == 'co_training':
        return X_train, y_train2, X_test, y_test2, X_unlabeled, y_unlabeled2
    else:
        return X_train, y_train2, X_test, y_test2


def get_imbalance_rations(sampled_dict):
    train_test_split('cluster1.info.txt', 5, sampled_dict, 'standard', 0)
    print(' ')
    # feature_freq_limit = 10
    features_frequencies_dict = calculate_feature_frequencies('chembl_datasets/train_split.csv', 0)
    train_outputfile = 'chembl_datasets/train_set.csv'
    print(' ')
    map_features('chembl_datasets/train_split.csv', features_frequencies_dict, train_outputfile)
    act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict = load_target_activities('chembl_datasets/targetActivities.txt')
    train_features = load_features('chembl_datasets/train_set.csv')
    train_compounds_list = list(train_features.keys())
    targets_over_limit = []
    train_act_inact_targets_per_compound_dict = load_labels(act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict, train_compounds_list, 0, targets_over_limit)
    targets_dict = {}
    features_dict = {}

    test_act_inact_targets_per_compound_dict = {}
    X_train, y_train = create_sparse_matrix(train_act_inact_targets_per_compound_dict,
                                             test_act_inact_targets_per_compound_dict, targets_dict, features_dict,
                                             'train', train_features)

    y_train2 = y_train.todense()

    y_train2 = np.asarray(y_train2)

    print('X_train.shape: '+str(X_train.shape))
    print('y_train.shape: '+str(y_train2.shape))
    imbalance_ratio_list = []
    for i in range(0, y_train2.shape[1]):
        print(str(i) + ')')
        minority_count = list(y_train2[:, i]).count(1)
        majority_count = list(y_train2[:, i]).count(0)
        if minority_count != 0:
            imbalance_ratio_list.append(majority_count / minority_count)

    print('sum(imbalance_ratio_list): '+str(sum(imbalance_ratio_list)))
    print('sum(imbalance_ratio_list) / y_train2.shape[1]: '+str(sum(imbalance_ratio_list)/y_train2.shape[1]))
    print('y_train2.shape[1]: '+str(y_train2.shape[1]))
    print('len(imbalance_ratio_list): '+str(len(imbalance_ratio_list)))
    print('np.mean(imbalance_ratio_list): '+str(np.mean(imbalance_ratio_list)))
    print('max(imbalance_ratio_list): '+str(max(imbalance_ratio_list)))
    print('min(imbalance_ratio_list): '+str(min(imbalance_ratio_list)))


def number_of_lines(filepath):
    print('filename: '+str(filepath))
    num_lines = sum(1 for line in open(filepath))
    print('File:'+str(filepath)+' has '+str(num_lines)+' samples')
    return num_lines


def random_sampling(dataset_dict, num_of_samples):
    keys = list(dataset_dict)
    random.shuffle(keys)
    sample = np.random.choice(keys, size=num_of_samples, replace=False)
    sampled_dict = dict((compound_id, dataset_dict[compound_id]) for compound_id in sample)
    print('dictionary successfully sampled: '+str(num_of_samples)+' samples out of '+str(len(dataset_dict)))
    return sampled_dict


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_dataset(filepath):
    number_of_lines(filepath)
    print('Loading the ' + str(filepath) + ' file...')
    fp = open(filepath)
    line = fp.readline()
    count = 0
    compound_features_dict = {}
    # read the file line by line
    while line:
        count += 1
        if count > 500000:
            if count % 50000 == 0:
                print(str(count)+' lines loaded')
            # extract the feature ids from each line-instance
            f = line.split(' ')
            features_list = []
            for feature in f[1:]:
                # append only the feature id to the list
                features_list.append(feature.split(':')[0])
            compound_features_dict[str(f[0])] = features_list
            line = fp.readline()
    fp.close()
    print('Loading completed')
    print('The compound_features_dictionary has '+str(len(compound_features_dict))+' entries')
    return compound_features_dict


def load_target_activities(filepath):
    print('')
    print('loading the '+str(filepath)+' file')
    compound_file_exists = True
    target_file_exists = True
    compounds_per_target_file = Path('chembl_datasets/pkl_files/compounds_per_target.pkl')
    if compounds_per_target_file.is_file():
        print('--file: '+str(compounds_per_target_file)+' already exists')
        act_inact_compounds_per_target_dict = load_obj('chembl_datasets/pkl_files/compounds_per_target')
    else:
        target_file_exists = False

    targets_per_compound_file = Path('chembl_datasets/pkl_files/targets_per_compound.pkl')
    if targets_per_compound_file.is_file():
        print('--file: '+str(targets_per_compound_file)+' already exists')
        act_inact_targets_per_compound_dict = load_obj('chembl_datasets/pkl_files/targets_per_compound')
    else:
        compound_file_exists = False

    if (target_file_exists == False) or (compound_file_exists == False):
        number_of_lines(filepath)
        fp = open(filepath)
        line = fp.readline()
        count = 0
        act_inact_targets_per_compound_dict = {}
        act_inact_compounds_per_target_dict = {}

        start_time = time.time()

        while line:
            count += 1
            if count % 100000 == 0:
                print(str(count) + '____lines loaded in ' + str((time.time() - start_time)) + ' secs')
                start_time = time.time()

            # extract the feature ids from each line-instance
            f = line.split(' ')
            activity = f[0]
            compound_id = f[1]
            target_id = f[2].strip()

            if (activity == '1') or (activity == '3'):
                if compound_id not in act_inact_targets_per_compound_dict:
                    act_inact_targets_per_compound_dict[compound_id] = [(target_id, activity)]
                elif (target_id, activity) not in act_inact_targets_per_compound_dict[compound_id]:
                    act_inact_targets_per_compound_dict[compound_id].append((target_id, activity))

                if target_id not in act_inact_compounds_per_target_dict:
                    act_inact_compounds_per_target_dict[target_id] = [(compound_id, activity)]
                elif (compound_id, activity) not in act_inact_compounds_per_target_dict[target_id]:
                    act_inact_compounds_per_target_dict[target_id].append((compound_id, activity))

            line = fp.readline()

        save_obj(act_inact_compounds_per_target_dict, 'chembl_datasets/pkl_files/compounds_per_target')
        save_obj(act_inact_targets_per_compound_dict, 'chembl_datasets/pkl_files/targets_per_compound')

    return act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict


def load_labels(act_inact_targets_per_compound_dict, act_inact_compounds_per_target_dict, compounds_list, compound_freq_limit_per_target, targets_over_limit):
    print(' ')
    print('starting to load labels for every compound')
    labels_per_compound_dict = {}
    compounds_set = set(compounds_list)
    if len(targets_over_limit) == 0:
        # iterate the act_inact_compounds_per_target_dict, calculate for each target the number of active and inactive
        # compounds and if they are over the limit keep them in the targets_over_limit list
        for target, compounds in act_inact_compounds_per_target_dict.items():
            active_counter = 0
            inact_counter = 0
            for compound_id, activity in compounds:
                if compound_id in compounds_set:
                    if activity == '1':
                        inact_counter += 1
                    elif activity == '3':
                        active_counter += 1

            if (active_counter >= compound_freq_limit_per_target and inact_counter >= compound_freq_limit_per_target):
                targets_over_limit.append(target)

    compounds_with_no_labels_counter = 0
    compounds_not_in_target_activities_file_counter = 0
    for compound_id in compounds_set:
        targets_list = []
        if compound_id in act_inact_targets_per_compound_dict:
            temp_targets_list = act_inact_targets_per_compound_dict[compound_id]
            for target_id, activity in temp_targets_list:
                if target_id in targets_over_limit:
                    targets_list.append((target_id, activity))
            if len(targets_list) > 0:
                labels_per_compound_dict[compound_id] = targets_list.copy()
            else:
                compounds_with_no_labels_counter += 1
        else:
            compounds_not_in_target_activities_file_counter += 1
    print('The set started with '+str(len(compounds_set))+' compounds')
    print('There are '+str(compounds_not_in_target_activities_file_counter)+' compounds that dont show up in the targetActivities.txt file')
    print('There are '+str(compounds_with_no_labels_counter)+' compounds that end up with no labels')
    print('The set ended with '+str(len(labels_per_compound_dict))+' compounds')
    print('finished reading labels')

    return labels_per_compound_dict


# load the compound-target interaction pairs. You keep targets that have a number of actives and inactives compounds
#  that is over a user specified limit
def load_labels_2nd_alternative_version(filepath, compound_freq_limit_per_target, train_compounds_list, test_compounds_list, mode, unlabeled_percentage):
    number_of_lines(filepath)
    print('load_labels) Loading the ' + str(filepath) + ' file...')
    fp = open(filepath)
    line = fp.readline()
    count = 0
    train_act_inact_targets_per_compound_dict = {}
    train_act_inact_compounds_per_target_dict = {}

    test_act_inact_targets_per_compound_dict = {}
    unlabeled_act_inact_targets_per_compound_dict = {}

    # in the cotraining mode we split the train set in the labeled and unlabeled sets.
    if mode == 'co_training':
        unlabeled_set_size = int(unlabeled_percentage * len(train_compounds_list))
        unlabeled_compounds_list = np.random.choice(list(train_compounds_list), size=unlabeled_set_size, replace=False)
        train_compounds_list = [compound_id for compound_id in train_compounds_list if compound_id not in unlabeled_compounds_list]

    missing_compounds_counter = 0
    # read the file line by line
    start_time = time.time()
    while line:
        count += 1
        if count % 100000 == 0:
            print(str(count)+' lines loaded in '+str((time.time() - start_time))+' secs')
            start_time = time.time()

        # extract the feature ids from each line-instance
        f = line.split(' ')
        activity = f[0]
        compound_id = f[1]
        target_id = f[2].strip()

        if compound_id in train_compounds_list: # check for compounds that are in the train set
            # print(str(count) + 'train---')
            if (activity == '1') or (activity == '3'):
                # print(str(count)+'train========')
                if compound_id not in train_act_inact_targets_per_compound_dict:
                    train_act_inact_targets_per_compound_dict[compound_id] = [(target_id, activity)]
                elif (target_id, activity) not in train_act_inact_targets_per_compound_dict[compound_id]:
                    train_act_inact_targets_per_compound_dict[compound_id].append((target_id, activity))

                if target_id not in train_act_inact_compounds_per_target_dict:
                    train_act_inact_compounds_per_target_dict[target_id] = [(compound_id, activity)]
                elif (compound_id, activity) not in train_act_inact_compounds_per_target_dict[target_id]:
                    train_act_inact_compounds_per_target_dict[target_id].append((compound_id, activity))
        elif compound_id in test_compounds_list:
            if (activity == '1') or (activity == '3'):
                if compound_id not in test_act_inact_targets_per_compound_dict:
                    test_act_inact_targets_per_compound_dict[compound_id] = [(target_id, activity)]
                elif not (target_id, activity) in test_act_inact_targets_per_compound_dict[compound_id]:
                    test_act_inact_targets_per_compound_dict[compound_id].append((target_id, activity))
        elif compound_id in unlabeled_compounds_list:
            if (activity == '1') or (activity == '3'):
                if compound_id not in unlabeled_act_inact_targets_per_compound_dict:
                    unlabeled_act_inact_targets_per_compound_dict[compound_id] = [(target_id, activity)]
                elif not (target_id, activity) in unlabeled_act_inact_targets_per_compound_dict[compound_id]:
                    unlabeled_act_inact_targets_per_compound_dict[compound_id].append((target_id, activity))
        else:
            missing_compounds_counter += 1
            # print(str(count)+') compound_id: '+compound_id+' doesnt appear in dataset')
        line = fp.readline()
    print('Loading the completed')
    print(str(missing_compounds_counter)+' compounds were missing from the dataset')


    # collect all the targets that have a small number of active and inactive compounds
    targets_to_be_removed_list = []
    for target, compounds_list in train_act_inact_compounds_per_target_dict.items():
        active_counter, inact_counter = act_inact_counter(compounds_list)
        if (active_counter == 0) or (inact_counter == 0) or ((active_counter+inact_counter) < compound_freq_limit_per_target):
            targets_to_be_removed_list.append(target)
    print('There are '+str(len(targets_to_be_removed_list))+' targets to be removed out of '+str(len(train_act_inact_compounds_per_target_dict)))

    print('Starting removing targets from train dict')
    removed_counter = 0
    for compound, targets_list in train_act_inact_targets_per_compound_dict.items():
        temp_targets_list = []
        for t_id, act in targets_list:
            if t_id in targets_to_be_removed_list:
                removed_counter += 1
            else:
                temp_targets_list.append((t_id, act))
        train_act_inact_targets_per_compound_dict[compound] = temp_targets_list
    print('finished removing '+str(removed_counter)+' targets from train dict in total')


    print('Starting removing targets from test dict')
    removed_counter = 0
    for compound, targets_list in test_act_inact_targets_per_compound_dict.items():
        temp_targets_list = []
        for t_id, act in targets_list:
            if t_id in targets_to_be_removed_list:
                removed_counter += 1
            else:
                temp_targets_list.append((t_id, act))
        test_act_inact_targets_per_compound_dict[compound] = temp_targets_list
    print('finished removing '+str(removed_counter)+' targets from test dict in total')


    print('Starting removing targets from unlabeled dict')
    removed_counter = 0
    for compound, targets_list in unlabeled_act_inact_targets_per_compound_dict.items():
        temp_targets_list = []
        for t_id, act in targets_list:
            if t_id in targets_to_be_removed_list:
                removed_counter += 1
            else:
                temp_targets_list.append((t_id, act))
            unlabeled_act_inact_targets_per_compound_dict[compound] = temp_targets_list
    print('finished removing '+str(removed_counter)+' targets from unlabeled dict in total')

    return train_act_inact_targets_per_compound_dict, test_act_inact_targets_per_compound_dict, train_act_inact_compounds_per_target_dict, targets_to_be_removed_list, unlabeled_act_inact_targets_per_compound_dict


def train_test_split(filepath, test_cluster_index, dataset_dict, mode, unlabeled_cluster_index):
    print(' ')
    print('beginning the train_test_split method')
    fp = open(filepath)
    line = fp.readline()
    cluster_assignments_dict = {}
    # read the cluster_info file line by line
    while line:
        # extract the feature ids from each line-instance
        f = line.split(' ')
        # entries in the cluster_info file are arranged in the format (cluster_id  compound_id)
        # there are three clusters in the dataset. The id's are 0,1,2
        cluster_assignments_dict[f[1].strip()] = f[0]
        line = fp.readline()
    fp.close()
    print('The cluster_assignments_dictionary has ' + str(len(cluster_assignments_dict)) + ' entries')

    train_file = open('chembl_datasets/train_split.csv', 'w')
    test_file = open('chembl_datasets/test_split.csv', 'w')
    if mode == 'co_training':
        unlabeled_file = open('chembl_datasets/unlabeled_split.csv', 'w')

    train_samples_count = 0
    unlabeled_samples_count = 0
    missing_compound_ids_counter = 0
    for id, f_list in dataset_dict.items():
        # if the dictionary with the cluster assignments contrains the compound_id get his cluster assignment.
        # Depending on the cluster id assign it to the testset or the trainset
        if id in cluster_assignments_dict:
            cluster_id = cluster_assignments_dict[id]
            if cluster_id == str(test_cluster_index):
                test_file.write(str(id)+' '+' '.join(str(f) for f in f_list)+'\n')
            elif mode == 'co_training':
                if cluster_id == str(unlabeled_cluster_index):
                    unlabeled_file.write(str(id) + ' ' + ' '.join(str(f) for f in f_list)+'\n')
                    unlabeled_samples_count += 1
                else:
                    train_file.write(str(id) + ' ' + ' '.join(str(f) for f in f_list) + '\n')
                    train_samples_count += 1
            else:
                train_file.write(str(id) + ' ' + ' '.join(str(f) for f in f_list)+'\n')
        else:
            missing_compound_ids_counter += 1

    print('')
    train_file.close()
    test_file.close()
    if mode == 'co_training':
        unlabeled_file.close()
    print('train-test split based on '+str(filepath)+' file completed')


def calculate_feature_frequencies(filepath, freq_limit):
    print(' ')
    print('calculating features frequencies')
    fp = open(filepath)
    line = fp.readline()
    features_counter = 0
    features_frequencies_dict = {}
    # read the file line by line
    while line:
        # extract the feature ids from each line-instance
        f = line.split(' ')
        # create a dictionary that contains the frequencies of every feature
        for feature in f[1:]:
            if not feature in features_frequencies_dict:
                # features_frequencies_dict[feature] = 1
                features_counter += 1
                features_frequencies_dict[feature] = (1, features_counter)
            else:
                # features_frequencies_dict[feature] += 1
                features_frequencies_dict[feature] = (features_frequencies_dict[feature][0]+1, features_frequencies_dict[feature][1])

        line = fp.readline()
    fp.close()

    # filter out all the features with frequencies under the limit
    final_dict = {}
    features_counter = 0
    for feature_id in features_frequencies_dict.keys():
        if features_frequencies_dict[feature_id][0] >= freq_limit:
            final_dict[feature_id] = (features_frequencies_dict[feature_id][0], features_counter)
            features_counter += 1

    print('features frequencies calculation completed')
    return final_dict


def map_features(filepath, features_frequencies_dict, output_file_name):
    print('')
    print('start mapping features')
    number_of_lines(filepath)
    fp = open(filepath)
    mapped_file = open(output_file_name, 'w')
    line = fp.readline()
    count = 0
    # read the cluster_info file line by line
    while line:
        count += 1
        # extract the feature ids from each line-instance
        f = line.split(' ')
        # add the compound_id
        features_line = f[0]
        # read every feature_id. Its frequency is bigger than the limit because of the previous method.
        # Add its new mapped id to the features list
        features_counter = 0
        for feature in f[1:]:
            if feature in features_frequencies_dict:
                temp_tuple = features_frequencies_dict[feature]
                # the tuple has the following format (frequency, mapped_freature_id)
                features_counter += 1
                features_line = features_line + ' ' + str(temp_tuple[1])
        if features_counter > 0:
            mapped_file.write(features_line+'\n')
        line = fp.readline()

    fp.close()
    mapped_file.close()
    print('Mapping completed')


def load_features(filepath):
    print(' ')
    print('loading features from '+str(filepath))
    fp = open(filepath)
    line = fp.readline()
    features_dict = {}
    while line:
        # extract the feature ids from each line-instance
        f = line.split(' ')
        compound_id = f[0]
        features_list = []
        for feature in f[1:]:
            features_list.append(feature.strip())

        features_dict[compound_id] = features_list
        line = fp.readline()

    fp.close()
    print(str(len(features_dict))+' compounds loaded')
    return features_dict


def act_inact_counter(elements_list):
    active_counter = 0
    inact_counter = 0
    for element_id, activity in elements_list:
        if activity == '1':
            inact_counter += 1
        elif activity == '3':
            active_counter += 1
    return active_counter, inact_counter


def create_sparse_matrix(train_act_inact_targets_per_compound_dict, test_act_inact_targets_per_compound_dict,
                          targets_dict, features_dict, mode, features):
    print(' ')
    print('creating the '+mode+' set')
    train_mode = False
    if mode == 'train':
        train_mode = True
    count = 0

    # for creating the features sparse matrix
    f_indptr = [0]
    f_indices = []
    f_data = []

    # for creating the labels sparse matrix
    l_indptr = [0]
    l_indices = []
    l_data = []

    # read the cluster_info file line by line
    if train_mode:

        for compound, targets_list in train_act_inact_targets_per_compound_dict.items():
            labels_counter = 0
            count += 1
            # iterate the target-lables and create another row in the labels sparse matrix
            for target_id, activity in targets_list:
                if activity == '3':
                    l_data.append(1)
                elif activity == '1':
                    l_data.append(-1)

                index = targets_dict.setdefault(target_id, (len(targets_dict), False))[0]
                l_indices.append(index)
                labels_counter += 1

                if len(l_data) != len(l_indices):
                    raise ValueError(
                        'Size problem!!!!!!!!!!!!!!!!!!!!!!!!!!!    labels_counter: ' + str(labels_counter)
                        + '          len(l_data): ' + str(len(l_data)) + '           len(l_indices): '
                        + str(len(l_indices)))

            l_indptr.append(len(l_indices))

            # iterate the instance's features and create another row in the feature sparse matrix
            # you must check the case where in the test phase there are no features available. In this case you
            # should probably add some zeros
            for feature in features[compound]:
                feature = int(feature.strip())
                if not feature in features_dict:
                    features_dict.setdefault(feature, (len(features_dict), False))

                feature_index = features_dict[feature][0]
                f_indices.append(feature_index)
                f_data.append(1)

                if len(f_data) != len(f_indices):
                    print('Size problem!!!!!!!!!!!!!!!!!!!!!!!!!!!    feature: ' +
                          str(feature) + '          len(f_data): ' + str(len(f_data)) +
                          '           len(l_indices): ' + str(len(f_indices)))

            f_indptr.append(len(f_indices))

    else:
        all_targets_inserted = False
        all_features_inserted = False
        for compound, targets_list in test_act_inact_targets_per_compound_dict.items():
            labels_counter = 0
            count += 1
            data_inserted_counter = 0
            for target_id, activity in targets_list:
                if target_id in targets_dict:
                    if activity == '3':
                        l_data.append(1)
                    elif activity == '1':
                        l_data.append(-1)
                    data_inserted_counter += 1
                    index = int(targets_dict[target_id][0])
                    l_indices.append(index)
                    labels_counter += 1
                    targets_dict[target_id] = (targets_dict[target_id][0], True)
                else:
                    print(str(count) + ') test sets target: ' + str(target_id) + ' doesnt appear in the train set')

            if labels_counter > 0:
                if all_targets_inserted == False:
                    for remaining_target in [compound_id for compound_id, used_in_train in targets_dict.items() if
                                             used_in_train[1] == False]:
                        target_index = targets_dict[remaining_target][0]
                        l_indices.append(target_index)
                        l_data.append(0)
                    all_targets_inserted = True
                l_indptr.append(len(l_indices))

                for feature in features[compound]:
                    feature = int(feature.strip())
                    if feature in features_dict:
                        features_dict[feature] = (features_dict[feature][0], True)

                        feature_index = features_dict[feature][0]
                        f_indices.append(feature_index)
                        f_data.append(1)

                if all_features_inserted == False:
                    counter = 0
                    for remaining_feature in [feature_id for feature_id, used_in_train in features_dict.items() if
                                              used_in_train[1] == False]:
                        feature_index = features_dict[remaining_feature][0]
                        f_indices.append(feature_index)
                        f_data.append(0)
                        counter += 1
                    all_features_inserted = True
                f_indptr.append(len(f_indices))
            else:
                for d in range(0, data_inserted_counter):
                    l_data.pop()
                    l_indices.pop()

    print('loading completed')

    X = csr_matrix((f_data, f_indices, f_indptr), dtype=int)
    y = csr_matrix((l_data, l_indices, l_indptr), dtype=int)

    return X, y



def get_density(y_train):
    sum = 0
    for i in range(0, y_train.shape[0]):
        label_counter = 0
        for j in range(0, y_train.shape[1]):
            if y_train[i, j] == 1:
                label_counter += 1
        sum += label_counter / y_train.shape[1]
    sum = (1 / y_train.shape[1]) * sum

    return sum