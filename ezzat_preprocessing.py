import numpy as np
import os
from collections import Counter
import random


def load_txt_file(filepath):
    num_rows = sum(1 for line in open(filepath))
    fp = open(filepath)
    line = fp.readline()
    num_columns = len(line.split("\t"))
    fp.close()
    data_matrix = np.zeros((num_rows, num_columns))

    fp = open(filepath)
    line = fp.readline()
    row_counter = 0
    # read the file line by line
    while line:
        for index, value in enumerate(line.split("\t")):
            if value.isdigit():
                data_matrix[row_counter, index] = int(value)
            else:
                data_matrix[row_counter, index] = float(value)
        line = fp.readline()
        row_counter += 1
    fp.close()

    return data_matrix


def load_dataset():
    os.chdir('../../datasets/ezzat_dataset/')

    # load the interaction matrix from the .txt file
    # interaction_matrix = load_data('interaction_matrix.txt')
    # np.save('interaction_matrix.npy', interaction_matrix)

    # load the target's feature vectors from the .txt file
    # target_feature_vectors = load_data('target_feature_vectors.txt')
    # np.save('target_feature_vectors.npy', target_feature_vectors)

    # load the interaction matrix from the .txt file
    # drug_feature_vectors = load_data('drug_feature_vectors.txt')
    # np.save('drug_feature_vectors.npy', drug_feature_vectors)

    # load the interaction matrix as a numpy array
    interaction_matrix = np.load('interaction_matrix.npy')
    # load the target feature vector matrix as a numpy array
    target_feature_vectors = np.load('target_feature_vectors.npy')
    # load the compound feature vector matrix as a numpy array
    drug_feature_vectors = np.load('drug_feature_vectors.npy')

    return interaction_matrix, drug_feature_vectors, target_feature_vectors


def check_labels_per_target(interaction_matrix):
    l = []
    for i in range(interaction_matrix.shape[1]):
        l.append(sum(interaction_matrix[:, i]))
        if sum(interaction_matrix[:, i]) == 0:
            print('target ' + str(i) + ' has no active pairs')
    freq_distribution = sorted(Counter(l).items())
    print(str(freq_distribution))


def get_folds(interaction_matrix, drug_feature_vectors, num_folds=5):
    num_folds=5

    multi_active_targets = []
    for i in range(interaction_matrix.shape[1]):
        if sum(interaction_matrix[:, i]) >= num_folds:
            multi_active_targets.append(i)

    print(str(len(multi_active_targets)) + ' / ' + str(interaction_matrix.shape[1]) + ' targets are kept')
    filtered_interaction_matrix = interaction_matrix[:, multi_active_targets]
    print(str(sum(sum(filtered_interaction_matrix))) + ' / ' + str(sum(sum(interaction_matrix))) + ' interactions are kept')

    folds = []
    num_rows = filtered_interaction_matrix.shape[0]
    num_columns = filtered_interaction_matrix.shape[1]

    # initialize the folds with the correct sizes
    avg_fold_size = int(num_rows / num_folds)
    last = 0
    for i in range(num_folds):
        if (last + 2 * avg_fold_size) <= num_rows:
            folds.append(
                [0, np.zeros((avg_fold_size, num_columns)), np.zeros((avg_fold_size, drug_feature_vectors.shape[1]))])
        else:
            folds.append([0, np.zeros((avg_fold_size + (num_rows % avg_fold_size), num_columns)),
                          np.zeros((avg_fold_size + (num_rows % avg_fold_size), drug_feature_vectors.shape[1]))])
        last += avg_fold_size

    # assigning at least ony active pair for every protein target of every fold. This way during training, every
    # target will have at least one sample from every class (active and inactive). The same will be true for the
    # validation setting. If we don't impose this limit a class with samples that belong to only one class will
    # cause problems during training as well as during validation.
    for l_index in range(num_columns):
        sampled_actives = [index for index, label in enumerate(filtered_interaction_matrix[:, l_index]) if label == 1]
        random.shuffle(sampled_actives)
        fold_index = 0
        active_index = 0
        while active_index < len(sampled_actives):
            if folds[fold_index][1][:, l_index].tolist().count(1) == 0:
                # the folds[fold_index][0] contains the index of the sample that was added last. This index must be incremented
                # after the addition of the new sample
                folds[fold_index][1][folds[fold_index][0]] = filtered_interaction_matrix[sampled_actives[active_index]]
                folds[fold_index][2][folds[fold_index][0]] = drug_feature_vectors[sampled_actives[active_index]]
                # marking all the compounds that are already added in one of the folds
                filtered_interaction_matrix[sampled_actives[active_index]] = num_columns * [-1]
                folds[fold_index][0] += 1
                active_index += 1
            fold_index += 1
            if fold_index == num_folds:
                break

    # gather all the indexes of samples that haven't been added in a fold
    remaining_actives_and_inactives = [index for index, label in enumerate(filtered_interaction_matrix[:, 0]) if
                                       label == 1 or label == 0]
    random.shuffle(remaining_actives_and_inactives)
    for sample_index in remaining_actives_and_inactives:
        has_been_added = False
        while has_been_added == False:
            i = random.randint(0, num_folds - 1)
            # check if this fold in full of samples
            if folds[i][0] < folds[i][1].shape[0]:
                folds[i][1][folds[i][0]] = filtered_interaction_matrix[sample_index]
                folds[i][2][folds[i][0]] = drug_feature_vectors[sample_index]
                folds[i][0] += 1
                has_been_added = True


    fold_features = [fold[2] for fold in folds]
    fold_labels = [fold[1] for fold in folds]
    return fold_features, fold_labels