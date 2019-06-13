from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup as soup
import urllib
from urllib.request import urlopen as uReq
#import rdkit
#from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix
import random
import numpy as np

def load_dataset(filepath):
    number_of_lines(filepath)
    print('Loading the ' + str(filepath) + ' file...')
    fp = open(filepath)
    line = fp.readline()
    count = 0
    targets_per_compound_dict = {}
    # read the file line by line
    while line:
        count += 1
        if count % 1000 == 0:
            print(str(count)+' lines loaded')
        # extract the feature ids from each line-instance
        f = line.split('\t')
        compound = f[1].strip()
        target = f[0]
        if not compound in targets_per_compound_dict:
            targets_per_compound_dict[compound] = []
            targets_per_compound_dict[compound].append(target)
        else:
            targets_per_compound_dict[compound].append(target)

        line = fp.readline()
    print('Loading completed')
    print('The dictionary has '+str(len(targets_per_compound_dict))+' entries')
    return targets_per_compound_dict


def kegg_database_crawler(compounds_list):

    print(str(len(compounds_list))+' compounds in total')
    chembl_ids_dict = {}

    counter = 0
    for compound_d_id in compounds_list:
        counter += 1
        print(str(counter) + ') Crawling on ' + compound_d_id)
        # open up connection
        my_url = 'https://www.kegg.jp/dbget-bin/www_bget?dr:' + compound_d_id
        # take the page
        uClient = uReq(my_url)
        # dump the html
        page_html = uClient.read()
        # close the connection
        uClient.close()
        page_soup = soup(page_html, "html.parser")

        # collect all nobr tags and find the one that has ChEMBL in its name
        nobrs_container= page_soup.findAll("nobr")

        for item in nobrs_container:
            if str(item).__contains__('ChEMBL'):
                # print(str(i))
                # print(i.parent.parent.find('a'))
                chembl_id = item.parent.parent.find('a').text
                chembl_url = item.parent.parent.find('a')['href']
                chembl_ids_dict[compound_d_id] = (chembl_id, chembl_url)

    print('Crawling of '+str(counter)+' compounds completed...')

    return chembl_ids_dict


def chembl_database_crawler(compounds_dict):

    print(str(len(compounds_dict))+' compounds in total')
    chembl_ids_dict = {}

    counter = 0
    for d_compound_id, chembl_id in compounds_dict.items():
        counter += 1
        print(str(counter) + ') Crawling on ' + chembl_id)
        # open up connection
        my_url = 'https://www.ebi.ac.uk/chembldb/compound/inspect/' + chembl_id
        # take the page
        uClient = uReq(my_url)
        # dump the html
        page_html = uClient.read()
        # close the connection
        uClient.close()
        page_soup = soup(page_html, "html.parser")

        # collect the container that contains the smiles representation
        compound_representations_container = page_soup.findAll("table", {"class": "contenttable_lmenu"})[1]

        representations = compound_representations_container.findAll("td")

        smiles_index = 0
        for name in representations:
            smiles_index += 1
            if name.text.encode('utf-8').__contains__('SMILES'):
                break
        if smiles_index < len(representations):
            if representations[smiles_index].find("a"):
                print('Had to download the smiles file')
                smiles_url = 'https://www.ebi.ac.uk'+representations[smiles_index].find("a")['href'].encode('utf-8').strip()
                data = urllib.urlopen(smiles_url)
                smiles_representation = ''
                for line in data:  # files are iterable
                    smiles_representation = smiles_representation + line

                print('The smiles is :  '+smiles_representation.strip())
                chembl_ids_dict[d_compound_id] = smiles_representation.strip()
            else:
                smiles_representation = representations[smiles_index].text.encode('utf-8').strip()
                chembl_ids_dict[d_compound_id] = smiles_representation.split('...')[0].strip()
        else:
            print('compound '+chembl_id+' doesnt have a smiles representation')


    print('Crawling of '+str(counter)+' ChEMBL ids completed...')

    return chembl_ids_dict

'''
def create_compound_features(filepath):
    number_of_lines(filepath)
    print('Loading the ' + str(filepath) + ' file...')
    fp = open(filepath)
    output_file = open(filepath.split('_smiles')[0]+'_features'+'.csv', 'w')
    line = fp.readline()
    count = 0
    # read the file line by line
    while line:
        count += 1
        if count % 1000 == 0:
            print(str(count)+' lines loaded')
        # print(line)
        # extract the feature ids from each line-instance

        f = line.split(',')
        print(str(count) + ')' + str(f))
        smiles = f[1]
        m = rdkit.Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprint(m, 2)
        output_file.write(f[0] + ' ' + ' '.join(str(f) for f in fingerprint.GetNonzeroElements()) + '\n')
        line = fp.readline()

    fp.close()
    output_file.close()
    print('Loading completed')
'''


def dict_tuple_to_csv(filename, dict):
    output_file = open(filename+'.csv', 'w')
    for compound_id, chembl_tuple in dict.items():
        output_file.write(compound_id + ',' + chembl_tuple[0] + ',' + chembl_tuple[1] +'\n')
    output_file.close()


def dict_to_csv(filename, dict):
    output_file = open(filename+'.csv', 'w')
    for compound_id, smiles in dict.items():
        output_file.write(compound_id + ',' + smiles +'\n')
    output_file.close()


def number_of_lines(filepath):
    print('filename: '+str(filepath))
    num_lines = sum(1 for line in open(filepath))
    print('File:'+str(filepath)+' has '+str(num_lines)+' lines')


def read_file(filepath):
    number_of_lines(filepath)
    print('Loading the ' + str(filepath) + ' file...')
    fp = open(filepath)
    line = fp.readline()
    count = 0
    compound_dict = {}
    # read the file line by line
    while line:
        count += 1
        if count % 1000 == 0:
            print(str(count)+' lines loaded')
        # extract the feature ids from each line-instance

        f = line.split(',')
        print(str(count) + ')' + str(f))
        d_compound_id = f[0]
        chembl_id = f[1]
        compound_dict[d_compound_id] = chembl_id
        line = fp.readline()

    print('Loading completed')
    print('The list has '+str(len(compound_dict))+' entries')
    return compound_dict


def get_kegg_data(datasets_dir):

    for inputfile in datasets_dir:
        compounds_targets_dict = load_dataset(inputfile+'.txt')
        chembl_ids_dict = kegg_database_crawler(list(compounds_targets_dict.keys()))
        dict_tuple_to_csv(inputfile, chembl_ids_dict)


def get_chembl_data(smiles_datasets_dir, datasets_dir):

    for smiles_input_file, inputfile in zip(smiles_datasets_dir, datasets_dir):
        compounds_dict = read_file(inputfile+'.csv')
        compounds_smiles_dict = chembl_database_crawler(compounds_dict)
        dict_to_csv(smiles_input_file, compounds_smiles_dict)
        # create_compound_features(smiles_input_file+'.csv')                            DONT FORGET TO UNCOMMMENT THIS IF Y0U WORK WITH PYTHON 2.7 !!!!!!!!!!!!!!


def sparse_matrix_generator(targets_per_compound_dict, features_per_compound_dict, features_dict, targets_dict, mode):

    train_mode = False
    if mode == 'train':
        train_mode = True

    # for creating the features sparse matrix
    f_indptr = [0]
    f_indices = []
    f_data = []

    # for creating the labels sparse matrix
    l_indptr = [0]
    l_indices = []
    l_data = []

    features_counter = 0
    targets_counter = 0

    if train_mode:
        for d_compound_id, features_list in features_per_compound_dict.items():
            if d_compound_id in targets_per_compound_dict:
                labels_counter = 0
                for target in targets_per_compound_dict[d_compound_id]:
                    if not target in targets_dict:
                        targets_dict[target] = (targets_counter, False)
                        targets_counter += 1

                    labels_counter += 1
                    target_index = targets_dict[target][0]
                    l_indices.append(target_index)
                    l_data.append(1)
                l_indptr.append(len(l_indices))

                for feature in features_list:
                    if not feature in features_dict:
                        features_dict[feature] = (features_counter, False)
                        features_counter += 1

                    feature_index = features_dict[feature][0]
                    f_indices.append(feature_index)
                    f_data.append(1)
                f_indptr.append(len(f_indices))

    else:

        all_targets_inserted = False
        all_features_inserted = False
        for d_compound_id, features_list in features_per_compound_dict.items():
            labels_counter = 0
            temp_l_indices = []
            temp_l_data = []
            if d_compound_id in targets_per_compound_dict:
                for target in targets_per_compound_dict[d_compound_id]:
                    if target in targets_dict:
                        targets_dict[target] = (targets_dict[target][0], True)
                        labels_counter += 1
                        target_index = targets_dict[target][0]
                        temp_l_indices.append(target_index)
                        temp_l_data.append(1)

            if labels_counter > 0:
                l_indices = list(np.concatenate((l_indices, temp_l_indices), axis=None))
                l_data = list(np.concatenate((l_data, temp_l_data), axis=None))
                if all_targets_inserted == False:
                    added_counter = 0
                    for remaining_target in [compound_id for compound_id, tupl in targets_dict.items() if tupl[1] == False]:
                        added_counter += 1
                        target_index = targets_dict[remaining_target][0]
                        l_indices.append(target_index)
                        l_data.append(0)
                all_targets_inserted = True
                l_indptr.append(len(l_indices))

                for feature in features_list:
                    if feature in features_dict:
                        features_dict[feature] = (features_dict[feature][0], True)
                        feature_index = features_dict[feature][0]
                        f_indices.append(feature_index)
                        f_data.append(1)

                if all_features_inserted == False:
                    added_counter = 0
                    for remaining_feature in [feature_id for feature_id, tupl in features_dict.items() if tupl[1] == False]:
                        added_counter += 1
                        feature_index = features_dict[remaining_feature][0]
                        f_indices.append(feature_index)
                        f_data.append(0)
                    print('Added '+str(added_counter)+' features')
                    all_features_inserted = True
                f_indptr.append(len(f_indices))


    print('Creation of X and Y matrices is done')

    X = csr_matrix((f_data, f_indices, f_indptr), dtype=int)
    y = csr_matrix((l_data, l_indices, l_indptr), dtype=int)

    return X, y



def load_features_per_compound(filepath):

    number_of_lines(filepath+'.csv')
    fp = open(filepath+'.csv')
    line = fp.readline()
    count = 0
    features_per_compound_dict = {}
    # read the file line by line
    while line:
        count += 1
        if count % 1000 == 0:
            print(str(count)+' lines loaded')
        # extract the feature ids from each line-instance

        f = line.split(' ')
        d_compound_id = f[0]
        features = f[1:]
        features[-1] = features[-1].rstrip()
        features_per_compound_dict[d_compound_id] = features
        line = fp.readline()

    print('Loading completed')
    print('The list has '+str(len(features_per_compound_dict))+' entries')

    return features_per_compound_dict


def k_fold_split(filepath, k):

    features_per_compound_dict = load_features_per_compound(filepath)

    keys = list(features_per_compound_dict.keys())
    random.shuffle(keys)

    #splits the list of keys-compounds into fold_size equal sized folds
    folds = np.array_split(np.asarray(keys), k)

    return folds


def get_train_and_test_dicts_from_fold(filepath, test_set_fold, mode, unlabeled_percentage = 0.9):

    features_per_compound_dict = load_features_per_compound(filepath)
    train_dict = {}
    test_dict = {}

    for key, features in features_per_compound_dict.items():
        if key in test_set_fold:
            test_dict[key] = features
        else:
            train_dict[key] = features

    if mode == 'co-training':
        unlabeled_dict = {}
        unlabeled_dict_size = int(len(train_dict)*unlabeled_percentage)
        keys = list(train_dict.keys())
        random.shuffle(keys)
        unlabeled_keys = keys[0: unlabeled_dict_size]
        train_keys = keys[unlabeled_dict_size: len(keys)]

        unlabeled_dict = {key: train_dict[key] for key in unlabeled_keys}
        train_dict = {key: train_dict[key] for key in train_keys}
        return train_dict, test_dict, unlabeled_dict
    else:

        return train_dict, test_dict


def random_split(filepath, precentage):

    features_per_compound_dict = load_features_per_compound(filepath)

    # split randomly the dataset to train and test set
    train_samples_num = int(len(features_per_compound_dict) * precentage)
    keys = list(features_per_compound_dict.keys())
    random.shuffle(keys)

    train_dict = {}
    test_dict = {}
    read_counter = 0
    for key in keys:
        read_counter += 1
        if read_counter < train_samples_num:
            train_dict[key] = features_per_compound_dict[key]
        else:
            test_dict[key] = features_per_compound_dict[key]

    return train_dict, test_dict


def crawl_and_create_datasets():
    datasets_dir = ['gold_standard_datasets/enzyme_dataset', 'gold_standard_datasets/GPCR_dataset',
                    'gold_standard_datasets/ion_channel_dataset', 'gold_standard_datasets/nuclear_receptor']

    get_kegg_data(datasets_dir)

    smiles_datasets_dir = ['gold_standard_datasets/enzyme_smiles_dataset', 'gold_standard_datasets/GPCR_smiles_dataset',
                     'gold_standard_datasets/ion_channel_smiles_dataset', 'gold_standard_datasets/nuclear_receptor_smiles_dataset']

    get_chembl_data(smiles_datasets_dir, datasets_dir)


def get_suffled_train_test_set():
    targets_per_compound_dict = load_dataset('gold_standard_datasets/enzyme_dataset'+'.txt')

    features_dict = {}
    targets_dict = {}

    train_dict, test_dict = random_split('gold_standard_datasets/enzyme_features', 0.7)

    X_train, y_train = sparse_matrix_generator(targets_per_compound_dict, train_dict, features_dict, targets_dict, 'train')
    X_test, y_test = sparse_matrix_generator(targets_per_compound_dict, test_dict, features_dict, targets_dict, 'test')

    print('X_train.shape: '+str(X_train.shape))
    print('y_train.shape: ' + str(y_train.shape))
    print('X_test.shape: ' + str(X_test.shape))
    print('y_test.shape: ' + str(y_test.shape))

    return X_train, y_train, X_test, y_test


def get_train_test_set_from_fold(targets_per_compound_dict, test_set_fold, features_filename, mode, unlabeled_percentage=0.9):
    features_dict = {}
    targets_dict = {}

    if mode == 'co-training':

        train_dict, test_dict, unlabeled_dict = get_train_and_test_dicts_from_fold(features_filename, test_set_fold, 'co-training', unlabeled_percentage)

        X_train, y_train = sparse_matrix_generator(targets_per_compound_dict, train_dict, features_dict, targets_dict, 'train')

        temp_features_dict = features_dict.copy()
        temp_targets_dict = targets_dict.copy()

        X_test, y_test = sparse_matrix_generator(targets_per_compound_dict, test_dict, features_dict, targets_dict, 'test')

        X_unlabeled, y_unlabeled = sparse_matrix_generator(targets_per_compound_dict, unlabeled_dict, temp_features_dict, temp_targets_dict, 'test')

    else:
        train_dict, test_dict = get_train_and_test_dicts_from_fold(features_filename, test_set_fold, 'standard')

        X_train, y_train = sparse_matrix_generator(targets_per_compound_dict, train_dict, features_dict, targets_dict, 'train')

        X_test, y_test = sparse_matrix_generator(targets_per_compound_dict, test_dict, features_dict, targets_dict, 'test')
    print('X_train.shape: '+str(X_train.shape))
    print('y_train.shape: ' + str(y_train.shape))
    print('X_test.shape: ' + str(X_test.shape))
    print('y_test.shape: ' + str(y_test.shape))

    if mode == 'co-training':
        print('X_unlabeled.shape: ' + str(X_unlabeled.shape))
        print('y_unlabeled.shape: ' + str(y_unlabeled.shape))
        return X_train, y_train.todense(), X_test, y_test.todense(), X_unlabeled, y_unlabeled.todense()
    else:
        return X_train, y_train.todense(), X_test, y_test.todense()


def main():

    print('golden_dataset')
    # crawl_and_create_datasets()

    # return get_suffled_train_test_set()



# main()