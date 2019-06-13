import deepchem
import numpy as np
from scipy import sparse
import os

class moleculenet_dataset():

    def __init__(self, dataset_name='tox21'):
        self.loading_functions = {
        'muv': (deepchem.molnet.load_muv, 'random'), # 17 targets, 93087 compounds
        'pcba': (deepchem.molnet.load_pcba, 'random'), # 128 targets, 437929 compounds
        'sider': (deepchem.molnet.load_sider, 'random'), # 27 targets, 1427 compounds
        'tox21': (deepchem.molnet.load_tox21, 'random'), # 12 targets, 7831 compounds
        'toxcast': (deepchem.molnet.load_toxcast, 'random') # 617 targets, 8575 compounds
        }
        try:
            self.loading_functions[dataset_name]
            self.dataset_name = dataset_name
            self.split_type = self.loading_functions[dataset_name][1]
        except KeyError:
            print("false dataset name")


    def get_dataset(self, save_dir=None):
        tasks, datasets, transformers = self.loading_functions[self.dataset_name][0](split=self.split_type)
        (train_dataset, valid_dataset, test_dataset) = datasets
        X_train, y_train = train_dataset.X, train_dataset.y
        X_test, y_test = test_dataset.X, test_dataset.y
        X_val, y_val = valid_dataset.X, valid_dataset.y

        sX_train = sparse.csr_matrix(X_train)
        sX_test = sparse.csr_matrix(X_test)
        sX_val = sparse.csr_matrix(X_val)

        '''
        if save_dir is not None:
            # save_dir = /datasets/tox21
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_dir+'/'+self.split_type+'_split/sX_train.npy', sX_train)
            np.save(save_dir+'/'+self.split_type+'_split/sX_test.npy', sX_test)
            np.save(save_dir+'/'+self.split_type+'_split/y_test.npy', y_test)
            np.save(save_dir+'/'+self.split_type+'_split/y_train.npy', y_train)
        '''
        return sX_train, y_train, sX_test, y_test, sX_val, y_val


