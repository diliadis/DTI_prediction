import numpy as np
import scipy.sparse as sp
import random
import sklearn
from sklearn.multioutput import ClassifierChain
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_array, check_X_y, check_random_state
from hypopt import GridSearch


class ClassifierChain_with_random_undesampling(ClassifierChain):

    def fit(self, X, Y, X_val, Y_val):
        param_grid = []

        val_mode = False
        if X_val is not None and Y_val is not None:
            val_mode = True
            if isinstance(self.base_estimator, sklearn.svm.SVC):
                self.base_estimator.set_params(probability=True)
                param_grid = [
                    {'C': [1, 10, 100], 'kernel': ['linear']},
                    {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                ]
            elif isinstance(self.base_estimator, sklearn.linear_model.LogisticRegression):
                param_grid = [
                    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.0001, 0.001, 0.01, 1, 100]}
                ]
        else:
            print('not in validation mode')
            if isinstance(self.base_estimator, sklearn.svm.SVC):
                self.base_estimator.set_params(gamma='auto')
                self.base_estimator.set_params(kernel='linear')
            elif isinstance(self.base_estimator, sklearn.linear_model.LogisticRegression):
                self.base_estimator.set_params(solver='liblinear')


        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Y : array-like, shape (n_samples, n_classes)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, Y = check_X_y(X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)

        check_array(X, accept_sparse=True)
        self.order_ = self.order
        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == 'random':
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        self.estimators_ = [clone(self.base_estimator)
                            for _ in range(Y.shape[1])]

        self.classes_ = []

        if self.cv is None:
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format='lil')

            if val_mode:
                Y_pred_chain_val = sp.lil_matrix((X_val.shape[0], Y.shape[1]))
                X_val_aug = sp.hstack((X_val, Y_pred_chain_val.copy()), format='lil')

        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format='lil')

            if val_mode:
                Y_pred_chain_val = sp.lil_matrix((X_val.shape[0], Y.shape[1]))
                X_val_aug = sp.hstack((X_val, Y_pred_chain_val.copy()), format='lil')

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

            if val_mode:
                Y_pred_chain_val = np.zeros((X_val.shape[0], Y.shape[1]))
                X_val_aug = sp.hstack((X_val, Y_pred_chain_val.copy()))

        del Y_pred_chain

        if val_mode:
            del Y_pred_chain_val

        class_1 = 1
        class_2 = 0
        if -1 in Y:
            class_2 = -1

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]

            # class_1_counter = np.count_nonzero(y[:, 0] == class_1)
            # class_2_counter = np.count_nonzero(y[:, 0] == class_2)
            class_1_counter = y.flatten().tolist().count(class_1)
            class_2_counter = y.flatten().tolist().count(class_2)

            if class_1_counter <= class_2_counter:
                minority_index = 1
                minority_counter = class_1_counter
                majority_index = 0
            else:
                minority_index = 0
                minority_counter = class_2_counter
                majority_index = 1

            # get all the minority samples
            sampled_index = [index for index, label in enumerate(y) if label == minority_index]
            sampled_y = [minority_index] * minority_counter
            #print('m'+str(len(sampled_y))+' '+str(minority_index)+'s')


            sampled_index.extend(sampled_index)
            sampled_y.extend(sampled_y)


            # sample the majority samples
            temp_sampled_index = [index for index, label in enumerate(y) if label == majority_index]
            #print(str(len(temp_sampled_index))+' '+str(majority_index)+'s')
            sampled_index.extend(random.sample(temp_sampled_index, minority_counter))
            sampled_y.extend([majority_index] * minority_counter)

            print(str(self.order_[chain_idx])+') training label:'+str(chain_idx)+' with '+str(len(sampled_y))+' instances ')
            #print('X_aug[np.array(sampled_index), :(X.shape[1] + chain_idx)]: '+str(X_aug[np.array(sampled_index), :(X.shape[1] + chain_idx)].shape))
            #print('np.array(sampled_y): '+str(np.array(sampled_y).shape))

            if val_mode:
                # for the grid search version
                gs = GridSearch(model=estimator, param_grid=param_grid)
                temp_estimator = gs.fit(X_aug[np.array(sampled_index), :(X.shape[1] + chain_idx)], np.array(sampled_y), X_val_aug[:, :(X.shape[1] + chain_idx)], Y_val[:, self.order_[chain_idx]], scoring='roc_auc')
            else:
                estimator.fit(X_aug[np.array(sampled_index), :(X.shape[1] + chain_idx)], np.array(sampled_y))


            if chain_idx < len(self.estimators_) - 1:
                # produce the predictions and add them as features for the next classifier in the chain
                col_idx = X.shape[1] + chain_idx
                # use all the available features(from X_train and all the predictions thus far)

                if val_mode:
                    previous_predictions_val = temp_estimator.predict(X_val_aug[:, :col_idx])
                    previous_predictions = temp_estimator.predict(X_aug[:, :col_idx])
                else:
                    previous_predictions = estimator.predict(X_aug[:, :col_idx])

                # insert the predictions as features to be used for the next classifier in the chain
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(previous_predictions, 1)
                    if val_mode:
                        X_val_aug[:, col_idx] = np.expand_dims(previous_predictions_val, 1)
                else:
                    X_aug[:, col_idx] = previous_predictions
                    if val_mode:
                        X_val_aug[:, col_idx] = previous_predictions_val


            # -------------------------------------------------------------------------------------------------------------------------------
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_idx],
                    y=y, cv=self.cv)
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result

            if val_mode:
                self.classes_.append(temp_estimator.classes_)
                self.estimators_[chain_idx] = temp_estimator
            else:
                self.classes_.append(estimator.classes_)

        return self