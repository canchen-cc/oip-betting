import numpy as np
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal, norm
import heapq
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, linear_kernel
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances
TYPES_KERNEL = ['rbf', 'laplace']


def compute_hyperparam(data: np.ndarray,
                       kernel_type: TYPES_KERNEL = 'rbf', style='median') -> float:
    """
    Median heuristic for computing kernel hyperparameters
    """
    if kernel_type == 'rbf':
        if data.ndim == 1:
            dist = pairwise_distances(data.reshape(-1, 1))**2
        else:
            dist = pairwise_distances(data)**2
    elif kernel_type == 'laplace':
        if data.ndim == 1:
            dist = pairwise_distances(data.reshape(-1, 1), metric='l1')
        else:
            dist = pairwise_distances(data, metric='l1')
    else:
        raise ValueError('Unknown kernel type')
    mask = np.ones_like(dist, dtype=bool)
    np.fill_diagonal(mask, 0)
    if style == 'median':
        return 1/(2*np.median(dist[mask]))
    elif style == 'mean':
        return 1/(2*np.mean(dist[mask]))


class kClosestFinder:
    def calculate_distance(self, x1, x2):
        return (math.sqrt(sum((x2-x1)**2)))

    def kClosest(self, points, query_point, K):
        heap_list = []
        for cur_ind, x in enumerate(points):
            dist = -self.calculate_distance(x, query_point)
            if len(heap_list) == K:
                heapq.heappushpop(heap_list, (dist, x, cur_ind))
            else:
                heapq.heappush(heap_list, (dist, x, cur_ind))

        return [cur_ind for (dist, x, cur_ind) in heap_list]


def form_new_fake_pts(X, Y):
    """ 
    Given two points from P_{XY}, form 4 points for training
    """
    labels_new = np.array([1, 1, -1, -1])

    orig_data_1 = np.hstack([X[0], Y[0]])
    orig_data_2 = np.hstack([X[1], Y[1]])
    perm_data_1 = np.hstack([X[0], Y[1]])
    perm_data_2 = np.hstack([X[1], Y[0]])

    features_new = np.vstack(
        [orig_data_1, orig_data_2, perm_data_1, perm_data_2])

    return features_new, labels_new


def form_new_training_set(features_new, labels_new, features_old=None, labels_old=None):
    """
    Add new training data to existing data

    Parameters:
        X,Y: 
            two pairs of new obs for P_{XY}
        train_old:
            prev_training_data for fitting a classifier
    """

    if features_old is None:
        ind_list = [i for i in range(4)]
        shuffle(ind_list)
        return features_new[ind_list], labels_new[ind_list]
    else:
        features_all = np.vstack([features_old, features_new])
        labels_all = np.hstack([labels_old, labels_new])
        N = features_all.shape[0]
        ind_list = [i for i in range(N)]
        shuffle(ind_list)
        return features_all[ind_list], labels_all[ind_list]


class RFPayoff(object):
    def __init__(self):
        self.clf = None
        self.depth_tuning = False
        self.param_grid = {
            'max_depth': [5, 10, 20]
        }
        self.features_train = None
        self.labels_train = None
        self.num_processed_pairs = 1
        self.retrain_ind = 1
        self.retrain_step = 10
        self.n_splits = 2

    def update_predictor(self):
        clf = RandomForestClassifier()
        if self.depth_tuning:
            grid_clf = GridSearchCV(clf, self.param_grid, cv=self.n_splits)
            grid_clf.fit(self.features_train, self.labels_train)
            self.clf = grid_clf.best_estimator_
        else:
            self.clf = clf.fit(self.features_train, self.labels_train)

    def evaluate_payoff(self, X, Y):
        if self.num_processed_pairs < 20:
            payoff_fn = 0
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            self.num_processed_pairs += 1
            return payoff_fn
        elif self.num_processed_pairs == 20:
            payoff_fn = 0
            # update training data
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            # train for the first time
            self.update_predictor()
            self.num_processed_pairs += 1
            self.retrain_ind += 1
            self.retrain_step = 10 * self.retrain_ind*(self.retrain_ind+1)
            return payoff_fn
        else:
            # evaluate payoff
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            confidences = abs(self.clf.predict_proba(
                fake_features).max(axis=1)-0.5)
            accuracies = 0.5 - (self.clf.predict(fake_features) !=
                                fake_labels).astype(float)
            payoff_fn = sum(confidences*accuracies)
            # add training data
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            # update predictor if needed
            if self.num_processed_pairs == self.retrain_step:
                self.update_predictor()
                self.retrain_ind += 1
                self.retrain_step = 10 * \
                    self.retrain_ind * (self.retrain_ind+1)
            self.num_processed_pairs += 1
            return payoff_fn


class MLPPayoff(object):
    def __init__(self):
        self.clf = None
        self.depth_tuning = False
        self.num_neurons = [128, 64, 32]
        # self.num_neurons = [128, 128, 64, 64, 32, 32]
        self.features_train = None
        self.labels_train = None
        self.num_processed_pairs = 1
        self.retrain_step = 5
        self.retrain_counter = 1
        self.low_percent = 0.01
        self.high_percent = 0.1
        self.bet_strategy = "confidence"
        self.callback = None

    def evaluate_payoff(self, X, Y):
        if self.num_processed_pairs < 10:
            payoff_fn = 0
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            self.num_processed_pairs += 1
            return payoff_fn
        elif self.num_processed_pairs == 10:
            payoff_fn = 0
            # update training data
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            # # train for the first time
            self.clf = MLPClassifier(hidden_layer_sizes=self.num_neurons, learning_rate='constant',
                                     early_stopping=True, validation_fraction=0.2, warm_start=True, max_iter=25, n_iter_no_change=3)

            self.clf.fit(self.features_train, self.labels_train)

            # self.clf = Sequential()
            # self.clf.add(
            #     Dense(128, input_dim=fake_features.shape[1], activation='relu'))
            # self.clf.add(Dense(64, activation='relu'))
            # self.clf.add(Dense(32, activation='relu'))

            # if self.bet_strategy == "accuracy" or self.bet_strategy == "confidence-accuracy":
            #     self.clf.add(Dense(1, activation='sigmoid'))
            #     self.clf.compile(loss='binary_crossentropy',
            #                   optimizer=Adam(), metrics=['accuracy'])
            #     self.callback = EarlyStopping(
            #         monitor="val_accuracy",
            #         min_delta=0,
            #         patience=3,
            #         verbose=0,
            #         mode="auto",
            #         baseline=None,
            #         restore_best_weights=True,
            #         start_from_epoch=0)

            #     hst =self.clf.fit(self.features_train, (1+self.labels_train)/2, validation_split=0.2, epochs=25,callbacks=[self.callback], verbose=0)
            # elif self.bet_strategy == "confidence":
            #     self.clf.add(Dense(1, activation='tanh'))
            #     self.clf.compile(loss='hinge', optimizer=Adam())
            #     self.callback = EarlyStopping(
            #         monitor="val_loss",
            #         min_delta=0,
            #         patience=3,
            #         verbose=0,
            #         mode="auto",
            #         baseline=None,
            #         restore_best_weights=True,
            #         start_from_epoch=0)

            #     hst =self.clf.fit(self.features_train, self.labels_train, validation_split=0.2, epochs=25,callbacks=[self.callback], verbose=0)

            self.num_processed_pairs += 1
            return payoff_fn
        else:
            # evaluate payoff
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            if self.bet_strategy == "confidence":
                # confidences = self.clf.predict(fake_features,verbose=0)
                # payoff_fn = (sum(confidences[:2])-sum(confidences[2:]))/4
                # print(self.clf.predict_proba(
                #     fake_features))
                # confidences = abs(self.clf.predict_proba(
                #     fake_features).max(axis=1)-0.5)
                # accuracies = 0.5 - (self.clf.predict(fake_features) !=
                #                     fake_labels).astype(float)
                # payoff_fn = sum(confidences*accuracies)
                pred_scores = 2*self.clf.predict_proba(fake_features)[:, 1]-1
                payoff_fn = (sum(pred_scores[:2])-sum(pred_scores[2:]))/4

            elif self.bet_strategy == 'accuracy':
                # confidences = self.clf.predict(fake_features,verbose=0)
                # class_preds = (confidences>=0.5).astype('float')
                # payoff_fn = class_preds[:2].mean()-class_preds[2:].mean()
                preds = self.clf.predict(fake_features)
                payoff_fn = (sum(preds[:2])-sum(preds[2:]))/4

            # add training data
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            # update predictor

            if self.num_processed_pairs % self.retrain_step == 0:

                self.clf.validation_scores_ = list()
                self.clf._no_improvement_count = 0
                self.clf.best_validation_score_ = 0.5
                self.clf.fit(self.features_train, self.labels_train)
                # if self.bet_strategy == "confidence":
                #     hst =self.clf.fit(self.features_train, self.labels_train, validation_split=0.2, epochs=25,callbacks=[self.callback], verbose=0)
                # elif self.bet_strategy == "accuracy" or self.bet_strategy == "confidence-accuracy":
                #     hst =self.clf.fit(self.features_train, (1+self.labels_train)/2, validation_split=0.2, epochs=25,callbacks=[self.callback], verbose=0)
            if self.retrain_step / self.num_processed_pairs <= self.low_percent:
                self.retrain_step += 5

            self.num_processed_pairs += 1
            return payoff_fn


class LDA(object):
    def __init__(self):
        self.mu_neg = None
        self.mu_pos = None
        self.sec_moment_pos = None
        self.sec_moment_neg = None
        self.sigma_neg = None
        self.sigma_pos = None
        self.num_pts = 0

    def update_predictor(self, x, y):
        if self.num_pts == 0:
            self.mu_pos = x[y == 1].sum(axis=0) / 2
            self.mu_neg = x[y == -1].sum(axis=0) / 2
            self.sec_moment_pos = x[y == 1].T.dot(x[y == 1]) / 2
            self.sec_moment_neg = x[y == -1].T.dot(x[y == -1]) / 2
            self.sigma_pos = self.sec_moment_pos - \
                np.outer(self.mu_pos, self.mu_pos)
            self.sigma_neg = self.sec_moment_neg - \
                np.outer(self.mu_pos, self.mu_neg)
            self.num_pts += 2
        else:
            self.mu_pos = self.num_pts / \
                (self.num_pts+2)*self.mu_pos + 1 / \
                (self.num_pts+2)*x[y == 1].sum(axis=0)
            self.mu_neg = self.num_pts / \
                (self.num_pts+2)*self.mu_neg + 1 / \
                (self.num_pts+2)*x[y == -1].sum(axis=0)
            self.sec_moment_pos = self.num_pts / \
                (self.num_pts+2)*self.sec_moment_pos + 1 / \
                (self.num_pts+2)*x[y == 1].T.dot(x[y == 1])
            self.sec_moment_neg = self.num_pts / \
                (self.num_pts+2)*self.sec_moment_neg + 1 / \
                (self.num_pts+2)*x[y == -1].T.dot(x[y == -1])
            self.sigma_pos = self.sec_moment_pos - \
                np.outer(self.mu_pos, self.mu_pos)
            self.sigma_neg = self.sec_moment_neg - \
                np.outer(self.mu_pos, self.mu_neg)
            self.num_pts += 2

    def predict_proba(self, x):
        neg_prob = multivariate_normal.pdf(
            x, mean=self.mu_neg, cov=self.sigma_neg)
        pos_prob = multivariate_normal.pdf(
            x, mean=self.mu_pos, cov=self.sigma_pos)
        denom = pos_prob+neg_prob
        ind_unc = (denom <= 1e-4)
        pos_prob[ind_unc] = 0.5
        denom[ind_unc] = 1
        # if pos_prob+neg_prob>=1e-4:
        return pos_prob/denom
        # else:
        # return 0.5

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5)


class LDAPayoff(object):
    def __init__(self):
        self.clf = None
        self.features_train = None
        self.labels_train = None
        self.num_processed_pairs = 0
        self.retrain_steps = 1
        self.bet_strategy = "accuracy"

    def evaluate_payoff(self, X, Y):
        if self.num_processed_pairs < self.retrain_steps:
            payoff_fn = 0
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            if self.num_processed_pairs == 0:
                self.clf = LDA()
            self.clf.update_predictor(fake_features, fake_labels)
            self.num_processed_pairs += 1
            return payoff_fn
        elif self.num_processed_pairs == self.retrain_steps:
            payoff_fn = 0
            # update training data
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            self.clf.update_predictor(fake_features, fake_labels)
            self.num_processed_pairs += 1
            return payoff_fn
        else:
            # evaluate payoff
            if self.bet_strategy == 'confidence':
                fake_features, fake_labels = form_new_fake_pts(X, Y)
                pred_scores = 2*self.clf.predict_proba(fake_features)-1
                payoff_fn = (sum(pred_scores[:2])-sum(pred_scores[2:]))/4
                # confidences = self.clf.predict_proba(
                #     fake_features) * (self.clf.predict(fake_features) >= 1/2).astype(float)
                # payoff_fn = confidences[:2].mean()-confidences[2:].mean()
            elif self.bet_strategy == 'accuracy':
                fake_features, fake_labels = form_new_fake_pts(X, Y)
                # preds are -1 and 1
                preds = 2*self.clf.predict(fake_features)-1
                payoff_fn = (sum(preds[:2])-sum(preds[2:]))/4
                # payoff_fn = 1 - 0.5*sum(self.clf.predict(fake_features) !=
                #                         fake_labels).astype(float)

                # payoff_fn = sum(confidences*accuracies)
            # elif self.bet_strategy == 'confidence':
            #     fake_features, fake_labels = form_new_fake_pts(X, Y)
            #     confidences = abs(self.clf.predict_proba(
            #         fake_features)-0.5)
            #     accuracies = 0.5 - (self.clf.predict(fake_features) !=
            #                         fake_labels).astype(float)
            #     payoff_fn = sum(confidences*accuracies)
            #     # add training data
            #     # update predictor if needed
            self.clf.update_predictor(fake_features, fake_labels)
            self.num_processed_pairs += 1
            return payoff_fn


class OracleLDAPayoff(object):
    def __init__(self):
        self.clf = None
        self.features_train = None
        self.labels_train = None
        self.num_processed_pairs = 0
        self.retrain_steps = 1
        self.true_beta = None

    def evaluate_payoff(self, X, Y):
        if self.num_processed_pairs == 0:
            self.clf = LDA()
            self.clf.mu_pos = np.array([0, 0])
            self.clf.mu_neg = np.array([0, 0])
            self.clf.sigma_pos = np.array(
                [[1, self.true_beta], [self.true_beta, 1+self.true_beta**2]])
            self.clf.sigma_neg = np.array([[1, 0], [0, 1+self.true_beta**2]])

        fake_features, fake_labels = form_new_fake_pts(X, Y)
        confidences = abs(self.clf.predict_proba(
            fake_features)-0.5)
        accuracies = 0.5 - (self.clf.predict(fake_features) !=
                            fake_labels).astype(float)
        payoff_fn = sum(confidences*accuracies)
        self.num_processed_pairs += 1
        return payoff_fn


class KNNPayoff(object):
    def __init__(self):
        self.num_pts = 0
        self.num_neighbors = 0
        self.features_train = None
        self.proc_type = 'old'
        self.labels_train = None
        self.num_processed_pairs = 1
        self.bet_strategy = "confidence"
        self.regularized = True

    def update_predictor(self):
        self.num_pts += 4
        # update num of neig
        self.num_neighbors = np.ceil(np.sqrt(self.num_pts)).astype('int')

    def evaluate_payoff(self, X, Y):
        if self.num_processed_pairs == 1:
            payoff_fn = 0
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            self.update_predictor()

            self.num_processed_pairs += 1
            return payoff_fn
        else:
            # find indices
            fake_features, fake_labels = form_new_fake_pts(X, Y)
            if self.bet_strategy == 'confidence':
                payoff_fn = 0
                for cur_ind, cur_value in enumerate(fake_features):
                    # if self.proc_type == 'old':
                    temp_set = ((self.features_train-cur_value)**2).sum(axis=1)
                    ind_to_consider = np.argsort(temp_set)[:self.num_neighbors]
                    if self.regularized:
                        # cur_conf = (
                        #     np.sum(self.labels_train[ind_to_consider])/(self.num_neighbors+1)+1)/2
                        cur_conf = np.sum(
                            self.labels_train[ind_to_consider])/(self.num_neighbors+1)
                    else:
                        cur_conf = np.mean(self.labels_train[ind_to_consider])
                        # cur_conf = (
                        # np.mean(self.labels_train[ind_to_consider])+1)/2
                    # cur_pred = 2*(cur_conf >= 0.5)-1
                    # cur_acc = 0.5 - \
                    #     (cur_pred != fake_labels[cur_ind]).astype(float)
                    if cur_ind in [0, 1]:
                        payoff_fn += cur_conf/4
                    else:
                        payoff_fn -= cur_conf/4
                    # payoff_fn += abs(cur_conf-0.5)*cur_acc
            # elif self.bet_strategy == 'confidence-accuracy':
            #     payoff_fn = 0
            #     for cur_ind, cur_value in enumerate(fake_features):
            #         # if self.proc_type == 'old':
            #         temp_set = ((self.features_train-cur_value)**2).sum(axis=1)
            #         ind_to_consider = np.argsort(temp_set)[:self.num_neighbors]
            #         if self.regularized:
            #             cur_conf = (
            #                 np.sum(self.labels_train[ind_to_consider])/(self.num_neighbors+1)+1)/2
            #         else:
            #             cur_conf = (
            #                 np.mean(self.labels_train[ind_to_consider])+1)/2
            #         # cur_pred = 2*(cur_conf >= 0.5)-1
            #         # cur_acc = 0.5 - \
            #         #     (cur_pred != fake_labels[cur_ind]).astype(float)
            #         if cur_ind in [0, 1]:
            #             payoff_fn += 0.5*cur_conf * \
            #                 (cur_conf >= 0.5).astype('float')
            #         else:
            #             payoff_fn -= 0.5*cur_conf * \
            #                 (cur_conf >= 0.5).astype('float')
            elif self.bet_strategy == 'accuracy':
                payoff_fn = 0
                for cur_ind, cur_value in enumerate(fake_features):
                    # if self.proc_type == 'old':
                    temp_set = ((self.features_train-cur_value)**2).sum(axis=1)
                    ind_to_consider = np.argsort(temp_set)[:self.num_neighbors]
                    if self.regularized:
                        # cur_conf = (
                        #     np.sum(self.labels_train[ind_to_consider])/(self.num_neighbors+1)+1)/2
                        cur_conf = np.sum(
                            self.labels_train[ind_to_consider])/(self.num_neighbors+1)
                    else:
                        cur_conf = np.mean(self.labels_train[ind_to_consider])
                        # cur_conf = (
                        #     np.mean(self.labels_train[ind_to_consider])+1)/2
                    cur_pred = (cur_conf >= 0).astype('float')
                    # cur_acc = 0.5 - \
                    #     (cur_pred != fake_labels[cur_ind]).astype(float)
                    if cur_ind in [0, 1]:
                        payoff_fn += cur_pred / 4
                    else:
                        payoff_fn -= cur_pred / 4
                    # payoff_fn += 0.5*cur_acc
                # elif self.proc_type == 'new':
                #     cl_fndr = kClosestFinder()
                #     ind_to_consider = cl_fndr.kClosest(self.features_train, cur_value, K=self.num_neighbors)
                #     cur_conf = (np.mean(self.labels_train[ind_to_consider])+1)/2
                #     cur_pred = 2*(cur_conf>=0.5)-1
                #     cur_acc = 0.5 - (cur_pred != fake_labels[cur_ind]).astype(float)
                #     payoff_fn += abs(cur_conf-0.5)*cur_acc
            # update model
            self.update_predictor()
            self.features_train, self.labels_train = form_new_training_set(
                fake_features, fake_labels, self.features_train, self.labels_train)
            self.num_processed_pairs += 1
            return payoff_fn


class KNNPayoff_2ST(object):
    def __init__(self):
        self.num_neighbors = 0
        self.features_train = None
        self.proc_type = 'old'
        self.labels_train = None
        self.num_processed_pts = 1
        self.regularized = True

    def evaluate_payoff(self, Z, W):
        if self.num_processed_pts == 1:
            payoff_fn = 0
            self.features_train = np.copy(Z)
            self.labels_train = np.copy(W)
            self.num_processed_pts += 1
            self.num_neighbors = np.ceil(
                np.sqrt(self.num_processed_pts)).astype('int')
            return payoff_fn
        elif self.num_processed_pts <= 5:
            payoff_fn = 0
            self.features_train = np.vstack([self.features_train, Z])
            self.labels_train = np.vstack([self.labels_train, W])
            self.num_processed_pts += 1
            self.num_neighbors = np.ceil(
                np.sqrt(self.num_processed_pts)).astype('int')
            return payoff_fn
        else:
            # find indices

            # if self.proc_type == 'old':
            temp_set = ((self.features_train-Z)**2).sum(axis=1)
            ind_to_consider = np.argsort(temp_set)[:self.num_neighbors]
            if self.regularized:
                cur_conf = np.sum(
                    self.labels_train[ind_to_consider])/(self.num_neighbors+1)
            else:
                cur_conf = np.mean(self.labels_train[ind_to_consider])
            payoff_fn = W*cur_conf

            self.features_train = np.vstack([self.features_train, Z])
            self.labels_train = np.vstack([self.labels_train, W])
            self.num_processed_pts += 1
            self.num_neighbors = np.ceil(
                np.sqrt(self.num_processed_pts)).astype('int')

            return payoff_fn


class CNNPayoff_2ST(object):
    def __init__(self):
        self.features_train = None
        self.labels_train = None
        self.num_processed_pts = 0
        self.clf = None
        self.num_classes = 2
        self.input_shape = (64, 64, 1)
        self.batch_size = 32
        self.epochs = 25
        self.patience = 10
        self.callback = None
        self.enc = None
        self.update_steps = 10

    def evaluate_payoff(self, Z, W):
        if self.num_processed_pts == 0:
            # fit an encoder
            self.enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.enc.fit(np.array([-1, 1]).reshape(-1, 1))
            # initialize CNN
            self.clf = keras.Sequential(
                [
                    keras.Input(shape=self.input_shape),
                    keras.layers.Conv2D(
                        16, kernel_size=(3, 3), activation="relu"),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Conv2D(
                        32, kernel_size=(3, 3), activation="relu"),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Conv2D(
                        32, kernel_size=(3, 3), activation="relu"),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Conv2D(
                        64, kernel_size=(3, 3), activation="relu"),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Flatten(),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(self.num_classes, activation="softmax"),
                ]
            )
            self.clf.compile(loss="binary_crossentropy",
                             optimizer="adam", metrics=["accuracy"])

            self.callback = keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=self.patience, restore_best_weights=True)
            # add first training point

            self.features_train = Z.copy()
            self.labels_train = self.enc.transform(W.reshape(-1, 1))

            self.num_processed_pts += 1
            return 0

        elif self.num_processed_pts == 1:
            # return 0 and collect training data
            self.features_train = np.stack(
                [self.features_train, Z.copy()])
            self.labels_train = np.vstack(
                [self.labels_train, self.enc.transform(W.reshape(-1, 1))])
            self.num_processed_pts += 1
            return 0
        elif self.num_processed_pts < 20:
            # return 0 and collect training data
            self.features_train = np.vstack(
                [self.features_train, np.expand_dims(Z, 0)])
            self.labels_train = np.vstack(
                [self.labels_train, self.enc.transform(W.reshape(-1, 1))])
            self.num_processed_pts += 1
            return 0
        elif self.num_processed_pts == 20:
            # first permute data
            ind = np.arange(self.num_processed_pts)
            shuffle(ind)
            self.features_train = self.features_train[ind]
            self.labels_train = self.labels_train[ind]
            # training
            self.clf.fit(self.features_train, self.labels_train, batch_size=self.batch_size,
                         epochs=self.epochs, callbacks=[self.callback], validation_split=0.2, verbose=0)
            # first bet (rescale pred to [-1,1]), only confidence bet
            payoff_fn = W * \
                (2*self.clf.predict(np.expand_dims(Z, 0), verbose=0)[0, 1]-1)
            self.features_train = np.vstack(
                [self.features_train, np.expand_dims(Z, 0)])
            self.labels_train = np.vstack(
                [self.labels_train, self.enc.transform(W.reshape(-1, 1))])
            self.num_processed_pts += 1
            return payoff_fn

        else:
            if self.num_processed_pts % self.update_steps == 0:
                # perform a model update
                ind = np.arange(self.num_processed_pts)
                shuffle(ind)
                self.features_train = self.features_train[ind]
                self.labels_train = self.labels_train[ind]
                self.clf.fit(self.features_train, self.labels_train, batch_size=self.batch_size,
                             epochs=self.epochs, callbacks=[self.callback], validation_split=0.2, verbose=0)
            payoff_fn = W * \
                (2*self.clf.predict(np.expand_dims(Z, 0), verbose=0)[0, 1]-1)
            self.features_train = np.vstack(
                [self.features_train, np.expand_dims(Z, 0)])
            self.labels_train = np.vstack(
                [self.labels_train, self.enc.transform(W.reshape(-1, 1))])
            self.num_processed_pts += 1
            return payoff_fn


class MMDPayoff_2ST(object):
    def __init__(self):
        self.pos_mean_norm_sq = 0
        self.neg_mean_norm_sq = 0
        self.num_proc_positive = 0
        self.num_proc_negative = 0
        self.kernel_hyper = None
        self.num_processed_pts = 0
        self.pos_samples = None
        self.neg_samples = None
        self.norm_const = 0
        self.cross_inner_product = 0
        self.pts_to_estimate_param = 20

    def evaluate_payoff(self, Z, W):
        if self.num_processed_pts < self.pts_to_estimate_param:
            # only accumulate data
            if W == 1:
                if self.num_proc_positive == 0:
                    self.pos_samples = Z.reshape(1, -1).copy()
                else:
                    self.pos_samples = np.vstack(
                        [self.pos_samples, Z.reshape(1, -1)])
                self.num_proc_positive+=1
            else:
                if self.num_proc_negative == 0:
                    self.neg_samples = Z.reshape(1, -1).copy()
                else:
                    self.neg_samples = np.vstack(
                        [self.neg_samples, Z.reshape(1, -1)])
                self.num_proc_negative+=1
            self.num_processed_pts += 1
            return 0
        elif self.num_processed_pts == self.pts_to_estimate_param:
            # compute kernel param
            self.kernel_hyper = compute_hyperparam(
                    np.vstack([self.pos_samples, self.neg_samples]))
            self.num_proc_positive = 0
            self.num_proc_negative = 0
            self.pos_samples = None
            self.neg_samples = None
            self.num_processed_pts += 1
            return 0
        else:
            # start betting
            if self.num_proc_positive != 0 and self.num_proc_negative != 0:
                # bet if only both embedding are defined
                if self.num_proc_positive > 1:
                    p_mat = rbf_kernel(self.pos_samples, Z.reshape(1, -1), gamma=self.kernel_hyper)
                else:
                    p_mat = rbf_kernel(
                        self.pos_samples.reshape(1, -1), Z.reshape(1, -1), gamma=self.kernel_hyper)
                if self.num_proc_negative > 1:
                    q_mat = rbf_kernel(self.neg_samples, Z.reshape(1, -1), gamma=self.kernel_hyper)
                else:
                    q_mat = rbf_kernel(
                        self.neg_samples.reshape(1, -1), Z.reshape(1, -1), gamma=self.kernel_hyper)
                payoff_fn = W * (p_mat.mean() - q_mat.mean())/self.norm_const
            else:
                payoff_fn = 0

            # print(self.norm_const)

            if W == 1:
                if self.num_proc_positive == 0:
                    if self.num_proc_negative == 0:
                        # the first point
                        self.pos_mean_norm_sq = 1
                        self.norm_const = 1
                    else:
                        # if there are already instances from Q
                        if self.num_proc_negative > 1:
                            q_mat = rbf_kernel(
                                self.neg_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            q_mat = rbf_kernel(
                                self.neg_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        # compute cross inner product for the first time
                        self.cross_inner_product *= (
                            self.num_proc_positive/(self.num_proc_positive+1))
                        self.cross_inner_product += q_mat.mean() / (self.num_proc_positive+1)
                        # initialize the positive norm squared
                        self.pos_mean_norm_sq = 1 
                        self.norm_const = np.sqrt(
                            self.pos_mean_norm_sq-2*self.cross_inner_product+self.neg_mean_norm_sq)
                    self.pos_samples = Z.reshape(1, -1).copy()
                    self.num_proc_positive += 1
                else:
                    # if there are already instances from P
                    if self.num_proc_negative == 0:
                        # if there are instances from P BUT NOT Q
                        # just update norm of P
                        if self.num_proc_positive > 1:
                            p_mat = rbf_kernel(
                                self.pos_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            p_mat = rbf_kernel(
                                self.pos_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        self.pos_mean_norm_sq *= (self.num_proc_positive**2 / \
                            ((self.num_proc_positive+1)**2))
                        self.pos_mean_norm_sq += 1 / \
                            ((self.num_proc_positive+1)**2)
                        self.pos_mean_norm_sq += 2*self.num_proc_positive / \
                            ((self.num_proc_positive+1)**2)*p_mat.mean()
                        self.norm_const = np.sqrt(self.pos_mean_norm_sq)
                    else:
                        # if there are instances from both P and Q
                        if self.num_proc_negative > 1:
                            q_mat = rbf_kernel(
                                self.neg_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            q_mat = rbf_kernel(
                                self.neg_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        # compute cross inner product for the first time
                        self.cross_inner_product *= (
                            self.num_proc_positive/(self.num_proc_positive+1))
                        self.cross_inner_product += q_mat.mean() / (self.num_proc_positive+1)
                        # update the positive norm squared
                        if self.num_proc_positive > 1:
                            p_mat = rbf_kernel(
                                self.pos_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            p_mat = rbf_kernel(
                                self.pos_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        self.pos_mean_norm_sq *= (self.num_proc_positive**2 / \
                            ((self.num_proc_positive+1)**2))
                        self.pos_mean_norm_sq += 1 / \
                            ((self.num_proc_positive+1)**2)
                        self.pos_mean_norm_sq += 2*self.num_proc_positive / \
                            ((self.num_proc_positive+1)**2)*p_mat.mean()
                        self.norm_const = np.sqrt(
                            self.pos_mean_norm_sq-2*self.cross_inner_product+self.neg_mean_norm_sq)
                    self.pos_samples = np.vstack(
                        [self.pos_samples, Z.reshape(1, -1)])
                    self.num_proc_positive += 1
            elif W == -1:
                if self.num_proc_negative == 0:
                    if self.num_proc_positive == 0:
                        # if it is the very first instance
                        self.neg_mean_norm_sq = 1
                        self.norm_const = 1
                    else:
                        # if it is first inst from Q but not from P
                        if self.num_proc_positive > 1:
                            p_mat = rbf_kernel(
                                self.pos_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            p_mat = rbf_kernel(
                                self.pos_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        # compute cross inner product for the first time
                        self.cross_inner_product *= (
                            self.num_proc_negative/(self.num_proc_negative+1))
                        self.cross_inner_product += p_mat.mean() / (self.num_proc_negative+1)
                        # initialize the negative norm squared
                        self.neg_mean_norm_sq = 1 
                        self.norm_const = np.sqrt(
                            self.pos_mean_norm_sq-2*self.cross_inner_product+self.neg_mean_norm_sq)
                    self.neg_samples = Z.reshape(1, -1).copy()
                    self.num_proc_negative += 1
                else:
                    # if it is not the first instance from Q
                    if self.num_proc_positive == 0:
                        # if there are only instances from Q
                        # just update norm of Q
                        if self.num_proc_negative > 1:
                            q_mat = rbf_kernel(
                                self.neg_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            q_mat = rbf_kernel(
                                self.neg_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        self.neg_mean_norm_sq *= (self.num_proc_negative**2 / \
                            ((self.num_proc_negative+1)**2))
                        self.neg_mean_norm_sq += 1 / \
                            ((self.num_proc_negative+1)**2)
                        self.neg_mean_norm_sq += 2*self.num_proc_negative / \
                            ((self.num_proc_negative+1)**2)*q_mat.mean()
                        self.norm_const = np.sqrt(self.neg_mean_norm_sq)
                    else:
                        # if there are only instances from both P and Q
                        if self.num_proc_positive > 1:
                            p_mat = rbf_kernel(
                                self.pos_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            p_mat = rbf_kernel(
                                self.pos_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        # compute cross inner product for the first time
                        self.cross_inner_product *= (
                            self.num_proc_negative/(self.num_proc_negative+1))
                        self.cross_inner_product += p_mat.mean() / (self.num_proc_negative+1)
                        # update the positive norm squared
                        if self.num_proc_negative > 1:
                            q_mat = rbf_kernel(
                                self.neg_samples, Z.reshape(1, -1),gamma=self.kernel_hyper)
                        else:
                            q_mat = rbf_kernel(
                                self.neg_samples.reshape(1, -1), Z.reshape(1, -1),gamma=self.kernel_hyper)
                        self.neg_mean_norm_sq *= (self.num_proc_negative**2 / \
                            ((self.num_proc_negative+1)**2))
                        self.neg_mean_norm_sq += 1 / \
                            ((self.num_proc_negative+1)**2)
                        self.neg_mean_norm_sq += 2*self.num_proc_negative / \
                            ((self.num_proc_negative+1)**2)*q_mat.mean()
                        self.norm_const = np.sqrt(
                            self.pos_mean_norm_sq-2*self.cross_inner_product+self.neg_mean_norm_sq)
                    self.neg_samples = np.vstack(
                        [self.neg_samples, Z.reshape(1, -1)])
                    self.num_proc_negative += 1

            self.num_processed_pts += 1

            return payoff_fn


class LDA_2ST(object):
    def __init__(self):
        self.oracle = True
        self.mean_pos = 1
        self.mean_neg = 1
        self.scale_pos = 1
        self.scale_neg = 1
        self.second_moment_pos = 1
        self.second_moment_neg = 1

        self.num_processed_pts = 1
        self.regularized = True

    def evaluate_payoff(self, Z, W):
        # first time in pred case will bet zero
        payoff_fn = W*(norm.pdf(Z, loc=self.mean_pos, scale=self.scale_pos)-norm.pdf(Z, loc=self.mean_neg, scale=self.scale_neg))/(
            norm.pdf(Z, loc=self.mean_pos, scale=self.scale_pos)+norm.pdf(Z, loc=self.mean_neg, scale=self.scale_neg))
        if not self.oracle:
            # add a regularizer

            if W == 1:
                # if self.regularized:
                self.mean_pos = (self.num_processed_pts / (self.num_processed_pts+1)
                                 ) * self.mean_pos + Z / (self.num_processed_pts+1)
                self.second_moment_pos = (self.num_processed_pts / (self.num_processed_pts+1)) * \
                    self.second_moment_pos + Z**2 / (self.num_processed_pts+1)
                self.scale_pos = np.sqrt(
                    self.second_moment_pos - self.mean_pos**2)
                # else:
                #     self.mean_pos = ((self.num_processed_pts-1)/self.num_processed_pts) * self.mean_pos + Z / (self.num_processed_pts)
                #     self.mean_pos = ((self.num_processed_pts-1)/self.num_processed_pts) * self.mean_pos + Z / (self.num_processed_pts)
            elif W == -1:
                # if self.regularized:
                self.mean_neg = (self.num_processed_pts / (self.num_processed_pts+1)
                                 ) * self.mean_neg + Z / (self.num_processed_pts+1)
                self.second_moment_neg = (self.num_processed_pts / (self.num_processed_pts+1)) * \
                    self.second_moment_neg + Z**2 / (self.num_processed_pts+1)
                self.scale_neg = np.sqrt(
                    self.second_moment_neg - self.mean_neg**2)
                # else:
                #     self.mean_neg = ((self.num_processed_pts-1)/self.num_processed_pts) * self.mean_neg + Z / (self.num_processed_pts)
                #     self.mean_neg = ((self.num_processed_pts-1)/self.num_processed_pts) * self.mean_neg + Z / (self.num_processed_pts)

            self.num_processed_pts += 1

        return payoff_fn


class RidgeRegressor(object):
    def __init__(self):
        self.low_truncation_level = None
        self.upper_truncation_level = None
        self.beta = None
        self.lmbd = 1
        self.inv_sigma = None
        self.cov_term = None

    def predict(self, x):
        if self.beta is None:
            raise ValueError('Model is not yet trained')
        else:
            cand_pred = self.beta[0]+self.beta[1]*x
            return np.minimum(np.maximum(cand_pred, self.low_truncation_level), self.upper_truncation_level)

    def update(self, x, y):
        if self.beta is None:
            temp_vec = np.array([0., 1.])
            cur_sigma = np.outer(temp_vec, temp_vec)
            for cur_ind, cur_x_val in enumerate(x):
                # form a feature vector
                x_vec = np.array([1., cur_x_val])
                cur_sigma += np.outer(x_vec, x_vec)
                if self.cov_term is None:
                    self.cov_term = x_vec * y[cur_ind]
                else:
                    self.cov_term += x_vec * y[cur_ind]
                if self.low_truncation_level is None:
                    self.low_truncation_level = y[cur_ind]
                else:
                    self.low_truncation_level = min(
                        self.low_truncation_level, y[cur_ind])
                if self.upper_truncation_level is None:
                    self.upper_truncation_level = y[cur_ind]
                else:
                    self.upper_truncation_level = max(
                        self.upper_truncation_level, y[cur_ind])

            self.inv_sigma = np.linalg.inv(cur_sigma)

        else:
            for cur_ind, cur_x_val in enumerate(x):
                # form a feature vector
                x_vec = np.array([1., cur_x_val])
                prod_term = self.inv_sigma@x_vec
                normalizer = 1+prod_term@x_vec
                self.inv_sigma -= np.outer(prod_term, prod_term)/normalizer
                self.cov_term += x_vec * y[cur_ind]
                if self.low_truncation_level is None:
                    self.low_truncation_level = y[cur_ind]
                else:
                    self.low_truncation_level = min(
                        self.low_truncation_level, y[cur_ind])
                if self.upper_truncation_level is None:
                    self.upper_truncation_level = y[cur_ind]
                else:
                    self.upper_truncation_level = max(
                        self.upper_truncation_level, y[cur_ind])

        self.beta = self.inv_sigma @ self.cov_term


class RidgeRegressorPayoff(object):
    def __init__(self):
        self.ridge_reg = None
        self.num_processed_pairs = 0
        self.scaling_factor = 0
        self.num_sum = 0
        self.denom_sum = 0
        self.denom_max_reg = 0
        self.mu_1 = 0
        self.mu_3 = 0
        self.mu_5 = 0
        self.scaling_scheme = 'second'

    def evaluate_payoff(self, X, Y):
        if self.num_processed_pairs == 0:
            payoff_fn = 0
            self.ridge_reg = RidgeRegressor()
            self.ridge_reg.update(X, Y)
            # compute abs losses on joint and product pts
            preds = self.ridge_reg.predict(X)
            losses_joint = abs(Y-preds)
            losses_product = abs(Y[::-1]-preds)
            # compute first scaling factor
            if self.scaling_scheme == 'first':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += 2*sum(losses_product**3)
                # self.denom_max_reg = max(
                #     self.denom_max_reg, max(2*losses_product**3))
            elif self.scaling_scheme == 'second':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += sum(losses_product**3)
                self.mu_3 -= sum(losses_joint**3)
                self.mu_5 -= 2*sum(losses_joint**5)
        else:

            if self.scaling_scheme == 'first':
                self.scaling_factor = np.sqrt(2*max(
                    0, self.mu_1)/(self.mu_3))
                pass
            elif self.scaling_scheme == 'second':
                t1 = self.mu_3**2 - (4/3)*self.mu_5*max(self.mu_1, 0)
                t2 = self.mu_3 - np.sqrt(t1)
                self.scaling_factor = np.sqrt(t2/(2/3*self.mu_5))
            # make predictions
            preds = self.ridge_reg.predict(X)
            # compute abs losses on correct points
            losses_joint = abs(Y-preds)
            losses_product = abs(Y[::-1]-preds)
            payoff_joint = np.tanh(self.scaling_factor*losses_joint)
            payoff_product = np.tanh(self.scaling_factor*losses_product)
            payoff_fn = (sum(payoff_product)-sum(payoff_joint))/4
            self.ridge_reg.update(X, Y)
            # update scaling factor
            if self.scaling_scheme == 'first':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += 2*sum(losses_product**3)
            elif self.scaling_scheme == 'second':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += sum(losses_product**3)
                self.mu_3 -= sum(losses_joint**3)
                self.mu_5 -= 2*sum(losses_joint**5)

        self.num_processed_pairs += 1
        return payoff_fn


class OracleRegressorPayoff(object):
    def __init__(self):
        self.beta = None
        self.num_processed_pairs = 0
        self.scaling_factor = 0
        self.num_sum = 0
        self.denom_sum = 0
        self.denom_max_reg = 0
        self.mu_1 = 0
        self.mu_3 = 0
        self.mu_5 = 0
        self.scaling_scheme = 'second'

    def evaluate_payoff(self, X, Y):
        if self.num_processed_pairs == 0:
            # compute abs losses on joint and product pts
            payoff_fn = 0
            preds = X * self.beta
            losses_joint = abs(Y-preds)
            losses_product = abs(Y[::-1]-preds)
            # compute first scaling factor
            if self.scaling_scheme == 'first':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += 2*sum(losses_product**3)
                # self.denom_max_reg = max(
                #     self.denom_max_reg, max(2*losses_product**3))
            elif self.scaling_scheme == 'second':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += sum(losses_product**3)
                self.mu_3 -= sum(losses_joint**3)
                self.mu_5 -= 2*sum(losses_joint**5)
        else:
            if self.scaling_scheme == 'first':
                self.scaling_factor = np.sqrt(2*max(
                    0, self.mu_1)/(self.mu_3))
                pass
            elif self.scaling_scheme == 'second':
                t1 = self.mu_3**2 - (4/3)*self.mu_5*max(self.mu_1, 0)
                t2 = self.mu_3 - np.sqrt(t1)
                self.scaling_factor = np.sqrt(t2/(2/3*self.mu_5))
            # make predictions
            preds = X * self.beta
            # compute abs losses on correct points
            losses_joint = abs(Y-preds)
            losses_product = abs(Y[::-1]-preds)
            payoff_joint = np.tanh(self.scaling_factor*losses_joint)
            payoff_product = np.tanh(self.scaling_factor*losses_product)
            payoff_fn = (sum(payoff_product)-sum(payoff_joint))/4

            # update scaling factor
            if self.scaling_scheme == 'first':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += 2*sum(losses_product**3)
            elif self.scaling_scheme == 'second':
                self.mu_1 += sum(losses_product)
                self.mu_1 -= sum(losses_joint)
                self.mu_3 += sum(losses_product**3)
                self.mu_3 -= sum(losses_joint**3)
                self.mu_5 -= 2*sum(losses_joint**5)

        self.num_processed_pairs += 1
        return payoff_fn


class HSICWitness(object):
    def __init__(self):
        # specify type of a kernel, default: RBF-kernel with scale parameter 1
        self.kernel_type = 'rbf'
        self.kernel_param_x = 1
        self.kernel_param_y = 1
        # number of processed pairs
        self.num_processed_pairs = 0
        # store normalization constant
        self.norm_constant = 1e-6
        # store intermediate vals for linear updates
        # K^t 1
        self.k_vec_of_ones_product = None
        # L^t 1
        self.l_vec_of_ones_product = None
        # tr(K^t L^t)
        self.trace_prod = None

    def initialize_norm_const(self, first_pair_x, first_pair_y):
        """
        Function used to initialize the normalization
            constant using the first data pair
        """
        if self.kernel_type == 'rbf':
            if first_pair_x.ndim == 1:
                k_mat = rbf_kernel(
                    first_pair_x.reshape(-1, 1), gamma=self.kernel_param_x)
            else:
                k_mat = rbf_kernel(
                    first_pair_x, gamma=self.kernel_param_x)
            if first_pair_y.ndim == 1:
                l_mat = rbf_kernel(
                    first_pair_y.reshape(-1, 1), gamma=self.kernel_param_y)
            else:
                l_mat = rbf_kernel(
                    first_pair_y, gamma=self.kernel_param_y)
        elif self.kernel_type == 'laplace':
            if first_pair_x.ndim == 1:
                k_mat = laplacian_kernel(
                    first_pair_x.reshape(-1, 1), gamma=self.kernel_param_x)
            else:
                k_mat = laplacian_kernel(
                    first_pair_x, gamma=self.kernel_param_x)
            if first_pair_y.ndim == 1:
                l_mat = laplacian_kernel(
                    first_pair_y.reshape(-1, 1), gamma=self.kernel_param_y)
            else:
                l_mat = laplacian_kernel(
                    first_pair_y, gamma=self.kernel_param_y)

        # compute delta_1 and cache it
        delta_1 = np.trace(k_mat@l_mat)
        self.trace_prod = delta_1
        # compute K^t 1 and delta_2, store K^t 1
        # self.k_vec_of_ones_product = k_mat @ np.ones(2)
        self.k_vec_of_ones_product = k_mat.sum(axis=1)
        delta_2 = self.k_vec_of_ones_product.sum()
        # same for L^t 1 and delta_3
        self.l_vec_of_ones_product = l_mat.sum(axis=1)
        delta_3 = self.l_vec_of_ones_product.sum()
        # compute delta_4
        delta_4 = self.k_vec_of_ones_product @ self.l_vec_of_ones_product

        # update number of processed pairs
        self.num_processed_pairs += 1

        # compute the first normalization constant
        self.norm_constant += np.sqrt(
            delta_1/4+delta_2*delta_3/16 - delta_4/4)

    def update_norm_const(self, next_pair_x, next_pair_y, prev_data_x, prev_data_y):
        """
        Function that updates the value of the normalization constant in linear time
            using cached values
        """
        # compute kernel matrices for new points
        if self.kernel_type == 'rbf':
            # kernel evaluations with previous pts: x's
            # if one feature, add reshaping
            if next_pair_x.ndim == 1:
                k_mat_old = rbf_kernel(
                    prev_data_x.reshape(-1, 1), next_pair_x.reshape(-1, 1), gamma=self.kernel_param_x)
                # kernel evaluations for a new pair: x's
                k_mat_new = rbf_kernel(
                    next_pair_x.reshape(-1, 1), gamma=self.kernel_param_x)
            else:
                k_mat_old = rbf_kernel(
                    prev_data_x, next_pair_x, gamma=self.kernel_param_x)
                # kernel evaluations for a new pair: x's
                k_mat_new = rbf_kernel(
                    next_pair_x, gamma=self.kernel_param_x)
            # kernel evaluations with previous pts: y's
            if next_pair_y.ndim == 1:
                l_mat_old = rbf_kernel(
                    prev_data_y.reshape(-1, 1), next_pair_y.reshape(-1, 1), gamma=self.kernel_param_y)
                # kernel evaluations for a new pair: y's
                l_mat_new = rbf_kernel(
                    next_pair_y.reshape(-1, 1), gamma=self.kernel_param_y)
            else:
                l_mat_old = rbf_kernel(
                    prev_data_y, next_pair_y, gamma=self.kernel_param_y)
                # kernel evaluations for a new pair: y's
                l_mat_new = rbf_kernel(
                    next_pair_y, gamma=self.kernel_param_y)
        elif self.kernel_type == 'laplace':
            # kernel evaluations with previous pts: x's
            # if one feature, add reshaping
            if next_pair_x.ndim == 1:
                k_mat_old = laplacian_kernel(
                    prev_data_x.reshape(-1, 1), next_pair_x.reshape(-1, 1), gamma=self.kernel_param_x)
                # kernel evaluations for a new pair: x's
                k_mat_new = laplacian_kernel(
                    next_pair_x.reshape(-1, 1), gamma=self.kernel_param_x)
            else:
                k_mat_old = laplacian_kernel(
                    prev_data_x, next_pair_x, gamma=self.kernel_param_x)
                # kernel evaluations for a new pair: x's
                k_mat_new = laplacian_kernel(
                    next_pair_x, gamma=self.kernel_param_x)
            # kernel evaluations with previous pts: y's
            if next_pair_y.ndim == 1:
                l_mat_old = laplacian_kernel(
                    prev_data_y.reshape(-1, 1), next_pair_y.reshape(-1, 1), gamma=self.kernel_param_y)
                # kernel evaluations for a new pair: y's
                l_mat_new = laplacian_kernel(
                    next_pair_y.reshape(-1, 1), gamma=self.kernel_param_y)
            else:
                l_mat_old = laplacian_kernel(
                    prev_data_y, next_pair_y, gamma=self.kernel_param_y)
                # kernel evaluations for a new pair: y's
                l_mat_new = laplacian_kernel(
                    next_pair_y, gamma=self.kernel_param_y)
        # update the value of tr(K^t L^t)
        self.trace_prod += 2*(k_mat_old*l_mat_old).sum() + \
            np.sum(k_mat_new*l_mat_new)
        # update the value of K^t 1
        term_1 = self.k_vec_of_ones_product + k_mat_old.sum(axis=1)
        term_2 = k_mat_old.T.sum(axis=1) + k_mat_new.sum(axis=1)
        self.k_vec_of_ones_product = np.hstack([term_1, term_2])
        # update the value of L^t 1
        term_1 = self.l_vec_of_ones_product + l_mat_old.sum(axis=1)
        term_2 = l_mat_old.T.sum(axis=1) + l_mat_new.sum(axis=1)
        self.l_vec_of_ones_product = np.hstack([term_1, term_2])

        # update number of processed pairs
        self.num_processed_pairs += 1

        # compute delta_1
        delta_1 = self.trace_prod
        # compute K^t 1 and delta_2
        delta_2 = self.k_vec_of_ones_product.sum()
        # same for L^t 1 and delta_3
        delta_3 = self.l_vec_of_ones_product.sum()
        # compute delta_4
        delta_4 = self.k_vec_of_ones_product @ self.l_vec_of_ones_product

        # compute normalization constant
        self.norm_constant = np.sqrt(
            delta_1/((2*self.num_processed_pairs)**2) +
            delta_2*delta_3/((2*self.num_processed_pairs)**4)
            - 2*delta_4/((2*self.num_processed_pairs)**3))

    def evaluate_wf(self, new_pt_x, new_pt_y, prev_data_x, prev_data_y):
        """
        Witness function evaluation
        """
        # if there is a single feature, reshape the array
        if self.kernel_type == 'rbf':
            if new_pt_x.ndim == 1:
                k_mat = rbf_kernel(new_pt_x.reshape(-1, 1),
                                   prev_data_x.reshape(-1, 1), gamma=self.kernel_param_x)
            else:
                k_mat = rbf_kernel(new_pt_x, prev_data_x,
                                   gamma=self.kernel_param_x)
            if new_pt_y.ndim == 1:
                l_mat = rbf_kernel(new_pt_y.reshape(-1, 1),
                                   prev_data_y.reshape(-1, 1), gamma=self.kernel_param_y)
            else:
                l_mat = rbf_kernel(new_pt_y, prev_data_y,
                                   gamma=self.kernel_param_y)
        elif self.kernel_type == 'laplace':
            if new_pt_x.ndim == 1:
                k_mat = laplacian_kernel(new_pt_x.reshape(-1, 1),
                                         prev_data_x.reshape(-1, 1), gamma=self.kernel_param_x)
            else:
                k_mat = laplacian_kernel(new_pt_x, prev_data_x,
                                         gamma=self.kernel_param_x)
            if new_pt_y.ndim == 1:
                l_mat = laplacian_kernel(new_pt_y.reshape(-1, 1),
                                         prev_data_y.reshape(-1, 1), gamma=self.kernel_param_y)
            else:
                l_mat = laplacian_kernel(new_pt_y, prev_data_y,
                                         gamma=self.kernel_param_y)
        elif self.kernel_type == 'linear':
            if new_pt_x.ndim == 1:
                k_mat = linear_kernel(new_pt_x.reshape(-1, 1),
                                      prev_data_x.reshape(-1, 1))
            else:
                k_mat = linear_kernel(new_pt_x, prev_data_x)
            if new_pt_y.ndim == 1:
                l_mat = linear_kernel(new_pt_y.reshape(-1, 1),
                                      prev_data_y.reshape(-1, 1))
            else:
                l_mat = linear_kernel(new_pt_y, prev_data_y)

        mu_joint = np.mean(l_mat*k_mat)
        mu_product = np.mean(l_mat, axis=1) @ np.mean(k_mat, axis=1)
        res = (mu_joint-mu_product) / self.norm_constant

        return res
