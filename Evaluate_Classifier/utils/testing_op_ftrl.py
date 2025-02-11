import numpy as np
from .payoff_fns import RFPayoff, LDAPayoff, KNNPayoff, MLPPayoff, HSICWitness, OracleLDAPayoff, RidgeRegressorPayoff, OracleRegressorPayoff, KNNPayoff_2ST, LDA_2ST, CNNPayoff_2ST,MMDPayoff_2ST
from sklearn.metrics import pairwise_distances
TYPES_KERNEL = ['rbf', 'laplace']


def compute_hyperparam(data: np.ndarray,
                       kernel_type: TYPES_KERNEL = 'rbf', style='median') -> float:
    """
    Use median heuristic to compute the hyperparameter
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


class Seq_C_2ST(object):
    def __init__(self, significance_level=0.05):
        # specify the payoff function style, default: hsic
        self.payoff_style = 'classification'
        self.pred_model = 'LDA'
        self.payoff_obj = None
        self.wf = None
        self.kernel_type = 'rbf'
        self.kernel_param_x = 1
        self.kernel_param_y = 1
        self.bet_scheme = 'OFB'
        self.payoff_strategy = 'accuracy'
        self.scaling_strategy = 'second'
        self.knn_comp = 'old'
        self.knn_reg = False
        # specify type of a kernel, default: RBF-kernel with scale parameter 1
        # lmbd params
        # choose fixed or mixture method
        # wealth process vals
        self.wealth = 1
        self.wealth_flag = False
        # store intermediate vals for linear updates
        self.payoff_hist = list()
        self.num_proc_pairs = 1
        self.mixed_wealth = None
        # for testing
        self.significance_level = significance_level
        self.null_rejected = False
        self.run_mean = 0
        self.run_second_moment = 0
        self.opt_lmbd = 0
        self.ons = True
        self.grad_sq_sum = 1
        self.lmbd_hist = list()
        self.truncation_level = 0.5
        self.oracle_beta = 0
        # mixture
        self.grid_of_lmbd = None
        self.lmbd_grid_size = 19
        self.lda_mean_pos = 0
        self.lda_mean_neg = 0
        self.lda_oracle = False
        self.eta = 1   
        self.learning_rate = 0.01  # FTRL learning rate
        self.cumulative_grad = 0 
        self.historical_grad = 0
        self.update_opt_lmbd = self.GD_with_barrier(self.eta, self.learning_rate)

    def GD_with_barrier(self, eta, learning_rate=0.01):

        def update_opt_lmbd(grad_loss_value):
            nonlocal eta, learning_rate
            self.historical_grad += grad_loss_value
            self.cumulative_grad = self.historical_grad + grad_loss_value
            for _ in range(500):  
                barrier_grad = 1 / (1 - self.opt_lmbd) - 1 / (1 + self.opt_lmbd)
                total_grad = eta * self.cumulative_grad + barrier_grad
                self.opt_lmbd -= learning_rate * total_grad
                convergence_threshold = 1e-6  # threshold
                if np.linalg.norm(total_grad) < convergence_threshold:
                    break
            return self.opt_lmbd
    
        return update_opt_lmbd

    def compute_predictive_payoff(self, next_Z, next_W):
        if self.num_proc_pairs == 1:
            if self.pred_model == 'kNN':
                self.payoff_obj = KNNPayoff_2ST()
                self.payoff_obj.proc_type = self.knn_comp
                self.payoff_obj.bet_strategy = self.payoff_strategy
                self.payoff_obj.regularized = self.knn_reg

            elif self.pred_model == 'LDA':
                self.payoff_obj = LDA_2ST()
                self.payoff_obj.mean_pos = self.lda_mean_pos
                self.payoff_obj.mean_neg = self.lda_mean_neg
                self.payoff_obj.oracle = self.lda_oracle
            elif self.pred_model == 'CNN':
                self.payoff_obj = CNNPayoff_2ST()
            elif self.pred_model == 'MMD':
                self.payoff_obj = MMDPayoff_2ST()
            if self.bet_scheme == 'fixed':
                self.opt_lmbd = 1
        cand_payoff = self.payoff_obj.evaluate_payoff(next_Z, next_W)
        self.payoff_hist+=[cand_payoff]
        if self.bet_scheme == 'fixed':
            payoff_fn = self.opt_lmbd * cand_payoff
        if self.bet_scheme == 'OFB':
            if self.num_proc_pairs == 1:
                payoff_fn = self.opt_lmbd * cand_payoff
                self.run_mean = np.copy(cand_payoff)
            else:
                grad = -self.run_mean/(1+self.run_mean*self.opt_lmbd)
                #self.grad_sq_sum += grad**2
                self.opt_lmbd = self.update_opt_lmbd(grad) # use optimistic-ftrl+barrier
                self.opt_lmbd = max(self.opt_lmbd, 0)
                payoff_fn = self.opt_lmbd * cand_payoff
                self.run_mean = np.copy(cand_payoff)

        self.num_proc_pairs += 1

        return payoff_fn

    def process_pair(self, next_Z, next_W, prev_data_x=None, prev_data_y=None):
        """
        Function to call to process next pair of datapoints:
        """
        # perform pairing to obtain points from the product
        # form points from joint dist and from product
        if self.payoff_style == 'classification' or self.payoff_style == 'regression' or self.payoff_style == 'kernel':
            payoff_fn = self.compute_predictive_payoff(next_Z, next_W)
            self.payoff_hist+=[payoff_fn]
        else:
            raise ValueError(
                'Unknown version of payoff function')
        # update wealth process value

        if self.bet_scheme == 'aGRAPA' or self.bet_scheme == 'OFB' or self.bet_scheme == 'fixed':
            cand_wealth = self.wealth * (1+payoff_fn)
            if cand_wealth >= 0 and self.wealth_flag is False:
                self.wealth = cand_wealth
                if self.wealth >= 1/self.significance_level:
                    self.null_rejected = True
            else:
                self.wealth_flag = True
        elif self.bet_scheme == 'mixing':
            # update wealth for each value of lmbd
            cand_wealth = [self.wealth[cur_ind] * (1+cur_lmbd*payoff_fn)
                           for cur_ind, cur_lmbd in enumerate(self.grid_of_lmbd)]
            # self.payoff_hist += [payoff_fn]
            for cur_ind in range(self.lmbd_grid_size):
                if cand_wealth[cur_ind] >= 0 and self.wealth_flag[cur_ind] is False:
                    self.wealth[cur_ind] = cand_wealth[cur_ind]
                    # update mixed wealth
                    # update whether null is rejected
                else:
                    self.wealth_flag[cur_ind] = True
                    self.wealth = [0 for i in range(self.lmbd_grid_size)]
                    break
            self.mixed_wealth = np.mean(self.wealth)
            if self.mixed_wealth >= 1/self.significance_level:
                self.null_rejected = True
