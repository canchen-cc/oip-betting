import numpy as np 

def test_by_ftrl_barrier(seq1, seq2, d, epsilon, alpha):
   
    wealth_A = 1
    wealth_B = 1
    wealth_hist_A = [1]
    wealth_hist_B = [1]
    theta_a = 0
    theta_b = 0
    eta = 1
    
    # Bet on the whole decision space
    update_theta_a = GD_with_barrier(eta, 1/d)
    update_theta_b = GD_with_barrier(eta, 1/d)

    for t in range(1, min(len(seq1), len(seq2))):
        At = 1 - theta_a * (seq1[t] - seq2[t] - epsilon)
        Bt = 1 - theta_b * (seq2[t] - seq1[t] - epsilon)
        wealth_A *= At
        wealth_B *= Bt
        wealth_hist_A.append(wealth_A)
        wealth_hist_B.append(wealth_B)

        if wealth_A > 2 / alpha:
            return wealth_hist_A, 'reject'
        elif wealth_B > 2 / alpha:
            return wealth_hist_B, 'reject'

        # Update grad_loss for θ_a and θ_b
        grad_a = (seq1[t] - seq2[t] - epsilon) / (1 - theta_a * (seq1[t] - seq2[t] - epsilon))
        grad_b = (seq2[t] - seq1[t] - epsilon) / (1 - theta_b * (seq2[t] - seq1[t] - epsilon))

        # Update the betting fraction theta via Optimistic-FTRL+Barrier       
        theta_a = min(update_theta_a(grad_a),0)
        theta_b = min(update_theta_b(grad_b),0)
       
    U = np.random.uniform()
    if wealth_A > (2 * U) / alpha:
        return wealth_hist_A, 'reject'
    elif wealth_B > (2 * U) / alpha:
        return wealth_hist_B, 'reject'

    return wealth_hist_B, 'sustain'


def betting_experiment(seq1, seq2, epsilon, alphas, iters, shift_time=None): 
    
    results = []
    rejections = []
    s1, s2 = seq1, seq2
    q1 = np.array(s1)
    q2 = np.array(s2)
    diff = np.abs(q1[:, np.newaxis] - q2)
    d = np.max(diff)
    for _ in range(iters): 
        if shift_time != None: 
            s1, s2 = shuffle_with_shift(seq1, seq2, shift_time)
        else: 
            np.random.shuffle(s1)
            np.random.shuffle(s2)
        
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_ftrl_barrier(s1, s2, d, epsilon, alpha=alpha)
            real_tau = len(wealth)
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections

def shuffle_with_shift(seq1, seq2, shift_time): 

    s1_pre, s1_post = seq1[:shift_time], seq1[shift_time:]
    s2_pre, s2_post = seq2[:shift_time], seq2[shift_time:]
    np.random.shuffle(s1_pre), np.random.shuffle(s1_post)
    np.random.shuffle(s2_pre), np.random.shuffle(s2_post)
    s1 = np.concatenate((s1_pre, s1_post))
    s2 = np.concatenate((s2_pre, s2_post))
    
    return s1, s2
        
def GD_with_barrier(eta, r): 
    
    learning_rate=0.01
    theta = 0  # Initialize theta
    historical_grad = 0  # To store cumulative gradient
    cumulative_grad=0
    
    def update_theta(grad_loss_value):
        nonlocal historical_grad, cumulative_grad, theta
        historical_grad += grad_loss_value
        cumulative_grad = historical_grad + grad_loss_value
        for _ in range(1000):  
            barrier_grad = 1 / (r - theta) - 1 / (r + theta)
            total_grad = eta * cumulative_grad + barrier_grad
            theta -= learning_rate * total_grad
            convergence_threshold = 1e-6  # threshold
            if np.linalg.norm(total_grad) < convergence_threshold:
                break
        return theta

    return update_theta


