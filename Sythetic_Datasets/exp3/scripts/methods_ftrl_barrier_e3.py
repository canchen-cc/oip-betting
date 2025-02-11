import numpy as np 


def test_by_ftrl_barrier(seq1, seq2, epsilon, alpha):

    # Initialization
    wealth_A = 1
    wealth_B = 1
    wealth_hist_A = [1]
    wealth_hist_B = [1]
    theta_a = 0
    theta_b = 0
    eta = 1
    update_theta_a = GD_with_barrier(eta)
    update_theta_b = GD_with_barrier(eta)

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

        # Update θ_a and θ_b
        theta_a = update_theta_a(grad_a)
        theta_a=min(theta_a,0)
        theta_b = update_theta_b(grad_b)
        theta_b=min(theta_b,0)
        

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
    for _ in range(iters): 
        # since s2 shift at t=300, we shuffle the orders seperately
        np.random.shuffle(s2[:300])
        np.random.shuffle(s2[300:]) 
        np.random.shuffle(s1)

        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_ftrl_barrier(s1, s2, epsilon, alpha=alpha)
            real_tau = len(wealth)
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections


def GD_with_barrier(eta): 
    learning_rate=0.01
    theta = 0  # Initialize theta
    cumulative_grad=0
    
    def update_theta(grad_loss_value):
        nonlocal cumulative_grad, theta
        cumulative_grad += grad_loss_value
        for _ in range(500):  
            barrier_grad = 1 / (1 - theta) - 1 / (1 + theta)
            total_grad = eta * cumulative_grad + barrier_grad
            theta -= learning_rate * total_grad
            convergence_threshold = 1e-6  # threshold
            if np.linalg.norm(total_grad) < convergence_threshold:
                break
        return theta

    return update_theta
    

