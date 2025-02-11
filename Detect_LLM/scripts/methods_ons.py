import numpy as np 

def test_by_betting(seq1, seq2, d, epsilon, alpha): 
    
    wealth_A = 1
    wealth_B = 1
    wealth_hist_A = [1]
    wealth_hist_B = [1] 
    const = 2 / (2 - np.log(3))
    theta_a = 0 
    theta_b = 0 
    zat2 = 0 
    zbt2 = 0
    for t in range(1,min(len(seq1), len(seq2))):        
        At = 1 - theta_a*(seq1[t] - seq2[t]- epsilon)
        Bt = 1 - theta_b*(seq2[t] - seq1[t]- epsilon)
        wealth_A = wealth_A * At
        wealth_B = wealth_B * Bt 
        wealth_hist_A.append(wealth_A)
        wealth_hist_B.append(wealth_B)
        if wealth_A > 2/alpha: 
            return wealth_hist_A, 'reject'
        elif wealth_B > 2/alpha: 
            return wealth_hist_B, 'reject'
            
        # Update the betting fraction theta via ONS  
        a = seq1[t] - seq2[t] - epsilon
        b = seq2[t] - seq1[t] - epsilon
        z_a = a / (1 - theta_a*a)
        z_b = b / (1 - theta_b*b)
       
        zat2 += z_a**2
        zbt2 += z_b**2

        # Bet on the half decision space
        theta_a = max(min(theta_a - const*z_a/(1 + zat2), 0), -1/(2*d))
        theta_b = max(min(theta_b - const*z_b/(1 + zbt2), 0), -1/(2*d))
    U = np.random.uniform()
    if wealth_A > (2*U)/alpha: 
        return wealth_hist_A, 'reject'
    elif wealth_B > (2*U)/alpha: 
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
            wealth, reject = test_by_betting(s1, s2, d, epsilon, alpha=alpha)
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
        
