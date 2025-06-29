import numpy as np
table1_omega_k = np.array([
        46, 68, 117, 167, 180, 191, 202, 243, 263, 284,
        291, 327, 366, 385, 404, 423, 440, 481, 541, 568,
        582, 597, 630, 638, 665, 684, 713, 726, 731, 750,
        761, 770, 795, 821, 856, 891, 900, 924, 929, 946,
        966, 984, 1004, 1037, 1058, 1094, 1104, 1123, 1130, 1162,
        1175, 1181, 1201, 1220, 1283, 1292, 1348, 1367, 1386, 1431,
        1503, 1545
    ], dtype=float)

# Huang-Rhys factors s_k
table1_s_k = np.array([
    0.011, 0.011, 0.009, 0.009, 0.010, 0.011, 0.011, 0.012, 0.003, 0.008,
    0.008, 0.003, 0.006, 0.002, 0.002, 0.002, 0.001, 0.002, 0.004, 0.007,
    0.004, 0.004, 0.003, 0.006, 0.004, 0.003, 0.007, 0.010, 0.005, 0.004,
    0.009, 0.018, 0.007, 0.006, 0.007, 0.003, 0.004, 0.001, 0.001, 0.002,
    0.002, 0.003, 0.001, 0.002, 0.002, 0.001, 0.001, 0.003, 0.003, 0.009,
    0.007, 0.010, 0.003, 0.005, 0.002, 0.004, 0.007, 0.002, 0.004, 0.002,
    0.003, 0.003
], dtype=float)
def Jh(omega, table_omega_k_vals, table_s_k_vals, gamma_k_val=5.0):
    omega_arr = np.asarray(omega, dtype=float)*100
    
    total_j_h_omega = np.zeros_like(omega_arr)

    positive_omega_mask = omega_arr > 0
    
    if not np.any(positive_omega_mask):
        if isinstance(omega, (int, float)):
            return 0.0
        else:
            return total_j_h_omega # All zeros

    # Process only positive omega values
    omega_pos = omega_arr[positive_omega_mask]
    
    # Accumulator for J_h for positive omegas
    j_h_omega_pos_accumulator = np.zeros_like(omega_pos)

    gamma_k_sq = gamma_k_val**2

    # Iterate over each mode k from Table I
    for omega_k, s_k in zip(table_omega_k_vals, table_s_k_vals):
        if omega_k <= 0: # Should not happen based on Table I, but good check
            continue

        omega_k_sq = omega_k**2
        
        numerator_k = (4 * omega_k * s_k * gamma_k_val * (omega_k_sq + gamma_k_sq) * omega_pos)

        # Denominator parts for mode k
        term_denom1 = (omega_pos + omega_k)**2 + gamma_k_sq
        term_denom2 = (omega_pos - omega_k)**2 + gamma_k_sq
        denominator_k = np.pi * term_denom1 * term_denom2
        
        # Add contribution of mode k
        j_h_omega_pos_accumulator += numerator_k / denominator_k
        
    # Assign calculated positive values to the corresponding positions
    total_j_h_omega[positive_omega_mask] = j_h_omega_pos_accumulator

    # If the original input was a scalar, return a scalar
    if isinstance(omega, (int, float)):
        return total_j_h_omega.item()
    else:
        return total_j_h_omega

def Jar(omega):

    omega_arr = np.asarray(omega, dtype=float)*100
    j_ar_omega = np.zeros_like(omega_arr)

    # Constants from the paper
    S = 0.29
    s1 = 0.8
    s2 = 0.5
    omega1_val = 0.069  # meV
    omega2_val = 0.24   # meV

    # Pre-calculated constant values
    s1_plus_s2 = s1 + s2  # 1.3
    seven_factorial_times_2 = 10080.0  # 7! * 2 = 5040 * 2

    # Calculate the main coefficient
    main_coefficient = S / s1_plus_s2

    
    positive_omega_mask = omega_arr > 0
    
    if np.any(positive_omega_mask):
        omega_pos = omega_arr[positive_omega_mask]

        # Term 1 components
        term1_s_factor = s1
        term1_omega_factor_denom = (omega1_val**4)
        term1_denom = seven_factorial_times_2 * term1_omega_factor_denom
        term1_exp_arg = np.sqrt(omega_pos / omega1_val) 
        term1 = (term1_s_factor / term1_denom) * (omega_pos**5) * np.exp(-term1_exp_arg)

        # Term 2 components
        term2_s_factor = s2
        term2_omega_factor_denom = (omega2_val**4)
        term2_denom = seven_factorial_times_2 * term2_omega_factor_denom
        term2_exp_arg = np.sqrt(omega_pos / omega2_val)
        term2 = (term2_s_factor / term2_denom) * (omega_pos**5) * np.exp(-term2_exp_arg)

        # Combine terms for positive omega values
        j_ar_omega_positive_values = main_coefficient * (term1 + term2)
        
        # Assign calculated values to the corresponding positions in the result array
        j_ar_omega[positive_omega_mask] = j_ar_omega_positive_values

    # If the original input was a scalar, return a scalar
    if isinstance(omega, (int, float)):
        return j_ar_omega.item()
    else:
        return j_ar_omega 

def spectral_density(w):
    return (Jh(w,table1_omega_k,table1_s_k)+Jar(w))/15000