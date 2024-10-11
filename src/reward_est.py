import cvxpy as cp
import numpy as np
import myenv
import gymnasium

def obtain_R1_tilde(eps=0.0,p=0.4):
    env = gymnasium.make("NoisyGridworld-v0",p=p,render_mode='rgb_array')
    # The following constant matrix and vector are given by NoisyGridworldEnv

    K = 16  # Example dimension
    Psi = env.unwrapped.get_Psi()  # constant matrix Psi
    Psi_bar = np.diag(np.sum(Psi,axis=0,keepdims=True).flatten())
    print(f"Optimal Rtilde_1(p={p}) Rank:", np.linalg.matrix_rank(Psi))
    
    R_1 = np.zeros((K,1))  
    R_1[(6,9,11,14),:] = 10 # constant colum vector R_1
    eps = eps  # A small positive constant

    # Define the variable
    Rtilde_1 = cp.Variable((K, 1))

    # Define the objective function: minimize norm(Rtilde_1, 2)^2
    objective = cp.Minimize(cp.quad_form(Rtilde_1,(Psi_bar- Psi.T @ Psi)))

    # Define the constraints: |Psi * Rtilde_1 - R_1| < eps * ones((K,1))
    constraints = [cp.norm(cp.abs(Psi @ Rtilde_1 - R_1),2)**2 <= eps]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    try:  
        problem.solve(solver=cp.MOSEK)
    except:
        return None,None
    if Rtilde_1.value is None:
        return None, None
    Rtilde_1 =  Rtilde_1.value.reshape((16,1))
    # Print the results
    print("VAR:",objective.value)
    print("R_tilde_1:")
    print(np.round(Rtilde_1.reshape(4,4),2))
    print("Corr true R:")
    print(np.round((Psi@Rtilde_1).reshape(4,4),2))

    return np.array(objective.value),Rtilde_1.flatten()

def estimated_reward(noisy_obs, action):
        # R1_tilde=obtain_R1_tilde(eps = 0.1)
        # print(R1_tilde)
        """ eps = 0 """
        # R1_tilde =[-30.71,  20.48,   6.19, -34.29,
        #            20.48, -16.43,   1.43,  39.52,
        #             6.19,   1.43,  33.57, -46.19,
        #           -34.29,  39.52, -46.19,  69.28]
        """ eps = 10 """
        # R1_tilde = [-12.2,    6.93,   9.74, -30.1, 
        #               6.93,  -7.41,   1.8,   30.8,
        #               9.74,   1.8,   25.45, -30.71,
        #             -30.1,   30.8,  -30.71,  48.63]
        """ eps = 20 """
        # R1_tilde = [ -6.02,   2.42,  10.35, -27.5, 
        #               2.42,  -4.05,   2.01,  26.92,
        #              10.35,   2.01,  22.52, -24.63,
        #             -27.5,   26.92, -24.63,  40.56]
        """ eps = 50 """
        R1_tilde = [  1.44,  -3.01,   9.43, -20.73,
                     -3.01,   1.03,   2.54,  19.2,
                      9.43,   2.54,  17.99, -14.08,
                    -20.73,   19.2, -14.08,  26.56]
        """ eps = 100 """
        # R1_tilde = [  2.62,  -3.79,   6.,   -12.2, 
        #              -3.79,   3.74,   3.38,  11.5,
        #               6.,     3.38,  14.4,   -5.08,
        #             -12.2,   11.51,  -5.08,  14.85,]
        """ eps = 150 """
        # R1_tilde = [ 0.99, -2.32,  2.99, -5.94,
        #             -2.32,  4.08,  4.13,  6.58,
        #              2.99,  4.13, 11.41,  0.3, 
        #             -5.94,  6.58,  0.3,   8.41,]
        
        reward = R1_tilde[noisy_obs]

        if action == 4:
            reward += 0 # stay
        else:
            reward -= 1 # move

        return reward

if __name__ == '__main__':
    obtain_R1_tilde(eps=0,p=0.4)
 