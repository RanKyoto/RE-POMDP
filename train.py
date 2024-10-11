from stable_baselines3 import PPO
from src.recppo import ConstrainedPPO
from src.policies import ConstrainedPolicy
from src.reward_est import obtain_R1_tilde,estimated_reward
import myenv
import numpy as np
import gymnasium

def recppo_train(gamma=0.6):
    env = gymnasium.make("NoisyGridworld-v0",render_mode='rgb_array')
    # Config epsilon in "rewrad_est.py" default eps=50
    model =  ConstrainedPPO(ConstrainedPolicy, env, verbose=1,gamma=gamma,
                            learning_rate=0.01,estimated_reward=estimated_reward)
    # Obtain the observation probability matrix from the POMDP
    Psi = env.unwrapped.get_Psi()
    # Pass the Phi matrix to the SNSF control policy
    model.set_Psi(Psi)
    
    model.learn(100000,log_interval=1,progress_bar=True)
    model.save("./data/recppo_eps50")

if __name__ == "__main__":             
    recppo_train()