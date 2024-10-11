
import numpy as np
import torch as th

class QTable():
    def __init__(self, state_dim, action_dim, lr=0.1) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lr = lr
        self.reset()
        pass
    
    def reset(self) -> None:
        self.table = np.zeros((self.state_dim,self.action_dim),dtype=np.float32)

    def update(self, state, action, episodic_reward):
        self.table[state,action] = (1-self.lr)*self.table[state,action] +self.lr*episodic_reward

    def to_torch(self,device:th.device)-> th.Tensor:
        return th.tensor(self.table,dtype=th.float32).to(device)
    
    def __str__(self) -> str:
        return np.round(self.table,3).__str__()
