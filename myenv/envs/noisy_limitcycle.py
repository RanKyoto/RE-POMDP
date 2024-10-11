import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Optional
import pygame


class NoisyLimitCycleEnv(gymnasium.Env):
    """
    Description:
        Note: This is a Limit Cycle case. 

    Observation:
        Type: Box(2)
        Num     Observation        Min               Max
        0       x1                - 10               10
        1       x2                - 10               10

    Actions:
        Type: Continuous(1)
        Num   Action
        u     [-10,10]

    System:
        x =[x_1 x_2]^T
        x1' = x1 - x2 - x1^3 - x1*x2^2
        x2' = x1 + x2 - x1^2*x2 - x2^3
        x' = f(x,u) + Fw  
        y  = x + Gv
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        }
    
    def __init__(self, dt = 0.01, render_mode: Optional[str] = None):
        '''
            default dt = 0.01
        '''
        self.render_mode = render_mode
        self.dt = dt  # sample period
        self.misunderstood = False

        self.V = np.zeros((2,2))

        self.R = 4

        # Boundary settings
        self.u_mag = 20.0 # max input magnitude
        self.x_max = 10

        x_mag = np.array(
            [
                self.x_max,self.x_max 
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-self.u_mag, high=self.u_mag,shape=(1,) , dtype=np.float32
        )
        self.observation_space = spaces.Box(-x_mag, x_mag, dtype=np.float32)

        self.screen_width = 500
        self.viewer = None
        self.clock = None
        self.isopen = True

        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self,u, Type =0):
        ''' 
            Type==1: evaluated; 2: misunderstood; Others: true
        '''
        if Type==1:
            y1,y2 = self.noisy_state
            return  y1**4-y1**2*(6*self.G1+2*self.G2+2*self.R**2)+\
                    y2**4-y2**2*(6*self.G2+2*self.G1+2*self.R**2)+\
                    2*y1**2*y2**2 - 8*y1*y2*self.G3 + self.C + np.abs(u)
        elif Type==2:
            y1,y2 = self.noisy_state
            return  (y1**2 + y2**2 - self.R**2)**2 + np.abs(u)
        else:
            x1,x2 = self.true_state
            return  (x1**2 + x2**2 - self.R**2)**2 + np.abs(u)

           
    def step(self, action):
        '''
            If IsDiscrete = True   , x'=Ax + Bu + Ww;  
            If IsDiscrete = False  , x'=x'+dt(Ax+Bu+Ww)

            Return: y, -r(x,u), done, _
        '''
        x1, x2 = self.true_state

        action = np.array(action,dtype=np.float32).reshape((1,))
        u = np.clip(action, -self.u_mag, self.u_mag)[0]
        if self.misunderstood:
            reward = - self.reward(u, Type=1) * self.dt
        else:
            reward = - self.reward(u, Type=0) * self.dt
   
        x1_dot = u*(x1 - x2) - (0.5)*( x1**3  + x1 * (x2**2))
        x2_dot = u*(x1 + x2) - (0.5)*((x1**2) * x2 +  x2**3 )  

        x1 = x1 + self.dt * x1_dot
        x2 = x2 + self.dt * x2_dot

        self.true_state = np.array([x1,x2],dtype=np.float32) #update state
        self.noisy_state = self._get_obs()

        if self.misunderstood:
            return self.noisy_state, reward, False, False, {"true_state":self.true_state}
        else:
            return self.true_state,  reward, False, False, {"noisy_state":self.noisy_state}

    def reset(self, seed: Optional[int] = None, 
              options: Optional[dict] = {"G":np.array([[1.0,0.2],[0.2,1.0]])}):
        ''' reset with x0 ~ d0 or some given x0, and A, B, W, V (if they are changed) '''

        self.true_state = np.random.uniform(-8.0, 8.0, size=2).astype(np.float32)
        self.noisy_state = self._get_obs()

        if options.get("x0") is not None:
           self.true_state = options.get("x0")
        if options.get("G") is not None:
           G = options.get("G")
           self.G = G
           self.G1 = G[0,0]**2 + G[0,1]**2
           self.G2 = G[1,0]**2 + G[1,1]**2
           self.G3 = G[0,0]*G[1,0] +G[0,1]*G[1,1]
           C1 = 3*G[0,0]**4+6*G[0,0]**2*G[0,1]**2+3*G[0,1]**4
           C2 = 3*G[1,0]**4+6*G[1,0]**2*G[1,1]**2+3*G[1,1]**4
           C3 = 2*self.R**2*self.G1
           C4 = 2*self.R**2*self.G2
           C5 = 3*(G[0,0]**2*G[1,0]**2+G[0,1]**2*G[1,1]**2)+4*G[0,0]*G[0,1]*G[1,0]*G[1,1]+(G[0,1]**2*G[1,0]**2+G[0,0]**2*G[1,1]**2)
           self.C = self.R**4 + C1+C2+C3+C4+C5*2
           self.V = G.T @ G
        if self.render_mode == "human":
            self.render()
        if self.misunderstood:
            return self.noisy_state, {"true_state":self.true_state}
        else:
            return self.true_state, {"noisy_state":self.noisy_state}


    def _get_obs(self):
        '''
            y = x + v, v N([0,0],V)
        ''' 
        return self.true_state + np.random.multivariate_normal([0,0],self.V).astype(np.float32)

    def render(self, mode="human"):
        screen_height = screen_width = self.screen_width

        world_width = self.x_max * 2
        scale = screen_width / world_width

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Noisy Limit Cycle")

        # Clear screen
        self.viewer.fill((255, 255, 255)) 

        pygame.draw.line(self.viewer, (0, 0, 0), (0, screen_height/2), (screen_width, screen_height/2), 2)  # XÖá
        pygame.draw.line(self.viewer, (0, 0, 0), (screen_width/2, 0), (screen_width/2, screen_height), 2)  # YÖá

        target_R = self.R * scale
        pygame.draw.circle(self.viewer, (128, 255, 128), (screen_width//2, screen_height//2), int(target_R), 2)
        pygame.draw.circle(self.viewer, (0, 0, 0), (screen_width//2, screen_height//2), 5)

        if self.true_state is None:
            return None

        x1, x2 = self.true_state
        y1, y2 = self.noisy_state

        x1 = int(x1 * scale + screen_width / 2.0)
        x2 = int(x2 * scale + screen_height / 2.0)
        y1 = int(y1 * scale + screen_width / 2.0)
        y2 = int(y2 * scale + screen_height / 2.0)

        pygame.draw.circle(self.viewer, (128, 128, 255), (x1, x2), 5)
        pygame.draw.circle(self.viewer, (255, 0, 0), (y1, y2), 3)
        pygame.display.flip()
        if mode == "human":
            return None
        else:
            return pygame.surfarray.array3d(self.viewer)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            
# Testing the environment
if __name__ == "__main__":
    env = NoisyLimitCycleEnv(render_mode="human")
    for _ in range(10):
        observation, info = env.reset()
        env.render()
        for i in range(100):
            action = np.random.rand()*5+2
            observation, reward, _, _ , info = env.step(action)
            env.render()
            pygame.time.wait(20)

