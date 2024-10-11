from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from src.buffers import QTable
from src.policies import ConstrainedPolicy
from stable_baselines3.common.distributions import Categorical
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor,get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv

SelfPPO = TypeVar("SelfPPO", bound="ConstrainedPPO")

class ConstrainedPPO(OnPolicyAlgorithm):
    policy: ConstrainedPolicy
    old_policy: ConstrainedPolicy

    def __init__(
        self,
        policy: Union[str, Type[ConstrainedPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 20,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        clip_range: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        estimated_reward = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Discrete
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.estimated_reward = estimated_reward
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.rollout_buffer = None
        self.policy = self.policy_class( 
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.old_policy = self.policy_class( 
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        self.old_policy = self.old_policy.to(self.device)
        self.old_policy.set_training_mode(False)

        self.qf = QTable(self.policy.observation_dim,
                         self.policy.action_dim,
                         lr=0.1)
        
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
    
    def set_Psi(self,Psi):
        """Psi[i,j]= psi(o=S_i|s=S_j)"""
        self.policy.set_Psi(Psi)
        self.old_policy.set_Psi(Psi)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: rollout for n episodes
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        self.noisy_obs_cnt = th.zeros((self.policy.observation_dim)).to(self.device)
      
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # [step 1] initial stage (s_0,a_0)
            init_states = self.env.buf_obs[None].copy()
            init_actions = np.random.randint(0,5,(1,))

            _, rewards, dones, infos = env.step(init_actions)  
            epi_rewards = rewards
            discount = self.gamma
        
            #[step 2] obtain (o_k,a_k) k > 0 to calculate Q_hat
            while(not dones.all()):
                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs = np.array([infos[0]['noisy_state']])
                    self.noisy_obs_cnt[obs[0]] += 1 # NOT support multi envs
                    obs_tensor = obs_as_tensor(obs, self.device)
                    actions = self.policy.forward(obs_tensor)

                actions = actions.cpu().numpy()
                _, _, dones, infos = env.step(actions)

                est_rewards = self.estimated_reward(obs[0],actions[0])
                epi_rewards += discount * est_rewards
                #epi_rewards += discount * rewards
                discount *= self.gamma

       
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)

            n_steps += 1
            self.num_timesteps += env.num_envs
 
            self.qf.update(init_states,init_actions,epi_rewards)

        callback.update_locals(locals())
        callback.on_rollout_end()
        #print(self.qf)
        return True
    
    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Update the value table for advantage calculating
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]

        pg_losses = []
        clip_fractions = []

        continue_training = True
        qf_torch = self.qf.to_torch(self.device)
        self.policy.evaluate_value(qf_torch)
        probs_tilde = th.abs(self.noisy_obs_cnt @ self.policy.PsiINV)
        state_distribution = Categorical(probs=probs_tilde)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):         
            states = state_distribution.sample((self.batch_size,))
            actions, advantages, log_prob = self.policy.forward_pibar(states,qf_torch)

            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            old_log_prob,_ = self.old_policy.evaluate_action(states,
                                                             actions.unsqueeze(1))
            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - old_log_prob)
            
            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
         
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "cPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
