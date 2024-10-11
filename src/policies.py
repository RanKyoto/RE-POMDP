from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from torch import nn
import torch as th

from stable_baselines3.common.preprocessing import preprocess_obs

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import create_mlp, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from stable_baselines3.common.distributions import make_proba_distribution,DiagGaussianDistribution,CategoricalDistribution
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer

class ConstrainedPolicy(BasePolicy):
    """
    Policy class for static noisy state feedback.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Discrete,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):


        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=True,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            # neural network structure of policy and Q-function
            net_arch = dict(pi=[64, 64], qf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        if isinstance(observation_space, spaces.Discrete):
            self.observation_dim = observation_space.n
        if isinstance(observation_space, spaces.Box):
            self.observation_dim = observation_space.shape[0]
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        if isinstance(action_space, spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_extractor = FlattenExtractor(observation_space)
        self.action_extractor = FlattenExtractor(action_space)

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, dist_kwargs=None)

        self._build(lr_schedule)

    def set_Psi(self,Psi):
        """Psi[i,j]= psi(o=S_i|s=S_j)"""
        self.Psi = th.tensor(Psi, dtype=th.float, device= self.device)
        self.PsiINV = th.inverse(self.Psi)
    

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        latent_action_net = create_mlp(input_dim=self.observation_dim,
                                output_dim=0,  # keep the latent dim
                                net_arch=self.net_arch["pi"],
                                activation_fn=self.activation_fn,
                                squash_output=False)
    
        
        self.latent_action_net = nn.Sequential(*latent_action_net).to(self.device)
        latent_dim_pi = self.net_arch["pi"][-1]

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=0.0
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data
   
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.forward(obs=observation,deterministic=deterministic)
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Forward pass in the actor network

        :param obs: Noisy Observation !!!!! ATENTION !!!!!
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation
        features = preprocess_obs(obs,self.observation_space)
        features = self.observation_extractor(features)

        latent_pi = self.latent_action_net(features)
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")
        actions = distribution.get_actions(deterministic=deterministic)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions
    
    def forward_pibar(self, state: th.Tensor,qf:th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        This is for SNSF control

        :param state: Observation without noise !!!!! ATENTION !!!!!
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        state_features = preprocess_obs(state,self.observation_space)
        state_features = self.observation_extractor(state_features)

        obs_features = th.eye(self.observation_dim,dtype=th.float32,device=self.device)
        latent_obs = self.latent_action_net(obs_features)
        mean_actions = self.action_net(latent_obs)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        
        # Evaluate the values for the given state
        values = self.value_table[state,:].flatten()
        actions = distribution.sample().unsqueeze(0)
        
        noisy_obs_dist = th.distributions.Categorical(self.Psi[state,:])
        noisy_obs = noisy_obs_dist.sample()
        actions = actions[0,noisy_obs].unsqueeze(1)

        prob = distribution.log_prob(actions).exp()
        log_prob = (self.Psi[state,:] * prob).sum(1).log()
        with th.no_grad():
            qvalue = qf[state,actions.flatten()]
            advatanges = qvalue-values
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, advatanges, log_prob
    
    def evaluate_action(self, init_states:th.Tensor, init_actions:th.Tensor)->Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """action evaluation only for the fully observable initial state"""
        # Preprocess init_states (we use states to refer to the observations without noise)
        state_features = preprocess_obs(init_states,self.observation_space)
        state_features = self.observation_extractor(state_features)

        obs_features = th.eye(self.observation_dim,dtype=th.float32,device=self.device)
        latent_obs = self.latent_action_net(obs_features)
        mean_actions = self.action_net(latent_obs)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
    
        prob = distribution.log_prob(init_actions).exp()
        init_states = init_states.long().flatten()
        log_prob = (self.Psi[init_states,:] * prob).sum(1).log()
        entropy = log_prob.exp()*log_prob
        return log_prob, entropy

    def evaluate_value(self, qf:th.Tensor):
        with th.no_grad():     
            obs_features = th.eye(self.observation_dim,dtype=th.float32,device=self.device)
            latent_obs = self.latent_action_net(obs_features)
            mean_actions = self.action_net(latent_obs)
            distribution = self.action_dist.proba_distribution(action_logits=mean_actions)          
            probs = distribution.distribution.probs

            self.value_table = (qf*(self.Psi@probs)).sum(1).unsqueeze(1)


