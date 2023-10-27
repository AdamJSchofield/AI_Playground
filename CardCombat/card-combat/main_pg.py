# System
import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple
# Packages
import gymnasium
import numpy as np
import torch

# Torch
from torchrl.modules.distributions import MaskedCategorical
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, PGPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

# Inner files
from env.card_combat_env import env as card_combat_env
from parameters import get_pg_parser

def get_args() -> argparse.Namespace:
    parser = get_pg_parser()
    return parser.parse_known_args()[0]

# Get learning agent and opponent from state or path. RandomPolicy opponenet if path not specified
def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    
    env = get_env()

    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )

    state_shape = observation_space.shape or observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    
    # Create learning agent/policy
    if agent_learn is None:
        # Network
        net = Net(
            state_shape,
            action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)

        # Optimizer
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)

        # Policy
        agent_learn = PGPolicy(
            net,
            optim,
            dist,
            discount_factor=args.gamma,
            deterministic_eval=True
        )

        # Load existing model
        if args.resume_path :
            agent_learn.load_state_dict(torch.load(args.resume_path))

    # Create opposing agent/policy
    if agent_opponent is None:
        if args.opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
        else:
            agent_opponent = RandomPolicy()

    # Set agent order
    if args.agent_id == 0:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]

    policy = MultiAgentPolicyManager(agents, env)

    return policy, optim, env.agents

# Distribution function for PG
def dist(p, mask):
    return MaskedCategorical(p, mask=mask)

# Create new environment with PettingZoo wrapper
def get_env(render_mode=None):
    return PettingZooEnv(card_combat_env(render_mode=render_mode))

# Define training parameters
def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    
    # Environment setup
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_envs)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_envs)])
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Agent setup
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )

    # Collector setup
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_episode=args.training_envs * args.batch_size)

    # Tensorboard setup
    log_path = os.path.join(args.logdir, "card_combat", "pg")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, train_interval=10, update_interval=10)

    # Callback functions
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "card_combat", "pg", "policy.pth"
            )
        torch.save(
            policy.policies[agents[args.agent_id]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        pass

    def test_fn(epoch, env_step):
        pass

    def reward_metric(rews):
        return rews[:, args.agent_id]

    # Trainer
    result = onpolicy_trainer(
        policy = policy,
        train_collector = train_collector,
        test_collector = test_collector,
        max_epoch = args.max_epoch,
        step_per_epoch = args.min_step_per_epoch,
        repeat_per_collect = args.repeat_per_collect,
        episode_per_test = args.episode_per_test,
        batch_size = args.batch_size,
        episode_per_collect = args.episode_per_collect,
        train_fn = train_fn,
        test_fn = test_fn,
        stop_fn = stop_fn,
        save_best_fn = save_best_fn,
        reward_metric = reward_metric,
        logger = logger,
        test_in_train = True
    )

    return result, policy.policies[agents[args.agent_id]]

# Watch a pretrained agent
def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])
    
    # Load policies from state or path. RandomPolicy opponent if path not specified
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent
    )

    # Set polict to evaluation mode
    policy.eval()

    # TODO: Preprocess batch function to support mutliagent PG and masked categorization without modifying libraries
    collector = Collector(policy, env, exploration_noise=True)

     # Render first step. Collector renders *after* step is called
    env.render()

    # Collect single episode
    result = collector.collect(n_episode=1, render=args.render_rate)

    # Display final reward and steps
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id].mean()}, length: {lens.mean()}")

# Entry point
if __name__ == "__main__":
    # Train the agent until max_epoch or mean_reward is reached then watch an episode
    args = get_args()
    result, agent = train_agent(args)
    watch(args, agent)