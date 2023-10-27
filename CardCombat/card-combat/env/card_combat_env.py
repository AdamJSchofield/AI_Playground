from env.game import game

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import Dict
from gymnasium.spaces import Box
from gymnasium.spaces import MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from env.cards import *

__all__ = ["ManualPolicy", "env",  "raw_env"]

ACTIONS = [CardType.FIREBALL, CardType.FIREBLAST, CardType.HEAL, None]
NUM_ITERS = 400

def env(render_mode=None):
    env = raw_env(render_mode)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-10)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):

    metadata = {"render_modes": ["human"], "name": "card_combat_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["Player" + str(r) for r in range(2)]
        
        self.game = game(self.possible_agents)

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(len(self.game.ACTIONS)) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Dict(spaces={
                "observation": MultiDiscrete(self.game.OBSERVATION_SHAPE),
                "action_mask": Box(low=0, high=1, shape=(len(self.game.ACTIONS),), dtype=np.int8)
            }) for agent in self.possible_agents
        }
        self.render_mode = render_mode
        
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        pass
    
    def observe(self, agent):
        return self.game.get_player_state(agent)

    def reset(self, seed=None, options=None):
        self.game = game(self.possible_agents)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: lambda agent: self.game.get_player_state(self.agent_name_mapping[agent]) for agent in self.agents}
        self.observations = {agent: lambda agent: self.game.get_player_state(self.agent_name_mapping[agent]) for agent in self.agents}
        self.num_moves = 0
        self.num_rounds = 0
        self.is_first_turn = True

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            if self.render_mode == "human":
                self.render_in_step(f'{self.agent_selection} is DEAD')
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        other_agent = self.agents[1 - self.agent_name_mapping[agent]]

        if self.is_first_turn:
            self.render_in_step(f'ROUND: {self.num_rounds}')
            self.render_in_step(f'\tDEBUG: "{agent}: {self.observations[agent]}\n\t\t{other_agent}: {self.observations[other_agent]}')
            self.is_first_turn = False

        
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0
        
        turn_over = self.game.take_action(action, agent, other_agent)
        self.render_in_step(f'{self.agent_selection + " plays " + CardType(action).name if action < len(CardType) else self.agent_selection + " END TURN"}')

        # Set agent states
        self.state[agent] = self.game.get_player_state(agent)
        self.state[other_agent] = self.game.get_player_state(other_agent)

        defeated_opponent = self.game.is_player_dead(other_agent)

        # Apply rewards
        if defeated_opponent:
            self.rewards[agent] = 1
            self.rewards[other_agent] = -1

        # Set observations
        for i in self.agents:
            self.observations[i] = self.state[i]

        # Terminate dead player
        self.terminations = {
            agent: self.game.is_player_dead(other_agent) for agent in self.agents
        }
        if defeated_opponent:
            self.render_in_step(f'{other_agent} is DEAD')

        # Truncate and reward both players when round limit is reached
        self.num_moves += 1
        if (self.num_moves >= NUM_ITERS):
            self.rewards[agent] = 0
            self.rewards[other_agent] = 0
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
            
        if (turn_over):
            self.is_first_turn = True
            self.num_rounds = self.num_rounds + 1
            self.agent_selection = self._agent_selector.next()
            self.game.players[self.agent_selection].reset()

    def seed(self, seed):
        np.random.seed(seed)

    def render_in_step(self, output):
        if self.render_mode == "human":
            print(output)